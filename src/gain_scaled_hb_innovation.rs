//! **Gain-scaled high-band excitation** (round r340 scope) — folds the
//! reconstructed high-band excitation gain into the raw high-band
//! innovation sub-vector, producing the magnitude-correct high-band
//! excitation `e_hb[n]` that drives the high-band LPC synthesis filter.
//!
//! ## Where this sits in the wideband decode path
//!
//! The wideband high band reconstructs an excitation signal exactly the
//! way the narrowband path does, with one simplification: *The Speex
//! Codec Manual* §10.2 states there is **no pitch prediction in the high
//! band**, so the §8.4 composition `e[n] = p[n] + c[n]` collapses to
//! `e_hb[n] = c_hb[n]` — the gain-scaled fixed-codebook (innovation)
//! contribution is the *entire* high-band excitation. This module is the
//! high-band analogue of the narrowband
//! [`crate::gain_scaled_innovation_subframe`], minus the
//! adaptive-codebook term.
//!
//! The three high-band layers, each already an independent primitive:
//!
//! 1. **Raw sub-vector lookup** —
//!    [`crate::decode_hb_subframe`] concatenates the per-sub-vector
//!    high-band codebook rows (`HbSv8_128` / `HbSv10_32`, with the
//!    `HbSv8_128` sign bit applied) into a raw `[i16; 40]` innovation
//!    vector — the *shape* of `c_hb[n]` but not its magnitude.
//! 2. **Scalar gain reconstruction** —
//!    [`crate::reconstruct_hb_exc_gain`] turns the index-only
//!    per-sub-frame [`crate::HbExcitationGainIndex`] (Table 10.1's
//!    `Excitation gain` field; 0 / 5 / 4 / 4 / 4 bits for modes 0..=4)
//!    into a single reconstructed scalar magnitude through the staged
//!    32-level folded-gain (mode 1) / 16-level gain-correction
//!    (modes 2..=4) tables.
//! 3. **This module** — multiplies the raw `[i16; 40]` sub-vector by the
//!    reconstructed scalar gain, producing the gain-scaled high-band
//!    excitation `e_hb[n] = g_hb · c_hb_raw[n]` as `[f32; 40]`. Because
//!    there is no high-band pitch term, this is the final high-band
//!    excitation the high-band synthesis filter `1/A_hb(z)` consumes.
//!
//! ## Spec basis (the gain-application law)
//!
//! * *The Speex Codec Manual* §10.3 ("Excitation Quantization"): *"The
//!   high-band excitation is coded in the same way as for narrowband."*
//!   The narrowband excitation magnitude is the codebook gain times the
//!   raw innovation sub-vector (companion §2.3), so the high band applies
//!   its own reconstructed `Excitation gain` factor the same way.
//! * *The Speex Codec Manual* §10.2 ("Pitch Prediction"): *"there's no
//!   pitch prediction for the high-band"* — so no adaptive-codebook term
//!   is summed; the scaled innovation is the whole excitation.
//! * Table 10.1: the high band carries exactly **one** gain field per
//!   sub-frame (no frame-level `OL Exc gain`, unlike narrowband
//!   Table 9.1), so the high-band gain is a single per-sub-frame scalar,
//!   not a `g_frame · g_subf` product.
//!
//! ## Numeric domain
//!
//! [`crate::reconstruct_hb_exc_gain`] produces the gain in the decoder's
//! normalised float signal domain (the same domain
//! [`crate::reconstruct_fixed_codebook_gain`] uses for the narrowband
//! path — see [`crate::gain_reconstruction`] module docs). Multiplying
//! the raw `i16` high-band innovation sub-vector by that float magnitude
//! keeps the product in `f32`, matching the Q-format-agnostic posture of
//! [`crate::gain_scaled_innovation`] / [`crate::synthesis`]; the downstream
//! high-band synthesis filter filters in floating point and consumes the
//! `[f32; 40]` directly.
//!
//! ## Silence behaviour
//!
//! For high-band modes 0 / 1 (silence / gain-only,
//! `excitation_vq_bits == 0`) [`crate::decode_hb_subframe`] returns the
//! all-zero sub-vector, so the scaled excitation is all-zero regardless
//! of the gain. For mode 0 the reconstructed gain is itself `0.0`
//! ([`crate::HbExcitationGainIndex::Absent`]). Either way the high-band
//! contribution vanishes for a silent high band.
//!
//! ## What this module DOES
//!
//! * [`gain_scaled_hb_innovation_subframe`] — scale a raw `[i16; 40]`
//!   high-band innovation sub-vector by a reconstructed scalar gain into
//!   a `[f32; 40]` high-band excitation.
//! * [`gain_scaled_hb_innovation_from_body`] — convenience that decodes
//!   the raw sub-vector + resolves and reconstructs the per-sub-frame
//!   high-band gain off a parsed [`crate::WidebandHighBandBody`] and
//!   applies it in one call.
//! * [`gain_scaled_hb_innovation_sample`] — single-sample helper.
//!
//! ## What this module DOES NOT do
//!
//! * No high-band pitch term (none exists — §10.2).
//! * No high-band synthesis filtering (that is the
//!   [`crate::hb_synthesis`] layer that consumes this excitation).
//! * No QMF recombination of the low + high half-bands (that is the
//!   wideband-synthesis-assembly layer).
//! * Mode 4 (80-bit two-stage) is decoded here via the float two-stage
//!   path ([`crate::hb_innovation::decode_hb_subframe_mode4_f32`],
//!   stage 2 at weight 0.4) rather than the `[i16; 40]`
//!   [`crate::decode_hb_subframe`] lookup, then scaled by the same gain
//!   / `INNOVATION_CODEBOOK_SCALE` — see
//!   `docs/audio/speex/hb-innovation-binding.md`. A residual on the
//!   absolute per-frame HB-innovation gain law remains (crate README).

use crate::gain_reconstruction::reconstruct_hb_exc_gain;
use crate::gain_scaled_innovation::INNOVATION_CODEBOOK_SCALE;
use crate::hb_excitation_gain::HbExcitationGainIndex;
use crate::hb_innovation::{decode_hb_subframe, HbInnovationError, HB_SUBFRAME_SAMPLES};
use crate::wideband::{WidebandHighBandBody, WidebandHighBandSubmode};

/// Polarity of the high-band innovation excitation (modes 2/3/4)
/// through *this crate's* synthesis conventions.
///
/// One bit of a convention *pair* with the QMF synthesis modulation
/// `g1[n] = −2·(−1)ⁿ·h0[n]` (like the r393 fold sign): the r450
/// crafted-stream probes measure the reference high band at
/// correlation −0.9998 against this crate's prior mode-4 decode, so the
/// r440 mode-4 flip is reverted — the direct (positive) reading is the
/// one that matches the reference PCM through this crate's chain, for
/// all three innovation sub-modes.
pub const HB_INNOVATION_POLARITY: f32 = 1.0;

/// The crossover-anchored absolute high-band innovation gain for the
/// gain-correction sub-modes (Table 10.1 modes 2/3/4), measured r450 by
/// crafted-bitstream black-box probing (`tests/fixtures/hb-gain-probes/
/// NOTES.md`): with the crate's own writers emitting wideband streams
/// whose per-sub-frame gain index, innovation content, low-band level,
/// low-band envelope, low-band innovation and high-band envelope were
/// varied **one at a time**, the reference decode's per-sub-frame
/// high-band gain is
///
/// ```text
///   g_hb = gc_recon · |A_hb(π)| · rms(e_lb) / |A_lb(π)|
///   gc_recon  = 0.87360 · gc_quant_bound[q]      (staged table + multiplier)
///   rms(e_lb) = same sub-frame's low-band excitation RMS
///   |A_lb(π)| = same sub-frame's interpolated low-band LPC analysis
///               response at the 4 kHz QMF crossover (z = −1)
///   |A_hb(π)| = same sub-frame's interpolated high-band LPC analysis
///               response at its own π edge — the same 4 kHz crossover
///               after the QMF fold
/// ```
///
/// with **no further constant**: an eight-point sweep of the high-band
/// LSP pair puts `g/(gc_bound·rms(e_lb)/|A_lb(π)|·|A_hb(π)|)` at
/// 0.852…0.884 (mean ≈ 0.873) over a 66× `|A_hb(π)|` range — the staged
/// `0.87360` reconstruction multiplier itself. Read as spectral
/// continuity: since the synthesis filter divides by `A_hb`, the
/// reconstructed high band's amplitude at the fold equals
/// `gc_recon ·` (the low band's spectral amplitude at the crossover) —
/// the transmitted 4-bit index codes the *ratio of the two bands at
/// 4 kHz*, mirroring the mode-1 folded-gain law's crossover shaping
/// (`crate::hb_fold`).
///
/// Measured properties (each pinned by a dedicated probe family): the
/// correction enters **linearly** (constant-gc grids give
/// `g/gc_bound[q]` flat to 0.3 % over the full 4-bit range), the
/// innovation energy does not enter (codebook-row sweeps move the
/// fitted gain < 1 %), the response is instantaneous with **no
/// backward-adaptive memory** (steps in any driver settle within the
/// same sub-frame; 32 frames of all-zero innovation leave the gain
/// unchanged on resumption), and one law with one constant fits modes
/// 2, 3 and 4. The earlier r440 fixture-fitted `(gc·lb_rms)²` reading —
/// and provenance/08's state-derivation direction — is superseded: the
/// apparent squared law was natural-speech co-variation between the low
/// band's level and its crossover response. Residual: the two deepest
/// swept envelopes (`|A_hb(π)| ≤ 0.06`) sit 7…40 % off the law through
/// this crate's `|A(π)|` — the same near-degenerate-envelope divergence
/// `hb-folded-gain.md` §7.6 records for the fold path.
///
/// `lb_base` is the per-sub-frame low-band crossover amplitude
/// `rms(e_lb)/|A_lb(π)|` (see
/// [`crate::NarrowbandDecoder::last_crossover_response`]); `hb_api` is
/// `|A_hb(π)|` from the same sub-frame's interpolated high-band LPC.
#[inline]
pub fn hb_gc_crossover_gain(gc_recon: f32, lb_base: f64, hb_api: f64) -> f64 {
    f64::from(gc_recon) * hb_api * lb_base
}

/// Number of gain-scaled high-band excitation samples per CELP
/// sub-frame. Restates [`HB_SUBFRAME_SAMPLES`] = `40` at the
/// gain-scaling layer so the public API names the dimension where the
/// consumer reads it.
pub const GAIN_SCALED_HB_INNOVATION_SAMPLES: usize = HB_SUBFRAME_SAMPLES;

/// Scale a raw high-band innovation sub-vector `c_raw[n]` (`[i16; 40]`)
/// by the reconstructed high-band excitation gain `g`, producing the
/// gain-scaled high-band excitation
/// `e_hb[n] = g · c_raw[n] · `[`INNOVATION_CODEBOOK_SCALE`]` ` as
/// `[f32; 40]`.
///
/// `gain` is the reconstructed scalar magnitude in the decoder's
/// normalised float signal domain (the output of
/// [`crate::reconstruct_hb_exc_gain`]). A `0.0` gain (silence) yields an
/// all-zero excitation. The staged high-band codebooks are the same
/// `signed char` Q5-fraction rows as the narrowband ones
/// (`tables/hb-innovation-cdbk-*.meta`), so the shared
/// [`INNOVATION_CODEBOOK_SCALE`] = 1/32 normalisation applies here too
/// (see [`crate::gain_scaled_innovation`] module docs for the external
/// calibration).
///
/// Because *The Speex Codec Manual* §10.2 specifies no high-band pitch
/// prediction, this product is the **entire** high-band excitation that
/// the high-band synthesis filter `1/A_hb(z)` consumes — there is no
/// `e[n] = p[n] + c[n]` sum in the high band.
#[inline]
pub fn gain_scaled_hb_innovation_subframe(
    c_raw: &[i16; GAIN_SCALED_HB_INNOVATION_SAMPLES],
    gain: f32,
) -> [f32; GAIN_SCALED_HB_INNOVATION_SAMPLES] {
    let mut out = [0.0f32; GAIN_SCALED_HB_INNOVATION_SAMPLES];
    for (slot, &c) in out.iter_mut().zip(c_raw.iter()) {
        *slot = gain * INNOVATION_CODEBOOK_SCALE * f32::from(c);
    }
    out
}

/// Scale a single high-band innovation sample `c_raw` by the
/// reconstructed gain. Matches
/// `gain_scaled_hb_innovation_subframe(c, gain)[n]` elementwise.
#[inline]
pub fn gain_scaled_hb_innovation_sample(c_raw: i16, gain: f32) -> f32 {
    gain * INNOVATION_CODEBOOK_SCALE * f32::from(c_raw)
}

/// Decode the raw high-band innovation sub-vector for sub-frame
/// `sub_idx` (`0..4`) off a parsed [`WidebandHighBandBody`], resolve and
/// reconstruct that sub-frame's high-band excitation gain, and return
/// the gain-scaled high-band excitation `e_hb[n]` as `[f32; 40]`.
///
/// This is the one-call high-band counterpart of the narrowband
/// [`crate::gain_scaled_innovation_from_indices`]. It composes:
///
/// * [`crate::decode_hb_subframe`] — raw `[i16; 40]` innovation shape
///   (sign-applied for `HbSv8_128`).
/// * [`crate::HbExcitationGainIndex::from_body`] +
///   [`crate::reconstruct_hb_exc_gain`] — the per-sub-frame scalar gain.
/// * [`gain_scaled_hb_innovation_subframe`] — the scaling.
///
/// Returns [`HbInnovationError::Undocumented`] for high-band mode 4
/// (whose codebook binding the staged inventory does not pin) and
/// [`HbInnovationError::IndexOutOfRange`] for a malformed index. For a
/// hand-built non-conforming sub-mode gain budget (none of {0, 4, 5})
/// the gain resolves to `0.0` and the excitation is all-zero — the same
/// graceful degradation [`crate::HbExcitationGainIndex::from_body`]
/// exposes (it returns `None`, treated here as "no gain").
pub fn gain_scaled_hb_innovation_from_body(
    body: &WidebandHighBandBody,
    submode: &WidebandHighBandSubmode,
    sub_idx: usize,
) -> Result<[f32; GAIN_SCALED_HB_INNOVATION_SAMPLES], HbInnovationError> {
    gain_scaled_hb_innovation_from_body_leveled(body, submode, sub_idx, None)
}

/// [`gain_scaled_hb_innovation_from_body`] with the r450
/// **crossover-anchored gain base** wired.
///
/// `hb_gain_base` is this sub-frame's
/// `(rms(e_lb)/|A_lb(π)|, |A_hb(π)|)` pair — the low band's crossover
/// amplitude and the high-band envelope's own crossover response, the
/// decoder state the r450 crafted-stream probes identify as the base of
/// the modes-2/3/4 absolute innovation gain (see
/// [`hb_gc_crossover_gain`]). When `Some`, the gain-correction
/// sub-modes scale their innovation by the measured absolute law; when
/// `None`, the legacy correction-only scaling is preserved (the
/// stateless single-frame entries keep their historical behaviour).
/// The 5-bit mode-1 folded gain is untouched either way — its absolute
/// law is the externally-arbitrated fold path ([`crate::hb_fold`]),
/// not this one.
pub fn gain_scaled_hb_innovation_from_body_leveled(
    body: &WidebandHighBandBody,
    submode: &WidebandHighBandSubmode,
    sub_idx: usize,
    hb_gain_base: Option<(f64, f64)>,
) -> Result<[f32; GAIN_SCALED_HB_INNOVATION_SAMPLES], HbInnovationError> {
    let gain_index = HbExcitationGainIndex::from_body(body, submode, sub_idx);
    let gc_recon = gain_index.map(reconstruct_hb_exc_gain).unwrap_or(0.0);
    let gain = match hb_gain_base {
        Some((lb_base, hb_api)) => {
            HB_INNOVATION_POLARITY * hb_gc_crossover_gain(gc_recon, lb_base, hb_api) as f32
        }
        None => gc_recon,
    };
    // Mode 4 (80-bit two-stage): float codebook shape (sign + 0.4 stage 2),
    // then the shared gain / codebook-scale (docs hb-innovation-binding.md).
    if submode.excitation_vq_bits == 80 {
        let shape = crate::hb_innovation::decode_hb_subframe_mode4_f32(
            body.subframes[sub_idx].excitation_vq_index,
        );
        let mut out = [0.0f32; GAIN_SCALED_HB_INNOVATION_SAMPLES];
        for (slot, &s) in out.iter_mut().zip(shape.iter()) {
            *slot = gain * INNOVATION_CODEBOOK_SCALE * s;
        }
        return Ok(out);
    }
    let c_raw = decode_hb_subframe(submode, body.subframes[sub_idx].excitation_vq_index)?;
    Ok(gain_scaled_hb_innovation_subframe(&c_raw, gain))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wideband::{HighBandSubFrameIndices, WIDEBAND_HIGH_BAND_SUBMODES};

    fn mk_body(lsp_index: u16, per_subframe: [(u8, u128); 4]) -> WidebandHighBandBody {
        let mut subframes = [HighBandSubFrameIndices::default(); 4];
        for (sf, (gain, vq)) in subframes.iter_mut().zip(per_subframe) {
            sf.excitation_gain_index = gain;
            sf.excitation_vq_index = vq;
        }
        WidebandHighBandBody {
            lsp_index,
            subframes,
        }
    }

    /// A unit gain applies exactly the shared Q5 row normalisation.
    #[test]
    fn unit_gain_applies_q5_row_normalisation() {
        let mut c = [0i16; GAIN_SCALED_HB_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 13 - 200;
        }
        let out = gain_scaled_hb_innovation_subframe(&c, 1.0);
        for (n, &v) in out.iter().enumerate() {
            assert_eq!(v, f32::from(c[n]) / 32.0, "n={n}");
        }
    }

    /// A zero gain (silence) zeroes every sample.
    #[test]
    fn zero_gain_zeroes_excitation() {
        let mut c = [0i16; GAIN_SCALED_HB_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 7 - 50;
        }
        let out = gain_scaled_hb_innovation_subframe(&c, 0.0);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Each scaled sample equals `gain · c_raw[n]` pointwise.
    #[test]
    fn pointwise_pin_matches_formula() {
        let mut c = [0i16; GAIN_SCALED_HB_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 23 - 17;
        }
        let gain = 1.5f32;
        let out = gain_scaled_hb_innovation_subframe(&c, gain);
        for (n, &v) in out.iter().enumerate() {
            assert_eq!(
                v,
                gain * INNOVATION_CODEBOOK_SCALE * f32::from(c[n]),
                "n={n}"
            );
        }
    }

    /// Scaling is linear in the gain.
    #[test]
    fn linear_in_gain() {
        let mut c = [0i16; GAIN_SCALED_HB_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 5 - 90;
        }
        let g = 1.25f32;
        let single = gain_scaled_hb_innovation_subframe(&c, g);
        let double = gain_scaled_hb_innovation_subframe(&c, 2.0 * g);
        for n in 0..GAIN_SCALED_HB_INNOVATION_SAMPLES {
            assert_eq!(double[n], 2.0 * single[n], "n={n}");
        }
    }

    /// The per-sample helper agrees with the batch path elementwise.
    #[test]
    fn per_sample_helper_matches_batch() {
        let mut c = [0i16; GAIN_SCALED_HB_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 17 - 250;
        }
        let gain = 0.42f32;
        let batch = gain_scaled_hb_innovation_subframe(&c, gain);
        for n in 0..GAIN_SCALED_HB_INNOVATION_SAMPLES {
            assert_eq!(
                gain_scaled_hb_innovation_sample(c[n], gain),
                batch[n],
                "n={n}"
            );
        }
    }

    /// Modes 0 / 1 (silence / gain-only) — the raw innovation is
    /// all-zero (no excitation-VQ field), so the gain-scaled high-band
    /// excitation is all-zero for every sub-frame regardless of the gain
    /// index.
    #[test]
    fn silence_modes_produce_zero_excitation() {
        for mode_id in [0u8, 1] {
            let submode = WIDEBAND_HIGH_BAND_SUBMODES[mode_id as usize];
            // Non-trivial gain index, but the raw sub-vector is zero.
            let body = mk_body(0, [(13, 0), (7, 0), (31, 0), (1, 0)]);
            for sub_idx in 0..4 {
                let e = gain_scaled_hb_innovation_from_body(&body, &submode, sub_idx)
                    .expect("silence modes decode");
                assert!(
                    e.iter().all(|&v| v == 0.0),
                    "mode {mode_id} sub {sub_idx} must be silent"
                );
            }
        }
    }

    /// Mode 4 now decodes via the two-stage (0.4-weighted) `sv8-128`
    /// binding (`docs/audio/speex/hb-innovation-binding.md`): a non-zero
    /// codeword + gain produces a non-silent excitation.
    #[test]
    fn mode_4_decodes_two_stage() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[4];
        // Stage 1 group 0 = index 5 (sign 0); everything else zero (index
        // 0). 80-bit field: top 8 bits = 0x05, rest 0.
        let vq: u128 = 0x05u128 << 72;
        let body = mk_body(0, [(5, vq), (0, 0), (0, 0), (0, 0)]);
        let e = gain_scaled_hb_innovation_from_body(&body, &submode, 0).unwrap();
        assert!(e.iter().all(|v| v.is_finite()), "finite");
        assert!(
            e.iter().any(|&v| v != 0.0),
            "mode 4 must produce excitation"
        );
        // Stage 2 adds a 0.4-weighted refinement: setting the stage-2
        // group changes the result (mode 4 ≠ mode 3).
        let vq2: u128 = (0x05u128 << 72) | (0x07u128 << 32); // stage-2 group 0 = idx 7
        let body2 = mk_body(0, [(5, vq2), (0, 0), (0, 0), (0, 0)]);
        let e2 = gain_scaled_hb_innovation_from_body(&body2, &submode, 0).unwrap();
        assert!(e2 != e, "stage 2 refinement must change the excitation");
    }

    /// Mode 2 (4 × `HbSv10_32`) — the convenience path equals the
    /// explicit decode → reconstruct → scale composition, and a non-zero
    /// gain index produces a non-silent excitation.
    #[test]
    fn mode_2_convenience_matches_explicit_composition() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
        // Pack 4 × 5-bit indices into the 20-bit VQ field.
        let indices: [u128; 4] = [3, 17, 9, 31];
        let mut vq: u128 = 0;
        for &idx in &indices {
            vq = (vq << 5) | idx;
        }
        // Mode 2 → 4-bit gain field.
        let body = mk_body(0, [(5, vq), (0, vq), (0, vq), (0, vq)]);

        let got = gain_scaled_hb_innovation_from_body(&body, &submode, 0).unwrap();

        // Explicit reference composition (mode 2 keeps the legacy law —
        // the r440 polarity/state-base is scoped to mode 4).
        let c_raw = decode_hb_subframe(&submode, vq).unwrap();
        let gain =
            reconstruct_hb_exc_gain(HbExcitationGainIndex::from_body(&body, &submode, 0).unwrap());
        let want = gain_scaled_hb_innovation_subframe(&c_raw, gain);
        assert_eq!(got, want);

        // With a non-zero gain and non-zero codebook rows, at least one
        // sample is non-zero (the excitation is not silent).
        let nonzero = got.iter().any(|&v| v != 0.0) || gain == 0.0;
        assert!(nonzero, "non-silent gain should produce non-silent e_hb");
    }
}
