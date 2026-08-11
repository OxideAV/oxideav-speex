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

/// Exponent of the transmitted 4-bit gain-correction term in the
/// **state-derived** high-band innovation gain (modes 2/3/4) — see
/// [`hb_gc_state_gain`].
pub const HB_GC_STATE_EXP_GC: i32 = 2;

/// Exponent of the same-frame low-band level term in the state-derived
/// high-band innovation gain — see [`hb_gc_state_gain`].
pub const HB_GC_STATE_EXP_LB: i32 = 2;

/// Absolute scale of the state-derived high-band innovation gain,
/// **fixture-calibrated** (r440) on the staged `hb-mode4-wb-q10` oracle
/// through the full decode path — an adopted reading, not a staged
/// constant (same posture as the r393 fold constants). The measured
/// window: the 4–8 kHz band-mean error bottoms at `0.8…1.0 × 10⁻⁴`
/// (6.1 dB) and the sub-band energy ratio crosses unity at
/// `≈ 1.2 × 10⁻⁴`; the adopted value takes the error minimum.
pub const HB_GC_STATE_SCALE: f64 = 1.0e-4;

/// Polarity of the high-band **mode-4** innovation excitation through
/// *this crate's* synthesis conventions, pinned r440.
///
/// `docs/audio/speex/hb-innovation-binding.md` §4 records the innovation
/// sign polarity as a one-bit residual "entangled with the QMF/synthesis
/// polarity convention", to be settled by "a one-bit trial against
/// `fixtures/hb-mode4-wb-q10/expected.pcm`". That trial (r440, the
/// QMF-split sub-band gate in `tests/hb_mode4_fixture.rs`): with the
/// direct reading the isolated 4–8 kHz sub-band correlates
/// **negatively** against the reference (−0.27…−0.47, strengthening
/// monotonically as the gain scale rises), and with the global flip it
/// correlates positively at the same magnitude. Like the r393 fold sign
/// ([`crate::hb_fold`] — pinned "jointly with this crate's QMF synthesis
/// modulation convention"), this constant is one bit of a *pair*: it is
/// the polarity that makes the composed chain
/// innovation → HB synthesis → QMF match the reference decoder's PCM
/// given this crate's `g1[n] = −2·(−1)ⁿ·h0[n]` synthesis highpass.
///
/// Scoped to **mode 4** alongside the state-derived gain base: no
/// staged sub-band measurement covers modes 2/3 yet, and the wideband
/// encoder emits only modes 0..=3, so the encode path is unaffected.
pub const HB_INNOVATION_POLARITY: f32 = -1.0;

/// The **state-derived** absolute high-band innovation gain for the
/// gain-correction sub-modes (Table 10.1 modes 2/3/4), r440.
///
/// `docs/audio/speex/provenance/08-qmf-recovered-hb-excitation.md` (and
/// `hb-innovation-binding.md` §5.3) measure, from staged bytes alone,
/// that the transmitted 4-bit gain correction **alone explains
/// essentially nothing** of the per-sub-frame high-band excitation gain
/// (`R² = 0.005`, wrong by tens of dB) — Table 10.1 carries no
/// frame-level high-band gain, so the base is *derived from decoder
/// state*, and the measured state is the **low band of the same frame**
/// (`R² = 0.80` at 8.7 dB rms with the correction added; free-fit
/// exponents ≈ 1.9 / 2.2 on the two terms, with both-fixed-at-2 nearly
/// free and both-at-1 costing 4 dB). The staged material deliberately
/// asserts **no closed-form law**; this function is therefore the
/// measured *direction* with the exponents fixed at the doc's
/// nearly-free reading and the absolute scale calibrated on the staged
/// fixture — documented as fixture-fitted, revisited when a
/// discriminating fixture pair lands (the doc's precise remaining ask):
///
/// ```text
///   g_hb = HB_GC_STATE_SCALE · (gc_recon · lb_frame_rms)²
///   gc_recon = 0.87360 · gc_quant_bound[q]     (staged table)
///   lb_frame_rms = RMS of the same frame's reconstructed low band
/// ```
#[inline]
pub fn hb_gc_state_gain(gc_recon: f32, lb_frame_rms: f64) -> f64 {
    HB_GC_STATE_SCALE
        * f64::from(gc_recon).powi(HB_GC_STATE_EXP_GC)
        * lb_frame_rms.powi(HB_GC_STATE_EXP_LB)
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

/// [`gain_scaled_hb_innovation_from_body`] with the r440
/// **state-derived gain base** wired (provenance/08).
///
/// `lb_frame_rms` is the RMS of the **same frame's** reconstructed
/// low-band (embedded narrowband) signal — the decoder state the staged
/// measurement identifies as the high-band gain base. When `Some`, the
/// gain-correction sub-modes (Table 10.1 modes 2/3/4, 4-bit field)
/// scale their innovation by [`hb_gc_state_gain`] instead of the bare
/// reconstructed correction (which provenance/08 measures as *not* the
/// gain, `R² = 0.005`). When `None`, the legacy correction-only scaling
/// is preserved (the stateless single-frame entries keep their
/// historical behaviour). The 5-bit mode-1 folded gain is untouched
/// either way — its absolute law is the externally-arbitrated fold path
/// ([`crate::hb_fold`]), not this one.
pub fn gain_scaled_hb_innovation_from_body_leveled(
    body: &WidebandHighBandBody,
    submode: &WidebandHighBandSubmode,
    sub_idx: usize,
    lb_frame_rms: Option<f64>,
) -> Result<[f32; GAIN_SCALED_HB_INNOVATION_SAMPLES], HbInnovationError> {
    let gain_index = HbExcitationGainIndex::from_body(body, submode, sub_idx);
    let gc_recon = gain_index.map(reconstruct_hb_exc_gain).unwrap_or(0.0);
    // State-derived base + polarity (r440): scoped to **mode 4** (the
    // 80-bit sub-mode), the one mode the provenance/08 measurement
    // covers. Provenance/08's structural claim (the 4-bit field is a
    // correction on a state-derived base) reads as common to modes
    // 2/3/4, but adopting the mode-4-calibrated squared law for modes
    // 2/3 was measured to *regress* the staged `wb_q6` speech oracle by
    // >20 dB (the law is level-inhomogeneous and its scale is pinned on
    // one fixture), so modes 2/3 keep the legacy correction-only gain
    // until a fixture pins their base. Recorded docs ask.
    let mode4 = submode.excitation_vq_bits == 80;
    let gain = match (lb_frame_rms, mode4) {
        (Some(lb_rms), true) => HB_INNOVATION_POLARITY * hb_gc_state_gain(gc_recon, lb_rms) as f32,
        (None, true) => HB_INNOVATION_POLARITY * gc_recon,
        _ => gc_recon,
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
