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
//! * No mode-4 coverage: [`crate::decode_hb_subframe`] returns
//!   [`crate::HbInnovationError::Undocumented`] for mode 4 (the staged
//!   inventory does not bind its codebook), so this module surfaces the
//!   same error for mode 4.

use crate::gain_reconstruction::reconstruct_hb_exc_gain;
use crate::hb_excitation_gain::HbExcitationGainIndex;
use crate::hb_innovation::{decode_hb_subframe, HbInnovationError, HB_SUBFRAME_SAMPLES};
use crate::wideband::{WidebandHighBandBody, WidebandHighBandSubmode};

/// Number of gain-scaled high-band excitation samples per CELP
/// sub-frame. Restates [`HB_SUBFRAME_SAMPLES`] = `40` at the
/// gain-scaling layer so the public API names the dimension where the
/// consumer reads it.
pub const GAIN_SCALED_HB_INNOVATION_SAMPLES: usize = HB_SUBFRAME_SAMPLES;

/// Scale a raw high-band innovation sub-vector `c_raw[n]` (`[i16; 40]`)
/// by the reconstructed high-band excitation gain `g`, producing the
/// gain-scaled high-band excitation `e_hb[n] = g · c_raw[n]` as
/// `[f32; 40]`.
///
/// `gain` is the reconstructed scalar magnitude in the decoder's
/// normalised float signal domain (the output of
/// [`crate::reconstruct_hb_exc_gain`]). A `0.0` gain (silence) yields an
/// all-zero excitation.
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
        *slot = gain * f32::from(c);
    }
    out
}

/// Scale a single high-band innovation sample `c_raw` by the
/// reconstructed gain. Matches
/// `gain_scaled_hb_innovation_subframe(c, gain)[n]` elementwise.
#[inline]
pub fn gain_scaled_hb_innovation_sample(c_raw: i16, gain: f32) -> f32 {
    gain * f32::from(c_raw)
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
    let c_raw = decode_hb_subframe(submode, body.subframes[sub_idx].excitation_vq_index)?;
    let gain = HbExcitationGainIndex::from_body(body, submode, sub_idx)
        .map(reconstruct_hb_exc_gain)
        .unwrap_or(0.0);
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

    /// A unit gain leaves the innovation unchanged (modulo i16→f32).
    #[test]
    fn unit_gain_is_identity() {
        let mut c = [0i16; GAIN_SCALED_HB_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 13 - 200;
        }
        let out = gain_scaled_hb_innovation_subframe(&c, 1.0);
        for (n, &v) in out.iter().enumerate() {
            assert_eq!(v, f32::from(c[n]), "n={n}");
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
            assert_eq!(v, gain * f32::from(c[n]), "n={n}");
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

    /// Mode 4 surfaces the documented `Undocumented` codebook-binding gap.
    #[test]
    fn mode_4_returns_undocumented() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[4];
        let body = mk_body(0, [(1, 0); 4]);
        let r = gain_scaled_hb_innovation_from_body(&body, &submode, 0);
        assert_eq!(r, Err(HbInnovationError::Undocumented));
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

        // Explicit reference composition.
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
