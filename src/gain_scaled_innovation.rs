//! **Gain-scaled fixed-codebook contribution** (round r326 scope) —
//! folds the reconstructed fixed-codebook gain into the raw innovation
//! sub-vector, producing the scaled contribution `c[n]` that enters the
//! Speex Codec Manual §8.4 excitation composition `e[n] = p[n] + c[n]`.
//!
//! ## Where this sits in the decode path
//!
//! The narrowband fixed-codebook (innovation) path has three layers,
//! each previously surfaced as an independent primitive:
//!
//! 1. **Raw sub-vector lookup** — [`crate::decode_innovation_subframe`]
//!    concatenates the per-sub-vector codebook rows into a raw
//!    `[i16; 40]` innovation vector. These are the unscaled codebook
//!    entries: the *shape* of `c[n]` but not its magnitude.
//! 2. **Scalar gain reconstruction** —
//!    [`crate::reconstruct_fixed_codebook_gain`] turns the index-only
//!    gain fields (frame-level OL excitation gain `g_frame` ×
//!    per-sub-frame correction `g_subf`) into a single reconstructed
//!    scalar magnitude `g = g_frame · g_subf` (companion §2.3 product
//!    structure).
//! 3. **This module** — multiplies the raw `[i16; 40]` sub-vector by
//!    the reconstructed scalar gain, producing the gain-scaled
//!    fixed-codebook contribution `c[n]` as `[f32; 40]`. This is the
//!    magnitude-correct `c[n]` the §8.4 composition adds to the
//!    adaptive-codebook contribution `p[n]`.
//!
//! ## Spec basis (the gain-application law)
//!
//! *The Speex Codec Manual* §8.4 ("Innovation") names the excitation
//! composition `e[n] = p[n] + c[n]`, where `c[n]` is the fixed-codebook
//! contribution. The companion §2.3 records that the **fixed-codebook
//! gain is the product** `g_frame · g_subf` applied to the innovation:
//!
//! > "Fixed-codebook gain = global frame gain `g_frame` × sub-frame gain
//! > correction `g_subf`"
//!
//! The clean-room extraction staged at
//! `docs/audio/speex/provenance/02-speex-gain-quant.md` pins the exact
//! decoder application: the innovation energy is formed as
//! `ener = MULT16_32_Q14(scal[q], ol_gain)`, i.e. the reconstructed
//! per-sub-frame correction level `scal[q]` multiplies the frame-level
//! gain `ol_gain` to give the contribution's scalar magnitude — exactly
//! the `g = g_frame · g_subf` product [`crate::reconstruct_fixed_codebook_gain`]
//! already exposes. This module multiplies that scalar through the raw
//! innovation sub-vector to produce the scaled `c[n]`.
//!
//! ## Numeric domain
//!
//! [`crate::reconstruct_fixed_codebook_gain`] produces the gain in the
//! decoder's **normalised float signal domain** (the Q15 `ol_gain` level
//! divided by `SIG_SCALING`, equivalently `exp(qe/3.5)` for the frame
//! factor — see [`crate::gain_reconstruction`] module docs). Multiplying
//! the raw `i16` innovation sub-vector by that float magnitude keeps the
//! product in `f32` rather than committing the result to a fixed-point
//! Q-format the in-repo manual does not specify for the synthesis
//! recurrence (the same Q-format-agnostic posture
//! [`crate::synthesis`] / [`crate::excitation`] adopt). The downstream
//! [`crate::SynthesisFilter`] already filters in floating point, so the
//! `[f32; 40]` contribution feeds it directly.
//!
//! ## Stream-start / silence behaviour
//!
//! When the frame is silent ([`crate::FrameInnovationGainIndex::Silence`],
//! mode 0) the reconstructed gain is `0.0`, so every scaled sample is
//! `0.0` regardless of the codebook shape — the contribution vanishes,
//! matching the silent-frame envelope the raw composition produced.
//!
//! ## What this module DOES
//!
//! * [`gain_scaled_innovation_subframe`] — scale a raw `[i16; 40]`
//!   innovation sub-vector by a reconstructed scalar gain into a
//!   `[f32; 40]` contribution.
//! * [`gain_scaled_innovation_from_indices`] — convenience that
//!   reconstructs the fixed-codebook gain from typed
//!   [`crate::FixedCodebookGainIndices`] and applies it in one call.
//! * [`gain_scaled_innovation_sample`] — single-sample helper matching
//!   the batch path elementwise.
//!
//! ## What this module DOES NOT do
//!
//! * No adaptive-codebook sum. Adding the gain-scaled `c[n]` to the
//!   pitch contribution `p[n]` is the §8.4 composition step
//!   ([`crate::raw_excitation_subframe`] performs the raw-integer
//!   analogue); composing the two float/integer domains into the final
//!   `e[n]` is the next synthesis layer.
//! * No high-band path. The wideband high band reconstructs its own
//!   excitation gain through [`crate::reconstruct_hb_exc_gain`] and a
//!   separate folding step (no shared narrowband `g_frame · g_subf`
//!   product).
//! * No encoder-side gain search.

use crate::fixed_codebook_gain::FixedCodebookGainIndices;
use crate::gain_reconstruction::reconstruct_fixed_codebook_gain;
use crate::innovation::SUBFRAME_SAMPLES;

/// Number of gain-scaled fixed-codebook samples per CELP sub-frame.
///
/// Restates [`crate::innovation::SUBFRAME_SAMPLES`] = `40` at the
/// gain-scaling layer so the public API names the dimension where the
/// consumer reads it.
pub const GAIN_SCALED_INNOVATION_SAMPLES: usize = SUBFRAME_SAMPLES;

/// Scale a raw innovation sub-vector `c_raw[n]` (`[i16; 40]`) by the
/// reconstructed fixed-codebook gain `g`, producing the gain-scaled
/// fixed-codebook contribution `c[n] = g · c_raw[n]` as `[f32; 40]`.
///
/// `gain` is the reconstructed scalar magnitude `g = g_frame · g_subf`
/// in the decoder's normalised float signal domain (the output of
/// [`crate::reconstruct_fixed_codebook_gain`]). A `0.0` gain (silent
/// frame) yields an all-zero contribution.
///
/// The product `g · c_raw[n]` is the magnitude-correct `c[n]` the
/// Speex Codec Manual §8.4 composition `e[n] = p[n] + c[n]` adds to the
/// adaptive-codebook contribution.
#[inline]
pub fn gain_scaled_innovation_subframe(
    c_raw: &[i16; GAIN_SCALED_INNOVATION_SAMPLES],
    gain: f32,
) -> [f32; GAIN_SCALED_INNOVATION_SAMPLES] {
    let mut out = [0.0f32; GAIN_SCALED_INNOVATION_SAMPLES];
    for (slot, &c) in out.iter_mut().zip(c_raw.iter()) {
        *slot = gain * f32::from(c);
    }
    out
}

/// Reconstruct the fixed-codebook gain from typed
/// [`FixedCodebookGainIndices`] and apply it to the raw innovation
/// sub-vector in one call.
///
/// Equivalent to
/// `gain_scaled_innovation_subframe(c_raw, reconstruct_fixed_codebook_gain(indices))`.
/// A silent frame ([`crate::FrameInnovationGainIndex::Silence`]) makes
/// the reconstructed gain `0.0`, so the contribution is all-zero.
#[inline]
pub fn gain_scaled_innovation_from_indices(
    c_raw: &[i16; GAIN_SCALED_INNOVATION_SAMPLES],
    indices: FixedCodebookGainIndices,
) -> [f32; GAIN_SCALED_INNOVATION_SAMPLES] {
    let gain = reconstruct_fixed_codebook_gain(indices);
    gain_scaled_innovation_subframe(c_raw, gain)
}

/// Scale a single innovation sample `c_raw` by the reconstructed gain.
///
/// Matches `gain_scaled_innovation_subframe(c, gain)[n]` for the same
/// `gain` and `c[n] = c_raw`. Provided for callers that walk samples one
/// at a time.
#[inline]
pub fn gain_scaled_innovation_sample(c_raw: i16, gain: f32) -> f32 {
    gain * f32::from(c_raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_codebook_gain::{FrameInnovationGainIndex, SubFrameInnovationGainCorrection};
    use crate::gain_reconstruction::{
        nb_ol_exc_gain_levels, nb_subframe_gain_levels_1bit, nb_subframe_gain_levels_3bit,
    };
    use crate::innovation::decode_subframe;
    use crate::submode::NARROWBAND_SUBMODES;

    /// A unit gain leaves the innovation unchanged (modulo the i16→f32
    /// widening), so the scaled contribution equals the raw sub-vector.
    #[test]
    fn unit_gain_is_identity() {
        let mut c = [0i16; GAIN_SCALED_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 11 - 200;
        }
        let out = gain_scaled_innovation_subframe(&c, 1.0);
        for (n, &v) in out.iter().enumerate() {
            assert_eq!(v, f32::from(c[n]), "n={n}");
        }
    }

    /// A zero gain (silent frame) zeroes every sample regardless of the
    /// codebook shape.
    #[test]
    fn zero_gain_zeroes_contribution() {
        let mut c = [0i16; GAIN_SCALED_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 7 - 50;
        }
        let out = gain_scaled_innovation_subframe(&c, 0.0);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Each scaled sample equals `gain · c_raw[n]` pointwise.
    #[test]
    fn pointwise_pin_matches_formula() {
        let mut c = [0i16; GAIN_SCALED_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 31 - 17;
        }
        let gain = 2.5f32;
        let out = gain_scaled_innovation_subframe(&c, gain);
        for (n, &v) in out.iter().enumerate() {
            assert_eq!(v, gain * f32::from(c[n]), "n={n}");
        }
    }

    /// Scaling is linear in the gain: scaling by `2g` is twice scaling
    /// by `g`.
    #[test]
    fn linear_in_gain() {
        let mut c = [0i16; GAIN_SCALED_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 5 - 90;
        }
        let g = 1.75f32;
        let single = gain_scaled_innovation_subframe(&c, g);
        let double = gain_scaled_innovation_subframe(&c, 2.0 * g);
        for n in 0..GAIN_SCALED_INNOVATION_SAMPLES {
            assert_eq!(double[n], 2.0 * single[n], "n={n}");
        }
    }

    /// The per-sample helper agrees with the batch path elementwise.
    #[test]
    fn per_sample_helper_matches_batch() {
        let mut c = [0i16; GAIN_SCALED_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 19 - 250;
        }
        let gain = 0.37f32;
        let batch = gain_scaled_innovation_subframe(&c, gain);
        for n in 0..GAIN_SCALED_INNOVATION_SAMPLES {
            let single = gain_scaled_innovation_sample(c[n], gain);
            assert_eq!(single, batch[n], "n={n}");
        }
    }

    /// The convenience entry point reconstructs the fixed-codebook gain
    /// (`g_frame · g_subf`) and applies it — matching the explicit
    /// two-step path.
    #[test]
    fn from_indices_matches_explicit_two_step() {
        let mut c = [0i16; GAIN_SCALED_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 3 - 40;
        }
        let indices = FixedCodebookGainIndices {
            frame: FrameInnovationGainIndex::Indexed(12),
            subframe: SubFrameInnovationGainCorrection::ThreeBit(4),
        };
        let g = reconstruct_fixed_codebook_gain(indices);
        let explicit = gain_scaled_innovation_subframe(&c, g);
        let combined = gain_scaled_innovation_from_indices(&c, indices);
        assert_eq!(explicit, combined);
    }

    /// A silent frame index drives the reconstructed gain to `0.0`, so
    /// the contribution vanishes through the convenience path too.
    #[test]
    fn silent_frame_via_indices_vanishes() {
        let mut c = [0i16; GAIN_SCALED_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 13 - 100;
        }
        let indices = FixedCodebookGainIndices {
            frame: FrameInnovationGainIndex::Silence,
            subframe: SubFrameInnovationGainCorrection::Absent,
        };
        let out = gain_scaled_innovation_from_indices(&c, indices);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// The scaled magnitude tracks the reconstructed `g_frame · g_subf`
    /// product: doubling neither factor but stepping the frame index up
    /// raises the per-sample magnitude monotonically (the OL gain table
    /// is strictly increasing).
    #[test]
    fn magnitude_tracks_frame_gain_monotonically() {
        let c = [1000i16; GAIN_SCALED_INNOVATION_SAMPLES];
        let ol = nb_ol_exc_gain_levels();
        let subf = SubFrameInnovationGainCorrection::Absent; // g_subf = 1.0
        let mut prev = f32::NEG_INFINITY;
        for i in 0..NARROWBAND_OL_LEVELS as u8 {
            let indices = FixedCodebookGainIndices {
                frame: FrameInnovationGainIndex::Indexed(i),
                subframe: subf,
            };
            let out = gain_scaled_innovation_from_indices(&c, indices);
            // With g_subf = 1.0, sample 0 = g_frame · 1000 = ol[i]·1000.
            let expected = ol[usize::from(i)] * 1000.0;
            assert!((out[0] - expected).abs() < 1e-1, "i={i}");
            assert!(out[0] > prev, "monotone i={i}");
            prev = out[0];
        }
    }

    const NARROWBAND_OL_LEVELS: usize = 32;

    /// End-to-end: a real innovation sub-vector (mode 6 documented
    /// dispatch) scaled by a reconstructed gain equals the raw sub-vector
    /// times that gain, sample for sample. Exercises the
    /// decode → reconstruct → scale chain.
    #[test]
    fn end_to_end_real_subvector_scaling() {
        let submode = NARROWBAND_SUBMODES[6];
        let indices_vq: [u32; 8] = [3, 7, 1, 12, 5, 9, 2, 6];
        let mut packed: u128 = 0;
        for &i in &indices_vq {
            packed = (packed << 8) | u128::from(i);
        }
        let c_raw = decode_subframe(&submode, packed).expect("mode 6 documented");

        // Reconstruct a 3-bit corrected gain (modes 5/6/7 carry 3-bit
        // innovation-gain corrections).
        let gain_indices = FixedCodebookGainIndices {
            frame: FrameInnovationGainIndex::Indexed(15),
            subframe: SubFrameInnovationGainCorrection::ThreeBit(6),
        };
        let g_frame = nb_ol_exc_gain_levels()[15];
        let g_subf = nb_subframe_gain_levels_3bit()[6];
        let expected_gain = g_frame * g_subf;

        let scaled = gain_scaled_innovation_from_indices(&c_raw, gain_indices);
        for n in 0..GAIN_SCALED_INNOVATION_SAMPLES {
            assert!(
                (scaled[n] - expected_gain * f32::from(c_raw[n])).abs() < 1e-2,
                "n={n}: {} vs {}",
                scaled[n],
                expected_gain * f32::from(c_raw[n])
            );
        }
    }

    /// 1-bit-correction modes (1/3/4) compose `g_frame · g_subf` through
    /// the 2-level correction table.
    #[test]
    fn one_bit_correction_composes() {
        let c = [500i16; GAIN_SCALED_INNOVATION_SAMPLES];
        let l1 = nb_subframe_gain_levels_1bit();
        for q in 0..2u8 {
            let indices = FixedCodebookGainIndices {
                frame: FrameInnovationGainIndex::Indexed(8),
                subframe: SubFrameInnovationGainCorrection::OneBit(q),
            };
            let out = gain_scaled_innovation_from_indices(&c, indices);
            let expected = nb_ol_exc_gain_levels()[8] * l1[usize::from(q)] * 500.0;
            assert!((out[0] - expected).abs() < 1e-1, "q={q}");
        }
    }

    /// Negating the innovation negates the scaled contribution (sign
    /// fidelity through the multiply).
    #[test]
    fn negation_commutes() {
        let mut c = [0i16; GAIN_SCALED_INNOVATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 9 - 80;
        }
        let g = 1.3f32;
        let out = gain_scaled_innovation_subframe(&c, g);
        let mut c_neg = [0i16; GAIN_SCALED_INNOVATION_SAMPLES];
        for (i, slot) in c_neg.iter_mut().enumerate() {
            *slot = -c[i];
        }
        let out_neg = gain_scaled_innovation_subframe(&c_neg, g);
        for n in 0..GAIN_SCALED_INNOVATION_SAMPLES {
            assert_eq!(out_neg[n], -out[n], "n={n}");
        }
    }
}
