//! **Float-domain excitation composition** (round r337 scope) — joins the
//! two gain-scaled contributions into the final per-sub-frame excitation
//! `e[n] = p[n] + c[n]` of *The Speex Codec Manual* §8.4 / CELP companion
//! §2.3, the float analogue of the r244 raw-integer
//! [`crate::raw_excitation_subframe`].
//!
//! ## Where this sits in the decode path
//!
//! The narrowband excitation is the sum of two contributions, each now
//! surfaced in the **same normalised float signal domain**:
//!
//! 1. **Adaptive-codebook (pitch) contribution `p[n]`** — the r331
//!    [`crate::gain_scaled_pitch_subframe`] output, the §9.2 long-term
//!    predictor dot product `g0·e[n−T−1] + g1·e[n−T] + g2·e[n−T+1]`
//!    divided by the staged **Q6** pitch-gain scaling
//!    ([`crate::PITCH_GAIN_SCALING`] = `64`), as `[f32; 40]`.
//! 2. **Fixed-codebook (innovation) contribution `c[n]`** — the r326
//!    [`crate::gain_scaled_innovation_subframe`] output, the raw
//!    `[i16; 40]` innovation sub-vector multiplied by the reconstructed
//!    fixed-codebook gain `g = g_frame · g_subf`, as `[f32; 40]`.
//!
//! Both contributions already occupy one shared float signal domain (see
//! the [`crate::gain_scaled_pitch`] module docs for the Q6 → float
//! division argument that brings `p[n]` into the same domain the
//! [`crate::gain_scaled_innovation`] `c[n]` occupies). This module
//! performs the final domain-coherent **per-sample sum** producing the
//! magnitude-correct excitation `e[n]` the synthesis filter consumes.
//!
//! ## Spec basis (the composition law)
//!
//! *The Speex Codec Manual* §8.4 ("Innovation") names the excitation
//! composition `e[n] = p[n] + c[n]`, where `p[n]` is the
//! long-term-predictor (adaptive-codebook) contribution and `c[n]` is
//! the fixed-codebook (innovation) contribution. The CELP companion §2.3
//! ("Innovation (fixed codebook)") paraphrases the same identity:
//!
//! > "Final excitation `e[n] = p[n] + c[n]`, c[n] from the fixed
//! > codebook"
//!
//! The r244 [`crate::raw_excitation_subframe`] already evaluated this
//! identity in **raw-integer** units, but its two terms carried different
//! un-divided Q-formats (the pitch dot product's un-divided Q6 gain and
//! the un-scaled `i16` innovation), so its sum was Q-format-agnostic and
//! *not* magnitude-correct. With the r331 Q6 division and the r326 gain
//! multiplication both terms are now in the one normalised float signal
//! domain, so this module's sum is the *magnitude-correct* excitation —
//! the closing step the README's "Not yet supported" tail flagged.
//!
//! ## Numeric domain
//!
//! Both inputs are `[f32; 40]` in the shared normalised float signal
//! domain, so the sum is a plain elementwise `f32` add — no Q-format
//! shift, matching the floating-point posture of the downstream
//! [`crate::SynthesisFilter`], which already filters in floating point
//! and consumes `e[n]` directly.
//!
//! The per-sample magnitudes are analytically bounded: `|p[n]|` by the
//! r331 argument (`≤ 2.5 × 10⁵` after the Q6 division of the
//! `3 × 159 × i16::MAX ≈ 1.6 × 10⁷` raw dot product), and `|c[n]|` by
//! the reconstructed gain `g` times `i16::MAX`. The sum stays well inside
//! `f32`'s dynamic range, so no overflow / non-finite value can arise
//! from any in-spec input.
//!
//! ## Stream-start / silence behaviour
//!
//! * **Stream start.** With the all-zero default
//!   [`crate::ExcitationBuffer`], the r331 `p[n]` term is identically
//!   `0.0` across the whole sub-frame for any tap triple, so `e[n]`
//!   equals the gain-scaled innovation `c[n]` verbatim — the same
//!   "envelope follows the first-frame innovation" property the r244 raw
//!   composition guaranteed, preserved through the float sum.
//! * **Silence.** A silent frame drives the reconstructed fixed-codebook
//!   gain to `0.0` (so `c[n] = 0.0`) and the silence taps
//!   ([`crate::PitchGainTaps::SILENCE`]) drive `p[n] = 0.0`, so the
//!   composed `e[n]` is identically `0.0` across the sub-frame.
//!
//! ## What this module DOES
//!
//! * [`gain_scaled_excitation_subframe`] — per-sample float sum of a
//!   gain-scaled pitch contribution `p[n]` (`[f32; 40]`) with a
//!   gain-scaled innovation contribution `c[n]` (`[f32; 40]`) into the
//!   composed excitation `e[n]` (`[f32; 40]`).
//! * [`gain_scaled_excitation_sample`] — single-sample helper matching
//!   the batch path elementwise.
//! * [`GAIN_SCALED_EXCITATION_SAMPLES`] — the restated `40` constant.
//!
//! ## What this module DOES NOT do
//!
//! * No gain reconstruction or Q6 division. Both inputs are already the
//!   magnitude-correct float contributions from
//!   [`crate::gain_scaled_pitch_subframe`] /
//!   [`crate::gain_scaled_innovation_subframe`]; this module only sums
//!   them.
//! * No excitation-buffer feedback. Pushing the composed `e[n]` back into
//!   the [`crate::ExcitationBuffer`] for the next sub-frame's r234/r331
//!   adaptive-codebook lookup is the caller's step (the buffer stores
//!   `i16`; the rounding/saturation policy for the float → `i16`
//!   reduction is the synthesis layer's choice and not pinned by the
//!   staged material).
//! * No synthesis filtering. Running `e[n]` through the
//!   [`crate::SynthesisFilter`] is the next layer.
//! * No high-band path. Per §10.2 the wideband high band has no adaptive
//!   codebook, so its excitation is the gain-scaled high-band innovation
//!   alone (no `p[n] + c[n]` sum).
//! * No encoder-side analysis-by-synthesis loop.

use crate::innovation::SUBFRAME_SAMPLES;

/// Number of composed-excitation samples per CELP sub-frame.
///
/// Restates [`crate::innovation::SUBFRAME_SAMPLES`] = `40` at the
/// composition layer so the public API names the dimension where the
/// consumer reads it (mirrors [`crate::GAIN_SCALED_PITCH_SAMPLES`] /
/// [`crate::GAIN_SCALED_INNOVATION_SAMPLES`]).
pub const GAIN_SCALED_EXCITATION_SAMPLES: usize = SUBFRAME_SAMPLES;

/// Compose one CELP sub-frame's float-domain excitation
/// `e[n] = p[n] + c[n]`.
///
/// Inputs:
///
/// * `pitch` — the r331 gain-scaled adaptive-codebook (pitch)
///   contribution `p[n]` (`[f32; 40]`) from
///   [`crate::gain_scaled_pitch_subframe`].
/// * `innovation` — the r326 gain-scaled fixed-codebook (innovation)
///   contribution `c[n]` (`[f32; 40]`) from
///   [`crate::gain_scaled_innovation_subframe`].
///
/// Both inputs are in the same normalised float signal domain, so the
/// result is the magnitude-correct excitation `e[n]` (`[f32; 40]`) the
/// [`crate::SynthesisFilter`] consumes — the closing step of *The Speex
/// Codec Manual* §8.4 / CELP companion §2.3 `e[n] = p[n] + c[n]`.
#[inline]
pub fn gain_scaled_excitation_subframe(
    pitch: &[f32; GAIN_SCALED_EXCITATION_SAMPLES],
    innovation: &[f32; GAIN_SCALED_EXCITATION_SAMPLES],
) -> [f32; GAIN_SCALED_EXCITATION_SAMPLES] {
    let mut out = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
    for (n, slot) in out.iter_mut().enumerate() {
        *slot = pitch[n] + innovation[n];
    }
    out
}

/// Compose one sample of the float-domain excitation at sub-frame
/// position `n`.
///
/// Convenience for callers that walk samples one at a time (e.g. an
/// encoder's analysis-by-synthesis loop). The result matches
/// `gain_scaled_excitation_subframe(pitch, innovation)[n]` for
/// `pitch[n] = p_n` and `innovation[n] = c_n`.
///
/// `n` participates only as a debug guard; the sum itself is index-free
/// once the two input scalars are in hand.
#[inline]
pub fn gain_scaled_excitation_sample(n: usize, p_n: f32, c_n: f32) -> f32 {
    debug_assert!(
        n < GAIN_SCALED_EXCITATION_SAMPLES,
        "sub-frame sample position out of range"
    );
    let _ = n;
    p_n + c_n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_codebook::ExcitationBuffer;
    use crate::fixed_codebook_gain::{
        FixedCodebookGainIndices, FrameInnovationGainIndex, SubFrameInnovationGainCorrection,
    };
    use crate::gain_scaled_innovation::gain_scaled_innovation_from_indices;
    use crate::gain_scaled_pitch::gain_scaled_pitch_subframe;
    use crate::innovation::decode_subframe;
    use crate::pitch_gain::PitchGainTaps;
    use crate::submode::NARROWBAND_SUBMODES;

    /// Both contributions zero → excitation zero.
    #[test]
    fn both_zero_yields_zero() {
        let p = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        let c = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        let out = gain_scaled_excitation_subframe(&p, &c);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Zero pitch → excitation equals the innovation verbatim
    /// (stream-start envelope property).
    #[test]
    fn zero_pitch_yields_innovation() {
        let p = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        let mut c = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as f32) * 1.5 - 30.0;
        }
        let out = gain_scaled_excitation_subframe(&p, &c);
        for (n, &v) in out.iter().enumerate() {
            assert_eq!(v, c[n], "n={n}");
        }
    }

    /// Zero innovation → excitation equals the pitch contribution
    /// verbatim.
    #[test]
    fn zero_innovation_yields_pitch() {
        let mut p = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        for (i, slot) in p.iter_mut().enumerate() {
            *slot = (i as f32) * -2.25 + 100.0;
        }
        let c = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        let out = gain_scaled_excitation_subframe(&p, &c);
        for (n, &v) in out.iter().enumerate() {
            assert_eq!(v, p[n], "n={n}");
        }
    }

    /// Pointwise pin: every output element equals `p[n] + c[n]`.
    #[test]
    fn pointwise_pin_matches_formula() {
        let mut p = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        let mut c = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        for i in 0..GAIN_SCALED_EXCITATION_SAMPLES {
            p[i] = (i as f32) * 12.5 - 60.0;
            c[i] = (i as f32) * -3.0 + 17.0;
        }
        let out = gain_scaled_excitation_subframe(&p, &c);
        for n in 0..GAIN_SCALED_EXCITATION_SAMPLES {
            assert_eq!(out[n], p[n] + c[n], "n={n}");
        }
    }

    /// Per-sample helper agrees with the batch path elementwise.
    #[test]
    fn per_sample_helper_matches_batch() {
        let mut p = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        let mut c = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        for i in 0..GAIN_SCALED_EXCITATION_SAMPLES {
            p[i] = (i as f32) * 7.0 - 40.0;
            c[i] = (i as f32) * 0.5 + 3.0;
        }
        let batch = gain_scaled_excitation_subframe(&p, &c);
        for n in 0..GAIN_SCALED_EXCITATION_SAMPLES {
            let single = gain_scaled_excitation_sample(n, p[n], c[n]);
            assert_eq!(single, batch[n], "n={n}");
        }
    }

    /// The composition is commutative (float addition): swapping the two
    /// contributions yields the same excitation.
    #[test]
    fn commutative() {
        let mut p = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        let mut c = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        for i in 0..GAIN_SCALED_EXCITATION_SAMPLES {
            p[i] = (i as f32) * 1.1 - 5.0;
            c[i] = (i as f32) * -0.7 + 9.0;
        }
        let a = gain_scaled_excitation_subframe(&p, &c);
        let b = gain_scaled_excitation_subframe(&c, &p);
        for n in 0..GAIN_SCALED_EXCITATION_SAMPLES {
            assert_eq!(a[n], b[n], "n={n}");
        }
    }

    /// Linearity: negating both contributions negates the excitation.
    #[test]
    fn negation_commutes() {
        let mut p = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        let mut c = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        for i in 0..GAIN_SCALED_EXCITATION_SAMPLES {
            p[i] = (i as f32) * 4.0 - 25.0;
            c[i] = (i as f32) * -1.5 + 8.0;
        }
        let out = gain_scaled_excitation_subframe(&p, &c);
        let mut p_neg = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        let mut c_neg = [0.0f32; GAIN_SCALED_EXCITATION_SAMPLES];
        for i in 0..GAIN_SCALED_EXCITATION_SAMPLES {
            p_neg[i] = -p[i];
            c_neg[i] = -c[i];
        }
        let out_neg = gain_scaled_excitation_subframe(&p_neg, &c_neg);
        for n in 0..GAIN_SCALED_EXCITATION_SAMPLES {
            assert_eq!(out_neg[n], -out[n], "n={n}");
        }
    }

    /// Stream-start end-to-end: empty buffer → `p[n] = 0.0`, so the
    /// composed excitation equals the gain-scaled innovation `c[n]`
    /// verbatim. Exercises the r331 pitch path + r326 innovation path +
    /// this composition together.
    #[test]
    fn stream_start_envelope_follows_innovation() {
        let buffer = ExcitationBuffer::new();
        let taps = PitchGainTaps { taps: [60, 70, 80] };
        let p = gain_scaled_pitch_subframe(50, taps, &buffer).expect("in-range pitch");
        // Stream-start sanity: p is all-zero.
        assert!(p.iter().all(|&v| v == 0.0), "stream-start p should be 0.0");

        // A real mode-6 innovation sub-vector scaled by a reconstructed
        // gain.
        let submode = NARROWBAND_SUBMODES[6];
        let indices_vq: [u32; 8] = [3, 7, 1, 12, 5, 9, 2, 6];
        let mut packed: u128 = 0;
        for &i in &indices_vq {
            packed = (packed << 8) | u128::from(i);
        }
        let c_raw = decode_subframe(&submode, packed).expect("mode 6 documented");
        let gain_indices = FixedCodebookGainIndices {
            frame: FrameInnovationGainIndex::Indexed(15),
            subframe: SubFrameInnovationGainCorrection::ThreeBit(6),
        };
        let c = gain_scaled_innovation_from_indices(&c_raw, gain_indices);

        let e = gain_scaled_excitation_subframe(&p, &c);
        for n in 0..GAIN_SCALED_EXCITATION_SAMPLES {
            assert_eq!(e[n], c[n], "n={n}: e should equal c when p=0");
        }
        // The excitation is non-trivial (the innovation is non-zero).
        assert!(e.iter().any(|&v| v != 0.0), "excitation should be non-zero");
    }

    /// Silence end-to-end: silent frame → `c[n] = 0.0`, silence taps →
    /// `p[n] = 0.0`, so the composed excitation is identically zero.
    #[test]
    fn silence_composes_to_zero() {
        let buffer = ExcitationBuffer::new();
        let p = gain_scaled_pitch_subframe(50, PitchGainTaps::SILENCE, &buffer).expect("in-range");

        // Silent-frame gain index drives the reconstructed gain to 0.0.
        let silence = NARROWBAND_SUBMODES[0];
        let c_raw = decode_subframe(&silence, 0).expect("silence yields zero c_raw");
        let gain_indices = FixedCodebookGainIndices {
            frame: FrameInnovationGainIndex::Silence,
            subframe: SubFrameInnovationGainCorrection::Absent,
        };
        let c = gain_scaled_innovation_from_indices(&c_raw, gain_indices);

        let e = gain_scaled_excitation_subframe(&p, &c);
        assert!(
            e.iter().all(|&v| v == 0.0),
            "silence composition should yield all-zero excitation"
        );
    }

    /// End-to-end with a non-stream-start buffer: a pre-loaded excitation
    /// history makes `p[n]` non-zero, and a real innovation makes `c[n]`
    /// non-zero; the composed excitation equals `p[n] + c[n]` elementwise.
    #[test]
    fn nonzero_pitch_and_innovation_compose() {
        let mut buffer = ExcitationBuffer::new();
        for i in 0..150 {
            let s = ((i * 13 + 7) & 0xff) as i16 - 128;
            buffer.push(s);
        }
        let taps = PitchGainTaps { taps: [30, 50, 20] };
        let period = 60u16;
        let p = gain_scaled_pitch_subframe(period, taps, &buffer).expect("in-range");
        assert!(p.iter().any(|&v| v != 0.0), "pitch should be non-zero");

        let submode = NARROWBAND_SUBMODES[8];
        let packed: u128 = (3u128 << 5) | 7u128;
        let c_raw = decode_subframe(&submode, packed).expect("mode 8 documented");
        let gain_indices = FixedCodebookGainIndices {
            frame: FrameInnovationGainIndex::Indexed(10),
            subframe: SubFrameInnovationGainCorrection::Absent,
        };
        let c = gain_scaled_innovation_from_indices(&c_raw, gain_indices);

        let e = gain_scaled_excitation_subframe(&p, &c);
        for n in 0..GAIN_SCALED_EXCITATION_SAMPLES {
            assert_eq!(e[n], p[n] + c[n], "n={n}");
            assert!(e[n].is_finite(), "n={n}: excitation must be finite");
        }
    }

    /// The constant restates the 40-sample sub-frame dimension.
    #[test]
    fn samples_constant_is_40() {
        assert_eq!(GAIN_SCALED_EXCITATION_SAMPLES, 40);
    }
}
