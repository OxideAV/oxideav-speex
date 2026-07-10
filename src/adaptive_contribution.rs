//! Narrowband adaptive-codebook contribution sum (round r241 scope).
//!
//! Composes the r208 pitch-gain reconstruction with the r234
//! index-resolution + excitation-history primitives into the
//! Q-format-agnostic raw-integer dot product of Speex Codec Manual §9.2
//! Eq. 9.1:
//!
//! > `ea[n] = g0·e[n − T − 1] + g1·e[n − T] + g2·e[n − T + 1]`
//!
//! ## Scope
//!
//! Given:
//!
//! * a pitch period `T ∈ [PITCH_PERIOD_MIN, PITCH_PERIOD_MAX]` =
//!   `[17, 144]` (de-biased from the per-sub-frame
//!   [`crate::NarrowbandSubFrameIndices::pitch_period`] field);
//! * three β tap coefficients
//!   ([`crate::PitchGainTaps`] from [`crate::reconstruct_pitch_gain`]),
//!   stored as post-bias signed integers in raw codebook units;
//! * the historical excitation buffer
//!   ([`crate::ExcitationBuffer`] from r234), holding the last
//!   `EXCITATION_HISTORY_LEN` = 145 samples of the emitted excitation;
//!
//! this module produces the per-sub-frame adaptive-codebook contribution
//! as a `[i32; 40]` vector. Each element is the raw integer dot product
//! of the post-bias tap triple with the three substituted historical
//! samples — no Q-format scaling, no rounding step, no widening shift.
//!
//! ## Why a Q-format-agnostic integer dot product
//!
//! The post-bias pitch-gain triple lives in raw codebook units (no
//! Q-format pin in the staged manual / CELP companion — see
//! [`crate::pitch_gain`] module docs for the recorded gap). The
//! historical excitation samples are stored as `i16` matching the
//! sub-vector codebook entry width (see [`crate::ExcitationBuffer`]).
//!
//! The product `g · e` is therefore an integer × integer multiplication
//! with no Q-format choice required at this layer: the output `ea[n]`
//! lives in the same units as `g · e`. Whatever Q-format the downstream
//! `e[n] = p[n] + c[n]` composition picks for `g · e` will trivially
//! apply to `ea[n] = sum(g · e)` with a single arithmetic shift (the
//! sum stays in the same units because all three terms share them).
//!
//! Storing the dot product as `i32` gives ample headroom: with the
//! documented `+32`-biased gain range `-96 ..= 159` and `i16`
//! excitation samples bounded by `i16::MIN/MAX`, each tap-product fits
//! in `i32` (`159 * 32767 ≈ 5.2e6`, well below `i32::MAX ≈ 2.1e9`); the
//! three-tap sum stays bounded by `3 * 5.2e6 ≈ 1.6e7`, also clear of
//! `i32::MAX`.
//!
//! ## Index resolution — composition with r234
//!
//! For each output sample position `n ∈ 0..40` within the sub-frame,
//! [`crate::sample_lookback_indices`] resolves the three per-tap
//! lookback offsets `[k0, k1, k2]` after applying the §9.2 repeat rule
//! for short pitches. Each `kj` is strictly negative and within
//! `[-EXCITATION_HISTORY_LEN, -1]`, so [`crate::ExcitationBuffer::lookup`]
//! can resolve it without surfacing an error for any in-spec pitch.
//!
//! ## Stream-start behaviour
//!
//! The historical excitation buffer initialises with all zeros (the
//! [`crate::ExcitationBuffer::new`] / [`Default`] convention from
//! r234). For the very first sub-frame of a stream, every historical
//! lookup returns `0`, so the adaptive-codebook contribution is
//! identically zero regardless of the tap values. This produces the
//! continuous "no spurious transient" envelope at stream start
//! (analogous to the r200 LSP interpolation first-frame convention).
//!
//! ## What this module does NOT do
//!
//! * No Q-format scaling. The output `ea[n]` is the raw integer dot
//!   product. A downstream step picks the post-multiplication Q-format
//!   (typically `Q6` per the common Speex pitch-gain convention) by a
//!   single arithmetic shift over the whole `[i32; 40]` vector. The
//!   in-repo manual / CELP companion are silent on the documented
//!   choice — see [`crate::pitch_gain`] module docs for the recorded
//!   docs gap.
//! * No final excitation composition. The §8.4 `e[n] = p[n] + c[n]`
//!   sum needs the gain-scaled fixed-codebook contribution `c[n]`
//!   (already produced by [`crate::decode_innovation_subframe`] but
//!   not yet gain-scaled), plus the post-Q-format `p[n] = ea[n]`. Both
//!   stay deferred behind the documented pitch-gain Q-format gap.
//! * No high-band path. Per §10.2 the wideband high-band has no
//!   adaptive codebook, so the high band never consults this module.
//! * No buffer mutation. The caller pushes new excitation samples
//!   into the buffer after emitting them; this module only reads.
//! * No silence-mode short-circuit through the sub-mode dispatcher.
//!   The caller is expected to use [`crate::PitchGainTaps::SILENCE`]
//!   (`= [0; 3]`) for mode 0, which by construction yields the
//!   all-zero contribution without any branching in this module.

use crate::adaptive_codebook::{
    sample_lookback_indices, ExcitationBuffer, ExcitationError, ADAPTIVE_CODEBOOK_TAPS,
};
use crate::innovation::SUBFRAME_SAMPLES;
use crate::narrowband_body::{PITCH_PERIOD_MAX, PITCH_PERIOD_MIN};
use crate::pitch_gain::PitchGainTaps;

/// Errors from [`adaptive_contribution_subframe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveContributionError {
    /// The pitch period is outside the §9.2 documented `[17, 144]`
    /// range. The bit-stream parser caps the 7-bit `fine_pitch` field
    /// at `[PITCH_PERIOD_MIN, PITCH_PERIOD_MAX]` after the documented
    /// `+ PITCH_PERIOD_MIN` bias, so this can only fire if a caller
    /// hand-builds a [`crate::NarrowbandSubFrameIndices`] with an
    /// out-of-range period.
    PitchOutOfRange { period: u16 },
    /// A historical-buffer lookup failed. With an in-spec pitch period
    /// every resolved lookback lies in `[-EXCITATION_HISTORY_LEN, -1]`
    /// and the lookup cannot fail — this variant exists only to
    /// surface the underlying [`ExcitationError`] if a caller passes
    /// a pitch period the range check missed (or a future module
    /// extension widens the range).
    Buffer(ExcitationError),
}

impl core::fmt::Display for AdaptiveContributionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AdaptiveContributionError::PitchOutOfRange { period } => write!(
                f,
                "adaptive contribution: pitch period {period} out of documented [{}, {}] range",
                PITCH_PERIOD_MIN, PITCH_PERIOD_MAX
            ),
            AdaptiveContributionError::Buffer(e) => write!(f, "adaptive contribution: {e}"),
        }
    }
}

impl std::error::Error for AdaptiveContributionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AdaptiveContributionError::Buffer(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ExcitationError> for AdaptiveContributionError {
    fn from(e: ExcitationError) -> Self {
        AdaptiveContributionError::Buffer(e)
    }
}

/// Compute the §9.2 Eq. 9.1 adaptive-codebook contribution `ea[n]` for
/// one entire 40-sample CELP sub-frame.
///
/// For each output sample position `n ∈ 0..SUBFRAME_SAMPLES`, the
/// returned vector's `n`-th element is the raw integer dot product
///
/// ```text
/// ea[n] = g0·e[k0(n)] + g1·e[k1(n)] + g2·e[k2(n)]
/// ```
///
/// where `[k0, k1, k2]` are the three substituted lookbacks from
/// [`sample_lookback_indices`] and `e[·]` is read from `buffer` via
/// [`ExcitationBuffer::lookup`].
///
/// The output is in Q-format-agnostic raw integer units (see module
/// docs for the rationale). Each element fits comfortably in `i32`.
///
/// # Arguments
///
/// * `pitch_period` — the de-biased pitch period (caller's
///   `NarrowbandSubFrameIndices::pitch_period` value). Must satisfy
///   `PITCH_PERIOD_MIN ..= PITCH_PERIOD_MAX`.
/// * `taps` — the three post-bias β tap coefficients
///   ([`crate::PitchGainTaps`] from [`crate::reconstruct_pitch_gain`]).
/// * `buffer` — the historical excitation buffer (read-only).
///
/// # Errors
///
/// Returns [`AdaptiveContributionError::PitchOutOfRange`] if the
/// supplied pitch period is outside `[17, 144]`. Returns
/// [`AdaptiveContributionError::Buffer`] only on the buffer-lookup
/// paths that should be unreachable for an in-spec pitch (preserved
/// so callers see structured diagnostics instead of a panic on a
/// future-spec-extension period value).
pub fn adaptive_contribution_subframe(
    pitch_period: u16,
    taps: PitchGainTaps,
    buffer: &ExcitationBuffer,
) -> Result<[i32; SUBFRAME_SAMPLES], AdaptiveContributionError> {
    if !(PITCH_PERIOD_MIN..=PITCH_PERIOD_MAX).contains(&pitch_period) {
        return Err(AdaptiveContributionError::PitchOutOfRange {
            period: pitch_period,
        });
    }
    let mut out = [0i32; SUBFRAME_SAMPLES];
    for (n, slot) in out.iter_mut().enumerate() {
        let lookbacks = sample_lookback_indices(n as u32, pitch_period);
        let mut acc: i32 = 0;
        for (tap, &lookback) in lookbacks.iter().enumerate().take(ADAPTIVE_CODEBOOK_TAPS) {
            let sample = buffer.lookup(lookback)?;
            acc += i32::from(taps.taps[tap]) * i32::from(sample);
        }
        *slot = acc;
    }
    Ok(out)
}

/// Compute one sample of the adaptive-codebook contribution at the
/// given output position `n` within a 40-sample sub-frame.
///
/// Convenience for callers that walk samples one at a time (e.g. an
/// encoder's analysis-by-synthesis loop). The result matches
/// `adaptive_contribution_subframe(pitch_period, taps, buffer)?[n]`.
///
/// # Errors
///
/// As [`adaptive_contribution_subframe`].
pub fn adaptive_contribution_sample(
    n: u32,
    pitch_period: u16,
    taps: PitchGainTaps,
    buffer: &ExcitationBuffer,
) -> Result<i32, AdaptiveContributionError> {
    if !(PITCH_PERIOD_MIN..=PITCH_PERIOD_MAX).contains(&pitch_period) {
        return Err(AdaptiveContributionError::PitchOutOfRange {
            period: pitch_period,
        });
    }
    debug_assert!(
        (n as usize) < SUBFRAME_SAMPLES,
        "sub-frame sample position out of range"
    );
    let lookbacks = sample_lookback_indices(n, pitch_period);
    let mut acc: i32 = 0;
    for (tap, &lookback) in lookbacks.iter().enumerate().take(ADAPTIVE_CODEBOOK_TAPS) {
        let sample = buffer.lookup(lookback)?;
        acc += i32::from(taps.taps[tap]) * i32::from(sample);
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty (default) buffer → all-zero contribution regardless of taps.
    #[test]
    fn empty_buffer_yields_zero_contribution() {
        let buffer = ExcitationBuffer::new();
        let taps = PitchGainTaps { taps: [50, 60, 70] };
        for period in [PITCH_PERIOD_MIN, 40, 80, PITCH_PERIOD_MAX] {
            let out = adaptive_contribution_subframe(period, taps, &buffer).unwrap();
            assert!(
                out.iter().all(|&v| v == 0),
                "empty buffer should yield zeros for period {period}"
            );
        }
    }

    /// Silence taps `[0; 3]` → all-zero contribution regardless of buffer
    /// content.
    #[test]
    fn silence_taps_yield_zero_contribution() {
        let mut buffer = ExcitationBuffer::new();
        // Push a non-trivial history pattern.
        for i in 0..150 {
            buffer.push(((i * 37 + 11) & 0xff) as i16 - 128);
        }
        for period in [PITCH_PERIOD_MIN, 50, PITCH_PERIOD_MAX] {
            let out =
                adaptive_contribution_subframe(period, PitchGainTaps::SILENCE, &buffer).unwrap();
            assert!(
                out.iter().all(|&v| v == 0),
                "silence taps must produce all-zero for period {period}"
            );
        }
    }

    /// Pitch-period range check refuses out-of-spec input.
    #[test]
    fn out_of_range_pitch_period_rejected() {
        let buffer = ExcitationBuffer::new();
        let taps = PitchGainTaps { taps: [32, 32, 32] };
        for bad in [0u16, 1, PITCH_PERIOD_MIN - 1, PITCH_PERIOD_MAX + 1, 1000] {
            let res = adaptive_contribution_subframe(bad, taps, &buffer);
            assert!(matches!(
                res,
                Err(AdaptiveContributionError::PitchOutOfRange { period }) if period == bad
            ));
            let res_sample = adaptive_contribution_sample(0, bad, taps, &buffer);
            assert!(matches!(
                res_sample,
                Err(AdaptiveContributionError::PitchOutOfRange { period }) if period == bad
            ));
        }
    }

    /// Single-tap (g1-only) buffer of constant value yields a deterministic
    /// scalar at every output position.
    #[test]
    fn single_tap_constant_buffer_yields_constant_contribution() {
        // Fill the buffer with a constant.
        let mut buffer = ExcitationBuffer::new();
        for _ in 0..150 {
            buffer.push(10);
        }
        // Only g1 active; g0 = g2 = 0 → ea[n] = 7 * 10 = 70 for every n.
        let taps = PitchGainTaps { taps: [0, 7, 0] };
        let period = 80u16;
        let out = adaptive_contribution_subframe(period, taps, &buffer).unwrap();
        for (n, &v) in out.iter().enumerate() {
            assert_eq!(v, 70, "constant buffer at n={n} expected 70, got {v}");
        }
    }

    /// Pin one sample analytically — Eq. 9.1 evaluated by hand.
    #[test]
    fn worked_example_pins_eq_9_1_arithmetic() {
        // Construct an excitation history where buffer.lookup(-k)
        // returns -k itself, so the analytics are easy to follow.
        //
        // ExcitationBuffer::push appends one sample; lookup(-1) returns
        // the most recently pushed sample. We want lookup(-k) to return
        // -k for k = 1..=145, so we push the sequence in the order
        // -145, -144, ..., -1; after 145 pushes lookup(-1) = -1,
        // lookup(-2) = -2, ..., lookup(-145) = -145.
        let mut buffer = ExcitationBuffer::new();
        for k_pos in (1..=145i16).rev() {
            buffer.push(-k_pos);
        }

        // Sanity-check the buffer wiring.
        for k in 1..=145i32 {
            let v = buffer.lookup(-k).unwrap();
            assert_eq!(v, -k as i16, "lookup({}) expected {}, got {v}", -k, -k);
        }

        // Now compute ea[0] for T = 50, taps = (g0, g1, g2) = (2, 5, 3).
        // Per Eq. 9.1: ea[0] = g0·e[-51] + g1·e[-50] + g2·e[-49]
        //                    = 2 * -51 + 5 * -50 + 3 * -49
        //                    = -102 + -250 + -147
        //                    = -499.
        let taps = PitchGainTaps { taps: [2, 5, 3] };
        let out = adaptive_contribution_subframe(50, taps, &buffer).unwrap();
        assert_eq!(out[0], -499, "ea[0] hand-computed mismatch");

        // ea[1] for T = 50: lookbacks are -50, -49, -48.
        //   = 2 * -50 + 5 * -49 + 3 * -48
        //   = -100 + -245 + -144 = -489.
        assert_eq!(out[1], -489, "ea[1] hand-computed mismatch");

        // ea[39] for T = 50: lookbacks are -12, -11, -10.
        //   = 2 * -12 + 5 * -11 + 3 * -10
        //   = -24 + -55 + -30 = -109.
        assert_eq!(out[39], -109, "ea[39] hand-computed mismatch");
    }

    /// The per-sample helper must agree with the per-sub-frame batch
    /// for every (period, n) pair.
    #[test]
    fn sample_helper_agrees_with_subframe_batch() {
        let mut buffer = ExcitationBuffer::new();
        for i in 0..150i32 {
            buffer.push(((i - 75) as i16).wrapping_mul(7));
        }
        let taps = PitchGainTaps { taps: [1, -3, 5] };
        for period in [PITCH_PERIOD_MIN, 17, 25, 40, 50, 80, 144, PITCH_PERIOD_MAX] {
            let batch = adaptive_contribution_subframe(period, taps, &buffer).unwrap();
            for (n, &expected) in batch.iter().enumerate() {
                let one = adaptive_contribution_sample(n as u32, period, taps, &buffer).unwrap();
                assert_eq!(
                    one, expected,
                    "sample helper disagreed at period={period}, n={n}"
                );
            }
        }
    }

    /// Short pitch (`T < SUBFRAME_SAMPLES`) exercises the §9.2 repeat
    /// rule — the contribution still composes only historical samples.
    #[test]
    fn short_pitch_repeat_rule_resolves_within_history() {
        let mut buffer = ExcitationBuffer::new();
        for i in 0..150 {
            buffer.push((i as i16) - 75);
        }
        // T = 17 is the documented minimum. The repeat rule fires
        // throughout the sub-frame (since SUBFRAME_SAMPLES = 40 > 17),
        // but every resolved lookback still lies in [-145, -1].
        let taps = PitchGainTaps { taps: [3, 4, 5] };
        let out = adaptive_contribution_subframe(17, taps, &buffer).unwrap();
        // Spot-check ea[0] manually:
        //   lookbacks at n=0: g0 = -18, g1 = -17, g2 = -16
        //   (no substitution; all already < 0).
        //   buffer.lookup(-k) returns the (149 - (k-1))-th pushed value
        //   = (149 - k + 1) = 150 - k pushed (0-indexed: push #(150-k)).
        //   Push index i pushes sample (i - 75); so lookup(-k) =
        //   ((150 - k) - 75) - ... wait — careful: pushes are i=0..149,
        //   sample value (i - 75). After 150 pushes the buffer holds
        //   values (i - 75) for i in 0..150; lookup(-1) returns the
        //   most recently pushed = (149 - 75) = 74; lookup(-2) = 73;
        //   ... lookup(-k) = (149 - (k-1) - 75) = (74 - k + 1) = 75 - k.
        //
        //   ea[0] = 3 * (75 - 18) + 4 * (75 - 17) + 5 * (75 - 16)
        //         = 3 * 57 + 4 * 58 + 5 * 59
        //         = 171 + 232 + 295 = 698.
        assert_eq!(out[0], 698, "short-pitch ea[0] mismatch");
    }

    /// Tap-product result fits in i32 even at the documented bias range.
    #[test]
    fn product_headroom_is_safe() {
        // Worst-case post-bias tap magnitude per the codebook range is
        // 159; multiplied by i16::MAX = 32767 the product is
        // ~5.2 million, well within i32 range.
        let max_product: i64 = 159i64 * i64::from(i16::MAX);
        assert!(
            max_product < i64::from(i32::MAX),
            "max tap-product {max_product} exceeds i32 headroom"
        );
        // Three-tap sum likewise.
        let max_sum: i64 = 3 * max_product;
        assert!(
            max_sum < i64::from(i32::MAX),
            "max three-tap sum {max_sum} exceeds i32 headroom"
        );
    }

    /// First-sub-frame stream-start behaviour: an all-zero buffer
    /// composes with any pitch period to yield the documented "no
    /// spurious transient" envelope.
    #[test]
    fn stream_start_first_sub_frame_zero_envelope() {
        let buffer = ExcitationBuffer::new();
        // The maximum bias-range taps and the entire valid pitch range
        // must all produce all-zeros.
        let extreme_taps = [
            PitchGainTaps {
                taps: [-96, -96, -96],
            },
            PitchGainTaps {
                taps: [159, 159, 159],
            },
            PitchGainTaps {
                taps: [0, 159, -96],
            },
        ];
        for taps in extreme_taps {
            for period in [PITCH_PERIOD_MIN, 25, 80, PITCH_PERIOD_MAX] {
                let out = adaptive_contribution_subframe(period, taps, &buffer).unwrap();
                for (n, &v) in out.iter().enumerate() {
                    assert_eq!(v, 0, "stream-start ea[{n}] for period {period} must be 0");
                }
            }
        }
    }

    /// Linearity in the gain triple: scaling every tap by `s` scales
    /// the contribution by `s`. (Pure-arithmetic consequence of the
    /// dot product; pinned as a regression guard.)
    #[test]
    fn contribution_linear_in_taps() {
        let mut buffer = ExcitationBuffer::new();
        for i in 0..150 {
            buffer.push((i as i16 - 75) * 3);
        }
        let base = PitchGainTaps { taps: [2, -3, 4] };
        let scaled = PitchGainTaps {
            taps: [base.taps[0] * 7, base.taps[1] * 7, base.taps[2] * 7],
        };
        for period in [PITCH_PERIOD_MIN, 40, 80, PITCH_PERIOD_MAX] {
            let a = adaptive_contribution_subframe(period, base, &buffer).unwrap();
            let b = adaptive_contribution_subframe(period, scaled, &buffer).unwrap();
            for (n, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    va * 7,
                    vb,
                    "linearity broke at period={period}, n={n}: a={va}, b={vb}"
                );
            }
        }
    }

    /// Linearity in the buffer: scaling every excitation sample by `s`
    /// scales the contribution by `s`.
    #[test]
    fn contribution_linear_in_buffer() {
        let mut buffer_a = ExcitationBuffer::new();
        let mut buffer_b = ExcitationBuffer::new();
        for i in 0..150 {
            let raw: i16 = (i as i16 - 75) * 2;
            buffer_a.push(raw);
            buffer_b.push(raw * 3);
        }
        let taps = PitchGainTaps { taps: [5, 7, 11] };
        for period in [PITCH_PERIOD_MIN, 50, PITCH_PERIOD_MAX] {
            let a = adaptive_contribution_subframe(period, taps, &buffer_a).unwrap();
            let b = adaptive_contribution_subframe(period, taps, &buffer_b).unwrap();
            for (n, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    va * 3,
                    vb,
                    "buffer linearity broke at period={period}, n={n}: a={va}, b={vb}"
                );
            }
        }
    }

    /// Decoder-side composition smoke test: assemble a contribution
    /// from r208's `reconstruct` + r234's buffer + this module, and
    /// confirm the result is finite + structurally consistent.
    #[test]
    fn integrates_with_r208_r234_modules() {
        use crate::pitch_gain::reconstruct;
        use crate::submode::PitchGainQuant;

        let mut buffer = ExcitationBuffer::new();
        for i in 0..150 {
            buffer.push((i as i16 - 75) * 5);
        }
        // 5-bit codebook index 1 → (16, -26, 1) per the r410
        // column-reversal pin (`pitch_gain` module docs).
        let taps = reconstruct(1, PitchGainQuant::Vq5Bit).unwrap();
        assert_eq!(taps.taps, [16, -26, 1]);
        let out = adaptive_contribution_subframe(80, taps, &buffer).unwrap();
        // Pin n=0 against a direct dot-product evaluation.
        //   lookbacks for T=80, n=0: -81, -80, -79.
        //   buffer.lookup(-k) = (150-k-75) * 5 = (75-k) * 5.
        //   ea[0] = 16 * (75-81)*5 + (-26) * (75-80)*5 + 1 * (75-79)*5
        //         = 16 * -30 + (-26) * -25 + 1 * -20
        //         = -480 + 650 + -20 = 150.
        assert_eq!(out[0], 150, "decoder composition ea[0] mismatch");
        // Smoke-test the rest are finite + within the documented
        // headroom band.
        for (n, &v) in out.iter().enumerate() {
            assert!(
                v.abs() < (3 * 159i32 * i32::from(i16::MAX)),
                "n={n} contribution {v} exceeds headroom"
            );
        }
    }
}
