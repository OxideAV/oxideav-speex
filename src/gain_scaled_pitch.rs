//! **Gain-scaled adaptive-codebook contribution** (round r331 scope) —
//! folds the staged Q6 pitch-gain scaling into the §9.2 long-term
//! predictor dot product, producing the adaptive-codebook (pitch)
//! contribution `p[n]` in the **same normalised float signal domain**
//! as the r326 gain-scaled fixed-codebook contribution `c[n]`, so the
//! Speex Codec Manual §8.4 composition `e[n] = p[n] + c[n]` can finally
//! add the two contributions in a single domain.
//!
//! ## Where this sits in the decode path
//!
//! The narrowband adaptive-codebook (pitch) path has two layers:
//!
//! 1. **Raw integer dot product** —
//!    [`crate::adaptive_contribution_subframe`] forms the §9.2 Eq. 9.1
//!    long-term predictor sum
//!    `ea[n] = g0·e[n−T−1] + g1·e[n−T] + g2·e[n−T+1]` as a raw
//!    `[i32; 40]`. The three taps `g0, g1, g2` are the post-bias
//!    integer β coefficients ([`crate::PitchGainTaps`]); the historical
//!    excitation samples `e[·]` are raw `i16` codebook units. The
//!    product is Q-format-agnostic — the integer dot product carries
//!    whatever Q-format the taps were stored in, undivided.
//! 2. **This module** — divides the raw dot product by the staged
//!    pitch-gain scaling `GAIN_SCALING = 64` (`GAIN_SHIFT = 6`, i.e.
//!    Q6), recovering the float-domain pitch contribution
//!    `p[n] = ea[n] / 64`. This is the magnitude-correct `p[n]` the
//!    §8.4 composition adds to the gain-scaled `c[n]`.
//!
//! ## Spec basis (the pitch-gain Q-format)
//!
//! The post-bias β tap coefficients [`crate::reconstruct_pitch_gain`]
//! resolves are stored in **Q6** fixed point: the long-term predictor
//! interprets each tap as `g / 64`. The clean-room extraction staged at
//! `docs/audio/speex/provenance/02-speex-gain-quant.md` pins this in the
//! "Scalar constants" table:
//!
//! > `GAIN_SCALING` / `GAIN_SHIFT` (gain domain scaling) = `64` / `6`
//!
//! i.e. the gain domain is Q6, so the raw integer dot product
//! `ea[n] = Σ gᵢ·e[·]` (whose gain factor carries the un-divided Q6
//! scale) divides by `64` to land in the same un-scaled signal domain
//! as the historical excitation samples `e[·]` themselves. Because the
//! r326 gain-scaled `c[n]` is already in that normalised float signal
//! domain (the `SIG_SCALING`-normalised float `ol_gain` × raw `i16`
//! sub-vector — see [`crate::gain_scaled_innovation`] module docs), the
//! Q6-divided `p[n]` and the float `c[n]` now share one domain and the
//! §8.4 sum `e[n] = p[n] + c[n]` is well-posed.
//!
//! ## Numeric domain
//!
//! The raw dot product `ea[n]` is bounded analytically by
//! `3 × |g|_max × |e|_max ≤ 3 × 159 × i16::MAX ≈ 1.6 × 10⁷` (see
//! [`crate::adaptive_contribution`] module docs). Dividing by `64`
//! yields `|p[n]| ≤ 2.5 × 10⁵`, comfortably inside `f32`'s exact-integer
//! range (`2²⁴ ≈ 1.7 × 10⁷`), so the Q6 → float division introduces no
//! rounding for any in-spec tap/excitation combination whose dot product
//! is a multiple of the float step — and at worst a single-ULP rounding
//! otherwise. The result is `[f32; 40]`, matching the r326
//! [`crate::gain_scaled_innovation_subframe`] output type so the two
//! contributions add directly.
//!
//! ## Stream-start / silence behaviour
//!
//! * **Stream start.** With the all-zero default
//!   [`crate::ExcitationBuffer`], every historical lookup returns `0`,
//!   so the raw dot product is identically zero and `p[n] = 0.0` across
//!   the whole sub-frame for any tap triple — the continuous
//!   "no spurious transient" envelope the raw layer already guarantees,
//!   preserved through the division.
//! * **Silence taps.** [`crate::PitchGainTaps::SILENCE`] (`[0; 3]`,
//!   mode 0 / no-pitch) makes the dot product zero, so `p[n] = 0.0`
//!   regardless of buffer content — the pitch contribution vanishes and
//!   the excitation reduces to `c[n]` alone.
//!
//! ## What this module DOES
//!
//! * [`gain_scaled_pitch_subframe`] — compute the float-domain pitch
//!   contribution `p[n]` for a whole 40-sample sub-frame from the pitch
//!   period, post-bias tap triple, and excitation history buffer.
//! * [`gain_scaled_pitch_sample`] — single-sample helper matching the
//!   batch path elementwise.
//!
//! ## What this module DOES NOT do
//!
//! * No raw dot product. It composes [`crate::adaptive_contribution`]
//!   and applies only the Q6 → float division.
//! * No final excitation composition. Adding `p[n]` to the gain-scaled
//!   `c[n]` is the next synthesis layer (the float analogue of
//!   [`crate::raw_excitation_subframe`]).
//! * No high-band path. Per §10.2 the wideband high band has no
//!   adaptive codebook.
//! * No encoder-side pitch search.

use crate::adaptive_codebook::ExcitationBuffer;
use crate::adaptive_contribution::{
    adaptive_contribution_sample, adaptive_contribution_subframe, AdaptiveContributionError,
};
use crate::innovation::SUBFRAME_SAMPLES;
use crate::pitch_gain::PitchGainTaps;

/// Number of gain-scaled pitch-contribution samples per CELP sub-frame.
///
/// Restates [`crate::innovation::SUBFRAME_SAMPLES`] = `40` at the
/// gain-scaling layer so the public API names the dimension where the
/// consumer reads it (mirrors
/// [`crate::GAIN_SCALED_INNOVATION_SAMPLES`]).
pub const GAIN_SCALED_PITCH_SAMPLES: usize = SUBFRAME_SAMPLES;

/// Pitch-gain domain scaling factor (`GAIN_SCALING`, `GAIN_SHIFT = 6`,
/// i.e. **Q6**).
///
/// The post-bias β tap coefficients [`crate::reconstruct_pitch_gain`]
/// resolves are Q6 fixed point: the long-term predictor interprets each
/// tap as `g / 64`. Staged in
/// `docs/audio/speex/provenance/02-speex-gain-quant.md` ("Scalar
/// constants": `GAIN_SCALING` / `GAIN_SHIFT` = `64` / `6`). Dividing the
/// raw §9.2 dot product by this factor lands the pitch contribution in
/// the normalised float signal domain the r326 gain-scaled `c[n]`
/// already occupies.
pub const PITCH_GAIN_SCALING: f32 = 64.0;

/// Compute the float-domain adaptive-codebook (pitch) contribution
/// `p[n]` for one entire 40-sample CELP sub-frame.
///
/// Forms the raw §9.2 Eq. 9.1 dot product via
/// [`adaptive_contribution_subframe`] and divides each element by the
/// staged Q6 pitch-gain scaling [`PITCH_GAIN_SCALING`] = `64`, giving
///
/// ```text
/// p[n] = ( g0·e[n−T−1] + g1·e[n−T] + g2·e[n−T+1] ) / 64
/// ```
///
/// in the normalised float signal domain. The `[f32; 40]` result shares
/// the domain of the r326 [`crate::gain_scaled_innovation_subframe`]
/// `c[n]`, so the §8.4 composition `e[n] = p[n] + c[n]` adds the two
/// directly.
///
/// # Arguments
///
/// * `pitch_period` — the de-biased pitch period in `[17, 144]`.
/// * `taps` — the three post-bias β tap coefficients
///   ([`crate::PitchGainTaps`] from [`crate::reconstruct_pitch_gain`]),
///   in Q6.
/// * `buffer` — the historical excitation buffer (read-only).
///
/// # Errors
///
/// Propagates [`AdaptiveContributionError`] from the raw dot-product
/// layer (out-of-range pitch period or buffer-lookup failure).
pub fn gain_scaled_pitch_subframe(
    pitch_period: u16,
    taps: PitchGainTaps,
    buffer: &ExcitationBuffer,
) -> Result<[f32; GAIN_SCALED_PITCH_SAMPLES], AdaptiveContributionError> {
    let raw = adaptive_contribution_subframe(pitch_period, taps, buffer)?;
    let mut out = [0.0f32; GAIN_SCALED_PITCH_SAMPLES];
    for (slot, &ea) in out.iter_mut().zip(raw.iter()) {
        *slot = ea as f32 / PITCH_GAIN_SCALING;
    }
    Ok(out)
}

/// Q6 bound on the tap sum used for the **in-sub-frame recursive**
/// reads of [`gain_scaled_pitch_subframe_recursive`]: `0.9 · 64 =
/// 57.6`.
///
/// When a short pitch (`T <` sub-frame length) makes the §9.2
/// long-term predictor read positions inside the current sub-frame,
/// the recursion re-applies the tap gains once per period. A tap sum
/// above unity would then grow geometrically inside the sub-frame.
/// The staged provenance records a `0.9` pitch-coefficient damping /
/// bounding constant family (`provenance/02` "Pitch-gain damping" and
/// the `pitch_coef` clamp rows); black-box arbitration against the
/// staged narrowband reference decodes
/// (`tests/fixtures/nb-conformance/`) pins the *placement*: bounding
/// only the **recursive** reads (history reads keep the transmitted
/// gains) scores best across every sub-mode — clamping the history
/// reads too collapses the fine-pitch modes' energy to ≈ 0.5, and no
/// bound at all overshoots the OL-pitch modes by ≈ +23 % energy.
pub const RECURSIVE_TAP_SUM_BOUND_Q6: f32 = 0.9 * PITCH_GAIN_SCALING;

/// Compute the float-domain adaptive-codebook (pitch) contribution
/// `p[n]` for one 40-sample sub-frame with the **in-sub-frame
/// recursion** the reference decode exhibits for short pitch periods
/// (`T < 40`), arbitrated in round r410 against the staged narrowband
/// reference decodes.
///
/// For each output position `n`, the three Eq. 9.1 reads at `k = n − T
/// + {−1, 0, +1}` resolve as:
///
/// * `k < 0` — a **history** read from `buffer` (the previous
///   sub-frames' composed excitation `e[·]`), weighted by the
///   transmitted tap gains unmodified;
/// * `k ≥ 0` — a **recursive** read of the *pitch-only partial*
///   `p[k]` already produced inside this sub-frame (not the composed
///   `e[k]`, which does not exist yet at this point of the recurrence),
///   weighted by the tap gains bounded to a `0.9` total
///   ([`RECURSIVE_TAP_SUM_BOUND_Q6`]).
///
/// The candidate semantics were fixture-arbitrated black-box
/// (`tests/nb_conformance_fixture.rs`):
///
/// * the manual's §9.2 repeat rule (`n−T+1 ≥ 0 → n−2T+1`, reading the
///   previous period's composed excitation) scores 6.8 dB / 9.3 dB on
///   the OL-pitch sub-modes 8/2 where this recursion scores
///   12.7–13.4 dB;
/// * recursing over the **composed** partial (`p[k] + c[k]`) inverts
///   the energy error (2.8× overshoot on sub-mode 8);
/// * the pitch-only recursion with the 0.9 recursive bound is the
///   best-scoring reading on all ten staged fixtures simultaneously.
///
/// For `T ≥ 40` no `k ≥ 0` read occurs and this function reduces
/// exactly to the plain history dot product.
///
/// # Errors
///
/// [`AdaptiveContributionError::PitchOutOfRange`] outside `[17, 144]`,
/// as the raw layer.
pub fn gain_scaled_pitch_subframe_recursive(
    pitch_period: u16,
    taps: PitchGainTaps,
    buffer: &ExcitationBuffer,
) -> Result<[f32; GAIN_SCALED_PITCH_SAMPLES], AdaptiveContributionError> {
    use crate::narrowband_body::{PITCH_PERIOD_MAX, PITCH_PERIOD_MIN};
    if !(PITCH_PERIOD_MIN..=PITCH_PERIOD_MAX).contains(&pitch_period) {
        return Err(AdaptiveContributionError::PitchOutOfRange {
            period: pitch_period,
        });
    }

    // Transmitted tap gains for the history reads.
    let hist_taps = [
        f32::from(taps.taps[0]),
        f32::from(taps.taps[1]),
        f32::from(taps.taps[2]),
    ];
    // Bounded tap gains for the recursive reads (module docs).
    let mut rec_taps = hist_taps;
    let sum: f32 = hist_taps.iter().sum();
    if sum > RECURSIVE_TAP_SUM_BOUND_Q6 {
        for t in rec_taps.iter_mut() {
            *t *= RECURSIVE_TAP_SUM_BOUND_Q6 / sum;
        }
    }

    let period = i32::from(pitch_period);
    let mut p = [0.0f32; GAIN_SCALED_PITCH_SAMPLES];
    for n in 0..GAIN_SCALED_PITCH_SAMPLES {
        let mut acc = 0.0f32;
        for (tap, off) in [-1i32, 0, 1].iter().enumerate() {
            let k = n as i32 - period + off;
            if k >= 0 {
                acc += rec_taps[tap] * p[k as usize];
            } else {
                // In-range by construction: |k| ≤ T + 1 ≤ 145.
                let sample = buffer.lookup(k)?;
                acc += hist_taps[tap] * f32::from(sample);
            }
        }
        p[n] = acc / PITCH_GAIN_SCALING;
    }
    Ok(p)
}

/// Compute one sample of the float-domain pitch contribution at the
/// given output position `n` within a 40-sample sub-frame.
///
/// Matches `gain_scaled_pitch_subframe(pitch_period, taps, buffer)?[n]`
/// for the same arguments. Provided for callers that walk samples one at
/// a time (e.g. an encoder's analysis-by-synthesis loop).
///
/// # Errors
///
/// As [`gain_scaled_pitch_subframe`].
pub fn gain_scaled_pitch_sample(
    n: u32,
    pitch_period: u16,
    taps: PitchGainTaps,
    buffer: &ExcitationBuffer,
) -> Result<f32, AdaptiveContributionError> {
    let raw = adaptive_contribution_sample(n, pitch_period, taps, buffer)?;
    Ok(raw as f32 / PITCH_GAIN_SCALING)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_codebook::EXCITATION_HISTORY_LEN;
    use crate::narrowband_body::{PITCH_PERIOD_MAX, PITCH_PERIOD_MIN};

    /// Build a buffer whose most-recent samples are a known ramp so the
    /// dot product is non-trivial and easy to cross-check.
    fn ramp_buffer() -> ExcitationBuffer {
        let mut buffer = ExcitationBuffer::new();
        for i in 0..EXCITATION_HISTORY_LEN {
            buffer.push((i as i16) - 64);
        }
        buffer
    }

    /// Empty (default) buffer → all-zero contribution regardless of taps,
    /// preserved through the Q6 division.
    #[test]
    fn empty_buffer_yields_zero_contribution() {
        let buffer = ExcitationBuffer::new();
        let taps = PitchGainTaps { taps: [50, 60, 70] };
        for period in [PITCH_PERIOD_MIN, 40, 80, PITCH_PERIOD_MAX] {
            let out = gain_scaled_pitch_subframe(period, taps, &buffer).unwrap();
            assert!(
                out.iter().all(|&v| v == 0.0),
                "empty buffer should yield 0.0 for period {period}"
            );
        }
    }

    /// Silence taps `[0; 3]` → all-zero contribution regardless of buffer
    /// content.
    #[test]
    fn silence_taps_yield_zero_contribution() {
        let buffer = ramp_buffer();
        for period in [PITCH_PERIOD_MIN, 50, PITCH_PERIOD_MAX] {
            let out = gain_scaled_pitch_subframe(period, PitchGainTaps::SILENCE, &buffer).unwrap();
            assert!(
                out.iter().all(|&v| v == 0.0),
                "silence taps should yield 0.0 for period {period}"
            );
        }
    }

    /// The scaled contribution equals the raw dot product divided by the
    /// Q6 factor `64`, element for element.
    #[test]
    fn matches_raw_over_64() {
        let buffer = ramp_buffer();
        let taps = PitchGainTaps { taps: [40, 96, 18] };
        for period in [PITCH_PERIOD_MIN, 33, 80, PITCH_PERIOD_MAX] {
            let raw = adaptive_contribution_subframe(period, taps, &buffer).unwrap();
            let scaled = gain_scaled_pitch_subframe(period, taps, &buffer).unwrap();
            for (n, (&s, &r)) in scaled.iter().zip(raw.iter()).enumerate() {
                assert_eq!(
                    s,
                    r as f32 / 64.0,
                    "period {period}, n={n}: scaled should be raw/64"
                );
            }
        }
    }

    /// The per-sample helper agrees with the batch path elementwise.
    #[test]
    fn per_sample_helper_matches_batch() {
        let buffer = ramp_buffer();
        let taps = PitchGainTaps {
            taps: [12, -30, 77],
        };
        let period = 60u16;
        let batch = gain_scaled_pitch_subframe(period, taps, &buffer).unwrap();
        for (n, &b) in batch.iter().enumerate() {
            let single = gain_scaled_pitch_sample(n as u32, period, taps, &buffer).unwrap();
            assert_eq!(single, b, "n={n}");
        }
    }

    /// The Q6 division is linear in the taps: doubling every tap doubles
    /// the contribution (the dot product is linear in the gains).
    #[test]
    fn linear_in_taps() {
        let buffer = ramp_buffer();
        let single = PitchGainTaps { taps: [20, 25, 15] };
        let double = PitchGainTaps { taps: [40, 50, 30] };
        let period = 47u16;
        let s = gain_scaled_pitch_subframe(period, single, &buffer).unwrap();
        let d = gain_scaled_pitch_subframe(period, double, &buffer).unwrap();
        for (n, (&dv, &sv)) in d.iter().zip(s.iter()).enumerate() {
            assert_eq!(dv, 2.0 * sv, "n={n}");
        }
    }

    /// A pitch period outside `[17, 144]` propagates the raw-layer error
    /// rather than panicking.
    #[test]
    fn out_of_range_period_errors() {
        let buffer = ramp_buffer();
        let taps = PitchGainTaps { taps: [10, 10, 10] };
        assert!(matches!(
            gain_scaled_pitch_subframe(16, taps, &buffer),
            Err(AdaptiveContributionError::PitchOutOfRange { period: 16 })
        ));
        assert!(matches!(
            gain_scaled_pitch_subframe(145, taps, &buffer),
            Err(AdaptiveContributionError::PitchOutOfRange { period: 145 })
        ));
    }

    /// The scaling constant is exactly the staged Q6 factor.
    #[test]
    fn scaling_constant_is_q6() {
        assert_eq!(PITCH_GAIN_SCALING, 64.0);
    }
}
