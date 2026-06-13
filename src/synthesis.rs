//! LPC synthesis filter (round r286 scope).
//!
//! The Speex decoder reconstructs each frame's PCM by running the
//! excitation signal `e[n]` through the all-pole synthesis filter
//! `1/A(z)`. *The Speex Codec Manual* §9.4 names it directly:
//!
//! > "the synthesis filter S(z) = 1/A(z)"
//!
//! and §8.2 defines the prediction-error filter `A(z)` through the
//! linear predictor
//!
//! > `e[n] = x[n] − Σ a[i]·x[n−i]`
//!
//! (manual §8.2, prediction error of an order-`N` LPC analysis).
//! Inverting that relation gives the decoder recurrence this module
//! implements:
//!
//! ```text
//! x[n] = e[n] + Σ_{i=1}^{N} a[i]·x[n−i]
//! ```
//!
//! where `a[i]` are the [`crate::lsp_to_lpc`]-produced LPC
//! coefficients of `A(z) = 1 − a[0]·z⁻¹ − … − a[N−1]·z⁻ᴺ` (the
//! `−a` sign fold is already baked into the coefficient vector so the
//! recurrence above is a plain *add*).
//!
//! ## Filter state across sub-frames and frames
//!
//! The synthesis filter is recursive: each output sample feeds back
//! into the next `N` outputs. The decoder therefore keeps a running
//! history of the last `N` synthesised samples ([`SynthesisFilter`]).
//! The narrowband frame is four 40-sample sub-frames (companion §1);
//! each sub-frame is filtered with its own interpolated LPC set
//! (r200 sub-frame LSP interpolation → r286 LSP→LPC) but the history
//! carries straight through sub-frame and frame boundaries — that
//! continuity is what makes `1/A(z)` an IIR filter rather than a
//! per-block reset.
//!
//! ## Numeric domain
//!
//! The LPC coefficients are produced in `f64` by [`crate::lsp_to_lpc`].
//! The excitation arrives as raw integer samples ([`crate::excitation`]
//! emits `i32`). This module filters in `f64` to avoid imposing a
//! fixed-point Q-format that the in-repo manual does not specify (the
//! reference decoder's exact `Q`-domains for the synthesis recurrence
//! are part of the documented Q-format gap; see [`crate::lsp_to_lpc`]).
//! [`SynthesisFilter::process_into_i16`] rounds + saturates the `f64`
//! output to `i16` PCM for callers that want 16-bit samples.

use crate::lsp_to_lpc::LPC_ORDER;

/// All-pole LPC synthesis filter `1/A(z)` with persistent history.
///
/// Construct with [`SynthesisFilter::new`] (zero history — the
/// stream-start convention, matching the all-zero excitation history
/// the decoder begins from), feed excitation through
/// [`SynthesisFilter::process`], and the filter carries its `N`-sample
/// memory across calls.
#[derive(Debug, Clone)]
pub struct SynthesisFilter {
    /// Last [`LPC_ORDER`] synthesised samples, most-recent last.
    /// `history[LPC_ORDER-1]` is `x[n−1]`, `history[0]` is `x[n−N]`.
    history: [f64; LPC_ORDER],
}

impl Default for SynthesisFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl SynthesisFilter {
    /// A fresh filter with zero history (stream start).
    pub fn new() -> Self {
        Self {
            history: [0.0; LPC_ORDER],
        }
    }

    /// Filter one excitation block `e[n]` through `1/A(z)` using the
    /// LPC coefficients `a` (the [`crate::lsp_to_lpc`] convention:
    /// `x[n] = e[n] + Σ a[i]·x[n−1−i]`), writing the synthesised
    /// samples into `out` and advancing the filter history.
    ///
    /// `excitation` and `out` must be the same length. The history
    /// carries across calls so successive sub-frames compose into a
    /// continuous IIR output.
    pub fn process(&mut self, lpc: &[f64; LPC_ORDER], excitation: &[f64], out: &mut [f64]) {
        debug_assert_eq!(excitation.len(), out.len());
        for (slot, &e) in out.iter_mut().zip(excitation.iter()) {
            // x[n] = e[n] + Σ_{i=0}^{N-1} a[i]·x[n−1−i].
            // history[LPC_ORDER-1] = x[n-1] (i=0),
            // history[LPC_ORDER-1-i] = x[n-1-i].
            let mut acc = e;
            for (i, &coeff) in lpc.iter().enumerate() {
                acc += coeff * self.history[LPC_ORDER - 1 - i];
            }
            // Shift history left, append the new sample.
            self.history.rotate_left(1);
            self.history[LPC_ORDER - 1] = acc;
            *slot = acc;
        }
    }

    /// Like [`SynthesisFilter::process`] but takes integer excitation
    /// (the [`crate::excitation`] raw-sample domain) and produces
    /// `f64` synthesised samples.
    pub fn process_i32(&mut self, lpc: &[f64; LPC_ORDER], excitation: &[i32], out: &mut [f64]) {
        debug_assert_eq!(excitation.len(), out.len());
        for (slot, &e) in out.iter_mut().zip(excitation.iter()) {
            let mut acc = f64::from(e);
            for (i, &coeff) in lpc.iter().enumerate() {
                acc += coeff * self.history[LPC_ORDER - 1 - i];
            }
            self.history.rotate_left(1);
            self.history[LPC_ORDER - 1] = acc;
            *slot = acc;
        }
    }

    /// Filter an integer excitation block and write rounded, saturated
    /// `i16` PCM into `out`. History advances in the full-precision
    /// `f64` domain (only the *output* is quantised), so feedback is
    /// not degraded by per-sample rounding.
    pub fn process_into_i16(
        &mut self,
        lpc: &[f64; LPC_ORDER],
        excitation: &[i32],
        out: &mut [i16],
    ) {
        debug_assert_eq!(excitation.len(), out.len());
        for (slot, &e) in out.iter_mut().zip(excitation.iter()) {
            let mut acc = f64::from(e);
            for (i, &coeff) in lpc.iter().enumerate() {
                acc += coeff * self.history[LPC_ORDER - 1 - i];
            }
            self.history.rotate_left(1);
            self.history[LPC_ORDER - 1] = acc;
            *slot = saturate_i16(acc);
        }
    }

    /// Read-only view of the current filter history (most-recent last).
    pub fn history(&self) -> &[f64; LPC_ORDER] {
        &self.history
    }
}

/// Round to nearest and saturate an `f64` synthesised sample into the
/// signed 16-bit PCM range `[-32768, 32767]`.
fn saturate_i16(v: f64) -> i16 {
    let r = v.round();
    if r >= f64::from(i16::MAX) {
        i16::MAX
    } else if r <= f64::from(i16::MIN) {
        i16::MIN
    } else {
        r as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_lpc_passes_excitation_through() {
        // With all-zero LPC coefficients, A(z) = 1 and 1/A(z) = 1, so
        // the synthesis output equals the excitation.
        let mut f = SynthesisFilter::new();
        let lpc = [0.0_f64; LPC_ORDER];
        let exc = [1.0_f64, -2.0, 3.0, -4.0, 5.0];
        let mut out = [0.0_f64; 5];
        f.process(&lpc, &exc, &mut out);
        assert_eq!(out, exc);
    }

    #[test]
    fn single_pole_geometric_response() {
        // A(z) = 1 − a·z⁻¹ ⇒ 1/A(z) is a one-pole filter:
        //   x[n] = e[n] + a·x[n−1].
        // A unit impulse excitation produces the geometric series
        // 1, a, a², a³, … .
        let mut f = SynthesisFilter::new();
        let a = 0.5_f64;
        let mut lpc = [0.0_f64; LPC_ORDER];
        lpc[0] = a; // coefficient for x[n-1]
        let exc = [1.0_f64, 0.0, 0.0, 0.0, 0.0];
        let mut out = [0.0_f64; 5];
        f.process(&lpc, &exc, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - a).abs() < 1e-12);
        assert!((out[2] - a * a).abs() < 1e-12);
        assert!((out[3] - a * a * a).abs() < 1e-12);
        assert!((out[4] - a.powi(4)).abs() < 1e-12);
    }

    #[test]
    fn history_carries_across_calls() {
        // Two successive process() calls must produce the same output
        // as one call over the concatenated block — proving the IIR
        // memory persists across sub-frame boundaries.
        let a = 0.5_f64;
        let mut lpc = [0.0_f64; LPC_ORDER];
        lpc[0] = a;
        let exc = [1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0];

        let mut whole = SynthesisFilter::new();
        let mut out_whole = [0.0_f64; 6];
        whole.process(&lpc, &exc, &mut out_whole);

        let mut split = SynthesisFilter::new();
        let mut out_a = [0.0_f64; 3];
        let mut out_b = [0.0_f64; 3];
        split.process(&lpc, &exc[0..3], &mut out_a);
        split.process(&lpc, &exc[3..6], &mut out_b);

        for i in 0..3 {
            assert!((out_whole[i] - out_a[i]).abs() < 1e-12);
            assert!((out_whole[3 + i] - out_b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn second_order_recurrence_matches_hand_computation() {
        // A(z) = 1 − a0 z⁻¹ − a1 z⁻²:
        //   x[n] = e[n] + a0·x[n-1] + a1·x[n-2].
        let mut lpc = [0.0_f64; LPC_ORDER];
        lpc[0] = 0.3;
        lpc[1] = -0.2;
        let exc = [2.0_f64, 1.0, 0.0, 0.0];
        let mut f = SynthesisFilter::new();
        let mut out = [0.0_f64; 4];
        f.process(&lpc, &exc, &mut out);
        // x0 = 2.
        // x1 = 1 + 0.3·2 = 1.6.
        // x2 = 0 + 0.3·1.6 + (−0.2)·2 = 0.48 − 0.4 = 0.08.
        // x3 = 0 + 0.3·0.08 + (−0.2)·1.6 = 0.024 − 0.32 = −0.296.
        assert!((out[0] - 2.0).abs() < 1e-12);
        assert!((out[1] - 1.6).abs() < 1e-12);
        assert!((out[2] - 0.08).abs() < 1e-12);
        assert!((out[3] - (-0.296)).abs() < 1e-12);
    }

    #[test]
    fn process_i32_matches_process_on_float_cast() {
        let mut lpc = [0.0_f64; LPC_ORDER];
        lpc[0] = 0.4;
        let exc_i = [10_i32, -5, 3, 0];
        let exc_f: Vec<f64> = exc_i.iter().map(|&v| f64::from(v)).collect();

        let mut fa = SynthesisFilter::new();
        let mut out_i = [0.0_f64; 4];
        fa.process_i32(&lpc, &exc_i, &mut out_i);

        let mut fb = SynthesisFilter::new();
        let mut out_f = [0.0_f64; 4];
        fb.process(&lpc, &exc_f, &mut out_f);

        for i in 0..4 {
            assert!((out_i[i] - out_f[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn saturate_i16_clamps_and_rounds() {
        assert_eq!(saturate_i16(0.4), 0);
        assert_eq!(saturate_i16(0.6), 1);
        assert_eq!(saturate_i16(-0.6), -1);
        assert_eq!(saturate_i16(40000.0), i16::MAX);
        assert_eq!(saturate_i16(-40000.0), i16::MIN);
        assert_eq!(saturate_i16(32767.0), i16::MAX);
        assert_eq!(saturate_i16(-32768.0), i16::MIN);
    }

    #[test]
    fn process_into_i16_saturates_loud_output() {
        // A near-unstable one-pole filter with a large impulse drives
        // the output past i16 range; the PCM must clamp, not wrap.
        let mut lpc = [0.0_f64; LPC_ORDER];
        lpc[0] = 0.99;
        let exc = [30000_i32, 30000, 30000, 30000];
        let mut f = SynthesisFilter::new();
        let mut out = [0_i16; 4];
        f.process_into_i16(&lpc, &exc, &mut out);
        // Output grows monotonically and must hit the ceiling.
        assert_eq!(out[3], i16::MAX);
    }

    #[test]
    fn default_is_zero_history() {
        let f = SynthesisFilter::default();
        assert_eq!(f.history(), &[0.0_f64; LPC_ORDER]);
    }
}
