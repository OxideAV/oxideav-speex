//! Perceptual noise-weighting filter `W(z) = A(z/γ1) / A(z/γ2)`
//! (encoder analysis-by-synthesis, round r382 scope).
//!
//! Speex — like every CELP coder — performs its closed-loop codebook
//! search in a *perceptually weighted* signal domain rather than the raw
//! signal domain, so that the quantisation noise is shaped to sit under
//! the formant peaks where the ear is least sensitive. *The Speex Codec
//! Manual* names the weighting filter directly (MAN §8.5 p.28, Eq. 8.1;
//! CELP companion §2.4):
//!
//! ```text
//! W(z) = A(z/γ1) / A(z/γ2)
//! ```
//!
//! with the reference perceptual constants **γ1 = 0.9** and
//! **γ2 = 0.6** (MAN §8.5 p.28 / companion §2.4). `A(z)` is the order-`N`
//! LPC analysis (prediction-error) filter of [`crate::lpc_analysis`], in
//! the same coefficient convention the rest of the crate uses:
//!
//! ```text
//! A(z) = 1 − Σ_{i=1}^{N} a[i]·z⁻ⁱ           (a stored 0-indexed:
//!                                            a[j] multiplies z⁻⁽ʲ⁺¹⁾)
//! ```
//!
//! ## Coefficient bandwidth expansion
//!
//! Substituting `z → z/γ` in `A(z)` scales the `i`-th tap by `γⁱ`:
//!
//! ```text
//! A(z/γ) = 1 − Σ_{i=1}^{N} (a[i]·γⁱ)·z⁻ⁱ
//! ```
//!
//! so the weighted coefficient vector is simply the analysis vector with
//! each tap multiplied by the corresponding power of `γ` (0-indexed:
//! `aw[j] = a[j]·γ^(j+1)`). This "bandwidth expansion" is a textbook
//! linear-prediction operation ([`weighted_coeffs`]); nothing here is
//! read from any external decoder — the two γ values and the filter
//! form are the only facts, both taken verbatim from the manual.
//!
//! ## Filter structure
//!
//! `W(z)` is a pole-zero filter: the numerator `A(z/γ1)` is an all-zero
//! (FIR prediction-error) filter and the denominator `1/A(z/γ2)` is an
//! all-pole (IIR synthesis) filter. [`PerceptualWeighting::process`]
//! runs the FIR then the IIR, carrying both histories across calls so a
//! frame can be filtered sub-frame by sub-frame with no block-boundary
//! discontinuity — exactly the continuity the analysis-by-synthesis
//! search requires when it weights a whole frame's target signal.
//!
//! ## Numeric domain
//!
//! Filtering is done in `f64`, consistent with [`crate::synthesis`] and
//! [`crate::lpc_analysis`]. The encoder search never needs a fixed-point
//! Q-format for the weighted domain (the *decoder's* fixed-point layout
//! is the documented Q-format gap, and it does not touch the weighting
//! filter, which is encoder-only).

use crate::lsp_to_lpc::LPC_ORDER;

/// Reference perceptual-weighting numerator bandwidth-expansion factor
/// `γ1` (MAN §8.5 p.28 / companion §2.4).
pub const WEIGHT_GAMMA1: f64 = 0.9;

/// Reference perceptual-weighting denominator bandwidth-expansion factor
/// `γ2` (MAN §8.5 p.28 / companion §2.4).
pub const WEIGHT_GAMMA2: f64 = 0.6;

/// Apply bandwidth expansion `aw[j] = a[j]·γ^(j+1)` to an order-`N` LPC
/// coefficient vector, producing the coefficients of `A(z/γ)`.
///
/// `a` is in the crate convention (`a[j]` multiplies `z⁻⁽ʲ⁺¹⁾` in
/// `A(z) = 1 − Σ a[i] z⁻ⁱ`); the returned vector is the same length and
/// convention, so it drops straight into the FIR / IIR filters below or
/// into [`crate::synthesis::SynthesisFilter`].
pub fn weighted_coeffs(a: &[f64; LPC_ORDER], gamma: f64) -> [f64; LPC_ORDER] {
    let mut out = [0.0_f64; LPC_ORDER];
    let mut g = gamma; // γ^(j+1) for j = 0
    for (slot, &coeff) in out.iter_mut().zip(a.iter()) {
        *slot = coeff * g;
        g *= gamma;
    }
    out
}

/// Perceptual noise-weighting filter `W(z) = A(z/γ1) / A(z/γ2)` with
/// persistent FIR + IIR history.
///
/// Construct with [`PerceptualWeighting::new`] (from an order-`N` LPC
/// analysis vector and the two γ factors), feed the signal through
/// [`PerceptualWeighting::process`], and the filter carries its memory
/// across calls. Recompute a new filter per frame from that frame's LPC
/// (bandwidth-expansion is per-frame), threading the previous filter's
/// history with [`PerceptualWeighting::with_history`] if sub-frame
/// continuity across an LPC update is desired.
#[derive(Debug, Clone)]
pub struct PerceptualWeighting {
    /// Numerator coefficients `A(z/γ1)` (bandwidth-expanded by γ1).
    num: [f64; LPC_ORDER],
    /// Denominator coefficients `A(z/γ2)` (bandwidth-expanded by γ2).
    den: [f64; LPC_ORDER],
    /// FIR input history: last `N` *input* samples, most-recent last.
    x_hist: [f64; LPC_ORDER],
    /// IIR output history: last `N` *output* samples, most-recent last.
    y_hist: [f64; LPC_ORDER],
}

impl PerceptualWeighting {
    /// Build a weighting filter from an order-`N` LPC analysis vector and
    /// the two bandwidth-expansion factors (`γ1` for the numerator,
    /// `γ2` for the denominator), with zero history.
    ///
    /// Pass [`WEIGHT_GAMMA1`] / [`WEIGHT_GAMMA2`] for the reference
    /// Speex weighting.
    pub fn new(a: &[f64; LPC_ORDER], gamma1: f64, gamma2: f64) -> Self {
        Self {
            num: weighted_coeffs(a, gamma1),
            den: weighted_coeffs(a, gamma2),
            x_hist: [0.0; LPC_ORDER],
            y_hist: [0.0; LPC_ORDER],
        }
    }

    /// Build the reference Speex weighting filter (`γ1 = 0.9`,
    /// `γ2 = 0.6`) from an LPC analysis vector.
    pub fn reference(a: &[f64; LPC_ORDER]) -> Self {
        Self::new(a, WEIGHT_GAMMA1, WEIGHT_GAMMA2)
    }

    /// Rebuild the filter for a new frame's LPC while preserving the
    /// input / output history of `prev`, so the weighted signal stays
    /// continuous across the per-frame LPC update.
    pub fn with_history(a: &[f64; LPC_ORDER], gamma1: f64, gamma2: f64, prev: &Self) -> Self {
        Self {
            num: weighted_coeffs(a, gamma1),
            den: weighted_coeffs(a, gamma2),
            x_hist: prev.x_hist,
            y_hist: prev.y_hist,
        }
    }

    /// Filter a signal block through `W(z) = A(z/γ1)/A(z/γ2)`, writing
    /// the weighted samples into `out` and advancing both histories.
    ///
    /// `input` and `out` must be the same length. The two-stage
    /// recurrence is
    ///
    /// ```text
    /// v[n] = x[n] − Σ_{i} num[i]·x[n−1−i]        (FIR: A(z/γ1))
    /// y[n] = v[n] + Σ_{i} den[i]·y[n−1−i]        (IIR: 1/A(z/γ2))
    /// ```
    pub fn process(&mut self, input: &[f64], out: &mut [f64]) {
        debug_assert_eq!(input.len(), out.len());
        for (slot, &x) in out.iter_mut().zip(input.iter()) {
            // FIR numerator A(z/γ1): v = x − Σ num[i]·x[n−1−i].
            let mut v = x;
            for (i, &c) in self.num.iter().enumerate() {
                v -= c * self.x_hist[LPC_ORDER - 1 - i];
            }
            self.x_hist.rotate_left(1);
            self.x_hist[LPC_ORDER - 1] = x;

            // IIR denominator 1/A(z/γ2): y = v + Σ den[i]·y[n−1−i].
            let mut y = v;
            for (i, &c) in self.den.iter().enumerate() {
                y += c * self.y_hist[LPC_ORDER - 1 - i];
            }
            self.y_hist.rotate_left(1);
            self.y_hist[LPC_ORDER - 1] = y;

            *slot = y;
        }
    }

    /// Read-only view of the numerator coefficients `A(z/γ1)`.
    pub fn numerator(&self) -> &[f64; LPC_ORDER] {
        &self.num
    }

    /// Read-only view of the denominator coefficients `A(z/γ2)`.
    pub fn denominator(&self) -> &[f64; LPC_ORDER] {
        &self.den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_coeffs_scale_by_gamma_powers() {
        let a = [
            1.0_f64, 0.5, -0.25, 0.1, -0.05, 0.02, -0.01, 0.005, -0.002, 0.001,
        ];
        let g = 0.9;
        let w = weighted_coeffs(&a, g);
        let mut expect_pow = g;
        for j in 0..LPC_ORDER {
            assert!((w[j] - a[j] * expect_pow).abs() < 1e-15, "tap {j}");
            expect_pow *= g;
        }
    }

    #[test]
    fn gamma_one_is_identity() {
        let a = [0.3_f64, -0.2, 0.1, 0.0, 0.05, -0.05, 0.02, -0.01, 0.0, 0.0];
        let w = weighted_coeffs(&a, 1.0);
        assert_eq!(w, a);
    }

    #[test]
    fn equal_gammas_give_unity_filter() {
        // W(z) = A(z/γ)/A(z/γ) = 1: the weighted output must equal the
        // input for any signal when γ1 == γ2.
        let a = [0.5_f64, -0.3, 0.2, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut w = PerceptualWeighting::new(&a, 0.8, 0.8);
        let x = [1.0_f64, -2.0, 3.0, 0.5, -1.5, 4.0, -0.25, 0.0];
        let mut out = [0.0_f64; 8];
        w.process(&x, &mut out);
        for i in 0..x.len() {
            assert!(
                (out[i] - x[i]).abs() < 1e-9,
                "sample {i}: {} vs {}",
                out[i],
                x[i]
            );
        }
    }

    #[test]
    fn zero_lpc_is_passthrough() {
        // A(z) = 1 for all-zero coefficients, so W(z) = 1/1 = 1.
        let a = [0.0_f64; LPC_ORDER];
        let mut w = PerceptualWeighting::reference(&a);
        let x = [7.0_f64, -3.0, 2.0, 9.0];
        let mut out = [0.0_f64; 4];
        w.process(&x, &mut out);
        assert_eq!(out, x);
    }

    #[test]
    fn history_carries_across_calls() {
        let a = [0.6_f64, -0.4, 0.25, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x = [1.0_f64, 2.0, -1.0, 3.0, 0.5, -2.0];

        let mut whole = PerceptualWeighting::reference(&a);
        let mut out_whole = [0.0_f64; 6];
        whole.process(&x, &mut out_whole);

        let mut split = PerceptualWeighting::reference(&a);
        let mut out_a = [0.0_f64; 3];
        let mut out_b = [0.0_f64; 3];
        split.process(&x[0..3], &mut out_a);
        split.process(&x[3..6], &mut out_b);

        for i in 0..3 {
            assert!((out_whole[i] - out_a[i]).abs() < 1e-12);
            assert!((out_whole[3 + i] - out_b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn first_sample_matches_hand_computation() {
        // With zero history, y[0] = x[0] − num·(0) + den·(0) = x[0];
        // y[1] = (x[1] − num[0]·x[0]) + den[0]·y[0].
        let a = [0.5_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let g1 = 0.9;
        let g2 = 0.6;
        let mut w = PerceptualWeighting::new(&a, g1, g2);
        let x = [4.0_f64, 8.0];
        let mut out = [0.0_f64; 2];
        w.process(&x, &mut out);
        let num0 = 0.5 * g1;
        let den0 = 0.5 * g2;
        let y0 = 4.0;
        let y1 = (8.0 - num0 * 4.0) + den0 * y0;
        assert!((out[0] - y0).abs() < 1e-12);
        assert!((out[1] - y1).abs() < 1e-12);
    }

    #[test]
    fn reference_gammas_are_documented_values() {
        assert_eq!(WEIGHT_GAMMA1, 0.9);
        assert_eq!(WEIGHT_GAMMA2, 0.6);
    }

    #[test]
    fn with_history_threads_state() {
        let a1 = [0.5_f64, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x = [1.0_f64, 2.0, 3.0, 4.0];
        let mut w = PerceptualWeighting::reference(&a1);
        let mut out = [0.0_f64; 2];
        w.process(&x[0..2], &mut out);
        // Rebuild with the same coefficients but preserved history:
        // continuing must equal one uninterrupted filter over x[0..4].
        let mut w2 = PerceptualWeighting::with_history(&a1, WEIGHT_GAMMA1, WEIGHT_GAMMA2, &w);
        let mut out_b = [0.0_f64; 2];
        w2.process(&x[2..4], &mut out_b);

        let mut whole = PerceptualWeighting::reference(&a1);
        let mut out_whole = [0.0_f64; 4];
        whole.process(&x, &mut out_whole);
        assert!((out_b[0] - out_whole[2]).abs() < 1e-12);
        assert!((out_b[1] - out_whole[3]).abs() < 1e-12);
    }
}
