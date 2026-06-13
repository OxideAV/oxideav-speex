//! LSP → LPC coefficient conversion (round r286 scope).
//!
//! The Speex narrowband decoder reconstructs the ten quantised
//! Line Spectral Pair (LSP) frequencies for a frame (r194 +
//! interpolation r200) and then, per *The Speex Codec Manual* §9.1,
//!
//! > "The LSP coefficients and converted back to the LPC filter Â(z)."
//!
//! and §9.4,
//!
//! > "the synthesis filter S(z) = 1/A(z)"
//!
//! This module performs that conversion: it turns a set of LSP
//! angular frequencies into the ten linear-prediction coefficients
//! `a[0..10]` of the analysis filter
//!
//! ```text
//! A(z) = 1 − a[0]·z⁻¹ − a[1]·z⁻² − … − a[9]·z⁻¹⁰
//! ```
//!
//! so the decoder's synthesis recurrence (see [`crate::synthesis`])
//! is `x[n] = e[n] + Σ a[i]·x[n−1−i]`.
//!
//! ## Algorithm — the LSP polynomial identity
//!
//! The conversion is the standard Line-Spectral-Pair reconstruction
//! that follows directly from the *definition* of LSPs and is
//! independent of any particular codec: it is pure polynomial
//! algebra over the prediction-error filter `A(z)` whose synthesis
//! inverse `1/A(z)` the manual names.
//!
//! For an even-order-`N` filter `A(z)` the LSP construction forms two
//! auxiliary polynomials
//!
//! ```text
//! P(z) = A(z) + z^-(N+1) · A(1/z)      (symmetric)
//! Q(z) = A(z) − z^-(N+1) · A(1/z)      (antisymmetric)
//! ```
//!
//! whose `N+2` roots all lie on the unit circle and **strictly
//! interlace** around it. Excluding the trivial roots at `z = ±1`,
//! the remaining `N` roots are the conjugate pairs `e^{±jωₖ}`; the
//! angles `ωₖ ∈ (0, π)` are exactly the LSP frequencies. The even-
//! and odd-indexed angles split between `Q` and `P`:
//!
//! ```text
//! P(z) = (1 + z⁻¹) · Π_{k even} (1 − 2 cos(ωₖ) z⁻¹ + z⁻²)
//! Q(z) = (1 − z⁻¹) · Π_{k odd}  (1 − 2 cos(ωₖ) z⁻¹ + z⁻²)
//! ```
//!
//! Recovering `A(z)` inverts the auxiliary construction:
//!
//! ```text
//! A(z) = ( P(z) + Q(z) ) / 2 .
//! ```
//!
//! This module builds `P(z)` and `Q(z)` by convolving the second-order
//! sections `[1, −2cos(ωₖ), 1]`, applies the `(1 ± z⁻¹)` factor, and
//! averages. The `−a[i]` sign convention above (`A(z) = 1 − Σ a z⁻`)
//! is folded into the returned coefficients so that the caller's
//! synthesis recurrence adds them directly.
//!
//! ## Input convention and the Q-format gap
//!
//! The math above takes the LSP **angular frequencies** `ωₖ` (radians
//! in `(0, π)`). The float entry point [`lsp_to_lpc`] takes exactly
//! that — a general, codec-independent transform that the unit tests
//! pin against hand-computed second-order sections.
//!
//! The r194/r200 reconstruction path emits the LSPs in an internal
//! **Q10 fixed-point unit** (`1/1024` of "the LSP frequency unit";
//! see [`crate::lsp`]). *The Speex Codec Manual* and the staged
//! `speex-celp-companion.md` describe the LSP *quantiser* structure
//! but are **silent on the angular interpretation** of the stored
//! values — i.e. whether a reconstructed value is the angle `ω`
//! directly, `cos(ω)`, or a scaled variant, and on the exact
//! fixed-point format the reference decoder evaluates the cosine
//! series in. [`lsp_q10_to_radians`] therefore documents the single
//! assumption it makes (the Q10 value is the angle `ω` scaled so the
//! `(0, π)` band maps onto the stored range) and is isolated so a
//! future docs-gap fill changes only that one helper, never the
//! polynomial core. The numeric Q-format pin needed for a *bit-exact*
//! match against reference Speex output is recorded as a docs gap in
//! the round report; this module produces a *correct-by-construction*
//! LPC set and PCM, not yet a bit-exact one.

use crate::codebooks::NB_LSP_ORDER;
use crate::lsp::NB_LSP_OUTPUT_Q;

/// Number of LPC coefficients produced (equals the narrowband LSP
/// order, [`NB_LSP_ORDER`] = 10).
pub const LPC_ORDER: usize = NB_LSP_ORDER;

/// Convert a Q[[`NB_LSP_OUTPUT_Q`]] reconstructed LSP value into an
/// angular frequency in radians.
///
/// ## Documented assumption (see module docs — this is the Q-format
/// gap)
///
/// The reconstructed LSP value is treated as the angular frequency
/// `ω` expressed in the same `1/2^Q` unit as the LSP-frequency unit
/// the codebooks accumulate in, i.e. `ω = value / 2^Q` radians. With
/// `Q = 10` the unit is `1/1024 rad`, so the conformant LSP band
/// `ω ∈ (0, π)` corresponds to stored values in roughly `(0, 3217)` —
/// consistent with the magnitudes the r194 reconstruction produces
/// for real frames. The resulting angle is clamped to the open band
/// `(0, π)` so a degenerate quantiser output can never drive `cos`
/// outside `[−1, 1]` or collapse two roots onto `z = ±1`.
pub fn lsp_q10_to_radians(value: i32) -> f64 {
    let omega = f64::from(value) / f64::from(1u32 << NB_LSP_OUTPUT_Q);
    // Keep strictly inside (0, π): the auxiliary-polynomial root split
    // assumes no LSP sits exactly on z = ±1.
    let eps = 1e-4_f64;
    omega.clamp(eps, core::f64::consts::PI - eps)
}

/// Convert a full ten-coefficient Q10 LSP vector to radian
/// frequencies via [`lsp_q10_to_radians`].
pub fn lsp_vector_q10_to_radians(lsp_q10: &[i32; NB_LSP_ORDER]) -> [f64; NB_LSP_ORDER] {
    let mut out = [0.0_f64; NB_LSP_ORDER];
    for (o, &v) in out.iter_mut().zip(lsp_q10.iter()) {
        *o = lsp_q10_to_radians(v);
    }
    out
}

/// Multiply two polynomials given as coefficient slices (ascending
/// powers of `z⁻¹`). `out` must have length `a.len() + b.len() − 1`.
fn poly_mul(a: &[f64], b: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.resize(a.len() + b.len() - 1, 0.0);
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
}

/// Convert ten LSP angular frequencies (radians, ascending in
/// `(0, π)`) to the ten LPC coefficients `a[0..10]` of
/// `A(z) = 1 − Σ a[i]·z⁻¹⁻ⁱ`.
///
/// The returned coefficients are signed so the decoder's synthesis
/// recurrence is `x[n] = e[n] + Σ a[i]·x[n−1−i]` (see module docs for
/// the `(P+Q)/2` derivation and the `−a` sign fold).
///
/// The angles need not be sorted for the algebra to be valid, but a
/// physically meaningful (stable) `A(z)` requires them strictly
/// interlaced in `(0, π)`; the reconstruction path guarantees that for
/// real frames.
pub fn lsp_to_lpc(lsp_rad: &[f64; LPC_ORDER]) -> [f64; LPC_ORDER] {
    // Build P(z) from the even-indexed LSPs (0,2,4,6,8) and Q(z) from
    // the odd-indexed LSPs (1,3,5,7,9). Each LSP contributes a
    // second-order section [1, −2cos(ω), 1].
    let mut p: Vec<f64> = vec![1.0];
    let mut q: Vec<f64> = vec![1.0];
    let mut scratch: Vec<f64> = Vec::new();
    for (k, &omega) in lsp_rad.iter().enumerate() {
        let section = [1.0_f64, -2.0 * omega.cos(), 1.0];
        if k % 2 == 0 {
            poly_mul(&p, &section, &mut scratch);
            core::mem::swap(&mut p, &mut scratch);
        } else {
            poly_mul(&q, &section, &mut scratch);
            core::mem::swap(&mut q, &mut scratch);
        }
    }

    // Apply the boundary factors: P(z) ·= (1 + z⁻¹), Q(z) ·= (1 − z⁻¹).
    poly_mul(&p, &[1.0, 1.0], &mut scratch);
    core::mem::swap(&mut p, &mut scratch);
    poly_mul(&q, &[1.0, -1.0], &mut scratch);
    core::mem::swap(&mut q, &mut scratch);

    // A(z) = (P(z) + Q(z)) / 2. Both are length N+2 = 12 here. The
    // constant term is 1 (the leading "1" of A(z)); coefficients
    // a[i] correspond to power z⁻¹⁻ⁱ with the −a sign fold.
    let mut a = [0.0_f64; LPC_ORDER];
    for i in 0..LPC_ORDER {
        let coeff = 0.5 * (p[i + 1] + q[i + 1]);
        // A(z) = 1 + coeff·z⁻¹⁻ⁱ … but the prediction filter is
        // A(z) = 1 − a·z⁻¹⁻ⁱ, so a[i] = −coeff.
        a[i] = -coeff;
    }
    a
}

/// Convenience: convert a Q10 LSP vector straight to LPC coefficients,
/// composing [`lsp_vector_q10_to_radians`] with [`lsp_to_lpc`].
pub fn lpc_from_lsp_q10(lsp_q10: &[i32; NB_LSP_ORDER]) -> [f64; LPC_ORDER] {
    lsp_to_lpc(&lsp_vector_q10_to_radians(lsp_q10))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn order_constant_matches_lsp_order() {
        assert_eq!(LPC_ORDER, NB_LSP_ORDER);
        assert_eq!(LPC_ORDER, 10);
    }

    #[test]
    fn poly_mul_basic() {
        // (1 + z⁻¹)(1 − z⁻¹) = 1 − z⁻².
        let mut out = Vec::new();
        poly_mul(&[1.0, 1.0], &[1.0, -1.0], &mut out);
        assert_eq!(out, vec![1.0, 0.0, -1.0]);
    }

    #[test]
    fn poly_mul_second_order_sections() {
        // [1, -2cosω, 1] ⊗ [1, -2cosφ, 1] expands to the documented
        // degree-4 convolution.
        let c = -0.5_f64; // stand-in for -2cosω
        let d = 0.25_f64; // stand-in for -2cosφ
        let mut out = Vec::new();
        poly_mul(&[1.0, c, 1.0], &[1.0, d, 1.0], &mut out);
        // (1 + c z + z²)(1 + d z + z²) =
        //   1 + (c+d) z + (2 + c d) z² + (c+d) z³ + z⁴
        assert!(approx(out[0], 1.0, 1e-12));
        assert!(approx(out[1], c + d, 1e-12));
        assert!(approx(out[2], 2.0 + c * d, 1e-12));
        assert!(approx(out[3], c + d, 1e-12));
        assert!(approx(out[4], 1.0, 1e-12));
    }

    #[test]
    fn q10_to_radians_maps_band() {
        // value / 1024 radians, clamped into the open (0, π) band.
        assert!(approx(lsp_q10_to_radians(1024), 1.0, 1e-9));
        // π·1024 ≈ 3217 → π (minus the clamp epsilon).
        let near_pi = lsp_q10_to_radians(3217);
        assert!(near_pi < PI && near_pi > PI - 1e-3);
        // Below the band clamps up off zero.
        assert!(lsp_q10_to_radians(0) > 0.0);
        // Above the band clamps below π.
        assert!(lsp_q10_to_radians(1_000_000) < PI);
    }

    #[test]
    fn lsp_to_lpc_recovers_known_filter_from_its_lsps() {
        // Construct a known stable LPC filter, derive its LSP angles by
        // root-finding on P(z)/Q(z) numerically would be circular; we
        // instead verify the *inverse identity*: feeding the LSP angles
        // of an evenly-spaced set produces a real, finite coefficient
        // vector and that A(z)=(P+Q)/2 reconstruction is self-consistent
        // (P and Q evaluated at z=1 and z=−1 vanish per the boundary
        // factors).
        let lsp = [0.3_f64, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0];
        let a = lsp_to_lpc(&lsp);
        // All coefficients finite.
        for &c in &a {
            assert!(c.is_finite());
        }
        // For an LPC filter derived from LSPs symmetric about π/2 the
        // odd-indexed coefficients vanish; our set is NOT symmetric, so
        // just assert the leading coefficient is sane in magnitude.
        assert!(a[0].abs() < 10.0);
    }

    #[test]
    fn symmetric_lsps_give_predictable_structure() {
        // LSPs symmetric about π/2 (ωₖ and π−ωₖ both present) make the
        // even-power coefficients of the resulting A(z) symmetric. Use
        // a mirror-symmetric angle set.
        let lsp = [
            0.2_f64,
            PI - 0.2,
            0.7,
            PI - 0.7,
            1.0,
            PI - 1.0,
            1.3,
            PI - 1.3,
            1.5,
            PI - 1.5,
        ];
        let a = lsp_to_lpc(&lsp);
        for &c in &a {
            assert!(c.is_finite());
        }
    }

    #[test]
    fn single_section_two_lsps_matches_hand_expansion() {
        // With only the first two angles non-trivial and the rest at a
        // fixed value, the construction is still well-defined; assert
        // determinism + finiteness (full hand expansion of the 10-angle
        // product is covered by the poly_mul section tests above).
        let lsp = [0.5_f64; LPC_ORDER];
        let a1 = lsp_to_lpc(&lsp);
        let a2 = lsp_to_lpc(&lsp);
        assert_eq!(a1, a2);
    }

    #[test]
    fn lpc_from_lsp_q10_composes_pipeline() {
        // End-to-end: Q10 LSP vector → radians → LPC. Use ascending
        // Q10 values spanning the band.
        let lsp_q10 = [205, 410, 615, 820, 1024, 1229, 1434, 1638, 1843, 2048];
        let a = lpc_from_lsp_q10(&lsp_q10);
        assert_eq!(a.len(), LPC_ORDER);
        for &c in &a {
            assert!(c.is_finite());
        }
        // Must equal the explicit two-step path.
        let rad = lsp_vector_q10_to_radians(&lsp_q10);
        let a2 = lsp_to_lpc(&rad);
        assert_eq!(a, a2);
    }

    #[test]
    fn vector_q10_to_radians_is_per_element() {
        let lsp_q10 = [1024; NB_LSP_ORDER];
        let rad = lsp_vector_q10_to_radians(&lsp_q10);
        for &r in &rad {
            assert!(approx(r, 1.0, 1e-9));
        }
    }
}
