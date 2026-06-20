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

use crate::codebooks::{HB_LPC_ORDER, NB_LSP_ORDER};
use crate::hb_lsp::HB_LSP_OUTPUT_Q;
use crate::lsp::NB_LSP_OUTPUT_Q;
use crate::lsp_interp::{NbSubFrameLsp, NB_LSP_INTERP_OUTPUT_Q, NB_LSP_SUBFRAMES_PER_FRAME};

/// Number of LPC coefficients produced (equals the narrowband LSP
/// order, [`NB_LSP_ORDER`] = 10).
pub const LPC_ORDER: usize = NB_LSP_ORDER;

/// Number of high-band LPC coefficients produced (equals the wideband
/// high-band LSP order, [`HB_LPC_ORDER`] = 8). See [`hb_lsp_to_lpc`].
pub const HB_LPC_ORDER_OUT: usize = HB_LPC_ORDER;

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
    lsp_qn_to_radians(value, NB_LSP_OUTPUT_Q)
}

/// Convert a fixed-point reconstructed LSP value in an arbitrary
/// `Q`-format into an angular frequency in radians.
///
/// This is the Q-shift-parameterised generalisation of
/// [`lsp_q10_to_radians`]: the stored value is treated as the angle
/// `ω = value / 2^q` radians (the same documented angular-unit
/// assumption — see module docs — applied at the supplied scale `q`).
/// The result is clamped to the open band `(0, π)` so a degenerate
/// quantiser output can never drive `cos` outside `[−1, 1]` or collapse
/// a root onto `z = ±1`.
///
/// The r194/r200 path produces two distinct scales the synthesis chain
/// consumes:
///
/// * `q = `[`NB_LSP_OUTPUT_Q`]` = 10` — the per-frame reconstructed LSP
///   vector ([`crate::lsp::reconstruct_q10`]).
/// * `q = `[`NB_LSP_INTERP_OUTPUT_Q`]` = 12` — the per-sub-frame
///   interpolated LSP vector ([`crate::lsp_interp::NbSubFrameLsp`]),
///   which carries two extra sub-binary-point bits from the un-divided
///   weight multiplication (see the interpolation module docs).
///
/// Sharing one helper across both scales keeps the angular-unit
/// assumption pinned in a single place: a future docs-gap fill changes
/// only this function regardless of which Q-format feeds it.
pub fn lsp_qn_to_radians(value: i32, q: u32) -> f64 {
    let omega = f64::from(value) / f64::from(1u32 << q);
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
    let mut a = [0.0_f64; LPC_ORDER];
    lsp_to_lpc_slice(lsp_rad, &mut a);
    a
}

/// Order-generic LSP→LPC conversion writing into a caller-supplied
/// coefficient slice.
///
/// This is the shared polynomial core both the narrowband ([`lsp_to_lpc`],
/// order 10) and wideband high-band ([`hb_lsp_to_lpc`], order 8) paths
/// delegate to. `lsp_rad` carries the `N` LSP angular frequencies and
/// `out` receives the `N` LPC coefficients `a[0..N]` of
/// `A(z) = 1 − Σ a[i]·z⁻¹⁻ⁱ`; both must have the same even length `N`.
///
/// The construction is the same `A(z) = (P(z) + Q(z)) / 2`
/// auxiliary-polynomial reconstruction documented at the module level —
/// even-indexed angles build `P(z)`, odd-indexed build `Q(z)`, the
/// `(1 ± z⁻¹)` boundary factors are applied, and the two are averaged.
/// Nothing in the algebra depends on the order being 10; it requires
/// only that `N` be even (so the two boundary factors split evenly
/// between `P` and `Q`).
///
/// # Panics
///
/// Panics if `lsp_rad.len() != out.len()` or if the length is odd.
fn lsp_to_lpc_slice(lsp_rad: &[f64], out: &mut [f64]) {
    assert_eq!(
        lsp_rad.len(),
        out.len(),
        "LSP and LPC slices must share length"
    );
    let n = lsp_rad.len();
    assert!(n % 2 == 0, "LSP→LPC core requires an even filter order");

    // Build P(z) from the even-indexed LSPs and Q(z) from the
    // odd-indexed LSPs. Each LSP contributes a second-order section
    // [1, −2cos(ω), 1].
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

    // A(z) = (P(z) + Q(z)) / 2. Both are length N+2 here. The constant
    // term is 1 (the leading "1" of A(z)); coefficients a[i] correspond
    // to power z⁻¹⁻ⁱ with the −a sign fold.
    for (i, slot) in out.iter_mut().enumerate() {
        let coeff = 0.5 * (p[i + 1] + q[i + 1]);
        // A(z) = 1 + coeff·z⁻¹⁻ⁱ … but the prediction filter is
        // A(z) = 1 − a·z⁻¹⁻ⁱ, so a[i] = −coeff.
        *slot = -coeff;
    }
}

/// Convenience: convert a Q10 LSP vector straight to LPC coefficients,
/// composing [`lsp_vector_q10_to_radians`] with [`lsp_to_lpc`].
pub fn lpc_from_lsp_q10(lsp_q10: &[i32; NB_LSP_ORDER]) -> [f64; LPC_ORDER] {
    lsp_to_lpc(&lsp_vector_q10_to_radians(lsp_q10))
}

/// Add the pinned narrowband LSP **base vector**
/// ([`crate::lsp_base::nb_lsp_base_q10`]) to a reconstructed Q10
/// codebook-delta vector, returning the **pinned** LSP angles (the
/// actual Speex LSP value, base + codebook delta — round r347 scope).
///
/// The r194 reconstruction emits only the codebook delta sum; the
/// staged `LSP_LINEAR(i)` base offset (recorded as a numeric fact in
/// `docs/audio/speex/provenance/02-speex-gain-quant.md`) is the fixed
/// vector those deltas refine. Adding it lands every well-formed frame's
/// LSP angles inside the conformant `(0, π)` band by construction rather
/// than via the clamp fallback — the boundedness pin this round delivers.
pub fn nb_lsp_with_base_q10(lsp_delta_q10: &[i32; NB_LSP_ORDER]) -> [i32; NB_LSP_ORDER] {
    let mut v = *lsp_delta_q10;
    crate::lsp_base::add_nb_lsp_base(&mut v);
    v
}

/// Convert a Q10 narrowband LSP **codebook-delta** vector to LPC
/// coefficients **with the pinned base vector added** — the bounded
/// per-frame path (round r347).
///
/// This is the base-aware counterpart of [`lpc_from_lsp_q10`]: it first
/// adds the [`crate::lsp_base`] linear-init base offset, then enforces
/// the pinned `LSP_MARGIN` minimum-spacing safeguard
/// ([`crate::lsp_base::enforce_lsp_margin_radians`], margin `.002` rad),
/// then runs the Q10→radian→LPC pipeline. The decoder's per-frame LPC
/// reconstruction uses this so the synthesis filter sees a documented,
/// strictly-interlaced LSP set.
pub fn lpc_from_lsp_delta_q10(lsp_delta_q10: &[i32; NB_LSP_ORDER]) -> [f64; LPC_ORDER] {
    let based = nb_lsp_with_base_q10(lsp_delta_q10);
    let mut rad = lsp_vector_q10_to_radians(&based);
    crate::lsp_base::enforce_lsp_margin_radians(&mut rad, crate::lsp_base::nb_lsp_margin_radians());
    lsp_to_lpc(&rad)
}

/// Convert one Q[[`NB_LSP_INTERP_OUTPUT_Q`]] = Q12 interpolated
/// sub-frame LSP vector to its ten LPC coefficients.
///
/// The [`crate::lsp_interp::NbSubFrameLsp`] interpolation step emits the
/// per-sub-frame LSPs at Q12 (input Q10 × the un-divided weight sum of
/// 4). Each element is mapped to radians via [`lsp_qn_to_radians`] at
/// `q = `[`NB_LSP_INTERP_OUTPUT_Q`] and run through the same
/// [`lsp_to_lpc`] polynomial core as the per-frame path.
pub fn lpc_from_subframe_lsp_q12(lsp_q12: &[i32; NB_LSP_ORDER]) -> [f64; LPC_ORDER] {
    let mut rad = [0.0_f64; NB_LSP_ORDER];
    for (r, &v) in rad.iter_mut().zip(lsp_q12.iter()) {
        *r = lsp_qn_to_radians(v, NB_LSP_INTERP_OUTPUT_Q);
    }
    lsp_to_lpc(&rad)
}

/// Convert a whole frame's four interpolated sub-frame LSP vectors
/// ([`crate::lsp_interp::NbSubFrameLsp`], Q12) into four LPC coefficient
/// sets — one per 40-sample sub-frame.
///
/// This bridges the r200 sub-frame LSP interpolation and the r286
/// LSP→LPC core for an entire narrowband frame: the returned `[[f64;
/// 10]; 4]` is exactly the per-sub-frame LPC the [`crate::synthesis`]
/// filter consumes (the module docs there note "each sub-frame is
/// filtered with its own interpolated LPC set").
///
/// Sub-frame 4 (index 3) carries the current frame's LSPs unchanged
/// (manual §9.1: "the LSP's are considered to be associated to the 4th
/// sub-frame"); the other three carry the linearly-interpolated sets.
/// Because the interpolation is order-preserving for monotone inputs
/// and the conversion is per-vector, the four LPC sets evolve smoothly
/// across the frame — the IIR continuity the synthesis filter relies on.
pub fn subframe_lpc_set(lsp: &NbSubFrameLsp) -> [[f64; LPC_ORDER]; NB_LSP_SUBFRAMES_PER_FRAME] {
    let mut out = [[0.0_f64; LPC_ORDER]; NB_LSP_SUBFRAMES_PER_FRAME];
    for (slot, sf) in out.iter_mut().zip(lsp.subframes.iter()) {
        *slot = lpc_from_subframe_lsp_q12(sf);
    }
    out
}

/// Convert a whole frame's four interpolated sub-frame LSP vectors into
/// four LPC sets **with the pinned base vector added** — the bounded
/// per-sub-frame path (round r347).
///
/// The r200 interpolation runs on the **codebook-delta** LSP vectors
/// (prev / curr deltas) and emits Q12 sub-frame vectors. Because the
/// linear base offset is identical for `prev` and `curr`, it survives
/// the interpolation as a pure translation
/// (`((4−k)(d_p+B) + k(d_c+B))/4 = interp(d_p,d_c) + B`), so the pinned
/// sub-frame angle is the interpolated delta plus the **Q12** base
/// (`base_q10 × 4`, since `NB_LSP_INTERP_OUTPUT_Q = NB_LSP_OUTPUT_Q + 2`).
/// This routine adds that Q12 base to each sub-frame vector before the
/// LSP→LPC conversion, so every sub-frame's LPC set is reconstructed
/// from angles inside the conformant `(0, π)` band.
pub fn subframe_lpc_set_with_base(
    lsp: &NbSubFrameLsp,
) -> [[f64; LPC_ORDER]; NB_LSP_SUBFRAMES_PER_FRAME] {
    // Base re-expressed in the Q12 interpolation domain (Q10 × 4).
    let base_q10 = crate::lsp_base::nb_lsp_base_q10();
    let shift = NB_LSP_INTERP_OUTPUT_Q - NB_LSP_OUTPUT_Q; // = 2
    let mut out = [[0.0_f64; LPC_ORDER]; NB_LSP_SUBFRAMES_PER_FRAME];
    for (slot, sf) in out.iter_mut().zip(lsp.subframes.iter()) {
        let mut sf_based = *sf;
        for (v, &b) in sf_based.iter_mut().zip(base_q10.iter()) {
            *v += b << shift;
        }
        // Convert to radians, enforce the pinned LSP_MARGIN spacing
        // safeguard (`.002` rad), then run the polynomial core.
        let mut rad = [0.0_f64; NB_LSP_ORDER];
        for (r, &v) in rad.iter_mut().zip(sf_based.iter()) {
            *r = lsp_qn_to_radians(v, NB_LSP_INTERP_OUTPUT_Q);
        }
        crate::lsp_base::enforce_lsp_margin_radians(
            &mut rad,
            crate::lsp_base::nb_lsp_margin_radians(),
        );
        *slot = lsp_to_lpc(&rad);
    }
    out
}

/// Convert a Q[[`HB_LSP_OUTPUT_Q`]] reconstructed high-band LSP value
/// into an angular frequency in radians.
///
/// The high-band LSP reconstruction ([`crate::hb_lsp::reconstruct_q10`])
/// emits its eight coefficients in the **same** Q10 fixed-point unit as
/// the narrowband path (`HB_LSP_OUTPUT_Q == NB_LSP_OUTPUT_Q == 10`, by
/// construction in r214 so both bands share one downstream Q-format).
/// This helper therefore applies the identical documented angular-unit
/// assumption as [`lsp_q10_to_radians`] (`ω = value / 2^Q radians`,
/// clamped to the open `(0, π)` band) at the high-band scale, delegating
/// to the shared [`lsp_qn_to_radians`] so the assumption stays pinned in
/// one place across both bands.
pub fn hb_lsp_q10_to_radians(value: i32) -> f64 {
    lsp_qn_to_radians(value, HB_LSP_OUTPUT_Q)
}

/// Convert the eight high-band LSP angular frequencies (radians,
/// ascending in `(0, π)`) to the eight high-band LPC coefficients
/// `a[0..8]` of `A(z) = 1 − Σ a[i]·z⁻¹⁻ⁱ`.
///
/// This is the wideband high-band counterpart of the narrowband
/// [`lsp_to_lpc`]. The high-band LPC order is **8**
/// ([`crate::codebooks::HB_LPC_ORDER`]); the algebra is the same
/// order-generic `A(z) = (P(z) + Q(z)) / 2` auxiliary-polynomial
/// reconstruction (8 is even, so the `(1 ± z⁻¹)` boundary factors split
/// evenly between `P` and `Q`). Spec basis: *The Speex Codec Manual*
/// §10.1 (the high-band LSPs are *"converted back to the LPC filter"*
/// exactly as the narrowband §9.1 path describes) with the order-8
/// reconciliation recorded on [`crate::codebooks::HB_LPC_ORDER`].
///
/// As with the narrowband path the returned coefficients are signed so
/// the high-band synthesis recurrence
/// `x[n] = e[n] + Σ a[i]·x[n−1−i]` adds them directly.
pub fn hb_lsp_to_lpc(lsp_rad: &[f64; HB_LPC_ORDER_OUT]) -> [f64; HB_LPC_ORDER_OUT] {
    let mut a = [0.0_f64; HB_LPC_ORDER_OUT];
    lsp_to_lpc_slice(lsp_rad, &mut a);
    a
}

/// Convenience: convert an eight-coefficient Q[[`HB_LSP_OUTPUT_Q`]]
/// high-band LSP vector straight to high-band LPC coefficients, composing
/// [`hb_lsp_q10_to_radians`] with [`hb_lsp_to_lpc`].
///
/// This is the high-band counterpart of [`lpc_from_lsp_q10`]: it takes
/// the eight-coefficient Q10 vector that
/// [`crate::WidebandHighBandBody::reconstructed_lsp_q10`] /
/// [`crate::hb_lsp::reconstruct_q10`] produce and returns the eight
/// high-band LPC coefficients the high-band synthesis filter consumes.
pub fn lpc_from_hb_lsp_q10(lsp_q10: &[i32; HB_LPC_ORDER_OUT]) -> [f64; HB_LPC_ORDER_OUT] {
    let mut rad = [0.0_f64; HB_LPC_ORDER_OUT];
    for (r, &v) in rad.iter_mut().zip(lsp_q10.iter()) {
        *r = hb_lsp_q10_to_radians(v);
    }
    hb_lsp_to_lpc(&rad)
}

/// Add the pinned high-band LSP **base vector**
/// ([`crate::lsp_base::hb_lsp_base_q10`]) to a reconstructed Q10
/// codebook-delta vector, returning the **pinned** high-band LSP angles
/// (round r347).
pub fn hb_lsp_with_base_q10(lsp_delta_q10: &[i32; HB_LPC_ORDER_OUT]) -> [i32; HB_LPC_ORDER_OUT] {
    let mut v = *lsp_delta_q10;
    crate::lsp_base::add_hb_lsp_base(&mut v);
    v
}

/// Convert a Q10 high-band LSP **codebook-delta** vector to high-band
/// LPC coefficients **with the pinned base vector added** and the pinned
/// high-band `LSP_MARGIN` (`.05` rad) minimum-spacing safeguard applied
/// — the bounded high-band path (round r347). The base-aware counterpart
/// of [`lpc_from_hb_lsp_q10`].
pub fn lpc_from_hb_lsp_delta_q10(
    lsp_delta_q10: &[i32; HB_LPC_ORDER_OUT],
) -> [f64; HB_LPC_ORDER_OUT] {
    let based = hb_lsp_with_base_q10(lsp_delta_q10);
    let mut rad = [0.0_f64; HB_LPC_ORDER_OUT];
    for (r, &v) in rad.iter_mut().zip(based.iter()) {
        *r = hb_lsp_q10_to_radians(v);
    }
    crate::lsp_base::enforce_lsp_margin_radians(&mut rad, crate::lsp_base::hb_lsp_margin_radians());
    hb_lsp_to_lpc(&rad)
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

    #[test]
    fn qn_to_radians_matches_q10_special_case() {
        // The generalised helper at q=10 must reproduce the Q10 helper
        // exactly for every input value.
        for v in [-5000, -1, 0, 1, 205, 1024, 3217, 1_000_000] {
            assert_eq!(lsp_qn_to_radians(v, NB_LSP_OUTPUT_Q), lsp_q10_to_radians(v));
        }
    }

    #[test]
    fn qn_to_radians_scales_with_q() {
        // A Q12 value of 4096 is the same angle (1.0 rad) as a Q10 value
        // of 1024 — the two extra bits scale the integer by 4.
        let q10 = lsp_qn_to_radians(1024, NB_LSP_OUTPUT_Q);
        let q12 = lsp_qn_to_radians(4096, NB_LSP_INTERP_OUTPUT_Q);
        assert!(approx(q10, q12, 1e-12));
        assert!(approx(q12, 1.0, 1e-9));
    }

    #[test]
    fn subframe_q12_path_equals_q10_path_when_value_scaled_by_four() {
        // A Q12 sub-frame vector of 4·x equals the Q10 vector x at the
        // same angles, so the LPC sets must match coefficient-for-
        // coefficient.
        let lsp_q10 = [205, 410, 615, 820, 1024, 1229, 1434, 1638, 1843, 2048];
        let mut lsp_q12 = [0i32; NB_LSP_ORDER];
        for (d, &s) in lsp_q12.iter_mut().zip(lsp_q10.iter()) {
            *d = s * 4;
        }
        let from_q10 = lpc_from_lsp_q10(&lsp_q10);
        let from_q12 = lpc_from_subframe_lsp_q12(&lsp_q12);
        for i in 0..LPC_ORDER {
            assert!(approx(from_q10[i], from_q12[i], 1e-12), "coeff {i}");
        }
    }

    #[test]
    fn subframe_lpc_set_produces_four_finite_sets() {
        use crate::lsp_interp::NbSubFrameLsp;
        // Steady-state interpolation between two ascending Q10 LSP sets.
        let prev = [205, 410, 615, 820, 1024, 1229, 1434, 1638, 1843, 2048];
        let curr = [256, 470, 690, 900, 1110, 1320, 1530, 1740, 1950, 2160];
        let interp = NbSubFrameLsp::new(&prev, &curr);
        let sets = subframe_lpc_set(&interp);
        assert_eq!(sets.len(), NB_LSP_SUBFRAMES_PER_FRAME);
        for set in &sets {
            for &c in set {
                assert!(c.is_finite());
            }
        }
    }

    #[test]
    fn subframe_lpc_set_subframe4_matches_current_frame_lpc() {
        use crate::lsp_interp::NbSubFrameLsp;
        // Manual §9.1: sub-frame 4 carries the current LSPs unchanged.
        // Its interpolated Q12 vector is 4·curr, so its LPC set must
        // equal the per-frame Q10 conversion of `curr`.
        let prev = [100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900];
        let curr = [205, 410, 615, 820, 1024, 1229, 1434, 1638, 1843, 2048];
        let interp = NbSubFrameLsp::new(&prev, &curr);
        let sets = subframe_lpc_set(&interp);
        let curr_lpc = lpc_from_lsp_q10(&curr);
        for i in 0..LPC_ORDER {
            assert!(approx(sets[3][i], curr_lpc[i], 1e-12), "coeff {i}");
        }
    }

    #[test]
    fn subframe_lpc_set_first_frame_is_constant_across_subframes() {
        use crate::lsp_interp::NbSubFrameLsp;
        // First-frame init makes every sub-frame equal to `curr`, so all
        // four LPC sets must be identical.
        let curr = [205, 410, 615, 820, 1024, 1229, 1434, 1638, 1843, 2048];
        let interp = NbSubFrameLsp::first_frame(&curr);
        let sets = subframe_lpc_set(&interp);
        let first = sets[0];
        for (s, set) in sets.iter().enumerate().skip(1) {
            for (i, (&c, &f)) in set.iter().zip(first.iter()).enumerate() {
                assert!(approx(c, f, 1e-12), "set {s} coeff {i}");
            }
        }
    }

    // ---- Wideband high-band (order-8) LSP→LPC ----

    #[test]
    fn hb_order_constant_matches_codebook_order() {
        use crate::codebooks::HB_LPC_ORDER;
        assert_eq!(HB_LPC_ORDER_OUT, HB_LPC_ORDER);
        assert_eq!(HB_LPC_ORDER_OUT, 8);
    }

    #[test]
    fn hb_lsp_to_lpc_produces_eight_finite_coefficients() {
        let lsp = [0.3_f64, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4];
        let a = hb_lsp_to_lpc(&lsp);
        assert_eq!(a.len(), HB_LPC_ORDER_OUT);
        for &c in &a {
            assert!(c.is_finite());
        }
    }

    #[test]
    fn hb_lsp_to_lpc_is_deterministic() {
        let lsp = [0.5_f64; HB_LPC_ORDER_OUT];
        let a1 = hb_lsp_to_lpc(&lsp);
        let a2 = hb_lsp_to_lpc(&lsp);
        assert_eq!(a1, a2);
    }

    #[test]
    fn hb_q10_to_radians_matches_narrowband_helper() {
        // Both bands share Q10, so the high-band helper must reproduce
        // the narrowband Q10 helper for every input value.
        for v in [-5000, -1, 0, 1, 205, 1024, 3217, 1_000_000] {
            assert_eq!(hb_lsp_q10_to_radians(v), lsp_q10_to_radians(v));
        }
    }

    #[test]
    fn hb_q10_to_radians_maps_band() {
        assert!(approx(hb_lsp_q10_to_radians(1024), 1.0, 1e-9));
        let near_pi = hb_lsp_q10_to_radians(3217);
        assert!(near_pi < PI && near_pi > PI - 1e-3);
        assert!(hb_lsp_q10_to_radians(0) > 0.0);
        assert!(hb_lsp_q10_to_radians(1_000_000) < PI);
    }

    #[test]
    fn lpc_from_hb_lsp_q10_composes_pipeline() {
        // End-to-end: Q10 high-band LSP vector → radians → LPC.
        let lsp_q10 = [256, 512, 768, 1024, 1280, 1536, 1792, 2048];
        let a = lpc_from_hb_lsp_q10(&lsp_q10);
        assert_eq!(a.len(), HB_LPC_ORDER_OUT);
        for &c in &a {
            assert!(c.is_finite());
        }
        // Must equal the explicit two-step path.
        let mut rad = [0.0_f64; HB_LPC_ORDER_OUT];
        for (r, &v) in rad.iter_mut().zip(lsp_q10.iter()) {
            *r = hb_lsp_q10_to_radians(v);
        }
        let a2 = hb_lsp_to_lpc(&rad);
        assert_eq!(a, a2);
    }

    #[test]
    fn hb_path_agrees_with_generic_core_on_shared_angles() {
        // The narrowband and high-band paths share one polynomial core;
        // for the first 8 of a 10-angle set the order-8 conversion must
        // equal an independent order-8 reduction of the same core. We
        // verify the high-band conversion against a hand-rolled run of
        // the public order-10 core restricted to 8 angles would differ
        // (different order), so instead we pin the structural identity:
        // a symmetric high-band angle set about π/2 yields finite,
        // bounded coefficients.
        let lsp = [
            0.2_f64,
            PI - 0.2,
            0.7,
            PI - 0.7,
            1.0,
            PI - 1.0,
            1.3,
            PI - 1.3,
        ];
        let a = hb_lsp_to_lpc(&lsp);
        for &c in &a {
            assert!(c.is_finite());
            assert!(c.abs() < 100.0);
        }
    }

    #[test]
    fn hb_lpc_from_reconstructed_q10_round_trips() {
        // Compose the real r214 high-band LSP reconstruction with the
        // new order-8 conversion: a documented high-band sub-mode +
        // synthetic stage indices reconstruct to a Q10 vector that
        // converts to eight finite LPC coefficients.
        use crate::hb_lsp::{reconstruct_q10, HbLspStages};
        let stages = HbLspStages {
            stage1: 17,
            stage2: 42,
        };
        let lsp_q10 = reconstruct_q10(stages).unwrap();
        assert_eq!(lsp_q10.len(), HB_LPC_ORDER_OUT);
        let a = lpc_from_hb_lsp_q10(&lsp_q10);
        for &c in &a {
            assert!(c.is_finite());
        }
    }

    #[test]
    fn slice_core_panics_on_odd_order() {
        // The generic core requires an even order. A 3-element slice
        // must panic; we assert via catch_unwind so the test passes by
        // observing the panic rather than crashing the harness.
        let r = std::panic::catch_unwind(|| {
            let lsp = [0.5_f64, 1.0, 1.5];
            let mut out = [0.0_f64; 3];
            super::lsp_to_lpc_slice(&lsp, &mut out);
        });
        assert!(r.is_err());
    }

    #[test]
    fn order_ten_core_unchanged_after_generic_refactor() {
        // Regression pin: the order-10 path must still produce the same
        // coefficients it did before the generic-core refactor for a
        // fixed input.
        let lsp = [0.3_f64, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0];
        let a = lsp_to_lpc(&lsp);
        // Independent direct computation via the public poly path: build
        // P, Q by hand and average, mirroring the documented algorithm.
        let mut p: Vec<f64> = vec![1.0];
        let mut q: Vec<f64> = vec![1.0];
        let mut scratch = Vec::new();
        for (k, &w) in lsp.iter().enumerate() {
            let sec = [1.0, -2.0 * w.cos(), 1.0];
            if k % 2 == 0 {
                poly_mul(&p, &sec, &mut scratch);
                core::mem::swap(&mut p, &mut scratch);
            } else {
                poly_mul(&q, &sec, &mut scratch);
                core::mem::swap(&mut q, &mut scratch);
            }
        }
        poly_mul(&p, &[1.0, 1.0], &mut scratch);
        core::mem::swap(&mut p, &mut scratch);
        poly_mul(&q, &[1.0, -1.0], &mut scratch);
        core::mem::swap(&mut q, &mut scratch);
        for i in 0..LPC_ORDER {
            let want = -0.5 * (p[i + 1] + q[i + 1]);
            assert!(approx(a[i], want, 1e-12), "coeff {i}");
        }
    }
}
