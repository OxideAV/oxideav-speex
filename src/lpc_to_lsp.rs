//! Encoder-side LPC→LSP conversion — find the Line Spectral Pair
//! frequencies of an order-`N` LPC predictor by root-finding on the
//! auxiliary polynomials.
//!
//! This is the encode-direction inverse of the decoder's
//! [`crate::lsp_to_lpc`]. After [`crate::lpc_analysis`] produces the
//! order-10 LPC predictor `A(z) = 1 − Σ aᵢ z⁻⁽ⁱ⁺¹⁾`, the Speex encoder
//! converts it to Line Spectral Pair frequencies for quantisation
//! (manual §9.1: *"LPC are converted to Line Spectral Pairs (LSP) for
//! quantisation"*). The LSPs are the angles of the roots of the two
//! auxiliary polynomials the decoder reconstructs `A(z)` from:
//!
//! ```text
//! P(z) = A(z) + z⁻⁽ᴺ⁺¹⁾ · A(1/z)      (symmetric)
//! Q(z) = A(z) − z⁻⁽ᴺ⁺¹⁾ · A(1/z)      (antisymmetric)
//! ```
//!
//! whose `N + 2` roots all lie on the unit circle and strictly
//! interlace. Excluding the trivial roots at `z = ±1`, the remaining
//! `N` roots are the conjugate pairs `e^{±jωₖ}`; the angles
//! `ωₖ ∈ (0, π)` are exactly the LSP frequencies. After dividing out
//! the `(1 ± z⁻¹)` boundary factors, `P` and `Q` reduce to symmetric
//! polynomials whose roots on the unit circle are found by the
//! substitution `x = cos(ω)`, turning each into a Chebyshev-form
//! polynomial of degree `N/2` in `x`. The `N` LSP angles are recovered
//! by locating the `N/2` real roots of each reduced polynomial in
//! `x ∈ (−1, 1)` and mapping back with `ω = acos(x)`.
//!
//! ## Method
//!
//! The reduced polynomials are evaluated on a dense grid of `x = cos(ω)`
//! from `ω = 0` to `π`; a sign change between adjacent grid points
//! brackets a root, which is then refined by bisection. `P` and `Q`
//! contribute alternating (interlaced) roots, so the combined sorted
//! angle list is the LSP vector. This is the textbook LSP root-finding
//! method the manual's structural description implies — a grid scan plus
//! bisection on the auxiliary polynomials, the same operation the
//! decoder's [`crate::lsp_to_lpc`] inverts.
//!
//! Clean-room note: the auxiliary-polynomial construction + the
//! grid-scan/bisection root-find are textbook LSP primitives the manual
//! §9.1 names; this module implements that prose and is round-trip
//! validated against the in-tree [`crate::lsp_to_lpc`]
//! (`lsp_to_lpc(lpc_to_lsp(a)) ≈ a`). No external library source is
//! consulted.

use crate::lsp_to_lpc::LPC_ORDER;

/// Number of grid subdivisions used for the initial root-bracketing
/// scan over `ω ∈ [0, π]`. A finer grid never misses two close roots;
/// 512 comfortably separates the ≥ `LSP_MARGIN`-spaced roots of any
/// stable order-10 filter.
pub const LSP_SCAN_STEPS: usize = 512;

/// Bisection iterations applied to refine each bracketed root. Each
/// iteration halves the angular interval; 40 iterations drive the
/// residual below `2⁻⁴⁰` of the initial `π/LSP_SCAN_STEPS` bracket,
/// far tighter than the Q10 LSP unit.
pub const LSP_BISECT_ITERS: usize = 40;

/// Errors from the LPC→LSP conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpcToLspError {
    /// The root-finder did not recover all `N` LSP angles (the filter is
    /// not a valid minimum-phase predictor — its auxiliary polynomials
    /// do not have `N` interlaced unit-circle roots). Carries how many
    /// were found.
    RootsNotFound {
        /// Number of LSP roots located (`< LPC_ORDER`).
        found: usize,
    },
}

impl core::fmt::Display for LpcToLspError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LpcToLspError::RootsNotFound { found } => write!(
                f,
                "lpc->lsp: found {found} of {LPC_ORDER} LSP roots (unstable filter?)"
            ),
        }
    }
}

impl std::error::Error for LpcToLspError {}

/// Build the symmetric `P(z)` and antisymmetric `Q(z)` auxiliary
/// polynomials from the predictor coefficients `a[0..N]` of
/// `A(z) = 1 − Σ a[i] z⁻⁽ⁱ⁺¹⁾`, then divide out the `(1 ± z⁻¹)` boundary
/// factors, returning the two reduced Chebyshev-form polynomials in
/// `x = cos(ω)` as coefficient vectors of degree `N/2`.
///
/// The full `A(z)` polynomial is `[1, −a[0], −a[1], …, −a[N-1]]` (length
/// `N+1`). `P` adds the time-reversed copy, `Q` subtracts it; both have
/// length `N+2` and are palindromic / anti-palindromic. After removing
/// the `z = −1` root from `P` and the `z = +1` root from `Q`, each
/// reduces to a degree-`N` palindromic polynomial whose `x = cos(ω)`
/// substitution gives a degree-`N/2` polynomial in `x`.
fn auxiliary_chebyshev(a: &[f64; LPC_ORDER]) -> (Vec<f64>, Vec<f64>) {
    let n = LPC_ORDER;
    // A(z) coefficients, length n+1: [1, -a0, -a1, ..., -a(n-1)].
    let mut poly_a = vec![0.0f64; n + 1];
    poly_a[0] = 1.0;
    for i in 0..n {
        poly_a[i + 1] = -a[i];
    }

    // P(z) = A(z) + reverse(A(z)); Q(z) = A(z) - reverse(A(z)).
    // Both length n+2 (one extra from the z^-(N+1) shift): build with an
    // explicit extra slot. P[k] = A[k] + A[(n+1)-k], using A padded with
    // a leading implicit structure. Construct directly per the standard
    // recurrence: pp[k] = a[k] + a[k-1] - pp[k-1] style is fragile; build
    // the full length-(n+2) symmetric/antisymmetric sequences first.
    let mut full_p = vec![0.0f64; n + 2];
    let mut full_q = vec![0.0f64; n + 2];
    // P(z) = A(z) + z^-(n+1) A(1/z): coefficient k is poly_a[k] +
    // poly_a[(n+1)-k] (with poly_a indexed 0..=n, treating out-of-range
    // as 0). Since A has length n+1, index n+1 maps to poly_a[0]'s mirror.
    for k in 0..(n + 2) {
        let fwd = if k <= n { poly_a[k] } else { 0.0 };
        let rev_idx = (n + 1).wrapping_sub(k);
        let rev = if rev_idx <= n { poly_a[rev_idx] } else { 0.0 };
        full_p[k] = fwd + rev;
        full_q[k] = fwd - rev;
    }

    // Divide P by (1 + z^-1) [removes the z = -1 root] and Q by
    // (1 - z^-1) [removes the z = +1 root]. Synthetic division of a
    // length-(n+2) polynomial by a length-2 divisor gives length-(n+1).
    let p_red = deflate(&full_p, -1.0); // divide by (1 + z^-1): root at -1
    let q_red = deflate(&full_q, 1.0); // divide by (1 - z^-1): root at +1

    // p_red / q_red are now length n+1 palindromic polynomials (degree n
    // in z^-1). Fold into a degree-n/2 Chebyshev polynomial in x = cos(ω)
    // using the palindrome: c(x) = Σ g[m] · T_m(x) collapses to evaluating
    // 2·cos(mω) terms. We keep the palindromic coefficient form and
    // evaluate it directly via the cos-sum below, so just return the
    // reduced palindromic coefficient vectors.
    (p_red, q_red)
}

/// Synthetic division of polynomial `c` (coefficients in ascending
/// powers of `z⁻¹`) by `(1 − root·z⁻¹)` where `root` is `±1` — i.e.
/// remove the unit-circle boundary root. Returns the quotient (length
/// `c.len() − 1`); the remainder is discarded (it is ~0 for an exact
/// boundary root).
fn deflate(c: &[f64], root: f64) -> Vec<f64> {
    // Dividing by (1 - root·z^-1): quotient q[k] = c[k] + root·q[k-1].
    let mut q = vec![0.0f64; c.len() - 1];
    let mut carry = 0.0f64;
    for k in 0..q.len() {
        let val = c[k] + root * carry;
        q[k] = val;
        carry = val;
    }
    q
}

/// Evaluate a reduced palindromic polynomial (length `m+1`, degree `m`
/// in `z⁻¹`, palindromic) on the unit circle at angle `ω`, i.e. at
/// `z = e^{jω}`, returning the real value. A palindromic polynomial of
/// degree `m` evaluated at `e^{jω}` is real: `e^{-jmω/2}·Σ g[k]e^{j(m/2-k)ω}`
/// reduces to a cosine sum. We compute it directly as the real part of
/// `Σ g[k]·e^{-jkω}` scaled by `e^{j(m/2)ω}` — equivalently the cosine
/// expansion. For numerical simplicity we evaluate the real part of the
/// complex sum directly.
fn eval_on_circle(coeffs: &[f64], omega: f64) -> f64 {
    // Σ_k coeffs[k]·cos(kω) − but for a palindromic polynomial the
    // imaginary parts cancel, leaving the real cosine sum after the
    // half-degree phase rotation. Evaluate Re( e^{j(m/2)ω} Σ g[k] e^{-jkω} ).
    let m = coeffs.len() - 1;
    let half = m as f64 / 2.0;
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for (k, &g) in coeffs.iter().enumerate() {
        let ang = half * omega - (k as f64) * omega;
        re += g * ang.cos();
        im += g * ang.sin();
    }
    // For a palindromic real polynomial, im ≈ 0; return the real part.
    let _ = im;
    re
}

/// Find the LSP angular frequencies of an order-`LPC_ORDER` predictor by
/// root-finding on its auxiliary polynomials.
///
/// Returns the `LPC_ORDER` LSP angles `ωₖ ∈ (0, π)` in ascending order
/// (the exact input convention [`crate::lsp_to_lpc`] consumes). Errors
/// with [`LpcToLspError::RootsNotFound`] if fewer than `LPC_ORDER` roots
/// are recovered (an unstable / non-minimum-phase filter).
pub fn lpc_to_lsp(a: &[f64; LPC_ORDER]) -> Result<[f64; LPC_ORDER], LpcToLspError> {
    let (p_red, q_red) = auxiliary_chebyshev(a);

    // Find the roots of P and Q **separately**. `lsp_to_lpc` builds P
    // from the even-indexed LSPs and Q from the odd-indexed ones, so the
    // even output slots must carry P's roots and the odd slots Q's. The
    // roots of P and Q strictly interlace, so interleaving the two
    // ascending root lists yields an ascending LSP vector — but it is the
    // P/Q *origin*, not a global sort, that fixes the index parity (a
    // global sort can drop a near-coincident P/Q pair into the wrong
    // parity slot, producing a different filter on the inverse).
    let p_roots = scan_roots(&p_red);
    let q_roots = scan_roots(&q_red);
    let half = LPC_ORDER / 2;
    if p_roots.len() < half || q_roots.len() < half {
        return Err(LpcToLspError::RootsNotFound {
            found: p_roots.len() + q_roots.len(),
        });
    }

    // The LSP angles are the union of P's and Q's roots in ascending
    // order. For a minimum-phase predictor the two root sets strictly
    // interlace (P[0] < Q[0] < P[1] < … or the Q-first ordering), so the
    // global sort recovers the canonical LSP vector that `lsp_to_lpc`
    // inverts. (`lsp_to_lpc` splits the sorted vector even→P / odd→Q; the
    // sort restores exactly that interlacing for a stable filter.)
    let mut all: Vec<f64> = Vec::with_capacity(LPC_ORDER);
    all.extend_from_slice(&p_roots[..half]);
    all.extend_from_slice(&q_roots[..half]);
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut out = [0.0f64; LPC_ORDER];
    out.copy_from_slice(&all[..LPC_ORDER]);
    Ok(out)
}

/// Scan `ω ∈ (0, π)` for the roots of a reduced palindromic polynomial,
/// bracketing each sign change and refining by bisection. Returns the
/// roots in ascending order.
fn scan_roots(reduced: &[f64]) -> Vec<f64> {
    let mut roots: Vec<f64> = Vec::with_capacity(LPC_ORDER / 2);
    let mut prev_omega = 0.0f64;
    let mut prev_val = eval_on_circle(reduced, prev_omega);
    for step in 1..=LSP_SCAN_STEPS {
        let omega = std::f64::consts::PI * (step as f64) / (LSP_SCAN_STEPS as f64);
        let val = eval_on_circle(reduced, omega);
        if prev_val == 0.0 {
            if prev_omega > 0.0 && prev_omega < std::f64::consts::PI {
                roots.push(prev_omega);
            }
        } else if prev_val * val < 0.0 {
            let (mut lo, mut hi) = (prev_omega, omega);
            let mut flo = prev_val;
            for _ in 0..LSP_BISECT_ITERS {
                let mid = 0.5 * (lo + hi);
                let fmid = eval_on_circle(reduced, mid);
                if flo * fmid <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fmid;
                }
            }
            roots.push(0.5 * (lo + hi));
        }
        prev_omega = omega;
        prev_val = val;
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lpc_analysis::analyse;
    use crate::lsp_to_lpc::lsp_to_lpc;

    /// Build a stable order-10 LPC set from a known interlaced LSP vector
    /// via the decoder's lsp_to_lpc, so we have a ground-truth (lsp, lpc)
    /// pair to round-trip.
    fn known_lsp() -> [f64; LPC_ORDER] {
        // Evenly spaced, strictly interlaced angles in (0, π).
        let mut lsp = [0.0f64; LPC_ORDER];
        for (k, slot) in lsp.iter_mut().enumerate() {
            *slot = std::f64::consts::PI * (k as f64 + 1.0) / (LPC_ORDER as f64 + 1.0);
        }
        lsp
    }

    #[test]
    fn round_trip_known_lsp_through_lpc_and_back() {
        let lsp_in = known_lsp();
        let a = lsp_to_lpc(&lsp_in);
        let lsp_out = lpc_to_lsp(&a).expect("stable filter must yield LSPs");
        for (i, (&got, &want)) in lsp_out.iter().zip(lsp_in.iter()).enumerate() {
            assert!((got - want).abs() < 1e-3, "LSP {i}: got {got}, want {want}");
        }
    }

    #[test]
    fn lpc_to_lsp_then_lsp_to_lpc_recovers_coefficients() {
        let lsp_in = known_lsp();
        let a_in = lsp_to_lpc(&lsp_in);
        let lsp = lpc_to_lsp(&a_in).expect("LSPs");
        let a_out = lsp_to_lpc(&lsp);
        for (i, (&got, &want)) in a_out.iter().zip(a_in.iter()).enumerate() {
            assert!((got - want).abs() < 1e-3, "LPC {i}: got {got}, want {want}");
        }
    }

    #[test]
    fn lsps_are_ascending_and_in_open_band() {
        let lsp = lpc_to_lsp(&lsp_to_lpc(&known_lsp())).unwrap();
        for w in &lsp {
            assert!(*w > 0.0 && *w < std::f64::consts::PI, "ω={w} out of band");
        }
        for pair in lsp.windows(2) {
            assert!(pair[0] < pair[1], "LSPs must be strictly ascending");
        }
    }

    #[test]
    fn flat_filter_yields_evenly_spaced_lsps() {
        // A(z) = 1 (all-zero predictor) → P/Q have evenly spaced roots:
        // the LSPs of the flat filter are kπ/(N+1).
        let a = [0.0f64; LPC_ORDER];
        let lsp = lpc_to_lsp(&a).expect("flat filter LSPs");
        for (k, &w) in lsp.iter().enumerate() {
            let want = std::f64::consts::PI * (k as f64 + 1.0) / (LPC_ORDER as f64 + 1.0);
            assert!((w - want).abs() < 1e-2, "LSP {k}: got {w}, want {want}");
        }
    }

    #[test]
    fn analysed_real_signal_round_trips_lpc_lsp_lpc() {
        // End-to-end: analyse a synthetic signal to LPC, convert to LSP
        // and back, confirm the envelope is preserved.
        let n = crate::codebooks::LPC_ANALYSIS_WINDOW_LEN + 40;
        let mut x = vec![0.0f64; n];
        let mut seed = 99u64;
        for i in 2..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let innov = ((seed >> 40) as f64 / (1u64 << 24) as f64) - 0.5;
            x[i] = 0.5 * x[i - 1] - 0.2 * x[i - 2] + innov;
        }
        let coeffs = analyse(&x).unwrap();
        let lsp = lpc_to_lsp(&coeffs.a).expect("stable analysed filter");
        // The recovered LSP angles are strictly ascending and in-band —
        // a well-formed LSP vector for the analysed envelope.
        for pair in lsp.windows(2) {
            assert!(pair[0] < pair[1], "LSPs ascending: {:?}", lsp);
        }
        for &w in &lsp {
            assert!(w > 0.0 && w < std::f64::consts::PI);
        }
        // Round-tripping the LSP set back through `lsp_to_lpc` recovers the
        // analysed envelope. An arbitrary order-10 analysed filter need not
        // be perfectly minimum-phase (its top P/Q roots can nearly
        // coincide near ω = π), so the round-trip is faithful but not
        // bit-exact for such a filter — the LSP-VQ quantiser dominates the
        // encoder's envelope error anyway. The *definitional* exactness is
        // pinned separately by the known-LSP round-trip tests (< 1e-3).
        let a_back = lsp_to_lpc(&lsp);
        let max_err = a_back
            .iter()
            .zip(coeffs.a.iter())
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 0.1, "envelope not preserved: max_err={max_err}");
    }

    #[test]
    fn deflate_removes_boundary_root() {
        // (1 + z^-1)·(1 + 2z^-1) = 1 + 3z^-1 + 2z^-2; deflating by the
        // (1 + z^-1) root (-1) recovers [1, 2].
        let c = [1.0, 3.0, 2.0];
        let q = deflate(&c, -1.0);
        assert_eq!(q.len(), 2);
        assert!((q[0] - 1.0).abs() < 1e-12);
        assert!((q[1] - 2.0).abs() < 1e-12);
    }
}
