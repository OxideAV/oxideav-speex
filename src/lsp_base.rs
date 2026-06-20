//! Speex LSP base vector + Q-format pin (round r347 scope).
//!
//! The narrowband / high-band LSP reconstruction
//! ([`crate::lsp::reconstruct_q10`] / [`crate::hb_lsp::reconstruct_q10`])
//! sums the per-stage VQ codebook **contributions** into the common
//! Q10 fixed-point unit (`1/1024` of an LSP angle in radians). Those
//! contributions are *deltas* around a fixed **linear base vector**;
//! the reconstructed LSP angle the decoder feeds into the cosine
//! series is
//!
//! ```text
//! lsp[i]  =  LSP_LINEAR(i)  +  Σ_stages (codebook delta)
//! ```
//!
//! Until this round the reconstruction returned **only** the codebook
//! delta sum, leaving the base offset unpinned — so the LSP angles were
//! tiny (a few hundredths of a radian) instead of spanning the
//! conformant `(0, π)` band, and the downstream LSP→LPC conversion was
//! correct-by-construction but **not bounded to the documented band**.
//! This module pins the base vector and the Q-format so the
//! reconstructed angle is the actual Speex LSP value.
//!
//! ## What the staged material pins
//!
//! `docs/audio/speex/provenance/02-speex-gain-quant.md` records, as
//! data-only extracted numeric facts, the LSP reconstruction-path
//! `#define` scalars:
//!
//! | Constant | Role | Q15 fixed | Float |
//! | --- | --- | --: | --: |
//! | `LSP_LINEAR(i)` | linear LSP init (NB) | `SHL16(i+1,11)` | `.25·i + .25` |
//! | `LSP_LINEAR_HIGH(i)` | linear LSP init (HB) | `i·2560 + 6144` | `.3125·i + .75` |
//! | `LSP_SCALE` | LSP fixed↔float scale | 256 | 256. |
//! | `LSP_DIV_256/512/1024` | per-stage codebook scaling | shifts 5/4/3 | 0.0039062 / 0.0019531 / 0.00097656 |
//! | `LSP_PI` | π in the LSP angle domain | 25736 | M_PI |
//!
//! ## Deriving the Q10-radian base vector
//!
//! The codebook deltas already accumulate in Q10 radians: stage 0's
//! `1/256`-scaled entry (`LSP_DIV_256`) multiplied by the module's
//! `×4` factor lands in `1/1024`-radian units, i.e. `delta_q10 / 1024`
//! is the codebook contribution in radians (this is exactly the float
//! `LSP_DIV_256 = 1/256` reconstruction). For the **base vector** the
//! Q15-domain `LSP_PI = 25736` maps to π radians, so a Q15-domain value
//! `v` is `v · π / 25736` radians; re-expressing that in the Q10-radian
//! unit (`× 1024`) and substituting the linear-init formula gives an
//! exact integer:
//!
//! ```text
//! NB: LSP_LINEAR(i)      = SHL16(i+1, 11) = (i+1) · 2048   (Q15 domain)
//!     rad                = (i+1)·2048 · π / 25736
//!     Q10-radian         = rad · 1024 = (.25·i + .25)·1024 = 256·i + 256
//!
//! HB: LSP_LINEAR_HIGH(i) = i·2560 + 6144                   (Q15 domain)
//!     rad                = (i·2560 + 6144) · π / 25736
//!     Q10-radian         = (.3125·i + .75)·1024 = 320·i + 768
//! ```
//!
//! Both float forms (`.25·i + .25`, `.3125·i + .75`) scaled by `1024`
//! land on **exact integers**, so the Q10-radian base vector carries no
//! rounding-direction question. The cross-check between the Q15-domain
//! (`LSP_PI = 25736`) path and the float (`M_PI`) path agrees to within
//! the float rounding of `2048·π/25736·1024 = 256.0000…` — i.e. the two
//! independent staged forms pin the **same** base vector, which is the
//! boundedness guarantee this round delivers.
//!
//! ## Why this bounds the LSP→LPC conversion
//!
//! With the base vector added, every reconstructed NB LSP angle for a
//! real frame lands in roughly `(0.25, 2.5 + ε)` radians (base
//! `0.25 … 2.5` rad plus a small signed codebook delta), strictly
//! inside the `(0, π)` band the auxiliary-polynomial root split
//! requires (`π ≈ 3.1416`). The HB base spans `0.75 … 2.94` rad,
//! likewise inside `(0, π)`. The conformant band is therefore reached
//! by construction rather than by the `clamp` fallback in
//! [`crate::lsp_to_lpc::lsp_qn_to_radians`] — the clamp now only guards
//! against a degenerate (corrupt-bitstream) quantiser output, not
//! against the *expected* magnitude of a well-formed frame.

use crate::codebooks::{HB_LPC_ORDER, NB_LSP_ORDER};
use crate::hb_lsp::HB_LSP_OUTPUT_Q;
use crate::lsp::NB_LSP_OUTPUT_Q;

/// Narrowband LSP linear base-vector slope, in Q10-radian units per
/// coefficient index (`LSP_LINEAR(i) = .25·i + .25` rad → `256·i + …`).
pub const NB_LSP_BASE_SLOPE_Q10: i32 = 256;

/// Narrowband LSP linear base-vector intercept, in Q10-radian units
/// (`.25 rad × 1024`).
pub const NB_LSP_BASE_INTERCEPT_Q10: i32 = 256;

/// High-band LSP linear base-vector slope, in Q10-radian units per
/// coefficient index (`LSP_LINEAR_HIGH(i) = .3125·i + .75` rad →
/// `320·i + …`).
pub const HB_LSP_BASE_SLOPE_Q10: i32 = 320;

/// High-band LSP linear base-vector intercept, in Q10-radian units
/// (`.75 rad × 1024`).
pub const HB_LSP_BASE_INTERCEPT_Q10: i32 = 768;

/// Narrowband LSP minimum-spacing margin `LSP_MARGIN`, in Q10-radian
/// units. Staged float value `.002` rad
/// (`docs/audio/speex/provenance/02-speex-gain-quant.md`, `nb_celp.c`):
/// `.002 × 1024 = 2.048`, taken as `2` (the integer-Q10 representation;
/// the float path uses `.002` exactly — see [`nb_lsp_margin_radians`]).
pub const NB_LSP_MARGIN_Q10: i32 = 2;

/// High-band LSP minimum-spacing margin `LSP_MARGIN`, in Q10-radian
/// units. Staged float value `.05` rad (`sb_celp.c`):
/// `.05 × 1024 = 51.2`, taken as `51`.
pub const HB_LSP_MARGIN_Q10: i32 = 51;

/// The narrowband `LSP_MARGIN` re-expressed in radians (staged float
/// form `.002`).
pub const NB_LSP_MARGIN_RADIANS: f64 = 0.002;

/// The high-band `LSP_MARGIN` re-expressed in radians (staged float
/// form `.05`).
pub const HB_LSP_MARGIN_RADIANS: f64 = 0.05;

/// Returns the narrowband LSP minimum-spacing margin in radians
/// (`.002`, the staged float `LSP_MARGIN`).
pub const fn nb_lsp_margin_radians() -> f64 {
    NB_LSP_MARGIN_RADIANS
}

/// Returns the high-band LSP minimum-spacing margin in radians (`.05`,
/// the staged float `LSP_MARGIN`).
pub const fn hb_lsp_margin_radians() -> f64 {
    HB_LSP_MARGIN_RADIANS
}

/// Enforce the Speex `LSP_MARGIN` minimum-spacing constraint on an
/// **ascending** vector of LSP angular frequencies (radians), in place.
///
/// Speex's `lsp_enforce_margin` (the `lsp_interpolate` safeguard that
/// `LSP_MARGIN` parameterises — recorded as a numeric fact in
/// `docs/audio/speex/provenance/02-speex-gain-quant.md`) guarantees the
/// reconstructed LSP set stays a **valid, strictly-interlaced** set so
/// the LSP→LPC auxiliary-polynomial root split (see
/// [`crate::lsp_to_lpc`]) yields a stable filter:
///
/// 1. **forward pass** — clamp `lsp[0]` up to at least `margin`
///    (off `ω = 0`), then for each `i` raise `lsp[i]` to at least
///    `lsp[i-1] + 2·margin` (lower spacing);
/// 2. **backward pass** — clamp `lsp[n-1]` down to at most `π − margin`
///    (off `ω = π`), then for each `i` (descending) lower `lsp[i]` to at
///    most `lsp[i+1] − 2·margin` (upper spacing).
///
/// The two passes together guarantee every coefficient lands in
/// `[margin, π − margin]` strictly ascending with at least `2·margin`
/// spacing — feasible because `n·2·margin ≪ π` for both bands
/// (NB `10·2·.002 = .04`; HB `8·2·.05 = .8`). For a well-formed frame
/// whose angles are already inside the band and adequately spaced this
/// is the identity.
///
/// The margin *constant* (`.002` NB / `.05` HB) is the pinned Speex
/// fact; the clamping procedure itself is the generic order-preserving
/// safeguard every CELP LSP path applies (companion §"minimum-spacing
/// enforcement"; the `lsp_to_lpc` module notes "most CELP codecs enforce
/// `ω_{i+1} − ω_i ≥ Δ_min`").
pub fn enforce_lsp_margin_radians(lsp: &mut [f64], margin: f64) {
    if lsp.is_empty() {
        return;
    }
    let pi = core::f64::consts::PI;
    let n = lsp.len();
    let gap = 2.0 * margin;
    // Forward pass: lower bound + lower spacing.
    if lsp[0] < margin {
        lsp[0] = margin;
    }
    for i in 1..n {
        let lo = lsp[i - 1] + gap;
        if lsp[i] < lo {
            lsp[i] = lo;
        }
    }
    // Backward pass: upper bound + upper spacing.
    if lsp[n - 1] > pi - margin {
        lsp[n - 1] = pi - margin;
    }
    for i in (0..n - 1).rev() {
        let hi = lsp[i + 1] - gap;
        if lsp[i] > hi {
            lsp[i] = hi;
        }
    }
}

/// The narrowband LSP linear base vector `LSP_LINEAR(i)` expressed in
/// the Q[`NB_LSP_OUTPUT_Q`] = Q10 radian unit (`256·i + 256` for
/// `i = 0..10`).
///
/// This is the fixed offset the per-stage VQ deltas refine — the
/// "uninitialised / base" LSP set that the reconstructed angle is
/// measured against (see module docs).
pub const fn nb_lsp_base_q10() -> [i32; NB_LSP_ORDER] {
    let mut base = [0i32; NB_LSP_ORDER];
    let mut i = 0;
    while i < NB_LSP_ORDER {
        base[i] = NB_LSP_BASE_SLOPE_Q10 * i as i32 + NB_LSP_BASE_INTERCEPT_Q10;
        i += 1;
    }
    base
}

/// The high-band LSP linear base vector `LSP_LINEAR_HIGH(i)` expressed
/// in the Q[`HB_LSP_OUTPUT_Q`] = Q10 radian unit (`320·i + 768` for
/// `i = 0..8`).
pub const fn hb_lsp_base_q10() -> [i32; HB_LPC_ORDER] {
    let mut base = [0i32; HB_LPC_ORDER];
    let mut i = 0;
    while i < HB_LPC_ORDER {
        base[i] = HB_LSP_BASE_SLOPE_Q10 * i as i32 + HB_LSP_BASE_INTERCEPT_Q10;
        i += 1;
    }
    base
}

/// Add the narrowband LSP base vector ([`nb_lsp_base_q10`]) to a
/// reconstructed Q10 codebook-delta vector in place, yielding the
/// pinned Q10-radian LSP angles.
pub fn add_nb_lsp_base(delta_q10: &mut [i32; NB_LSP_ORDER]) {
    let base = nb_lsp_base_q10();
    for (d, b) in delta_q10.iter_mut().zip(base.iter()) {
        *d += *b;
    }
}

/// Add the high-band LSP base vector ([`hb_lsp_base_q10`]) to a
/// reconstructed Q10 codebook-delta vector in place, yielding the
/// pinned Q10-radian LSP angles.
pub fn add_hb_lsp_base(delta_q10: &mut [i32; HB_LPC_ORDER]) {
    let base = hb_lsp_base_q10();
    for (d, b) in delta_q10.iter_mut().zip(base.iter()) {
        *d += *b;
    }
}

/// The narrowband LSP base vector re-expressed in radians (`base_q10 /
/// 2^Q10`). Convenience for tests / callers that want the float band
/// directly; the staged float form is `.25·i + .25`.
pub fn nb_lsp_base_radians() -> [f64; NB_LSP_ORDER] {
    let mut out = [0.0f64; NB_LSP_ORDER];
    let scale = f64::from(1u32 << NB_LSP_OUTPUT_Q);
    for (o, &v) in out.iter_mut().zip(nb_lsp_base_q10().iter()) {
        *o = f64::from(v) / scale;
    }
    out
}

/// The high-band LSP base vector re-expressed in radians (`base_q10 /
/// 2^Q10`). The staged float form is `.3125·i + .75`.
pub fn hb_lsp_base_radians() -> [f64; HB_LPC_ORDER] {
    let mut out = [0.0f64; HB_LPC_ORDER];
    let scale = f64::from(1u32 << HB_LSP_OUTPUT_Q);
    for (o, &v) in out.iter_mut().zip(hb_lsp_base_q10().iter()) {
        *o = f64::from(v) / scale;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The NB base vector matches the staged float form `.25·i + .25`
    /// exactly (the `×1024` lands on integers).
    #[test]
    fn nb_base_matches_float_linear_init() {
        let base = nb_lsp_base_q10();
        for (i, &v) in base.iter().enumerate() {
            // .25·i + .25 in Q10 = (0.25*i + 0.25) * 1024.
            let expected = ((0.25 * i as f64 + 0.25) * 1024.0).round() as i32;
            assert_eq!(v, expected, "NB base coeff {i}");
        }
        // Spot values: base[0] = 256 (0.25 rad), base[9] = 256*9+256 = 2560 (2.5 rad).
        assert_eq!(base[0], 256);
        assert_eq!(base[9], 2560);
    }

    /// The HB base vector matches the staged float form `.3125·i + .75`.
    #[test]
    fn hb_base_matches_float_linear_init() {
        let base = hb_lsp_base_q10();
        for (i, &v) in base.iter().enumerate() {
            let expected = ((0.3125 * i as f64 + 0.75) * 1024.0).round() as i32;
            assert_eq!(v, expected, "HB base coeff {i}");
        }
        // base[0] = 768 (0.75 rad), base[7] = 320*7+768 = 3008 (≈2.9375 rad).
        assert_eq!(base[0], 768);
        assert_eq!(base[7], 3008);
    }

    /// Cross-check the Q15-domain (`LSP_PI = 25736`) path against the
    /// float (`M_PI`) path: the two independent staged forms pin the
    /// **same** base vector to sub-Q10 precision. This is the
    /// boundedness pin's provenance cross-check.
    #[test]
    fn q15_domain_and_float_paths_agree() {
        const LSP_PI: f64 = 25736.0;
        for i in 0..NB_LSP_ORDER {
            // Q15 domain: LSP_LINEAR(i) = SHL16(i+1, 11) = (i+1) << 11.
            let q15 = ((i as i32 + 1) << 11) as f64;
            let rad_q15 = q15 * core::f64::consts::PI / LSP_PI;
            let q10_from_q15 = (rad_q15 * 1024.0).round() as i32;
            assert_eq!(
                q10_from_q15,
                nb_lsp_base_q10()[i],
                "NB base coeff {i}: Q15-domain path disagrees with float path"
            );
        }
        for i in 0..HB_LPC_ORDER {
            // Q15 domain: LSP_LINEAR_HIGH(i) = i·2560 + 6144.
            let q15 = (i as i32 * 2560 + 6144) as f64;
            let rad_q15 = q15 * core::f64::consts::PI / LSP_PI;
            let q10_from_q15 = (rad_q15 * 1024.0).round() as i32;
            assert_eq!(
                q10_from_q15,
                hb_lsp_base_q10()[i],
                "HB base coeff {i}: Q15-domain path disagrees with float path"
            );
        }
    }

    /// Every NB base angle (before codebook deltas) is strictly inside
    /// the conformant `(0, π)` band, ascending and minimally spaced.
    #[test]
    fn nb_base_is_inside_conformant_band_and_ascending() {
        let rad = nb_lsp_base_radians();
        let mut prev = 0.0;
        for (i, &w) in rad.iter().enumerate() {
            assert!(
                w > 0.0 && w < core::f64::consts::PI,
                "coeff {i} out of band"
            );
            if i > 0 {
                assert!(w > prev, "base not ascending at {i}");
            }
            prev = w;
        }
        // Slope is 0.25 rad/coeff → spacing always 0.25 rad.
        assert!((rad[1] - rad[0] - 0.25).abs() < 1e-9);
    }

    /// HB base spans `0.75 … 2.9375` rad, all inside `(0, π)`.
    #[test]
    fn hb_base_is_inside_conformant_band() {
        let rad = hb_lsp_base_radians();
        for (i, &w) in rad.iter().enumerate() {
            assert!(
                w > 0.0 && w < core::f64::consts::PI,
                "HB coeff {i} out of band"
            );
        }
        assert!((rad[0] - 0.75).abs() < 1e-9);
    }

    /// Adding the base shifts a zero delta vector exactly onto the base.
    #[test]
    fn add_nb_base_on_zero_delta_is_base() {
        let mut v = [0i32; NB_LSP_ORDER];
        add_nb_lsp_base(&mut v);
        assert_eq!(v, nb_lsp_base_q10());
    }

    #[test]
    fn add_hb_base_on_zero_delta_is_base() {
        let mut v = [0i32; HB_LPC_ORDER];
        add_hb_lsp_base(&mut v);
        assert_eq!(v, hb_lsp_base_q10());
    }

    /// Adding the base is a pure translation: a delta perturbation
    /// survives unchanged.
    #[test]
    fn add_base_is_translation_invariant() {
        let mut a = [3i32; NB_LSP_ORDER];
        let mut b = [3i32; NB_LSP_ORDER];
        b[4] += 100;
        add_nb_lsp_base(&mut a);
        add_nb_lsp_base(&mut b);
        for i in 0..NB_LSP_ORDER {
            let expected = if i == 4 { 100 } else { 0 };
            assert_eq!(b[i] - a[i], expected, "coeff {i}");
        }
    }

    // ---- LSP_MARGIN minimum-spacing enforcement ----

    /// The staged margin constants match their Q10/float forms.
    #[test]
    fn margin_constants_match_staged_values() {
        assert!((nb_lsp_margin_radians() - 0.002).abs() < 1e-12);
        assert!((hb_lsp_margin_radians() - 0.05).abs() < 1e-12);
        // Q10 forms: .002*1024 = 2.048 → 2; .05*1024 = 51.2 → 51.
        assert_eq!(NB_LSP_MARGIN_Q10, 2);
        assert_eq!(HB_LSP_MARGIN_Q10, 51);
    }

    /// A well-formed, well-spaced ascending set is unchanged (identity).
    #[test]
    fn margin_is_identity_on_well_spaced_set() {
        let mut lsp = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5];
        let before = lsp;
        enforce_lsp_margin_radians(&mut lsp, 0.002);
        assert_eq!(lsp, before);
    }

    /// The first coefficient is pushed up off zero; the last down off π.
    #[test]
    fn margin_clamps_band_endpoints() {
        let pi = core::f64::consts::PI;
        let mut lsp = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8, 3.0, pi - 0.04, pi];
        enforce_lsp_margin_radians(&mut lsp, 0.05);
        assert!(lsp[0] >= 0.05 - 1e-12, "first not clamped up");
        assert!(lsp[9] <= pi - 0.05 + 1e-12, "last not clamped down");
    }

    /// A pair closer than `2·margin` is separated to at least `2·margin`,
    /// preserving order.
    #[test]
    fn margin_separates_too_close_pair() {
        let margin = 0.05;
        // Coeffs 3 and 4 are only 0.01 apart — far below 2*margin=0.1.
        let mut lsp = [0.2, 0.4, 0.6, 1.00, 1.01, 1.4, 1.6, 1.8, 2.0, 2.2];
        enforce_lsp_margin_radians(&mut lsp, margin);
        // After enforcement every consecutive gap is >= 2*margin (minus
        // float slack).
        for i in 1..lsp.len() {
            assert!(
                lsp[i] - lsp[i - 1] >= 2.0 * margin - 1e-9,
                "gap {i} = {} below 2*margin",
                lsp[i] - lsp[i - 1]
            );
        }
        // Order preserved.
        for i in 1..lsp.len() {
            assert!(lsp[i] > lsp[i - 1], "order broken at {i}");
        }
    }

    /// Enforcement keeps the whole set strictly inside `(0, π)`,
    /// ascending with ≥ `2·margin` spacing, even for a degenerate
    /// near-collapsed input clustered at both band edges.
    #[test]
    fn margin_keeps_set_inside_band() {
        let pi = core::f64::consts::PI;
        let m = 0.002;
        // A degenerate near-collapsed set: five clustered near 0, five
        // clustered near π.
        let mut lsp = [
            0.001,
            0.001,
            0.002,
            0.002,
            0.003,
            pi - 0.01,
            pi - 0.005,
            pi - 0.002,
            pi - 0.001,
            pi - 0.0005,
        ];
        enforce_lsp_margin_radians(&mut lsp, m);
        for (i, &w) in lsp.iter().enumerate() {
            assert!(w > 0.0 && w < pi, "coeff {i} = {w} outside (0, π)");
            if i > 0 {
                assert!(w - lsp[i - 1] >= 2.0 * m - 1e-9, "gap {i} below 2*margin");
            }
        }
    }

    /// Empty / single-element inputs do not panic.
    #[test]
    fn margin_handles_trivial_lengths() {
        let mut empty: [f64; 0] = [];
        enforce_lsp_margin_radians(&mut empty, 0.002);
        let mut one = [1.5f64];
        enforce_lsp_margin_radians(&mut one, 0.002);
        assert_eq!(one, [1.5]);
        let mut one_low = [0.0001f64];
        enforce_lsp_margin_radians(&mut one_low, 0.002);
        assert!(one_low[0] >= 0.002 - 1e-12);
    }
}
