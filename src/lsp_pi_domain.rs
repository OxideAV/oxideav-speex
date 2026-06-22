//! Speex **`LSP_PI` angular-domain pin** (round r359 scope).
//!
//! The Speex decoder stores and accumulates LSP frequencies in a
//! fixed-point *angular domain* in which the constant
//!
//! ```text
//! LSP_PI = 25736
//! ```
//!
//! represents the angle `π`. This constant is recorded as a **numeric
//! fact** (data-only extraction, *Feist* posture) in
//! `docs/audio/speex/provenance/02-speex-gain-quant.md` (the
//! "LSP→LPC reconstruction path constants" table: *"`LSP_PI` | π in the
//! LSP angle domain | 25736 | M_PI"*). It is the missing pin that the
//! earlier rounds flagged as the LSP "angular-unit gap": until it was
//! staged, [`crate::lsp_to_lpc::lsp_qn_to_radians`] could only
//! *document an assumption* that the stored value was the angle scaled
//! so the `(0, π)` band mapped onto the stored range. With `LSP_PI`
//! pinned, that assumption is now a **provenance-confirmed fact**:
//!
//! ```text
//! ω (radians)  =  v_storage · π / LSP_PI            (LSP_PI = 25736)
//! ```
//!
//! where `v_storage` is an LSP value in the codec's own angular storage
//! domain (the domain the `LSP_LINEAR` base vector and the per-stage
//! `LSP_DIV_*` codebook scalings are expressed in).
//!
//! ## Relationship to the crate's internal Q10-radian unit
//!
//! The reconstruction path ([`crate::lsp`]) accumulates the per-stage VQ
//! deltas directly into an internal **Q10-radian** unit (`1/1024` rad
//! per LSB) by folding the float `LSP_DIV_256/512/1024` scalings into
//! integer `×4 / ×2 / ×1` multiplies (see that module's docs). The
//! storage domain and the Q10-radian unit are therefore two coordinate
//! systems for the *same* angle, related by
//!
//! ```text
//! ω_q10  =  round( v_storage · 2^10 · π / LSP_PI )
//!        =  round( v_storage · 1024 · π / 25736 )
//!        ≈  v_storage / 8                          (1024·π/25736 = 0.125028…)
//! ```
//!
//! This module pins `LSP_PI` and the storage→radian conversion, and its
//! tests **cross-check** that the [`crate::lsp_base`] Q10-radian base
//! vector — derived independently from the float `LSP_LINEAR` form —
//! agrees with the `LSP_PI`-domain conversion of the Q15-storage
//! `LSP_LINEAR` form to within sub-Q10 rounding. Two independently
//! staged numeric facts (`LSP_LINEAR` and `LSP_PI`) pinning the same
//! angle is the provenance cross-check that closes the angular-unit gap.
//!
//! ## What is still NOT pinned (the remaining bit-exactness gap)
//!
//! `LSP_PI` fixes the *angular interpretation* of a stored LSP value.
//! It does **not** fix the exact fixed-point evaluation order of the
//! `cos(ω)` series the reference decoder uses inside its LSP→LPC
//! conversion (Speex evaluates `cos` through a fixed-point `lsp_cos`
//! lookup table + interpolation whose table is not staged). That
//! cosine-table detail — the last increment needed for a *bit-exact*
//! LSP→LPC match — remains a recorded docs gap, isolated to the
//! cosine evaluation and independent of this angular-unit pin.

use crate::codebooks::{HB_LPC_ORDER, NB_LSP_ORDER};
use crate::lsp::NB_LSP_OUTPUT_Q;

/// `LSP_PI` — the value representing the angle `π` in the Speex LSP
/// angular storage domain.
///
/// Numeric fact from `docs/audio/speex/provenance/02-speex-gain-quant.md`
/// ("LSP→LPC reconstruction path constants" table). The float build
/// uses `M_PI`; the fixed-point build uses this integer.
pub const LSP_PI: i32 = 25736;

/// Convert an LSP value in the codec's angular **storage domain** (the
/// domain `LSP_PI` measures `π` in) to an angle in radians.
///
/// ```text
/// ω  =  v_storage · π / LSP_PI            (LSP_PI = 25736)
/// ```
///
/// This is the exact, provenance-pinned angular interpretation (see the
/// module docs). It is *not* clamped — callers feeding a reconstructed,
/// margin-enforced vector get a value already inside `(0, π)`; the
/// clamp guard lives in [`crate::lsp_to_lpc::lsp_qn_to_radians`] for the
/// internal Q10-radian path.
pub fn lsp_storage_to_radians(v_storage: i32) -> f64 {
    f64::from(v_storage) * core::f64::consts::PI / f64::from(LSP_PI)
}

/// Convert a radian angle back into the LSP angular storage domain,
/// rounding to the nearest integer storage unit.
///
/// The exact inverse of [`lsp_storage_to_radians`] up to integer
/// rounding: `round(ω · LSP_PI / π)`.
pub fn radians_to_lsp_storage(omega: f64) -> i32 {
    (omega * f64::from(LSP_PI) / core::f64::consts::PI).round() as i32
}

/// Convert an LSP value in the angular **storage domain** to the crate's
/// internal **Q[`NB_LSP_OUTPUT_Q`] = Q10-radian** unit, rounding to the
/// nearest Q10 LSB.
///
/// ```text
/// ω_q10  =  round( v_storage · 2^Q · π / LSP_PI )
/// ```
///
/// This is the bridge between the two coordinate systems described in
/// the module docs: the codec's `LSP_PI` storage domain and the Q10-rad
/// unit the [`crate::lsp`] reconstruction accumulates the VQ deltas in.
pub fn lsp_storage_to_q10(v_storage: i32) -> i32 {
    let q10 = f64::from(v_storage) * f64::from(1i32 << NB_LSP_OUTPUT_Q) * core::f64::consts::PI
        / f64::from(LSP_PI);
    q10.round() as i32
}

/// The narrowband `LSP_LINEAR(i)` base vector in the **Q15 storage
/// domain** (`SHL16(i+1, 11) = (i+1) << 11`), as recorded in
/// `provenance/02`.
///
/// This is the codec's own storage-domain form of the base vector;
/// [`crate::lsp_base::nb_lsp_base_q10`] is the same vector re-expressed
/// in the crate's internal Q10-radian unit. The module tests confirm
/// the two agree under the `LSP_PI` conversion.
pub const fn nb_lsp_linear_storage() -> [i32; NB_LSP_ORDER] {
    let mut out = [0i32; NB_LSP_ORDER];
    let mut i = 0;
    while i < NB_LSP_ORDER {
        out[i] = ((i as i32) + 1) << 11;
        i += 1;
    }
    out
}

/// The high-band `LSP_LINEAR_HIGH(i)` base vector in the **Q15 storage
/// domain** (`i·2560 + 6144`), as recorded in `provenance/02`.
pub const fn hb_lsp_linear_storage() -> [i32; HB_LPC_ORDER] {
    let mut out = [0i32; HB_LPC_ORDER];
    let mut i = 0;
    while i < HB_LPC_ORDER {
        out[i] = (i as i32) * 2560 + 6144;
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp_base::{hb_lsp_base_q10, nb_lsp_base_q10};

    /// `LSP_PI` represents π: dividing it back out recovers π to better
    /// than one storage LSB.
    #[test]
    fn lsp_pi_represents_pi() {
        let pi_back = lsp_storage_to_radians(LSP_PI);
        assert!((pi_back - core::f64::consts::PI).abs() < 1e-12);
        // 25736 is the nearest integer to 8192·π (the Q13-radian domain):
        // 8192·π = 25735.93…, which rounds to 25736.
        let q13_pi = (8192.0 * core::f64::consts::PI).round() as i32;
        assert_eq!(LSP_PI, q13_pi);
    }

    /// `lsp_storage_to_radians` and `radians_to_lsp_storage` round-trip.
    #[test]
    fn storage_radian_round_trips() {
        for &v in &[0, 1, 2048, 6144, 12868, 25736] {
            let omega = lsp_storage_to_radians(v);
            assert_eq!(radians_to_lsp_storage(omega), v, "round-trip v={v}");
        }
    }

    /// The narrowband Q15-storage `LSP_LINEAR` vector, converted through
    /// the `LSP_PI` domain, lands on the **same** Q10-radian base vector
    /// the [`crate::lsp_base`] module derived independently from the
    /// float `.25·i + .25` form. This is the provenance cross-check that
    /// closes the angular-unit gap: two staged numeric facts
    /// (`LSP_LINEAR`, `LSP_PI`) pin one angle.
    #[test]
    fn nb_linear_storage_matches_q10_base_via_lsp_pi() {
        let storage = nb_lsp_linear_storage();
        let base_q10 = nb_lsp_base_q10();
        for i in 0..NB_LSP_ORDER {
            assert_eq!(
                lsp_storage_to_q10(storage[i]),
                base_q10[i],
                "NB coeff {i}: LSP_PI path disagrees with lsp_base Q10 base",
            );
        }
    }

    /// Same cross-check for the high-band `LSP_LINEAR_HIGH` vector.
    #[test]
    fn hb_linear_storage_matches_q10_base_via_lsp_pi() {
        let storage = hb_lsp_linear_storage();
        let base_q10 = hb_lsp_base_q10();
        for i in 0..HB_LPC_ORDER {
            assert_eq!(
                lsp_storage_to_q10(storage[i]),
                base_q10[i],
                "HB coeff {i}: LSP_PI path disagrees with lsp_base Q10 base",
            );
        }
    }

    /// The storage form matches the staged literal expressions exactly.
    #[test]
    fn linear_storage_matches_staged_literals() {
        let nb = nb_lsp_linear_storage();
        // SHL16(i+1, 11) = (i+1)·2048.
        for (i, &v) in nb.iter().enumerate() {
            assert_eq!(v, (i as i32 + 1) * 2048, "NB linear storage {i}");
        }
        assert_eq!(nb[0], 2048);
        assert_eq!(nb[9], 20480);

        let hb = hb_lsp_linear_storage();
        for (i, &v) in hb.iter().enumerate() {
            assert_eq!(v, i as i32 * 2560 + 6144, "HB linear storage {i}");
        }
        assert_eq!(hb[0], 6144);
        assert_eq!(hb[7], 7 * 2560 + 6144);
    }

    /// The storage→Q10 bridge is ≈ ÷8 (1024·π/25736 = 0.125028…), so a
    /// storage value is roughly one-eighth as many Q10 LSBs.
    #[test]
    fn storage_to_q10_is_approximately_eighth() {
        // LSP_PI storage units = π rad = round(1024·π) Q10.
        let q10_pi = lsp_storage_to_q10(LSP_PI);
        let expected_pi = (1024.0 * core::f64::consts::PI).round() as i32;
        assert_eq!(q10_pi, expected_pi);
        // 8192 storage ≈ 8192·0.125028 ≈ 1024 Q10 (the ≈÷8 relation).
        assert_eq!(lsp_storage_to_q10(8192), 1024);
    }

    /// A whole reconstructed-and-based NB LSP vector converts to the same
    /// radians whether taken through the internal Q10 path
    /// ([`crate::lsp_to_lpc::lsp_q10_to_radians`]) or — for the base
    /// component — through the `LSP_PI` storage path. We check the base
    /// vector itself: its Q10 form / 1024 equals its storage form ·
    /// π / LSP_PI to sub-LSB precision.
    #[test]
    fn base_vector_radians_agree_across_domains() {
        let storage = nb_lsp_linear_storage();
        let base_q10 = nb_lsp_base_q10();
        for i in 0..NB_LSP_ORDER {
            let rad_storage = lsp_storage_to_radians(storage[i]);
            let rad_q10 = f64::from(base_q10[i]) / f64::from(1i32 << NB_LSP_OUTPUT_Q);
            // The Q10 base is the rounded storage→Q10 conversion, so the
            // two radian values agree to within half a Q10 LSB.
            let half_lsb = 0.5 / f64::from(1i32 << NB_LSP_OUTPUT_Q);
            assert!(
                (rad_storage - rad_q10).abs() <= half_lsb,
                "coeff {i}: storage rad {rad_storage} vs Q10 rad {rad_q10}",
            );
        }
    }
}
