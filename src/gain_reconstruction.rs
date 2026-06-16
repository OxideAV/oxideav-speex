//! **Scalar gain reconstruction from the staged exact quantiser tables**
//! (r321 scope) — turns the index-only gain fields surfaced by
//! [`crate::fixed_codebook_gain`] (narrowband frame-level OL excitation
//! gain + per-sub-frame innovation-gain correction) and
//! [`crate::hb_excitation_gain`] (high-band per-sub-frame excitation
//! gain) into reconstructed scalar magnitudes via the **exact**
//! reconstruction lookup tables staged for the codec.
//!
//! ## Spec basis
//!
//! Earlier rounds modelled these quantisers with a *parametric*
//! log-domain grid `g = 10^((index − offset)/slope)` because the codec
//! author's exact reconstruction points were not yet available — only
//! the bit widths (manual Table 9.1 / Table 10.1) and the log-domain
//! shape were pinned. The clean-room extraction staged at
//! `docs/audio/speex/tables/` (`provenance/02-speex-gain-quant.md`) now
//! pins the **exact reconstruction levels and decision boundaries** for
//! every scalar gain quantiser, so this module reconstructs from those
//! tables directly. The parametric grid is retired.
//!
//! The staged tables and their documented reconstruction laws are:
//!
//! ### Narrowband open-loop frame excitation gain (5-bit)
//!
//! `tables/nb-ol-gain-table-q15.csv` — 32 levels addressed by the
//! 5-bit `OL Exc gain` index `qe`. The fixed-point decoder reconstructs
//! `ol_gain = MULT16_32_Q15(28406, ol_gain_table[qe])`; the float
//! decoder uses the equivalent closed form `ol_gain = exp(qe / 3.5) ·
//! SIG_SCALING`. The two agree: applying the Q15 reconstruction
//! multiplier `28406` to the staged level and dividing by `SIG_SCALING`
//! (`= 16384`, the fixed-point signal-domain unit) recovers the float
//! magnitude `exp(qe/3.5)` to within one Q15 quantisation step. This
//! module exposes that normalised float magnitude, which is the
//! decoder-domain gain a synthesis stage multiplies the innovation by:
//!
//! ```text
//! g(qe) = (28406 · ol_gain_table[qe] / 2^15) / SIG_SCALING
//!       ≈ exp(qe / 3.5)
//! ```
//!
//! ### Narrowband per-sub-frame innovation-gain correction (1-/3-bit)
//!
//! The fixed-codebook gain is `g_frame · g_subf` (companion §2.3). The
//! sub-frame correction `g_subf` is a scalar quantiser:
//!
//! * 3-bit (modes 5/6/7): `tables/nb-exc-gain-scal3-float.csv` — 8
//!   reconstruction levels, with `tables/nb-exc-gain-scal3-bound-float.csv`
//!   the 7 encode-side decision boundaries.
//! * 1-bit (modes 1/3/4): `tables/nb-exc-gain-scal1-float.csv` — 2
//!   reconstruction levels, with `tables/nb-exc-gain-scal1-bound-float.csv`
//!   the single decision boundary.
//!
//! The decoder forms the final innovation energy as
//! `ener = MULT16_32_Q14(scal[q], ol_gain)`, i.e. the reconstructed
//! correction level multiplies the frame-level gain.
//!
//! ### High-band excitation gain
//!
//! * 4-bit gain-correction (Table 10.1 `Excitation gain`, HB modes
//!   2..=4): `tables/hb-gc-quant-bound-float.csv` — 16 decision
//!   boundaries; reconstruction level `gc = 0.87360 · gc_bound[qgc]`.
//! * 5-bit folded gain (HB mode 1):
//!   `tables/hb-fold-quant-bound-float.csv` — 32 boundaries that double
//!   as the reconstruction levels.
//!
//! ## What this module DOES
//!
//! * [`reconstruct_frame_ol_exc_gain`] — exact NB frame-level gain from
//!   a [`crate::FrameInnovationGainIndex`] (silence → `0.0`).
//! * [`reconstruct_subframe_gain_correction`] — exact NB per-sub-frame
//!   innovation-gain correction multiplier from a
//!   [`crate::SubFrameInnovationGainCorrection`] (absent → `1.0`, the
//!   identity multiplier).
//! * [`reconstruct_fixed_codebook_gain`] — the composed
//!   `g_frame · g_subf` fixed-codebook gain from a
//!   [`crate::FixedCodebookGainIndices`].
//! * [`reconstruct_hb_exc_gain`] — exact HB per-sub-frame gain from a
//!   [`crate::HbExcitationGainIndex`] (absent → `0.0`), dispatching the
//!   5-bit folded / 4-bit gain-correction surface to the matching table.
//! * Raw-table accessors ([`nb_ol_exc_gain_levels`],
//!   [`nb_subframe_gain_levels_3bit`], [`nb_subframe_gain_levels_1bit`],
//!   [`hb_gain_correction_levels`], [`hb_folded_gain_levels`]) for
//!   callers that want the full reconstruction sweep.
//!
//! ## What this module DOES NOT do
//!
//! * No excitation scaling. Multiplying the §8.4 excitation `c[n]` by
//!   the reconstructed gain is a downstream synthesis layer that
//!   consumes these primitives.
//! * No encoder-side index selection (the decision boundaries are
//!   exposed for completeness / round-trip tests, not an encoder).

use crate::fixed_codebook_gain::{
    FixedCodebookGainIndices, FrameInnovationGainIndex, SubFrameInnovationGainCorrection,
};
use crate::hb_excitation_gain::HbExcitationGainIndex;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Embedded exact reconstruction tables (vendored from
// docs/audio/speex/tables/, byte-identical).
// ---------------------------------------------------------------------------

/// `ol_gain_table` (32 entries, Q15 / SIG_SCALING domain). The float
/// reconstruction magnitude is the level divided by `SIG_SCALING`.
const NB_OL_GAIN_Q15_CSV: &str = include_str!("../tables/nb-ol-gain-table-q15.csv");
/// `exc_gain_quant_scal3` (8 float reconstruction levels).
const NB_SCAL3_CSV: &str = include_str!("../tables/nb-exc-gain-scal3-float.csv");
/// `exc_gain_quant_scal3_bound` (7 float decision boundaries).
const NB_SCAL3_BOUND_CSV: &str = include_str!("../tables/nb-exc-gain-scal3-bound-float.csv");
/// `exc_gain_quant_scal1` (2 float reconstruction levels).
const NB_SCAL1_CSV: &str = include_str!("../tables/nb-exc-gain-scal1-float.csv");
/// `exc_gain_quant_scal1_bound` (1 float decision boundary).
const NB_SCAL1_BOUND_CSV: &str = include_str!("../tables/nb-exc-gain-scal1-bound-float.csv");
/// `gc_quant_bound` (16 float decision boundaries).
const HB_GC_BOUND_CSV: &str = include_str!("../tables/hb-gc-quant-bound-float.csv");
/// `fold_quant_bound` (32 float decision boundaries / levels).
const HB_FOLD_BOUND_CSV: &str = include_str!("../tables/hb-fold-quant-bound-float.csv");

/// `SIG_SCALING` in the fixed-point build (the signal-domain unit). The
/// reconstructed `ol_gain` is divided by this to recover the normalised
/// float gain (`exp(qe/3.5)`). Documented in
/// `provenance/02-speex-gain-quant.md`.
const SIG_SCALING: f32 = 16384.0;

/// `ol_gain` reconstruction multiplier: the fixed-point decoder forms
/// `ol_gain = MULT16_32_Q15(28406, ol_gain_table[qe])`, i.e.
/// `28406 · table / 2^15`. Documented in
/// `provenance/02-speex-gain-quant.md`.
const OL_GAIN_RECON_MULT: f32 = 28406.0;
/// Q15 scaling shift used by `MULT16_32_Q15` (`/ 2^15`).
const Q15: f32 = 32768.0;

/// High-band gain-correction reconstruction multiplier
/// (`QCONST16(0.87360, 15)` — `gc = 0.87360 · gc_bound[qgc]`).
/// Documented in `provenance/02-speex-gain-quant.md`.
const HB_GC_RECONSTRUCTION_MULT: f32 = 0.873_60;

/// Number of narrowband OL excitation-gain levels (5-bit field).
pub const NB_OL_EXC_GAIN_LEVELS: usize = 32;
/// Number of 3-bit sub-frame innovation-gain correction levels.
pub const NB_SUBFRAME_GAIN_LEVELS_3BIT: usize = 8;
/// Number of 1-bit sub-frame innovation-gain correction levels.
pub const NB_SUBFRAME_GAIN_LEVELS_1BIT: usize = 2;
/// Number of high-band 4-bit gain-correction levels.
pub const HB_GAIN_CORRECTION_LEVELS: usize = 16;
/// Number of high-band 5-bit folded-gain levels.
pub const HB_FOLDED_GAIN_LEVELS: usize = 32;

fn parse_floats(body: &str, expected: usize) -> Vec<f32> {
    let v: Vec<f32> = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim()
                .parse::<f32>()
                .expect("speex gain CSV: token must be a decimal float")
        })
        .collect();
    assert_eq!(
        v.len(),
        expected,
        "speex gain CSV row count mismatch: got {}, expected {}",
        v.len(),
        expected
    );
    v
}

fn parse_i64(body: &str, expected: usize) -> Vec<i64> {
    let v: Vec<i64> = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim()
                .parse::<i64>()
                .expect("speex gain CSV: token must be a decimal integer")
        })
        .collect();
    assert_eq!(
        v.len(),
        expected,
        "speex gain CSV row count mismatch: got {}, expected {}",
        v.len(),
        expected
    );
    v
}

/// The 32 narrowband OL excitation-gain reconstruction magnitudes, in
/// the decoder's normalised float signal domain
/// (`28406 · ol_gain_table[qe] / 2^15 / SIG_SCALING`, equivalently
/// `exp(qe/3.5)`).
pub fn nb_ol_exc_gain_levels() -> &'static [f32; NB_OL_EXC_GAIN_LEVELS] {
    static T: OnceLock<[f32; NB_OL_EXC_GAIN_LEVELS]> = OnceLock::new();
    T.get_or_init(|| {
        let q15 = parse_i64(NB_OL_GAIN_Q15_CSV, NB_OL_EXC_GAIN_LEVELS);
        let mut out = [0.0f32; NB_OL_EXC_GAIN_LEVELS];
        for (o, &v) in out.iter_mut().zip(q15.iter()) {
            // ol_gain = MULT16_32_Q15(28406, table) / SIG_SCALING.
            // Compute in f64 to avoid intermediate f32 overflow on the
            // large high-index Q15 levels (~1.3e8).
            *o = ((OL_GAIN_RECON_MULT as f64) * (v as f64) / (Q15 as f64) / (SIG_SCALING as f64))
                as f32;
        }
        out
    })
}

/// The 8 narrowband 3-bit sub-frame innovation-gain correction levels.
pub fn nb_subframe_gain_levels_3bit() -> &'static [f32; NB_SUBFRAME_GAIN_LEVELS_3BIT] {
    static T: OnceLock<[f32; NB_SUBFRAME_GAIN_LEVELS_3BIT]> = OnceLock::new();
    T.get_or_init(|| {
        let v = parse_floats(NB_SCAL3_CSV, NB_SUBFRAME_GAIN_LEVELS_3BIT);
        let mut out = [0.0f32; NB_SUBFRAME_GAIN_LEVELS_3BIT];
        out.copy_from_slice(&v);
        out
    })
}

/// The 7 narrowband 3-bit sub-frame correction encode-side decision
/// boundaries (exposed for round-trip / quantiser-shape tests).
pub fn nb_subframe_gain_bounds_3bit() -> &'static [f32] {
    static T: OnceLock<Vec<f32>> = OnceLock::new();
    T.get_or_init(|| parse_floats(NB_SCAL3_BOUND_CSV, NB_SUBFRAME_GAIN_LEVELS_3BIT - 1))
}

/// The 2 narrowband 1-bit sub-frame innovation-gain correction levels.
pub fn nb_subframe_gain_levels_1bit() -> &'static [f32; NB_SUBFRAME_GAIN_LEVELS_1BIT] {
    static T: OnceLock<[f32; NB_SUBFRAME_GAIN_LEVELS_1BIT]> = OnceLock::new();
    T.get_or_init(|| {
        let v = parse_floats(NB_SCAL1_CSV, NB_SUBFRAME_GAIN_LEVELS_1BIT);
        let mut out = [0.0f32; NB_SUBFRAME_GAIN_LEVELS_1BIT];
        out.copy_from_slice(&v);
        out
    })
}

/// The single narrowband 1-bit sub-frame correction decision boundary.
pub fn nb_subframe_gain_bound_1bit() -> f32 {
    static T: OnceLock<f32> = OnceLock::new();
    *T.get_or_init(|| parse_floats(NB_SCAL1_BOUND_CSV, 1)[0])
}

/// The 16 high-band 4-bit gain-correction reconstruction levels
/// (`0.87360 · gc_quant_bound[qgc]`).
pub fn hb_gain_correction_levels() -> &'static [f32; HB_GAIN_CORRECTION_LEVELS] {
    static T: OnceLock<[f32; HB_GAIN_CORRECTION_LEVELS]> = OnceLock::new();
    T.get_or_init(|| {
        let bound = parse_floats(HB_GC_BOUND_CSV, HB_GAIN_CORRECTION_LEVELS);
        let mut out = [0.0f32; HB_GAIN_CORRECTION_LEVELS];
        for (o, &b) in out.iter_mut().zip(bound.iter()) {
            *o = HB_GC_RECONSTRUCTION_MULT * b;
        }
        out
    })
}

/// The 32 high-band 5-bit folded-gain reconstruction levels
/// (`fold_quant_bound[q]`, which double as both boundary and level).
pub fn hb_folded_gain_levels() -> &'static [f32; HB_FOLDED_GAIN_LEVELS] {
    static T: OnceLock<[f32; HB_FOLDED_GAIN_LEVELS]> = OnceLock::new();
    T.get_or_init(|| {
        let v = parse_floats(HB_FOLD_BOUND_CSV, HB_FOLDED_GAIN_LEVELS);
        let mut out = [0.0f32; HB_FOLDED_GAIN_LEVELS];
        out.copy_from_slice(&v);
        out
    })
}

/// Reconstruct the narrowband frame-level OL excitation-gain magnitude
/// from a typed [`FrameInnovationGainIndex`].
///
/// * [`FrameInnovationGainIndex::Silence`] (mode 0) reconstructs to
///   `0.0` — no excitation gain is transmitted and the frame is silent.
/// * [`FrameInnovationGainIndex::Indexed`] reconstructs through the
///   exact 32-level [`nb_ol_exc_gain_levels`] table.
pub fn reconstruct_frame_ol_exc_gain(index: FrameInnovationGainIndex) -> f32 {
    match index {
        FrameInnovationGainIndex::Silence => 0.0,
        FrameInnovationGainIndex::Indexed(i) => {
            nb_ol_exc_gain_levels()[usize::from(i) % NB_OL_EXC_GAIN_LEVELS]
        }
    }
}

/// Reconstruct the narrowband per-sub-frame innovation-gain
/// **correction** multiplier `g_subf` from a typed
/// [`SubFrameInnovationGainCorrection`].
///
/// * [`SubFrameInnovationGainCorrection::Absent`] (0-bit budget, modes
///   0/2/8) reconstructs to `1.0` — the identity multiplier, i.e. the
///   fixed-codebook gain is the frame-level gain unchanged.
/// * [`SubFrameInnovationGainCorrection::OneBit`] reconstructs through
///   the exact 2-level [`nb_subframe_gain_levels_1bit`] table.
/// * [`SubFrameInnovationGainCorrection::ThreeBit`] reconstructs through
///   the exact 8-level [`nb_subframe_gain_levels_3bit`] table.
pub fn reconstruct_subframe_gain_correction(correction: SubFrameInnovationGainCorrection) -> f32 {
    match correction {
        SubFrameInnovationGainCorrection::Absent => 1.0,
        SubFrameInnovationGainCorrection::OneBit(i) => {
            nb_subframe_gain_levels_1bit()[usize::from(i) % NB_SUBFRAME_GAIN_LEVELS_1BIT]
        }
        SubFrameInnovationGainCorrection::ThreeBit(i) => {
            nb_subframe_gain_levels_3bit()[usize::from(i) % NB_SUBFRAME_GAIN_LEVELS_3BIT]
        }
    }
}

/// Reconstruct the composed narrowband fixed-codebook gain
/// `g = g_frame · g_subf` from a typed [`FixedCodebookGainIndices`]
/// (companion §2.3 product structure).
///
/// When the frame is silent the frame factor is `0.0`, so the product
/// is `0.0` regardless of the (absent) correction.
pub fn reconstruct_fixed_codebook_gain(indices: FixedCodebookGainIndices) -> f32 {
    let g_frame = reconstruct_frame_ol_exc_gain(indices.frame);
    let g_subf = reconstruct_subframe_gain_correction(indices.subframe);
    g_frame * g_subf
}

/// Reconstruct the high-band per-sub-frame excitation-gain magnitude
/// from a typed [`HbExcitationGainIndex`].
///
/// * [`HbExcitationGainIndex::Absent`] (mode 0) reconstructs to `0.0`.
/// * [`HbExcitationGainIndex::FiveBit`] (HB mode 1) reconstructs through
///   the exact 32-level folded-gain table [`hb_folded_gain_levels`].
/// * [`HbExcitationGainIndex::FourBit`] (HB modes 2..=4) reconstructs
///   through the exact 16-level gain-correction table
///   [`hb_gain_correction_levels`].
pub fn reconstruct_hb_exc_gain(index: HbExcitationGainIndex) -> f32 {
    match index {
        HbExcitationGainIndex::Absent => 0.0,
        HbExcitationGainIndex::FiveBit(i) => {
            hb_folded_gain_levels()[usize::from(i) % HB_FOLDED_GAIN_LEVELS]
        }
        HbExcitationGainIndex::FourBit(i) => {
            hb_gain_correction_levels()[usize::from(i) % HB_GAIN_CORRECTION_LEVELS]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reconstruction table has exactly the bit-width-implied
    /// number of entries.
    #[test]
    fn table_lengths_match_bit_widths() {
        assert_eq!(nb_ol_exc_gain_levels().len(), 32);
        assert_eq!(nb_subframe_gain_levels_3bit().len(), 8);
        assert_eq!(nb_subframe_gain_bounds_3bit().len(), 7);
        assert_eq!(nb_subframe_gain_levels_1bit().len(), 2);
        assert_eq!(hb_gain_correction_levels().len(), 16);
        assert_eq!(hb_folded_gain_levels().len(), 32);
    }

    /// Every reconstruction level is finite and strictly positive.
    #[test]
    fn all_levels_finite_and_positive() {
        let mut all: Vec<f32> = Vec::new();
        all.extend_from_slice(nb_ol_exc_gain_levels());
        all.extend_from_slice(nb_subframe_gain_levels_3bit());
        all.extend_from_slice(nb_subframe_gain_levels_1bit());
        all.extend_from_slice(hb_gain_correction_levels());
        all.extend_from_slice(hb_folded_gain_levels());
        for v in all {
            assert!(
                v.is_finite() && v > 0.0,
                "level {v} must be finite + positive"
            );
        }
    }

    /// Each scalar quantiser's reconstruction levels are strictly
    /// monotone increasing in the index (the staged tables are sorted
    /// log-domain quantisers).
    #[test]
    fn levels_strictly_monotone() {
        for tbl in [
            &nb_ol_exc_gain_levels()[..],
            &nb_subframe_gain_levels_3bit()[..],
            &nb_subframe_gain_levels_1bit()[..],
            &hb_gain_correction_levels()[..],
            &hb_folded_gain_levels()[..],
        ] {
            for w in tbl.windows(2) {
                assert!(w[1] > w[0], "levels must increase: {} !> {}", w[1], w[0]);
            }
        }
    }

    /// The narrowband OL gain table reproduces the documented float law
    /// `exp(qe/3.5)` (the Q15 staged levels divided by SIG_SCALING).
    #[test]
    fn nb_ol_gain_matches_exp_law() {
        let levels = nb_ol_exc_gain_levels();
        for (qe, &g) in levels.iter().enumerate() {
            let expected = (qe as f32 / 3.5).exp();
            // The Q15 source quantises exp(qe/3.5)·16384 to an integer,
            // so the float law is reproduced to the integer-rounding
            // tolerance of one Q15 level over the magnitude.
            let rel = (g - expected).abs() / expected;
            assert!(
                rel < 1e-3,
                "qe {qe}: level {g} vs exp law {expected} (rel {rel})"
            );
        }
    }

    /// The decision boundaries strictly interleave the reconstruction
    /// levels for the 3-bit quantiser: `level[i] < bound[i] < level[i+1]`.
    #[test]
    fn scal3_bounds_interleave_levels() {
        let levels = nb_subframe_gain_levels_3bit();
        let bounds = nb_subframe_gain_bounds_3bit();
        for (i, &b) in bounds.iter().enumerate() {
            assert!(
                levels[i] < b && b < levels[i + 1],
                "bound {b} must lie between levels {} and {}",
                levels[i],
                levels[i + 1]
            );
        }
    }

    /// The single 1-bit boundary lies between the two 1-bit levels.
    #[test]
    fn scal1_bound_interleaves_levels() {
        let levels = nb_subframe_gain_levels_1bit();
        let b = nb_subframe_gain_bound_1bit();
        assert!(levels[0] < b && b < levels[1], "bound {b} between levels");
    }

    /// HB gain-correction levels equal `0.87360 · gc_quant_bound`.
    #[test]
    fn hb_gain_correction_applies_reconstruction_mult() {
        let levels = hb_gain_correction_levels();
        let bounds = parse_floats(HB_GC_BOUND_CSV, HB_GAIN_CORRECTION_LEVELS);
        for (l, b) in levels.iter().zip(bounds.iter()) {
            assert!((*l - HB_GC_RECONSTRUCTION_MULT * b).abs() < 1e-6);
        }
    }

    /// NB frame reconstruction: silence → 0; an indexed field hits the
    /// exact level for its index.
    #[test]
    fn nb_frame_reconstruction_dispatch() {
        assert_eq!(
            reconstruct_frame_ol_exc_gain(FrameInnovationGainIndex::Silence),
            0.0
        );
        let levels = nb_ol_exc_gain_levels();
        for i in 0..32u8 {
            assert_eq!(
                reconstruct_frame_ol_exc_gain(FrameInnovationGainIndex::Indexed(i)),
                levels[usize::from(i)],
                "index {i}"
            );
        }
    }

    /// Sub-frame correction: absent → identity 1.0; 1-bit / 3-bit hit
    /// the exact level.
    #[test]
    fn subframe_correction_dispatch() {
        assert_eq!(
            reconstruct_subframe_gain_correction(SubFrameInnovationGainCorrection::Absent),
            1.0
        );
        let l1 = nb_subframe_gain_levels_1bit();
        for i in 0..2u8 {
            assert_eq!(
                reconstruct_subframe_gain_correction(SubFrameInnovationGainCorrection::OneBit(i)),
                l1[usize::from(i)]
            );
        }
        let l3 = nb_subframe_gain_levels_3bit();
        for i in 0..8u8 {
            assert_eq!(
                reconstruct_subframe_gain_correction(SubFrameInnovationGainCorrection::ThreeBit(i)),
                l3[usize::from(i)]
            );
        }
    }

    /// Composed fixed-codebook gain is the product of frame and
    /// sub-frame factors; silence frame zeroes the product.
    #[test]
    fn fixed_codebook_gain_is_product() {
        let frame = FrameInnovationGainIndex::Indexed(10);
        let subf = SubFrameInnovationGainCorrection::ThreeBit(5);
        let g = reconstruct_fixed_codebook_gain(FixedCodebookGainIndices {
            frame,
            subframe: subf,
        });
        let expected =
            reconstruct_frame_ol_exc_gain(frame) * reconstruct_subframe_gain_correction(subf);
        assert_eq!(g, expected);

        let silent = reconstruct_fixed_codebook_gain(FixedCodebookGainIndices {
            frame: FrameInnovationGainIndex::Silence,
            subframe: SubFrameInnovationGainCorrection::Absent,
        });
        assert_eq!(silent, 0.0);
    }

    /// HB reconstruction: absent → 0; 5-bit → folded table; 4-bit →
    /// gain-correction table.
    #[test]
    fn hb_reconstruction_dispatch() {
        assert_eq!(reconstruct_hb_exc_gain(HbExcitationGainIndex::Absent), 0.0);
        let fold = hb_folded_gain_levels();
        for i in 0..32u8 {
            assert_eq!(
                reconstruct_hb_exc_gain(HbExcitationGainIndex::FiveBit(i)),
                fold[usize::from(i)],
                "5-bit {i}"
            );
        }
        let gc = hb_gain_correction_levels();
        for i in 0..16u8 {
            assert_eq!(
                reconstruct_hb_exc_gain(HbExcitationGainIndex::FourBit(i)),
                gc[usize::from(i)],
                "4-bit {i}"
            );
        }
    }

    /// Spot-check exact staged values against the provenance manifest's
    /// recorded extremes (defends against a vendoring / parse slip).
    #[test]
    fn staged_value_spot_checks() {
        // ol_gain_table Q15 first/last: 18900 / 132760927; float-domain
        // = 28406 · table / 2^15 / 2^14.
        let ol = nb_ol_exc_gain_levels();
        let f = |t: f64| (28406.0 * t / 32768.0 / 16384.0) as f32;
        assert!((ol[0] - f(18900.0)).abs() < 1e-4);
        assert!((ol[31] - f(132_760_927.0)).abs() < 1.0);
        // scal3 float endpoints 0.061130 / 1.326874.
        let s3 = nb_subframe_gain_levels_3bit();
        assert!((s3[0] - 0.061130).abs() < 1e-6);
        assert!((s3[7] - 1.326874).abs() < 1e-6);
        // scal1 float levels 0.70469 / 1.05127.
        let s1 = nb_subframe_gain_levels_1bit();
        assert!((s1[0] - 0.70469).abs() < 1e-6);
        assert!((s1[1] - 1.05127).abs() < 1e-6);
        // gc bound first 0.97979 → level 0.87360 · it.
        let gc = hb_gain_correction_levels();
        assert!((gc[0] - 0.873_60 * 0.97979).abs() < 1e-5);
        // fold bound first/last 0.30498 / 14.69497.
        let fold = hb_folded_gain_levels();
        assert!((fold[0] - 0.30498).abs() < 1e-5);
        assert!((fold[31] - 14.69497).abs() < 1e-4);
    }
}
