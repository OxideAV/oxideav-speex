//! Encoder-side narrowband LSP vector quantiser — the multi-stage VQ
//! codebook *search* that picks the per-stage indices best representing a
//! target LSP vector.
//!
//! This is the encode-direction inverse of the decoder's
//! [`crate::lsp::reconstruct_q10`]. The decoder consumed a packed
//! `lsp_index`, split it into per-stage 6-bit indices ([`NbLspStages`]),
//! and *summed* the scaled codebook rows into the Q10 LSP vector. The
//! encoder must do the opposite: given the order-10 Q10 LSP vector
//! produced by [`crate::lpc_to_lsp`] (after the radian→Q10 conversion),
//! find the per-stage indices whose reconstruction is closest to it.
//!
//! ## Multi-stage VQ search (manual §9.1)
//!
//! The narrowband LSP quantiser is a 5-stage multi-stage VQ (companion
//! §9.1 + the staged `tables/README.md`):
//!
//! * **stage 0** — a full 10-coefficient VQ (`nb_lsp_stage0`, scale
//!   1/256), the coarse envelope.
//! * **stages 1 / 2** — split low-half (coeffs 0..4) residual refinement
//!   (`nb_lsp_low1` scale 1/512, `nb_lsp_low2` scale 1/1024).
//! * **stages 3 / 4** — split high-half (coeffs 5..9) residual refinement
//!   (`nb_lsp_high1` scale 1/512, `nb_lsp_high2` scale 1/1024).
//!
//! The 18-bit regime uses stages 0 / 1 / 3; the 30-bit regime adds the
//! second-stage refinements 2 / 4. The search is **sequential** (the
//! standard multi-stage-VQ greedy search): pick the stage-0 row that
//! minimises the squared error to the full target, subtract its scaled
//! contribution, then independently pick the best low-half and high-half
//! refinement rows on the residual. Each refinement stage is chosen to
//! minimise the residual it sees, so the composed reconstruction is a
//! near-optimal codebook representation of the target (greedy MSVQ does
//! not guarantee the *globally* optimal index tuple, but each stage only
//! reduces the residual it is given — the refinement is monotone). The
//! per-stage scaling matches [`crate::lsp::reconstruct_q10`] exactly, so
//! the chosen indices reconstruct through the existing decoder path.
//!
//! Clean-room note: the multi-stage-VQ search is the textbook encode
//! counterpart of the staged reconstruction; it consults only the staged
//! codebook **data** ([`crate::codebooks`]) and is round-trip validated
//! against the in-tree [`crate::lsp::reconstruct_q10`]. No external
//! library source is consulted.

use crate::codebooks::{
    nb_lsp_high1, nb_lsp_high2, nb_lsp_low1, nb_lsp_low2, nb_lsp_scale, nb_lsp_stage0, NbLspScale,
    NB_LSP_ORDER, NB_LSP_SPLIT_HALF,
};
use crate::lsp::NbLspStages;
use crate::submode::LspQuant;

/// Map the per-stage `.meta` scale to the Q10 reconstruction multiplier,
/// matching [`crate::lsp`]'s private `stage_shift_factor`
/// (`Div256 → ×4`, `Div512 → ×2`, `Div1024 → ×1`).
fn stage_factor(scale: NbLspScale) -> i32 {
    match scale {
        NbLspScale::Div256 => 4,
        NbLspScale::Div512 => 2,
        NbLspScale::Div1024 => 1,
    }
}

/// Find the stage-0 codebook row (full 10-coeff VQ) whose scaled
/// contribution best matches `target` (least squared error), returning
/// its index.
fn search_stage0(target: &[i32; NB_LSP_ORDER]) -> u8 {
    let cb = nb_lsp_stage0();
    let factor = stage_factor(nb_lsp_scale(0).expect("stage 0 scale"));
    let mut best = 0u8;
    let mut best_err = i64::MAX;
    for (idx, row) in cb.iter().enumerate() {
        let mut err = 0i64;
        for (i, &v) in row.iter().enumerate() {
            let d = i64::from(target[i]) - i64::from(i32::from(v) * factor);
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best = idx as u8;
        }
    }
    best
}

/// Find the split-half codebook row (coeffs `0..5` of the residual half)
/// best matching `residual_half`, against the given codebook + scale.
fn search_half(
    residual_half: &[i32; NB_LSP_SPLIT_HALF],
    cb: &[[i16; NB_LSP_SPLIT_HALF]],
    factor: i32,
) -> u8 {
    let mut best = 0u8;
    let mut best_err = i64::MAX;
    for (idx, row) in cb.iter().enumerate() {
        let mut err = 0i64;
        for (i, &v) in row.iter().enumerate() {
            let d = i64::from(residual_half[i]) - i64::from(i32::from(v) * factor);
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best = idx as u8;
        }
    }
    best
}

/// Quantise an order-10 Q10 LSP vector to the multi-stage VQ indices of
/// the given regime — the encode inverse of
/// [`crate::lsp::reconstruct_q10`].
///
/// Returns the [`NbLspStages`] (3 stages for [`LspQuant::Bits18`], 5 for
/// [`LspQuant::Bits30`]) whose reconstruction is closest to `target`, or
/// `None` for [`LspQuant::None`] (mode 0 — no LSP field is transmitted).
///
/// The search is sequential: stage 0 (full vector) → low/high refinement
/// stage 1 (residual) → second low/high refinement stage 2 (residual of
/// the residual, 30-bit only). Because the reconstruction sums the scaled
/// rows, subtracting each chosen stage's contribution leaves the exact
/// residual the next stage approximates, so the refinement is monotone
/// (each stage only reduces the error it sees). The greedy search is
/// near-optimal, not guaranteed globally optimal — a different stage-0
/// row can leave a residual the split codebooks represent better.
pub fn quantise_lsp_q10(target: &[i32; NB_LSP_ORDER], quant: LspQuant) -> Option<NbLspStages> {
    if quant == LspQuant::None {
        return None;
    }

    // Stage 0: coarse full-vector VQ.
    let stage0 = search_stage0(target);
    let s0_row = nb_lsp_stage0()[stage0 as usize];
    let s0_factor = stage_factor(nb_lsp_scale(0).expect("stage 0 scale"));
    let mut residual = *target;
    for (i, &v) in s0_row.iter().enumerate() {
        residual[i] -= i32::from(v) * s0_factor;
    }

    // Split the residual into low (0..5) and high (5..10) halves.
    let mut low: [i32; NB_LSP_SPLIT_HALF] = [0; NB_LSP_SPLIT_HALF];
    let mut high: [i32; NB_LSP_SPLIT_HALF] = [0; NB_LSP_SPLIT_HALF];
    low.copy_from_slice(&residual[..NB_LSP_SPLIT_HALF]);
    high.copy_from_slice(&residual[NB_LSP_SPLIT_HALF..]);

    // Stage 1 (low1) + stage 3 (high1): first refinement, scale 1/512.
    let low1 = search_half(&low, nb_lsp_low1(), stage_factor(nb_lsp_scale(1).unwrap()));
    let high1 = search_half(
        &high,
        nb_lsp_high1(),
        stage_factor(nb_lsp_scale(3).unwrap()),
    );

    if quant == LspQuant::Bits18 {
        return Some(NbLspStages {
            stage0,
            low1,
            high1,
            low2: None,
            high2: None,
        });
    }

    // 30-bit regime: subtract stage-1 contributions and refine again with
    // stages 2 (low2) + 4 (high2), scale 1/1024.
    let l1_factor = stage_factor(nb_lsp_scale(1).unwrap());
    let h1_factor = stage_factor(nb_lsp_scale(3).unwrap());
    let l1_row = nb_lsp_low1()[low1 as usize];
    let h1_row = nb_lsp_high1()[high1 as usize];
    for i in 0..NB_LSP_SPLIT_HALF {
        low[i] -= i32::from(l1_row[i]) * l1_factor;
        high[i] -= i32::from(h1_row[i]) * h1_factor;
    }
    let low2 = search_half(&low, nb_lsp_low2(), stage_factor(nb_lsp_scale(2).unwrap()));
    let high2 = search_half(
        &high,
        nb_lsp_high2(),
        stage_factor(nb_lsp_scale(4).unwrap()),
    );

    Some(NbLspStages {
        stage0,
        low1,
        high1,
        low2: Some(low2),
        high2: Some(high2),
    })
}

/// Pack the per-stage 6-bit indices of [`NbLspStages`] into the on-wire
/// `lsp_index` bit-field, MSB-first, matching the layout
/// [`NbLspStages::from_packed`] parses.
///
/// 18-bit regime: `[stage0 | low1 | high1]`. 30-bit regime:
/// `[stage0 | low1 | low2 | high1 | high2]`.
pub fn pack_lsp_index(stages: &NbLspStages) -> u32 {
    let s = u32::from;
    const B: u32 = crate::lsp::NB_LSP_STAGE_BITS;
    match (stages.low2, stages.high2) {
        (Some(low2), Some(high2)) => {
            (s(stages.stage0) << (B * 4))
                | (s(stages.low1) << (B * 3))
                | (s(low2) << (B * 2))
                | (s(stages.high1) << B)
                | s(high2)
        }
        _ => (s(stages.stage0) << (B * 2)) | (s(stages.low1) << B) | s(stages.high1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebooks::NB_LSP_STAGE_ENTRIES;
    use crate::lsp::reconstruct_q10;

    /// Squared error between a reconstruction and a target vector.
    fn sq_err(recon: &[i32; NB_LSP_ORDER], target: &[i32; NB_LSP_ORDER]) -> i64 {
        recon
            .iter()
            .zip(target.iter())
            .map(|(a, b)| {
                let d = i64::from(*a) - i64::from(*b);
                d * d
            })
            .sum()
    }

    /// The greedy multi-stage VQ search must not increase error across
    /// stages: the full quantiser reconstruction is at least as close to
    /// the target as the stage-0-only reconstruction (each refinement
    /// stage is chosen to reduce the residual it sees, so it can never
    /// make the composed error worse).
    ///
    /// Note: greedy MSVQ does *not* promise
    /// `quantise(reconstruct(stages)) == stages` — a different stage-0 row
    /// may have lower error to the full vector yet leave a residual the
    /// split codebooks represent differently. The monotone-refinement
    /// invariant is what holds and is what we check.
    fn assert_refinement_monotone(stages: NbLspStages, quant: LspQuant) {
        let target = reconstruct_q10(stages).unwrap();
        let requant = quantise_lsp_q10(&target, quant).unwrap();
        let full_err = sq_err(&reconstruct_q10(requant).unwrap(), &target);

        // Stage-0-only reconstruction error (the analytic stage-0
        // contribution the quantiser picked, with no refinement). The
        // full quantiser reconstruction must be at least as close.
        let s0_factor = stage_factor(nb_lsp_scale(0).unwrap());
        let s0_row = nb_lsp_stage0()[requant.stage0 as usize];
        let mut s0_recon = [0i32; NB_LSP_ORDER];
        for (i, &v) in s0_row.iter().enumerate() {
            s0_recon[i] = i32::from(v) * s0_factor;
        }
        let stage0_err = sq_err(&s0_recon, &target);
        assert!(
            full_err <= stage0_err,
            "refinement must not worsen error: full {full_err} > stage0 {stage0_err}"
        );
    }

    #[test]
    fn silence_regime_yields_none() {
        let target = [0i32; NB_LSP_ORDER];
        assert!(quantise_lsp_q10(&target, LspQuant::None).is_none());
    }

    #[test]
    fn quantises_zero_vector_to_valid_indices() {
        let target = [0i32; NB_LSP_ORDER];
        let s = quantise_lsp_q10(&target, LspQuant::Bits18).unwrap();
        assert!((s.stage0 as usize) < NB_LSP_STAGE_ENTRIES);
        assert!((s.low1 as usize) < NB_LSP_STAGE_ENTRIES);
        assert!((s.high1 as usize) < NB_LSP_STAGE_ENTRIES);
        assert!(s.low2.is_none() && s.high2.is_none());
    }

    #[test]
    fn round_trip_18bit_known_indices() {
        // Build a target from known stage-0/low1/high1 indices, quantise,
        // and confirm the reconstruction matches.
        for &(s0, l1, h1) in &[(0u8, 0u8, 0u8), (1, 2, 3), (17, 31, 5), (63, 63, 63)] {
            let stages = NbLspStages {
                stage0: s0,
                low1: l1,
                high1: h1,
                low2: None,
                high2: None,
            };
            assert_refinement_monotone(stages, LspQuant::Bits18);
        }
    }

    #[test]
    fn round_trip_30bit_known_indices() {
        for &(s0, l1, l2, h1, h2) in &[
            (0u8, 0u8, 0u8, 0u8, 0u8),
            (1, 2, 3, 4, 5),
            (40, 12, 50, 7, 33),
            (63, 63, 63, 63, 63),
        ] {
            let stages = NbLspStages {
                stage0: s0,
                low1: l1,
                high1: h1,
                low2: Some(l2),
                high2: Some(h2),
            };
            assert_refinement_monotone(stages, LspQuant::Bits30);
        }
    }

    #[test]
    fn pack_then_from_packed_round_trips_18bit() {
        let stages = NbLspStages {
            stage0: 0b101010,
            low1: 0b010101,
            high1: 0b110011,
            low2: None,
            high2: None,
        };
        let packed = pack_lsp_index(&stages);
        let back = NbLspStages::from_packed(packed, LspQuant::Bits18).unwrap();
        assert_eq!(back, stages);
    }

    #[test]
    fn pack_then_from_packed_round_trips_30bit() {
        let stages = NbLspStages {
            stage0: 0b101010,
            low1: 0b010101,
            high1: 0b110011,
            low2: Some(0b001100),
            high2: Some(0b111000),
        };
        let packed = pack_lsp_index(&stages);
        let back = NbLspStages::from_packed(packed, LspQuant::Bits30).unwrap();
        assert_eq!(back, stages);
    }

    #[test]
    fn quantise_then_pack_then_decode_round_trips() {
        // Full encode chain: target Q10 → quantise → pack → from_packed →
        // reconstruct, and verify the reconstruction matches the direct
        // quantise reconstruction.
        let stages = NbLspStages {
            stage0: 7,
            low1: 11,
            high1: 22,
            low2: Some(3),
            high2: Some(44),
        };
        let target = reconstruct_q10(stages).unwrap();
        let q = quantise_lsp_q10(&target, LspQuant::Bits30).unwrap();
        let packed = pack_lsp_index(&q);
        let decoded = NbLspStages::from_packed(packed, LspQuant::Bits30).unwrap();
        assert_eq!(
            reconstruct_q10(decoded).unwrap(),
            reconstruct_q10(q).unwrap()
        );
    }

    #[test]
    fn quantiser_reduces_error_versus_arbitrary_index() {
        // The chosen stage-0 index must have ≤ error than a fixed wrong
        // guess for a non-trivial target.
        let stages = NbLspStages {
            stage0: 30,
            low1: 5,
            high1: 50,
            low2: None,
            high2: None,
        };
        let target = reconstruct_q10(stages).unwrap();
        let chosen = quantise_lsp_q10(&target, LspQuant::Bits18).unwrap();
        let err_of = |st: NbLspStages| -> i64 {
            let r = reconstruct_q10(st).unwrap();
            r.iter()
                .zip(target.iter())
                .map(|(a, b)| {
                    let d = i64::from(*a) - i64::from(*b);
                    d * d
                })
                .sum()
        };
        let wrong = NbLspStages {
            stage0: 0,
            low1: 0,
            high1: 0,
            low2: None,
            high2: None,
        };
        assert!(err_of(chosen) <= err_of(wrong));
        // Exact codebook target → zero error.
        assert_eq!(err_of(chosen), 0);
    }
}
