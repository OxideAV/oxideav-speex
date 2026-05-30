//! Narrowband LSP-VQ reconstruction (round r194 scope).
//!
//! Wires the narrowband LSP vector-quantisation codebooks staged in
//! [`crate::codebooks`] into a typed reconstruction path. Given a
//! per-stage 6-bit index sequence (either three stages for the 18-bit
//! LSP regime or five stages for the 30-bit regime — see
//! `docs/audio/speex/speex-celp-companion.md` §9.1 / §9 Table for
//! quantiser shape, and `docs/audio/speex/tables/README.md` for the
//! per-stage scaling factors), this module reconstructs the
//! ten-coefficient LSP frequency vector that the §9.1 LPC-from-LSP
//! conversion would take as input.
//!
//! ## What is reconstructed
//!
//! Per the companion's table inventory:
//!
//! * `nb-lsp-cdbk-stage0` — 64 × 10, scale 1/256, spans all 10 LSP
//!   coefficients.
//! * `nb-lsp-cdbk-low1` — 64 × 5, scale 1/512, spans coefficients 0..4.
//! * `nb-lsp-cdbk-low2` — 64 × 5, scale 1/1024, spans coefficients 0..4.
//! * `nb-lsp-cdbk-high1` — 64 × 5, scale 1/512, spans coefficients 5..9.
//! * `nb-lsp-cdbk-high2` — 64 × 5, scale 1/1024, spans coefficients 5..9.
//!
//! The 18-bit narrowband LSP field uses stage 0 + low1 + high1 (3
//! stages × 6 bits = 18 bits). The 30-bit field adds low2 + high2 (5
//! stages × 6 bits = 30 bits). Mode 0 ("silence", `LspQuant::None`)
//! transmits no LSP field at all.
//!
//! ## Fixed-point output convention
//!
//! Each codebook entry is a `signed char` (Speex source dimensioning;
//! kept as `i16` in the staged CSVs for sign-safety). Per the
//! `.meta` sidecars the per-stage decoder scaling is `1/256`,
//! `1/512`, or `1/1024`. To keep the reconstructed coefficients in
//! integer arithmetic with no loss, this module sums each stage's
//! contribution scaled to a **common Q10 fixed-point** representation
//! (output unit = 1/1024 of the underlying LSP frequency unit). The
//! shift per stage is therefore:
//!
//! ```text
//! Div256  → contribution = entry × 4   // 1/256  ≡ 4/1024
//! Div512  → contribution = entry × 2   // 1/512  ≡ 2/1024
//! Div1024 → contribution = entry × 1   // 1/1024 ≡ 1/1024
//! ```
//!
//! Choosing Q10 (the LCM divisor) means every per-stage scaling is an
//! exact integer multiplication, no rounding direction question
//! arises, and any later round that wants to convert to a different
//! Q-format can do so with a single arithmetic shift.
//!
//! ## Documented assumption: bit-stream stage ordering
//!
//! The Speex manual fixes the per-stage shapes + scaling factors but
//! is silent on the **order** in which the per-stage 6-bit indices
//! are concatenated into the 18-bit / 30-bit packed LSP field on the
//! wire. Three signals lead this module to read **stage 0 first**
//! (most-significant 6 bits of the packed index), followed by the
//! low / high split-band stages in the same numeric order their
//! codebooks are named in `lsp_tables_nb.c`:
//!
//! 1. Stage 0 is the only stage that spans the full LSP order; the
//!    low/high split stages refine it. Emitting the coarse stage
//!    first matches every multi-stage VQ scheme in the staged CELP
//!    family (and is what the in-crate wideband module already
//!    assumes for its 12-bit `(stage1, stage2)` MSVQ split, see
//!    `src/wideband.rs`).
//! 2. The companion's table inventory lists the stages in this exact
//!    order (`stage0`, `low1`, `low2`, `high1`, `high2`); the §9
//!    table summary calls them out as "5-stage 6-bit VQ".
//! 3. The 18-bit budget exactly fits the first three stages in this
//!    order (stage 0 + low1 + high1 = 18 bits), and the 30-bit budget
//!    fits all five (stage 0 + low1 + low2 + high1 + high2 = 30 bits).
//!
//! If a future docs-gap fill contradicts this ordering, the splitter
//! function [`NbLspStages::from_packed`] is the only place that has
//! to change: the per-stage reconstruction in
//! [`reconstruct_q10`] consumes the resolved indices directly and is
//! independent of how they were unpacked from the wire.
//!
//! ## What this module does NOT do
//!
//! * No LSP→LPC conversion (the §9.1 Levinson-Durbin / Chebyshev
//!   root-find that turns ten LSP frequencies into ten LP filter
//!   coefficients). That is a follow-up round.
//! * No sub-frame interpolation. §9.1 prescribes linear interpolation
//!   between the previous and current frame's LSPs for sub-frames
//!   1..3 (sub-frame 4 carries the current LSPs directly). That is
//!   also a follow-up round.
//! * No encoder-side codebook search. The reverse direction
//!   (LSPs → quantised indices) is a separate analysis path.

use crate::codebooks::{
    nb_lsp_high1, nb_lsp_high2, nb_lsp_low1, nb_lsp_low2, nb_lsp_scale, nb_lsp_stage0, NbLspScale,
    NB_LSP_ORDER, NB_LSP_SPLIT_HALF, NB_LSP_STAGE_ENTRIES,
};
use crate::submode::LspQuant;

/// Number of bits per narrowband LSP VQ stage (6-bit index = 64 entries).
pub const NB_LSP_STAGE_BITS: u32 = 6;

/// Number of stages in the 18-bit LSP field (stage 0 + low1 + high1).
pub const NB_LSP_STAGES_18BIT: u32 = 3;

/// Number of stages in the 30-bit LSP field (stage 0 + low1 + low2 +
/// high1 + high2).
pub const NB_LSP_STAGES_30BIT: u32 = 5;

/// Maximum codebook index that fits in a 6-bit per-stage field.
pub const NB_LSP_INDEX_MASK: u32 = (1 << NB_LSP_STAGE_BITS) - 1;

/// Q-shift applied to per-stage contributions so they accumulate in a
/// single common fixed-point format (output unit = 1/1024 of the LSP
/// frequency unit; see module docs).
pub const NB_LSP_OUTPUT_Q: u32 = 10;

/// Per-stage 6-bit indices for a single narrowband LSP frame. Stages
/// 3 (`low2`) and 4 (`high2`) are `None` for the 18-bit LSP regime
/// and `Some(_)` for the 30-bit regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NbLspStages {
    /// 6-bit index into [`nb_lsp_stage0`] (full 10-coeff VQ).
    pub stage0: u8,
    /// 6-bit index into [`nb_lsp_low1`] (low-half coeffs 0..4).
    pub low1: u8,
    /// 6-bit index into [`nb_lsp_high1`] (high-half coeffs 5..9).
    pub high1: u8,
    /// 6-bit index into [`nb_lsp_low2`] — present in the 30-bit
    /// regime only.
    pub low2: Option<u8>,
    /// 6-bit index into [`nb_lsp_high2`] — present in the 30-bit
    /// regime only.
    pub high2: Option<u8>,
}

impl NbLspStages {
    /// Split the packed bit-field `lsp_index` parsed off the wire
    /// into per-stage 6-bit indices, given the sub-mode's
    /// quantisation regime.
    ///
    /// Returns `None` for [`LspQuant::None`] (mode 0 / silence — no
    /// LSP field is transmitted).
    ///
    /// See the module-level documentation on the bit-stream stage
    /// ordering convention used here.
    pub fn from_packed(lsp_index: u32, quant: LspQuant) -> Option<Self> {
        match quant {
            LspQuant::None => None,
            LspQuant::Bits18 => {
                // 18 bits laid out MSB-first as [stage0 | low1 | high1].
                let stage0 = ((lsp_index >> (NB_LSP_STAGE_BITS * 2)) & NB_LSP_INDEX_MASK) as u8;
                let low1 = ((lsp_index >> NB_LSP_STAGE_BITS) & NB_LSP_INDEX_MASK) as u8;
                let high1 = (lsp_index & NB_LSP_INDEX_MASK) as u8;
                Some(Self {
                    stage0,
                    low1,
                    high1,
                    low2: None,
                    high2: None,
                })
            }
            LspQuant::Bits30 => {
                // 30 bits MSB-first as [stage0 | low1 | low2 | high1 | high2].
                let stage0 = ((lsp_index >> (NB_LSP_STAGE_BITS * 4)) & NB_LSP_INDEX_MASK) as u8;
                let low1 = ((lsp_index >> (NB_LSP_STAGE_BITS * 3)) & NB_LSP_INDEX_MASK) as u8;
                let low2 = ((lsp_index >> (NB_LSP_STAGE_BITS * 2)) & NB_LSP_INDEX_MASK) as u8;
                let high1 = ((lsp_index >> NB_LSP_STAGE_BITS) & NB_LSP_INDEX_MASK) as u8;
                let high2 = (lsp_index & NB_LSP_INDEX_MASK) as u8;
                Some(Self {
                    stage0,
                    low1,
                    high1,
                    low2: Some(low2),
                    high2: Some(high2),
                })
            }
        }
    }

    /// Returns the number of stages contributing to this record (3 for
    /// the 18-bit regime, 5 for the 30-bit regime).
    pub const fn stage_count(&self) -> u32 {
        if self.low2.is_some() && self.high2.is_some() {
            NB_LSP_STAGES_30BIT
        } else {
            NB_LSP_STAGES_18BIT
        }
    }
}

/// Multiplier applied to a per-stage codebook entry so its
/// contribution lands in the common Q[`NB_LSP_OUTPUT_Q`] format.
///
/// `Div256 → ×4`, `Div512 → ×2`, `Div1024 → ×1` — see module docs.
const fn stage_shift_factor(scale: NbLspScale) -> i32 {
    match scale {
        NbLspScale::Div256 => 4,
        NbLspScale::Div512 => 2,
        NbLspScale::Div1024 => 1,
    }
}

/// Reconstruct the ten narrowband LSP frequency coefficients in the
/// common Q[`NB_LSP_OUTPUT_Q`] fixed-point format from the per-stage
/// indices.
///
/// Output unit: `(1 / 2^NB_LSP_OUTPUT_Q) ≈ 9.77e-4` of an LSP
/// frequency. Sign is preserved — codebook entries are signed bytes.
///
/// Algorithm (companion §9.1 + `tables/README.md` per-stage scaling):
/// for each contributing stage `s` over its coefficient range `r`,
/// `out[i] += cb_s[idx][i - r.start] * stage_shift_factor(scale_s)`,
/// where `scale_s` is the `.meta`-documented divisor for that stage.
///
/// `idx` is bounds-checked against the staged codebook entry count
/// (64 per stage); out-of-range indices return `None`. Since the
/// per-stage index field is 6 bits wide, a value in `0..64` is always
/// representable on the wire — `None` here would only fire if the
/// caller hand-constructed an out-of-range [`NbLspStages`].
pub fn reconstruct_q10(stages: NbLspStages) -> Option<[i32; NB_LSP_ORDER]> {
    let stage0_idx = stages.stage0 as usize;
    let low1_idx = stages.low1 as usize;
    let high1_idx = stages.high1 as usize;
    if stage0_idx >= NB_LSP_STAGE_ENTRIES
        || low1_idx >= NB_LSP_STAGE_ENTRIES
        || high1_idx >= NB_LSP_STAGE_ENTRIES
    {
        return None;
    }
    if let Some(low2) = stages.low2 {
        if (low2 as usize) >= NB_LSP_STAGE_ENTRIES {
            return None;
        }
    }
    if let Some(high2) = stages.high2 {
        if (high2 as usize) >= NB_LSP_STAGE_ENTRIES {
            return None;
        }
    }

    let mut out = [0i32; NB_LSP_ORDER];

    // Stage 0: full 10-coeff codebook, scale 1/256 → ×4.
    let stage0_row = nb_lsp_stage0()[stage0_idx];
    let s0_factor = stage_shift_factor(nb_lsp_scale(0).expect("stage 0 scale"));
    for (out_i, &v) in out.iter_mut().zip(stage0_row.iter()) {
        *out_i += i32::from(v) * s0_factor;
    }

    // Stage 1 low half: coeffs 0..4, scale 1/512 → ×2.
    let low1_row = nb_lsp_low1()[low1_idx];
    let s1_factor = stage_shift_factor(nb_lsp_scale(1).expect("stage 1 scale"));
    for i in 0..NB_LSP_SPLIT_HALF {
        out[i] += i32::from(low1_row[i]) * s1_factor;
    }

    // Stage 3 high half: coeffs 5..9, scale 1/512 → ×2.
    let high1_row = nb_lsp_high1()[high1_idx];
    let s3_factor = stage_shift_factor(nb_lsp_scale(3).expect("stage 3 scale"));
    for i in 0..NB_LSP_SPLIT_HALF {
        out[NB_LSP_SPLIT_HALF + i] += i32::from(high1_row[i]) * s3_factor;
    }

    // Stages 2 + 4 only contribute in the 30-bit regime.
    if let Some(low2_idx) = stages.low2 {
        let row = nb_lsp_low2()[low2_idx as usize];
        let factor = stage_shift_factor(nb_lsp_scale(2).expect("stage 2 scale"));
        for i in 0..NB_LSP_SPLIT_HALF {
            out[i] += i32::from(row[i]) * factor;
        }
    }
    if let Some(high2_idx) = stages.high2 {
        let row = nb_lsp_high2()[high2_idx as usize];
        let factor = stage_shift_factor(nb_lsp_scale(4).expect("stage 4 scale"));
        for i in 0..NB_LSP_SPLIT_HALF {
            out[NB_LSP_SPLIT_HALF + i] += i32::from(row[i]) * factor;
        }
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_from_packed_none_for_silence() {
        assert!(NbLspStages::from_packed(0, LspQuant::None).is_none());
        assert!(NbLspStages::from_packed(0xFFFF_FFFF, LspQuant::None).is_none());
    }

    #[test]
    fn split_from_packed_18bit_msb_first() {
        // Pack [stage0=0b101010, low1=0b010101, high1=0b110011]
        // → 18-bit big-endian field = 0b101010_010101_110011 = 0x2A5733
        let packed = 0b101010_010101_110011u32;
        let s = NbLspStages::from_packed(packed, LspQuant::Bits18).unwrap();
        assert_eq!(s.stage0, 0b101010);
        assert_eq!(s.low1, 0b010101);
        assert_eq!(s.high1, 0b110011);
        assert_eq!(s.low2, None);
        assert_eq!(s.high2, None);
        assert_eq!(s.stage_count(), NB_LSP_STAGES_18BIT);
    }

    #[test]
    fn split_from_packed_30bit_msb_first() {
        // Pack [stage0=1, low1=2, low2=3, high1=4, high2=5]
        let packed = (1u32 << 24) | (2 << 18) | (3 << 12) | (4 << 6) | 5;
        let s = NbLspStages::from_packed(packed, LspQuant::Bits30).unwrap();
        assert_eq!(s.stage0, 1);
        assert_eq!(s.low1, 2);
        assert_eq!(s.low2, Some(3));
        assert_eq!(s.high1, 4);
        assert_eq!(s.high2, Some(5));
        assert_eq!(s.stage_count(), NB_LSP_STAGES_30BIT);
    }

    #[test]
    fn split_from_packed_extracts_only_low_18_bits() {
        // Stray bits above bit 17 must not leak into the per-stage
        // indices (caller is expected to pass a value that fits the
        // sub-mode's field width, but we mask defensively).
        let stray = 0xFFFC_0000u32; // bits 18..=31 set
        let s = NbLspStages::from_packed(stray, LspQuant::Bits18).unwrap();
        assert_eq!(s.stage0, 0);
        assert_eq!(s.low1, 0);
        assert_eq!(s.high1, 0);
    }

    #[test]
    fn reconstruct_q10_18bit_uses_three_stages() {
        // Indices all zero → all stages contribute their row-0
        // entries. With the documented per-stage shifts the output
        // is a deterministic linear sum.
        let stages = NbLspStages {
            stage0: 0,
            low1: 0,
            high1: 0,
            low2: None,
            high2: None,
        };
        let out = reconstruct_q10(stages).unwrap();

        // Cross-check against the staged row-0 values:
        //   stage0[0] = [30, 19, 38, 34, 40, 32, 46, 43, 58, 43]
        //   low1[0]   = [-34, -52, -15, 45, 2]
        //   high1[0]  = [-26, -8, 29, 21, 4]
        // factors: stage0 ×4, low1 ×2, high1 ×2.
        // Coefficient 0 = 30*4 + (-34)*2 = 120 - 68 = 52.
        assert_eq!(out[0], 30 * 4 + (-34) * 2);
        // Coefficient 4 (last of low half) = 40*4 + 2*2 = 160 + 4 = 164.
        assert_eq!(out[4], 40 * 4 + 2 * 2);
        // Coefficient 5 (first of high half) = 32*4 + (-26)*2 = 128 - 52 = 76.
        assert_eq!(out[5], 32 * 4 + (-26) * 2);
        // Coefficient 9 (last) = 43*4 + 4*2 = 172 + 8 = 180.
        assert_eq!(out[9], 43 * 4 + 4 * 2);
    }

    #[test]
    fn reconstruct_q10_30bit_adds_two_more_stages() {
        // Indices all zero, 30-bit regime: low2[0] and high2[0]
        // contribute on top of the 18-bit result.
        //   low2[0]  = [-6, 53, -21, -24, 4]
        //   high2[0] = [11, 47, 16, -9, -46]
        // factors: low2 ×1 (Div1024), high2 ×1 (Div1024).
        let stages = NbLspStages {
            stage0: 0,
            low1: 0,
            high1: 0,
            low2: Some(0),
            high2: Some(0),
        };
        let out = reconstruct_q10(stages).unwrap();
        // Coefficient 0 = stage0*4 + low1*2 + low2*1 = 120 - 68 - 6 = 46.
        assert_eq!(out[0], 30 * 4 + (-34) * 2 + (-6));
        // Coefficient 4 = stage0*4 + low1*2 + low2*1 = 160 + 4 + 4 = 168.
        assert_eq!(out[4], 40 * 4 + 2 * 2 + 4);
        // Coefficient 5 = stage0*4 + high1*2 + high2*1 = 128 - 52 + 11 = 87.
        assert_eq!(out[5], 32 * 4 + (-26) * 2 + 11);
        // Coefficient 9 = stage0*4 + high1*2 + high2*1 = 172 + 8 - 46 = 134.
        assert_eq!(out[9], 43 * 4 + 4 * 2 + (-46));
    }

    #[test]
    fn reconstruct_q10_is_linear_per_stage() {
        // Swapping a single per-stage index changes only the
        // coefficients that stage spans (and by exactly the staged
        // row-difference × stage_factor).
        let base = NbLspStages {
            stage0: 0,
            low1: 0,
            high1: 0,
            low2: None,
            high2: None,
        };
        let bumped_low1 = NbLspStages { low1: 1, ..base };
        let a = reconstruct_q10(base).unwrap();
        let b = reconstruct_q10(bumped_low1).unwrap();
        // High half (coeffs 5..=9) must be unchanged (low1 doesn't touch them).
        for i in NB_LSP_SPLIT_HALF..NB_LSP_ORDER {
            assert_eq!(a[i], b[i], "high half changed by low-half stage bump");
        }
        // Low half delta must equal (low1[1] - low1[0]) * factor 2.
        // low1[1] = [23, 21, 52, 24, -33]; low1[0] = [-34, -52, -15, 45, 2].
        let row0 = [-34i32, -52, -15, 45, 2];
        let row1 = [23i32, 21, 52, 24, -33];
        for i in 0..NB_LSP_SPLIT_HALF {
            assert_eq!(b[i] - a[i], (row1[i] - row0[i]) * 2);
        }
    }

    #[test]
    fn reconstruct_q10_handles_max_stage0_index() {
        // Index 63 (max for a 6-bit field) on stage 0 must succeed
        // and the contribution must reference the last staged row,
        // not panic.
        let stages = NbLspStages {
            stage0: 63,
            low1: 0,
            high1: 0,
            low2: None,
            high2: None,
        };
        let out = reconstruct_q10(stages).unwrap();
        let last_row = nb_lsp_stage0()[63];
        // Coefficient 0 contribution from stage 0 = last_row[0] × 4.
        // Add low1[0][0] × 2 = -34 × 2 = -68.
        assert_eq!(out[0], i32::from(last_row[0]) * 4 + (-34) * 2);
    }

    #[test]
    fn reconstruct_q10_rejects_out_of_range_index() {
        // 6-bit field can only address 0..64 on the wire, but a
        // hand-constructed `NbLspStages` with `stage0 = 64` is
        // out-of-range — must return `None` rather than panic.
        let stages = NbLspStages {
            stage0: 64,
            low1: 0,
            high1: 0,
            low2: None,
            high2: None,
        };
        assert!(reconstruct_q10(stages).is_none());
    }

    #[test]
    fn from_packed_round_trips_through_reconstruct_q10_18bit() {
        // End-to-end probe: feed a packed 18-bit field through the
        // splitter, then through reconstruction. Must succeed and
        // return ten coefficients (we do not assert content here —
        // the per-stage shift test above covers that).
        let packed = (5u32 << 12) | (17 << 6) | 42;
        let s = NbLspStages::from_packed(packed, LspQuant::Bits18).unwrap();
        assert_eq!(s.stage0, 5);
        assert_eq!(s.low1, 17);
        assert_eq!(s.high1, 42);
        let out = reconstruct_q10(s).unwrap();
        assert_eq!(out.len(), NB_LSP_ORDER);
    }

    #[test]
    fn stage_shift_factors_match_module_doc() {
        assert_eq!(stage_shift_factor(NbLspScale::Div256), 4);
        assert_eq!(stage_shift_factor(NbLspScale::Div512), 2);
        assert_eq!(stage_shift_factor(NbLspScale::Div1024), 1);
        // Sanity: the common Q-format unit divisor.
        assert_eq!(1 << NB_LSP_OUTPUT_Q, 1024);
    }
}
