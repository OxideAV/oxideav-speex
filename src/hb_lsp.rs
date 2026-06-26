//! Wideband high-band LSP MSVQ reconstruction (round r214 scope).
//!
//! Wires the two-stage 6-bit MSVQ codebooks staged in
//! [`crate::codebooks`] into a typed reconstruction path for the
//! wideband high-band's eight-coefficient LSP frequency vector. Given
//! the 12-bit `lsp_index` field already surfaced by
//! [`crate::WidebandHighBandBody::lsp_index`] (round-r160) plus the
//! resolved sub-mode (round-r160), this module yields an
//! eight-coefficient Q10 fixed-point LSP vector that the §10.1
//! high-band LSP→LPC conversion would take as input.
//!
//! ## What is reconstructed
//!
//! Per the staged Speex Codec Manual §10.1 *"we use only 12 bits to
//! encode the high-band LSP's using a multi-stage vector quantizer
//! (MSVQ). The first level quantizes the 10 coefficients with 6 bits
//! and the error is then quantized using 6 bits, too."* The
//! companion `docs/audio/speex/speex-celp-companion.md` §9
//! corroborates this as **"2-stage 6-bit MSVQ, order 8 (12 bits
//! total)"** and the `.meta` sidecars cross-check the codebook
//! dimensions:
//!
//! * `hb-lsp-cdbk-stage1` — 64 × 8, scale 1/256.
//! * `hb-lsp-cdbk-stage2` — 64 × 8, scale 1/512.
//!
//! The high-band LPC order is **8** (not 10 like the narrowband):
//! [`crate::codebooks::HB_LPC_ORDER`] = 8. The manual's *"10
//! coefficients"* prose in §10.1 is reconciled by the companion §9
//! and the `.meta` sidecar (`order=8`) — the in-crate
//! [`crate::codebooks::HB_LPC_ORDER`] constant has been the
//! authoritative value since r191. The eight-coefficient output of
//! this module matches that constant.
//!
//! The 12-bit packed `lsp_index` is split most-significant 6 bits =
//! stage 1 index (level-1 codebook), low 6 bits = stage 2 index
//! (residual codebook). The in-crate `WidebandHighBandBody` docs
//! comment fixes this ordering ("top 6 bits = level-1 codebook,
//! low 6 bits = level-2 codebook").
//!
//! ## Fixed-point output convention
//!
//! Each codebook entry is a `signed char` (kept as `i16` in the
//! staged CSVs for sign-safety). Per the `.meta` sidecars the
//! per-stage decoder scaling is `1/256` (stage 1) and `1/512`
//! (stage 2). To keep the reconstructed coefficients in integer
//! arithmetic with no loss, this module sums each stage's
//! contribution scaled to a **common Q10 fixed-point**
//! representation (output unit = 1/1024 of the underlying LSP
//! frequency unit). The shift per stage is therefore:
//!
//! ```text
//! Div256 → contribution = entry × 4   // 1/256 ≡ 4/1024
//! Div512 → contribution = entry × 2   // 1/512 ≡ 2/1024
//! ```
//!
//! Choosing Q10 (matching r194's narrowband choice) means every
//! per-stage scaling is an exact integer multiplication, no rounding
//! direction question arises, and the eventual high-band LSP→LPC
//! converter can rescale with a single arithmetic shift.
//!
//! ## Silence mode
//!
//! High-band mode 0 (silence / comfort-noise) skips the LSP field
//! entirely — `submode.lsp_bits == 0` — and the body reports
//! `lsp_index == 0`. [`reconstruct_q10`] returns `None` for any
//! sub-mode with `lsp_bits == 0` so callers can distinguish
//! "no LSP transmitted, fall back to LSP state" from "LSPs decoded".
//!
//! ## What this module does NOT do
//!
//! * No LSP→LPC conversion. Manual §9.1 / §10.1 only states the
//!   LSPs are *"converted back to the LPC filter Â(z)"* without
//!   giving the conversion procedure. Recorded as a docs gap
//!   alongside the narrowband LSP→LPC gap.
//! * No sub-frame interpolation. §10 does not state a sub-frame LSP
//!   interpolation rule for the high band; the staged manual is
//!   silent on whether high-band LSPs participate in the same
//!   r200-style four-way linear interpolation as the narrowband
//!   LSPs (see Spec gaps noted).
//! * Encoder-side codebook search **lands in round r372**: the reverse
//!   direction (LSPs → 12-bit MSVQ index) is [`quantise_q10`], a
//!   sequential greedy two-stage residual search against the same staged
//!   codebooks, with [`pack_hb_lsp_index`] producing the on-wire field.
//!   It uses an unweighted squared-error metric (the perceptual weighting
//!   the reference encoder applies is a refinement, not staged).
//! * No high-band innovation codebook lookup. The high-band
//!   excitation VQ is staged at
//!   [`crate::codebooks::hb_innovation_8_128`] and
//!   [`crate::codebooks::hb_innovation_10_32`] but its per-mode
//!   shape mapping (and gain × VQ assembly) lands in a follow-up
//!   round.

use crate::codebooks::{
    hb_lsp_scale, hb_lsp_stage1, hb_lsp_stage2, NbLspScale, HB_LPC_ORDER, HB_LSP_STAGE_ENTRIES,
};
use crate::wideband::WidebandHighBandSubmode;

/// Number of bits per high-band LSP MSVQ stage (6-bit index =
/// [`HB_LSP_STAGE_ENTRIES`] = 64 entries).
pub const HB_LSP_STAGE_BITS: u32 = 6;

/// Total width of the packed high-band LSP MSVQ field
/// (stage1 6 bits + stage2 6 bits = 12 bits). Mirrors
/// [`WidebandHighBandSubmode::lsp_bits`] for modes 1..=4.
pub const HB_LSP_PACKED_BITS: u32 = 2 * HB_LSP_STAGE_BITS;

/// Maximum codebook index that fits in a 6-bit per-stage field.
pub const HB_LSP_INDEX_MASK: u16 = (1 << HB_LSP_STAGE_BITS) - 1;

/// Q-shift applied to per-stage contributions so they accumulate in
/// a single common fixed-point format (output unit = 1/1024 of the
/// LSP frequency unit; see module docs). Matches r194's narrowband
/// choice so both bands speak the same Q-format downstream.
pub const HB_LSP_OUTPUT_Q: u32 = 10;

/// Per-stage 6-bit indices for a single high-band LSP frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HbLspStages {
    /// 6-bit index into [`hb_lsp_stage1`] (level-1 MSVQ codebook,
    /// full 8-coefficient vector). Top 6 bits of the packed
    /// 12-bit `lsp_index` field.
    pub stage1: u8,
    /// 6-bit index into [`hb_lsp_stage2`] (level-2 residual MSVQ
    /// codebook). Bottom 6 bits of the packed 12-bit `lsp_index`
    /// field.
    pub stage2: u8,
}

impl HbLspStages {
    /// Split the packed 12-bit `lsp_index` parsed off the wire into
    /// per-stage 6-bit indices.
    ///
    /// Returns `None` for a sub-mode that skips the LSP field
    /// (mode 0 — `submode.lsp_bits == 0`).
    ///
    /// The 12-bit ordering convention is "top 6 bits = level-1
    /// codebook, low 6 bits = level-2 codebook", as already
    /// documented on [`crate::WidebandHighBandBody::lsp_index`].
    pub fn from_packed(lsp_index: u16, submode: &WidebandHighBandSubmode) -> Option<Self> {
        if submode.lsp_bits == 0 {
            return None;
        }
        let stage1 = ((lsp_index >> HB_LSP_STAGE_BITS) & HB_LSP_INDEX_MASK) as u8;
        let stage2 = (lsp_index & HB_LSP_INDEX_MASK) as u8;
        Some(Self { stage1, stage2 })
    }
}

/// Multiplier applied to a per-stage codebook entry so its
/// contribution lands in the common Q[`HB_LSP_OUTPUT_Q`] format.
///
/// `Div256 → ×4`, `Div512 → ×2`. The high-band MSVQ uses only
/// these two scales (no 1/1024 stage); the `_ => 1` arm exists
/// only to keep the conversion total over [`NbLspScale`].
const fn stage_shift_factor(scale: NbLspScale) -> i32 {
    match scale {
        NbLspScale::Div256 => 4,
        NbLspScale::Div512 => 2,
        NbLspScale::Div1024 => 1,
    }
}

/// Reconstruct the eight high-band LSP frequency coefficients in
/// the common Q[`HB_LSP_OUTPUT_Q`] fixed-point format from the
/// per-stage indices.
///
/// Output unit: `(1 / 2^HB_LSP_OUTPUT_Q) ≈ 9.77e-4` of an LSP
/// frequency. Sign is preserved — codebook entries are signed bytes.
///
/// Algorithm: stage 1 contributes its full eight-coefficient row
/// scaled by `1/256` (×4 in Q10); stage 2 adds its residual row
/// scaled by `1/512` (×2 in Q10).
///
/// `idx` is bounds-checked against the staged codebook entry count
/// ([`HB_LSP_STAGE_ENTRIES`] = 64 per stage); out-of-range indices
/// return `None`. Since the per-stage index field is 6 bits wide,
/// a value in `0..64` is always representable on the wire — `None`
/// here would only fire if the caller hand-constructed an
/// out-of-range [`HbLspStages`].
pub fn reconstruct_q10(stages: HbLspStages) -> Option<[i32; HB_LPC_ORDER]> {
    let s1 = stages.stage1 as usize;
    let s2 = stages.stage2 as usize;
    if s1 >= HB_LSP_STAGE_ENTRIES || s2 >= HB_LSP_STAGE_ENTRIES {
        return None;
    }

    let mut out = [0i32; HB_LPC_ORDER];

    // Stage 1: full eight-coefficient codebook, scale 1/256 → ×4.
    let stage1_row = hb_lsp_stage1()[s1];
    let s1_factor = stage_shift_factor(hb_lsp_scale(1).expect("hb stage 1 scale"));
    for (out_i, &v) in out.iter_mut().zip(stage1_row.iter()) {
        *out_i += i32::from(v) * s1_factor;
    }

    // Stage 2: residual codebook, scale 1/512 → ×2.
    let stage2_row = hb_lsp_stage2()[s2];
    let s2_factor = stage_shift_factor(hb_lsp_scale(2).expect("hb stage 2 scale"));
    for (out_i, &v) in out.iter_mut().zip(stage2_row.iter()) {
        *out_i += i32::from(v) * s2_factor;
    }

    Some(out)
}

/// Quantise an order-8 Q10 high-band LSP vector to the 2-stage MSVQ
/// indices — the **encode-direction inverse** of [`reconstruct_q10`].
///
/// The high-band LSP quantiser is a 2-stage MSVQ (manual §10.1 / the
/// staged `hb-lsp-cdbk-stage1/stage2`): stage 1 is a full 8-coefficient
/// VQ (scale 1/256 → ×4 in Q10), stage 2 a residual VQ (scale 1/512 →
/// ×2). The search is sequential greedy: pick the stage-1 row minimising
/// squared error to `target`, subtract its scaled contribution, then pick
/// the stage-2 row minimising the residual. Each stage uses the exact
/// Q10 scaling [`reconstruct_q10`] applies, so the chosen indices
/// reconstruct through the existing decoder path. The refinement is
/// monotone (stage 2 only reduces the residual it sees); the greedy
/// search is near-optimal, not guaranteed globally optimal.
pub fn quantise_q10(target: &[i32; HB_LPC_ORDER]) -> HbLspStages {
    let s1_factor = stage_shift_factor(hb_lsp_scale(1).expect("hb stage 1 scale"));
    let s2_factor = stage_shift_factor(hb_lsp_scale(2).expect("hb stage 2 scale"));

    // Stage 1: full-vector coarse VQ.
    let stage1 = search_hb_stage(target, hb_lsp_stage1(), s1_factor);
    let s1_row = hb_lsp_stage1()[stage1 as usize];
    let mut residual = *target;
    for (r, &v) in residual.iter_mut().zip(s1_row.iter()) {
        *r -= i32::from(v) * s1_factor;
    }

    // Stage 2: residual refinement VQ.
    let stage2 = search_hb_stage(&residual, hb_lsp_stage2(), s2_factor);

    HbLspStages { stage1, stage2 }
}

/// Find the codebook row whose scaled contribution best matches `target`
/// (least squared error) among an 8-coefficient high-band codebook.
fn search_hb_stage(target: &[i32; HB_LPC_ORDER], cb: &[[i16; HB_LPC_ORDER]], factor: i32) -> u8 {
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

/// Pack the per-stage 6-bit indices of [`HbLspStages`] into the on-wire
/// 12-bit `lsp_index` field (MSB-first: stage 1 high, stage 2 low),
/// matching the layout [`HbLspStages::from_packed`] parses.
pub fn pack_hb_lsp_index(stages: &HbLspStages) -> u16 {
    (u16::from(stages.stage1) << HB_LSP_STAGE_BITS) | u16::from(stages.stage2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wideband::WIDEBAND_HIGH_BAND_SUBMODES;

    #[test]
    fn from_packed_none_for_silence_mode_0() {
        let mode0 = WIDEBAND_HIGH_BAND_SUBMODES[0];
        assert_eq!(mode0.lsp_bits, 0);
        // Even with a non-zero lsp_index (which mode 0 would never
        // produce on a conforming bit-stream) the splitter returns
        // None for mode 0.
        assert!(HbLspStages::from_packed(0, &mode0).is_none());
        assert!(HbLspStages::from_packed(0x0FFF, &mode0).is_none());
    }

    #[test]
    fn from_packed_msb_first_for_documented_modes() {
        for sm in &WIDEBAND_HIGH_BAND_SUBMODES[1..] {
            assert_eq!(sm.lsp_bits, 12);
            // Pack [stage1=0b101010, stage2=0b010101] = 0b101010_010101
            let packed = 0b101010_010101u16;
            let s = HbLspStages::from_packed(packed, sm).unwrap();
            assert_eq!(s.stage1, 0b101010);
            assert_eq!(s.stage2, 0b010101);
        }
    }

    #[test]
    fn from_packed_index_mask_caps_at_six_bits() {
        let mode1 = WIDEBAND_HIGH_BAND_SUBMODES[1];
        // The 12-bit field can only hold values 0..4096; check that
        // extra high bits in the u16 are masked off.
        let s = HbLspStages::from_packed(0xFFFF, &mode1).unwrap();
        assert_eq!(s.stage1, 0x3F);
        assert_eq!(s.stage2, 0x3F);
    }

    #[test]
    fn from_packed_round_trip_over_full_index_range() {
        let mode1 = WIDEBAND_HIGH_BAND_SUBMODES[1];
        for s1 in 0u8..=63 {
            for s2 in 0u8..=63 {
                let packed = (u16::from(s1) << HB_LSP_STAGE_BITS) | u16::from(s2);
                let stages = HbLspStages::from_packed(packed, &mode1).unwrap();
                assert_eq!(stages.stage1, s1);
                assert_eq!(stages.stage2, s2);
            }
        }
    }

    #[test]
    fn reconstruct_returns_eight_coefficients() {
        let stages = HbLspStages {
            stage1: 0,
            stage2: 0,
        };
        let out = reconstruct_q10(stages).unwrap();
        assert_eq!(out.len(), HB_LPC_ORDER);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn reconstruct_zero_indices_yields_combined_row_0_q10() {
        // With stage1 = stage2 = 0 the output should equal
        //   stage1_row[i] * 4 + stage2_row[i] * 2
        // for every coefficient i.
        let row1 = hb_lsp_stage1()[0];
        let row2 = hb_lsp_stage2()[0];
        let out = reconstruct_q10(HbLspStages {
            stage1: 0,
            stage2: 0,
        })
        .unwrap();
        for i in 0..HB_LPC_ORDER {
            assert_eq!(
                out[i],
                i32::from(row1[i]) * 4 + i32::from(row2[i]) * 2,
                "coeff {} mismatch (stage1 row[0] + stage2 row[0])",
                i,
            );
        }
    }

    #[test]
    fn reconstruct_stage1_only_isolates_div256_scaling() {
        // Pick an index whose stage2 row is known to be all-zero —
        // we cannot assume that without reading the table. Instead,
        // exercise the *difference* between two reconstructions
        // sharing the same stage2 index to isolate stage1's
        // contribution.
        let a = reconstruct_q10(HbLspStages {
            stage1: 0,
            stage2: 7,
        })
        .unwrap();
        let b = reconstruct_q10(HbLspStages {
            stage1: 1,
            stage2: 7,
        })
        .unwrap();
        let stage1_row0 = hb_lsp_stage1()[0];
        let stage1_row1 = hb_lsp_stage1()[1];
        for i in 0..HB_LPC_ORDER {
            let want = (i32::from(stage1_row1[i]) - i32::from(stage1_row0[i])) * 4;
            assert_eq!(b[i] - a[i], want, "coeff {} stage1 delta mismatch", i);
        }
    }

    #[test]
    fn reconstruct_stage2_only_isolates_div512_scaling() {
        // Same idea, isolate stage2's contribution by holding
        // stage1 fixed and stepping stage2.
        let a = reconstruct_q10(HbLspStages {
            stage1: 11,
            stage2: 0,
        })
        .unwrap();
        let b = reconstruct_q10(HbLspStages {
            stage1: 11,
            stage2: 1,
        })
        .unwrap();
        let stage2_row0 = hb_lsp_stage2()[0];
        let stage2_row1 = hb_lsp_stage2()[1];
        for i in 0..HB_LPC_ORDER {
            let want = (i32::from(stage2_row1[i]) - i32::from(stage2_row0[i])) * 2;
            assert_eq!(b[i] - a[i], want, "coeff {} stage2 delta mismatch", i);
        }
    }

    #[test]
    fn reconstruct_full_index_range_never_panics() {
        // Every 12-bit lsp_index value must reconstruct without
        // panic. Exhaustive search: 4096 inputs, cheap.
        for s1 in 0u8..=63 {
            for s2 in 0u8..=63 {
                let out = reconstruct_q10(HbLspStages {
                    stage1: s1,
                    stage2: s2,
                });
                assert!(out.is_some(), "stage1={} stage2={} produced None", s1, s2);
                let v = out.unwrap();
                // Q10 sanity: |entry| ≤ 127, factor ≤ 4, two stages
                // → |out[i]| ≤ 127*4 + 127*2 = 762. Well below i32
                // range; assert defensively.
                for c in &v {
                    assert!(c.abs() <= 127 * 4 + 127 * 2, "coeff out of range");
                }
            }
        }
    }

    #[test]
    fn reconstruct_returns_none_for_oob_stage1_index() {
        let stages = HbLspStages {
            stage1: 64,
            stage2: 0,
        };
        assert!(reconstruct_q10(stages).is_none());
    }

    #[test]
    fn reconstruct_returns_none_for_oob_stage2_index() {
        let stages = HbLspStages {
            stage1: 0,
            stage2: 64,
        };
        assert!(reconstruct_q10(stages).is_none());
    }

    #[test]
    fn from_packed_then_reconstruct_matches_direct_path() {
        // End-to-end: split a synthetic packed index, then
        // reconstruct, and check against a direct construction.
        let mode2 = WIDEBAND_HIGH_BAND_SUBMODES[2];
        let packed = (17u16 << HB_LSP_STAGE_BITS) | 42u16;
        let stages = HbLspStages::from_packed(packed, &mode2).unwrap();
        assert_eq!(stages.stage1, 17);
        assert_eq!(stages.stage2, 42);
        let via_split = reconstruct_q10(stages).unwrap();
        let direct = reconstruct_q10(HbLspStages {
            stage1: 17,
            stage2: 42,
        })
        .unwrap();
        assert_eq!(via_split, direct);
    }

    #[test]
    fn output_q_format_constant_matches_narrowband() {
        // The two bands share the same downstream Q-format so a
        // future LSP→LPC stage can consume both with identical
        // arithmetic.
        assert_eq!(HB_LSP_OUTPUT_Q, crate::lsp::NB_LSP_OUTPUT_Q);
    }

    #[test]
    fn packed_field_width_matches_submode_lsp_bits() {
        // The HB_LSP_PACKED_BITS constant must equal the lsp_bits
        // field on every documented (mode 1..=4) high-band sub-mode.
        for sm in &WIDEBAND_HIGH_BAND_SUBMODES[1..] {
            assert_eq!(u32::from(sm.lsp_bits), HB_LSP_PACKED_BITS);
        }
    }

    // ---- High-band LSP quantiser (encode inverse) ----

    fn hb_sq_err(recon: &[i32; HB_LPC_ORDER], target: &[i32; HB_LPC_ORDER]) -> i64 {
        recon
            .iter()
            .zip(target.iter())
            .map(|(a, b)| {
                let d = i64::from(*a) - i64::from(*b);
                d * d
            })
            .sum()
    }

    #[test]
    fn quantise_zero_yields_valid_indices() {
        let s = quantise_q10(&[0i32; HB_LPC_ORDER]);
        assert!((s.stage1 as usize) < HB_LSP_STAGE_ENTRIES);
        assert!((s.stage2 as usize) < HB_LSP_STAGE_ENTRIES);
    }

    #[test]
    fn quantise_refinement_is_monotone() {
        // For every codebook reconstruction the full quantiser error is
        // no worse than the stage-1-only error.
        for &(i1, i2) in &[(0u8, 0u8), (1, 2), (30, 50), (63, 63)] {
            let target = reconstruct_q10(HbLspStages {
                stage1: i1,
                stage2: i2,
            })
            .unwrap();
            let q = quantise_q10(&target);
            let full = hb_sq_err(&reconstruct_q10(q).unwrap(), &target);

            let s1_factor = stage_shift_factor(hb_lsp_scale(1).unwrap());
            let s1_row = hb_lsp_stage1()[q.stage1 as usize];
            let mut s1_recon = [0i32; HB_LPC_ORDER];
            for (o, &v) in s1_recon.iter_mut().zip(s1_row.iter()) {
                *o = i32::from(v) * s1_factor;
            }
            let stage1_err = hb_sq_err(&s1_recon, &target);
            assert!(full <= stage1_err, "refinement worsened error");
        }
    }

    #[test]
    fn pack_then_from_packed_round_trips() {
        let stages = HbLspStages {
            stage1: 0b101010,
            stage2: 0b010101,
        };
        let packed = pack_hb_lsp_index(&stages);
        // Use a documented (mode 1) sub-mode so from_packed accepts it.
        let sm = WIDEBAND_HIGH_BAND_SUBMODES[1];
        let back = HbLspStages::from_packed(packed, &sm).unwrap();
        assert_eq!(back, stages);
    }

    #[test]
    fn quantise_then_pack_then_decode_round_trips() {
        let stages = HbLspStages {
            stage1: 12,
            stage2: 44,
        };
        let target = reconstruct_q10(stages).unwrap();
        let q = quantise_q10(&target);
        let packed = pack_hb_lsp_index(&q);
        let sm = WIDEBAND_HIGH_BAND_SUBMODES[1];
        let decoded = HbLspStages::from_packed(packed, &sm).unwrap();
        assert_eq!(
            reconstruct_q10(decoded).unwrap(),
            reconstruct_q10(q).unwrap()
        );
    }

    #[test]
    fn exact_codebook_stage1_target_quantises_to_zero_error() {
        // A target that is exactly a stage-1 row (stage2 = 0 contributes a
        // non-zero row, so use a target built from stage1 alone is not
        // directly possible; instead confirm stage-1 search finds the
        // generating row when stage 2 is the zero-row's complement). Simpler
        // invariant: the chosen stage-1 row's error is the minimum over all
        // rows for a stage-1-only target.
        let s1_factor = stage_shift_factor(hb_lsp_scale(1).unwrap());
        let row = hb_lsp_stage1()[20];
        let mut target = [0i32; HB_LPC_ORDER];
        for (t, &v) in target.iter_mut().zip(row.iter()) {
            *t = i32::from(v) * s1_factor;
        }
        let picked = search_hb_stage(&target, hb_lsp_stage1(), s1_factor);
        // The exact row must achieve zero error, so the search picks a
        // zero-error row (index 20 unless a duplicate row exists).
        let picked_row = hb_lsp_stage1()[picked as usize];
        let mut err = 0i64;
        for (i, &v) in picked_row.iter().enumerate() {
            let d = i64::from(target[i]) - i64::from(i32::from(v) * s1_factor);
            err += d * d;
        }
        assert_eq!(err, 0, "stage-1 search must find an exact-match row");
    }
}
