//! Narrowband **fixed-codebook gain index composition** primitive (r261
//! scope).
//!
//! Surfaces the structural composition documented in Speex Codec Manual
//! §9.2 / CELP companion §2.3 — the fixed-codebook gain that scales the
//! r220 innovation sub-vector `c[n]` before the §8.4 / §2.3 final
//! excitation sum `e[n] = p[n] + c[n]`:
//!
//! > "Fixed-codebook gain = global frame gain `g_frame` × sub-frame gain
//! > correction `g_subf`; the correction uses 0–3 bits per sub-frame
//! > and is encoded *before* the codebook search (open loop),
//! > eliminating inter-frame dependency for packet-loss robustness."
//! > — CELP companion §2.3
//!
//! ## Scope
//!
//! Two raw codebook indices are already surfaced by the round-3 frame
//! body parser:
//!
//! * `ol_exc_gain_index` — the 5-bit (or 0-bit for silence) frame-level
//!   open-loop excitation gain field (Table 9.1, row "OL Exc gain").
//!   Stored as `u8` on [`crate::NarrowbandFrameBody::ol_exc_gain_index`].
//! * `innovation_gain_index` — the 0 / 1 / 3-bit per-sub-frame
//!   innovation-gain correction (Table 9.1, row "Innovation gain").
//!   Stored as `u8` on
//!   [`crate::NarrowbandSubFrameIndices::innovation_gain_index`].
//!
//! This module composes the two raw indices into a single typed
//! [`FixedCodebookGainIndices`] per sub-frame, reflecting the §9.2 /
//! §2.3 product structure (one frame-level factor × four per-sub-frame
//! correction factors). The typed surface lets the downstream consumer
//! reason about the composition without re-deriving the index plumbing
//! at every callsite.
//!
//! ## Why an index-only typed primitive
//!
//! Both factors are pure raw codebook indices off the bit-stream; the
//! actual gain magnitudes depend on the open-loop scalar quantiser that
//! CELP companion §9 records as *"computed, not a lookup array"* (see
//! the "Items NOT extracted" subsection — there is no static gain table
//! to consult). The Q-format and the precise dequantisation curve for
//! both fields are therefore a documented gap pending a docs round
//! staging the quantiser specification.
//!
//! Surfacing the composition at the **index** layer mirrors the
//! r234 / r241 / r244 design pattern: each step lands an algebra-only
//! primitive that the eventual Q-format pin commutes through with a
//! single arithmetic scaling step. The downstream `g_frame × g_subf`
//! product can later land as a separate gain-magnitude reconstruction
//! module once the docs gap closes, without re-deriving the index pair
//! plumbing.
//!
//! ## Frame-level field budget (5 bits or absent)
//!
//! Per Table 9.1 the `OL Exc gain` row width is one of:
//!
//! * `0` for mode 0 (silence): no field, the gain is implicitly zero;
//! * `5` for every other documented narrowband mode (1, 2, 3, 4, 5, 6,
//!   7, 8): a 5-bit field carrying the index in `0..=31`.
//!
//! The two values appear directly in the
//! [`crate::NarrowbandSubmode::ol_exc_gain_bits`] field.
//!
//! ## Sub-frame field budget (0 / 1 / 3 bits)
//!
//! Per Table 9.1 the per-sub-frame `Innovation gain` correction width is
//! one of:
//!
//! * `0` — modes 0, 2, 8 (no correction transmitted, implicit unity);
//! * `1` — modes 1, 3, 4 (single-bit correction);
//! * `3` — modes 5, 6, 7 (three-bit correction, 8 entries).
//!
//! The value is in the
//! [`crate::NarrowbandSubmode::innovation_gain_bits`] field. The
//! companion §2.3 phrase "0–3 bits per sub-frame" matches this set
//! exactly (no documented mode uses 2 bits).
//!
//! ## Silence convention
//!
//! For mode 0 (silence), both budgets are `0`. The composed
//! [`FixedCodebookGainIndices`] surfaces the absence via the
//! [`FixedCodebookGainIndices::is_absent`] predicate, which lets the
//! caller short-circuit the downstream `g_frame × g_subf` magnitude
//! reconstruction without an out-of-band branch.
//!
//! For non-silence modes the frame-level factor is always present
//! (5-bit width); only the sub-frame correction may be absent
//! (0-bit budget). The typed [`SubFrameInnovationGainCorrection::Absent`]
//! variant distinguishes "no correction transmitted" from "correction
//! transmitted with value 0" — the former preserves the frame-level
//! gain unchanged, the latter selects the row-0 correction entry.
//!
//! ## What this module DOES
//!
//! * [`FrameInnovationGainIndex`] — typed wrapper over the 5-bit
//!   frame-level OL excitation-gain index (or the silence variant).
//! * [`SubFrameInnovationGainCorrection`] — typed wrapper over the
//!   0 / 1 / 3-bit per-sub-frame correction (with the `Absent` variant
//!   for 0-bit budgets).
//! * [`FixedCodebookGainIndices`] — typed pair composing the
//!   frame-level factor with one sub-frame correction.
//! * [`fixed_codebook_gain_indices`] — sub-frame batch helper returning
//!   `[FixedCodebookGainIndices; 4]` over an entire frame's
//!   [`crate::NarrowbandFrameBody`].
//! * [`crate::NarrowbandFrameBody::fixed_codebook_gain_indices`] —
//!   convenience method composing the typed pair off the existing
//!   per-sub-frame indices.
//!
//! ## What this module DOES NOT do
//!
//! * No gain magnitude reconstruction. The scalar quantiser that maps
//!   the frame-level 5-bit index + sub-frame 0/1/3-bit correction into
//!   a numeric `g_frame × g_subf` product is the documented "computed,
//!   not a lookup array" gap from CELP companion §9; until a docs round
//!   stages the quantiser, the magnitude reconstruction module cannot
//!   land. This primitive surfaces only the raw index pair.
//! * No Q-format pin. The downstream magnitude reconstruction step
//!   will pick the Q-format once the quantiser is documented; this
//!   index-only primitive is Q-format-agnostic.
//! * No fixed-codebook scaling. Multiplying `c[n]` by the reconstructed
//!   gain is a downstream layer that consumes both this primitive and
//!   the eventual quantiser; this module surfaces only the indices.
//! * No high-band path. Per Table 10.1 the wideband high band has its
//!   own per-sub-frame `Excitation gain` field (5/4 bits) without a
//!   frame-level factor — the gain composition for the high band is
//!   structurally simpler (gain × c[n]) and lives separately.
//! * No encoder-side index selection. The reverse direction
//!   (gain magnitude → quantised indices) is the encoder's responsibility
//!   and requires the same documented quantiser the decoder magnitude
//!   step is blocked on.

use crate::narrowband_body::NarrowbandFrameBody;
use crate::submode::{NarrowbandSubmode, SUBFRAMES_PER_FRAME};
use core::fmt;

/// Width in bits of the frame-level open-loop excitation-gain field
/// when transmitted (Table 9.1 "OL Exc gain" row, modes 1..=8). Mode 0
/// (silence) carries no field.
pub const FRAME_OL_EXC_GAIN_BITS: u8 = 5;

/// Number of distinct entries the 5-bit frame-level OL excitation gain
/// field can take when transmitted.
pub const FRAME_OL_EXC_GAIN_ENTRIES: u32 = 1u32 << FRAME_OL_EXC_GAIN_BITS;

/// Typed wrapper over the frame-level open-loop excitation-gain index.
///
/// Distinguishes the silence case (no field transmitted) from the
/// 5-bit-field case (one of `0..=31`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameInnovationGainIndex {
    /// Mode 0 (silence): no field transmitted, frame-level gain is
    /// implicitly zero. The downstream magnitude reconstruction can
    /// short-circuit the per-sub-frame `g_frame × g_subf` product to
    /// zero without consulting the correction field.
    Silence,
    /// Mode 1..=8: a 5-bit index in `0..=31` resolved against the
    /// (gap-blocked) scalar quantiser.
    Indexed(u8),
}

impl FrameInnovationGainIndex {
    /// Resolve the frame-level index from a parsed
    /// [`NarrowbandFrameBody`] against the active sub-mode's
    /// [`NarrowbandSubmode::ol_exc_gain_bits`] budget.
    ///
    /// * 0-bit budget → [`Self::Silence`] (the parser stores `0` in the
    ///   absent field; this constructor never reads it).
    /// * 5-bit budget → [`Self::Indexed`] carrying the raw index.
    ///
    /// Returns `None` if the sub-mode's budget is neither 0 nor
    /// [`FRAME_OL_EXC_GAIN_BITS`] (`= 5`) — only possible with a
    /// hand-built non-conforming sub-mode literal.
    pub fn resolve(body: &NarrowbandFrameBody, submode: &NarrowbandSubmode) -> Option<Self> {
        match submode.ol_exc_gain_bits {
            0 => Some(Self::Silence),
            FRAME_OL_EXC_GAIN_BITS => Some(Self::Indexed(body.ol_exc_gain_index)),
            _ => None,
        }
    }

    /// `true` if the frame-level field was not transmitted.
    pub fn is_silence(&self) -> bool {
        matches!(self, Self::Silence)
    }

    /// Returns the raw 5-bit index when the field is present, else
    /// `None`. Useful for callers that want to walk the index value
    /// directly without pattern-matching.
    pub fn raw_index(&self) -> Option<u8> {
        match self {
            Self::Silence => None,
            Self::Indexed(i) => Some(*i),
        }
    }
}

impl fmt::Display for FrameInnovationGainIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Silence => f.write_str("silence (no OL Exc gain field)"),
            Self::Indexed(i) => write!(f, "OL Exc gain index {}", i),
        }
    }
}

/// Typed wrapper over the per-sub-frame innovation-gain correction
/// index, distinguishing the absent case (0-bit budget) from the
/// 1-bit and 3-bit cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubFrameInnovationGainCorrection {
    /// 0-bit budget — no correction transmitted (modes 0, 2, 8). The
    /// downstream consumer applies the frame-level gain unchanged.
    Absent,
    /// 1-bit correction (modes 1, 3, 4). Index in `0..=1`.
    OneBit(u8),
    /// 3-bit correction (modes 5, 6, 7). Index in `0..=7`.
    ThreeBit(u8),
}

impl SubFrameInnovationGainCorrection {
    /// Resolve the per-sub-frame correction from a raw
    /// `innovation_gain_index` field against the active sub-mode's
    /// [`NarrowbandSubmode::innovation_gain_bits`] budget.
    ///
    /// * 0-bit budget → [`Self::Absent`] (parser-stored `0` is ignored).
    /// * 1-bit budget → [`Self::OneBit`] carrying the index.
    /// * 3-bit budget → [`Self::ThreeBit`] carrying the index.
    ///
    /// Returns `None` if the sub-mode's budget is none of {0, 1, 3} —
    /// only possible with a hand-built non-conforming sub-mode literal.
    pub fn resolve(raw_index: u8, submode: &NarrowbandSubmode) -> Option<Self> {
        match submode.innovation_gain_bits {
            0 => Some(Self::Absent),
            1 => Some(Self::OneBit(raw_index)),
            3 => Some(Self::ThreeBit(raw_index)),
            _ => None,
        }
    }

    /// `true` if no correction field is transmitted.
    pub fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }

    /// Bit budget (0, 1, or 3) of the surface form.
    pub fn bit_budget(&self) -> u8 {
        match self {
            Self::Absent => 0,
            Self::OneBit(_) => 1,
            Self::ThreeBit(_) => 3,
        }
    }

    /// Raw index when the correction is present, else `None`. The index
    /// is in `0..=1` for [`Self::OneBit`] and `0..=7` for
    /// [`Self::ThreeBit`].
    pub fn raw_index(&self) -> Option<u8> {
        match self {
            Self::Absent => None,
            Self::OneBit(i) | Self::ThreeBit(i) => Some(*i),
        }
    }
}

impl fmt::Display for SubFrameInnovationGainCorrection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absent => f.write_str("absent (no Innovation gain field)"),
            Self::OneBit(i) => write!(f, "1-bit Innovation gain index {}", i),
            Self::ThreeBit(i) => write!(f, "3-bit Innovation gain index {}", i),
        }
    }
}

/// Typed pair composing the frame-level OL excitation-gain factor with
/// one sub-frame's innovation-gain correction factor.
///
/// Reflects the §9.2 / CELP companion §2.3 product structure
/// `fixed-codebook gain = g_frame × g_subf` at the index layer (one
/// raw bit-stream index per factor). The numeric reconstruction is
/// gap-blocked behind the §9 "computed, not a lookup array" quantiser
/// note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedCodebookGainIndices {
    /// Frame-level factor.
    pub frame: FrameInnovationGainIndex,
    /// Per-sub-frame correction factor.
    pub subframe: SubFrameInnovationGainCorrection,
}

impl FixedCodebookGainIndices {
    /// Compose the typed pair from a parsed frame body, a sub-mode, and
    /// a sub-frame slot in `0..4`.
    ///
    /// Returns `None` if either the frame-level field budget is not
    /// `{0, 5}` or the sub-frame field budget is not `{0, 1, 3}` —
    /// only possible with a hand-built non-conforming sub-mode.
    /// Returns `None` if `subframe_index >= 4`.
    pub fn from_body(
        body: &NarrowbandFrameBody,
        submode: &NarrowbandSubmode,
        subframe_index: usize,
    ) -> Option<Self> {
        if subframe_index >= SUBFRAMES_PER_FRAME {
            return None;
        }
        let frame = FrameInnovationGainIndex::resolve(body, submode)?;
        let subframe = SubFrameInnovationGainCorrection::resolve(
            body.subframes[subframe_index].innovation_gain_index,
            submode,
        )?;
        Some(Self { frame, subframe })
    }

    /// `true` iff the frame-level factor is the silence variant.
    ///
    /// When silent, the downstream `g_frame × g_subf` product collapses
    /// to zero (silence-mode 0 contribution), so the consumer can
    /// short-circuit the multiplication.
    pub fn is_absent(&self) -> bool {
        self.frame.is_silence()
    }

    /// Total bit footprint of the index pair on the wire
    /// (5 + 0 / 1 / 3 or 0 + 0 for silence). Pins the spec's per-mode
    /// gain budget at the type layer.
    pub fn wire_bit_budget(&self) -> u8 {
        let frame_bits = match self.frame {
            FrameInnovationGainIndex::Silence => 0,
            FrameInnovationGainIndex::Indexed(_) => FRAME_OL_EXC_GAIN_BITS,
        };
        frame_bits + self.subframe.bit_budget()
    }
}

impl fmt::Display for FixedCodebookGainIndices {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} × {}", self.frame, self.subframe)
    }
}

/// Compose the four per-sub-frame [`FixedCodebookGainIndices`] for one
/// parsed frame. Returns `None` if any of the index resolutions fails
/// (only possible with a hand-built non-conforming sub-mode).
pub fn fixed_codebook_gain_indices(
    body: &NarrowbandFrameBody,
    submode: &NarrowbandSubmode,
) -> Option<[FixedCodebookGainIndices; SUBFRAMES_PER_FRAME]> {
    let mut out = [FixedCodebookGainIndices {
        frame: FrameInnovationGainIndex::Silence,
        subframe: SubFrameInnovationGainCorrection::Absent,
    }; SUBFRAMES_PER_FRAME];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = FixedCodebookGainIndices::from_body(body, submode, i)?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrowband_body::NarrowbandSubFrameIndices;
    use crate::submode::NARROWBAND_SUBMODES;

    fn mk_body(ol_exc: u8, sub_gains: [u8; SUBFRAMES_PER_FRAME]) -> NarrowbandFrameBody {
        let mut subframes = [NarrowbandSubFrameIndices::default(); SUBFRAMES_PER_FRAME];
        for (i, sf) in subframes.iter_mut().enumerate() {
            sf.innovation_gain_index = sub_gains[i];
        }
        NarrowbandFrameBody {
            lsp_index: 0,
            ol_pitch_index: 0,
            ol_pitch_gain_index: 0,
            ol_exc_gain_index: ol_exc,
            subframes,
        }
    }

    /// Mode 0 silence — frame-level field absent, every sub-frame
    /// correction absent, composition flags `is_absent()`.
    #[test]
    fn mode0_silence_is_absent_everywhere() {
        let submode = NARROWBAND_SUBMODES[0];
        // Silence sub-mode budgets: 0 + 0.
        assert_eq!(submode.ol_exc_gain_bits, 0);
        assert_eq!(submode.innovation_gain_bits, 0);
        let body = mk_body(0, [0, 0, 0, 0]);
        let composed = fixed_codebook_gain_indices(&body, &submode).expect("budgets in spec");
        for (i, slot) in composed.iter().enumerate() {
            assert!(slot.is_absent(), "sf {i} should flag absent");
            assert_eq!(slot.frame, FrameInnovationGainIndex::Silence);
            assert_eq!(slot.subframe, SubFrameInnovationGainCorrection::Absent);
            assert_eq!(slot.wire_bit_budget(), 0);
        }
    }

    /// Mode 5 — 30-bit LSP, 5-bit OL Exc gain, 3-bit innovation gain per
    /// Table 9.1. Compose a non-zero index pattern and pin the surface.
    #[test]
    fn mode5_three_bit_correction_surface() {
        let submode = NARROWBAND_SUBMODES[5];
        assert_eq!(submode.ol_exc_gain_bits, 5);
        assert_eq!(submode.innovation_gain_bits, 3);
        let body = mk_body(0b1_0110, [0, 3, 5, 7]);
        let composed = fixed_codebook_gain_indices(&body, &submode).expect("budgets in spec");
        for (i, slot) in composed.iter().enumerate() {
            assert!(!slot.is_absent());
            assert_eq!(slot.frame, FrameInnovationGainIndex::Indexed(0b1_0110));
            match slot.subframe {
                SubFrameInnovationGainCorrection::ThreeBit(idx) => {
                    assert_eq!(idx, [0u8, 3, 5, 7][i], "sf {i}");
                }
                other => panic!("expected ThreeBit, got {:?}", other),
            }
            assert_eq!(slot.wire_bit_budget(), 5 + 3);
        }
    }

    /// Mode 1 — 1-bit innovation gain per Table 9.1.
    #[test]
    fn mode1_one_bit_correction_surface() {
        let submode = NARROWBAND_SUBMODES[1];
        assert_eq!(submode.ol_exc_gain_bits, 5);
        assert_eq!(submode.innovation_gain_bits, 1);
        let body = mk_body(31, [0, 1, 1, 0]);
        let composed = fixed_codebook_gain_indices(&body, &submode).expect("budgets in spec");
        for (i, slot) in composed.iter().enumerate() {
            assert_eq!(slot.frame, FrameInnovationGainIndex::Indexed(31));
            match slot.subframe {
                SubFrameInnovationGainCorrection::OneBit(idx) => {
                    assert_eq!(idx, [0u8, 1, 1, 0][i], "sf {i}");
                }
                other => panic!("expected OneBit, got {:?}", other),
            }
            assert_eq!(slot.wire_bit_budget(), 5 + 1);
        }
    }

    /// Mode 2 — 5-bit OL Exc gain, 0-bit innovation gain per Table 9.1
    /// (the correction is absent even though the frame factor is
    /// present).
    #[test]
    fn mode2_frame_factor_present_correction_absent() {
        let submode = NARROWBAND_SUBMODES[2];
        assert_eq!(submode.ol_exc_gain_bits, 5);
        assert_eq!(submode.innovation_gain_bits, 0);
        let body = mk_body(17, [0, 0, 0, 0]);
        let composed = fixed_codebook_gain_indices(&body, &submode).expect("budgets in spec");
        for slot in &composed {
            assert!(!slot.is_absent(), "frame-factor present so not absent");
            assert_eq!(slot.frame, FrameInnovationGainIndex::Indexed(17));
            assert_eq!(slot.subframe, SubFrameInnovationGainCorrection::Absent);
            assert_eq!(slot.wire_bit_budget(), 5);
            assert!(slot.subframe.is_absent());
            assert_eq!(slot.subframe.bit_budget(), 0);
            assert_eq!(slot.subframe.raw_index(), None);
        }
    }

    /// Mode 8 — special 3.95 kbps mode: 5-bit OL Exc gain, 0-bit
    /// innovation gain (same correction-absent pattern as mode 2).
    #[test]
    fn mode8_special_low_bitrate_pattern() {
        let submode = NARROWBAND_SUBMODES[8];
        assert_eq!(submode.ol_exc_gain_bits, 5);
        assert_eq!(submode.innovation_gain_bits, 0);
        let body = mk_body(7, [0, 0, 0, 0]);
        let composed = fixed_codebook_gain_indices(&body, &submode).expect("budgets in spec");
        for slot in &composed {
            assert_eq!(slot.frame, FrameInnovationGainIndex::Indexed(7));
            assert_eq!(slot.subframe, SubFrameInnovationGainCorrection::Absent);
        }
    }

    /// Walk every documented narrowband mode (0..=8) and check the
    /// per-mode bit-budget surface matches Table 9.1.
    #[test]
    fn every_documented_mode_has_in_spec_budgets() {
        // Table 9.1 expected: (ol_exc_bits, innovation_gain_bits) per
        // narrowband mode id 0..=8.
        let expected: [(u8, u8); 9] = [
            (0, 0), // mode 0
            (5, 1), // mode 1
            (5, 0), // mode 2
            (5, 1), // mode 3
            (5, 1), // mode 4
            (5, 3), // mode 5
            (5, 3), // mode 6
            (5, 3), // mode 7
            (5, 0), // mode 8
        ];
        for (id, submode) in NARROWBAND_SUBMODES.iter().enumerate() {
            assert_eq!(
                (submode.ol_exc_gain_bits, submode.innovation_gain_bits),
                expected[id],
                "mode {id}"
            );
            // Pick non-zero indices within each budget; the resolver
            // accepts any value the parser would have stored.
            let ol_index = if expected[id].0 == 0 { 0 } else { 13 };
            let gain_index = match expected[id].1 {
                0 => 0,
                1 => 1,
                3 => 5,
                _ => unreachable!(),
            };
            let body = mk_body(ol_index, [gain_index; SUBFRAMES_PER_FRAME]);
            let composed =
                fixed_codebook_gain_indices(&body, submode).expect("documented mode in spec");
            // Frame-level surface variant.
            match composed[0].frame {
                FrameInnovationGainIndex::Silence => assert_eq!(expected[id].0, 0),
                FrameInnovationGainIndex::Indexed(i) => {
                    assert_eq!(expected[id].0, 5);
                    assert_eq!(i, ol_index);
                }
            }
            // Sub-frame surface variant.
            match composed[0].subframe {
                SubFrameInnovationGainCorrection::Absent => assert_eq!(expected[id].1, 0),
                SubFrameInnovationGainCorrection::OneBit(i) => {
                    assert_eq!(expected[id].1, 1);
                    assert_eq!(i, gain_index);
                }
                SubFrameInnovationGainCorrection::ThreeBit(i) => {
                    assert_eq!(expected[id].1, 3);
                    assert_eq!(i, gain_index);
                }
            }
        }
    }

    /// `from_body` returns `None` for an out-of-range sub-frame slot.
    #[test]
    fn out_of_range_subframe_slot_returns_none() {
        let submode = NARROWBAND_SUBMODES[5];
        let body = mk_body(0, [0; 4]);
        assert!(FixedCodebookGainIndices::from_body(&body, &submode, 4).is_none());
        assert!(FixedCodebookGainIndices::from_body(&body, &submode, 100).is_none());
    }

    /// Hand-built non-conforming sub-mode (frame budget = 2 bits) is
    /// rejected by the resolver (`None`).
    #[test]
    fn non_conforming_frame_budget_rejected() {
        let mut submode = NARROWBAND_SUBMODES[5];
        submode.ol_exc_gain_bits = 2; // never appears in Table 9.1
        let body = mk_body(0, [0; 4]);
        assert!(FrameInnovationGainIndex::resolve(&body, &submode).is_none());
    }

    /// Hand-built non-conforming sub-mode (correction budget = 2 bits)
    /// is rejected.
    #[test]
    fn non_conforming_subframe_budget_rejected() {
        let mut submode = NARROWBAND_SUBMODES[5];
        submode.innovation_gain_bits = 2; // never appears in Table 9.1
        assert!(SubFrameInnovationGainCorrection::resolve(0, &submode).is_none());
    }

    /// Wire-budget total matches the sum of frame and sub-frame parts.
    #[test]
    fn wire_bit_budget_decomposes_into_factors() {
        let cases: [(
            u8,
            u8,
            FrameInnovationGainIndex,
            SubFrameInnovationGainCorrection,
        ); 4] = [
            (
                0,
                0,
                FrameInnovationGainIndex::Silence,
                SubFrameInnovationGainCorrection::Absent,
            ),
            (
                5,
                0,
                FrameInnovationGainIndex::Indexed(0),
                SubFrameInnovationGainCorrection::Absent,
            ),
            (
                5,
                1,
                FrameInnovationGainIndex::Indexed(0),
                SubFrameInnovationGainCorrection::OneBit(0),
            ),
            (
                5,
                3,
                FrameInnovationGainIndex::Indexed(0),
                SubFrameInnovationGainCorrection::ThreeBit(0),
            ),
        ];
        for (frame_bits, sub_bits, frame, subframe) in cases.iter().copied() {
            let pair = FixedCodebookGainIndices { frame, subframe };
            assert_eq!(pair.wire_bit_budget(), frame_bits + sub_bits);
        }
    }

    /// Display impls render the expected wire-shape phrases.
    #[test]
    fn display_strings_pin_human_readable_form() {
        assert_eq!(
            format!("{}", FrameInnovationGainIndex::Silence),
            "silence (no OL Exc gain field)"
        );
        assert_eq!(
            format!("{}", FrameInnovationGainIndex::Indexed(7)),
            "OL Exc gain index 7"
        );
        assert_eq!(
            format!("{}", SubFrameInnovationGainCorrection::Absent),
            "absent (no Innovation gain field)"
        );
        assert_eq!(
            format!("{}", SubFrameInnovationGainCorrection::OneBit(1)),
            "1-bit Innovation gain index 1"
        );
        assert_eq!(
            format!("{}", SubFrameInnovationGainCorrection::ThreeBit(5)),
            "3-bit Innovation gain index 5"
        );
        let pair = FixedCodebookGainIndices {
            frame: FrameInnovationGainIndex::Indexed(17),
            subframe: SubFrameInnovationGainCorrection::ThreeBit(4),
        };
        assert_eq!(
            format!("{}", pair),
            "OL Exc gain index 17 × 3-bit Innovation gain index 4"
        );
    }

    /// Independent index helpers (`raw_index`) match the embedded
    /// indices.
    #[test]
    fn raw_index_helpers_match_embedded_values() {
        assert_eq!(FrameInnovationGainIndex::Silence.raw_index(), None);
        assert_eq!(FrameInnovationGainIndex::Indexed(11).raw_index(), Some(11));
        assert_eq!(SubFrameInnovationGainCorrection::Absent.raw_index(), None);
        assert_eq!(
            SubFrameInnovationGainCorrection::OneBit(1).raw_index(),
            Some(1)
        );
        assert_eq!(
            SubFrameInnovationGainCorrection::ThreeBit(6).raw_index(),
            Some(6)
        );
    }

    /// Cross-check: 5-bit field index covers `0..=31` (no overflow at
    /// the boundary).
    #[test]
    fn frame_index_covers_full_5_bit_range() {
        assert_eq!(FRAME_OL_EXC_GAIN_BITS, 5);
        assert_eq!(FRAME_OL_EXC_GAIN_ENTRIES, 32);
        let submode = NARROWBAND_SUBMODES[5];
        for idx in 0..32u8 {
            let body = mk_body(idx, [0; 4]);
            let resolved = FrameInnovationGainIndex::resolve(&body, &submode).unwrap();
            assert_eq!(resolved, FrameInnovationGainIndex::Indexed(idx));
            assert_eq!(resolved.raw_index(), Some(idx));
            assert!(!resolved.is_silence());
        }
    }

    /// Cross-check: composed `fixed_codebook_gain_indices` over a
    /// mode-5 body matches per-position `from_body` calls.
    #[test]
    fn batch_matches_per_subframe_helper() {
        let submode = NARROWBAND_SUBMODES[5];
        let body = mk_body(9, [1, 2, 3, 7]);
        let batch = fixed_codebook_gain_indices(&body, &submode).unwrap();
        for (i, slot) in batch.iter().enumerate() {
            let single = FixedCodebookGainIndices::from_body(&body, &submode, i).unwrap();
            assert_eq!(single, *slot, "sf {i}");
        }
    }

    /// `is_absent()` mirrors the frame-level silence variant exactly
    /// (correction-absent alone does NOT make the pair "absent" for
    /// the composition's purposes — mode 2 has a frame-level factor).
    #[test]
    fn is_absent_tracks_frame_factor_only() {
        // Frame absent, correction absent (mode 0) → absent.
        let absent = FixedCodebookGainIndices {
            frame: FrameInnovationGainIndex::Silence,
            subframe: SubFrameInnovationGainCorrection::Absent,
        };
        assert!(absent.is_absent());
        // Frame present, correction absent (mode 2) → NOT absent.
        let frame_only = FixedCodebookGainIndices {
            frame: FrameInnovationGainIndex::Indexed(0),
            subframe: SubFrameInnovationGainCorrection::Absent,
        };
        assert!(!frame_only.is_absent());
        // Frame present, correction present → NOT absent.
        let both = FixedCodebookGainIndices {
            frame: FrameInnovationGainIndex::Indexed(0),
            subframe: SubFrameInnovationGainCorrection::OneBit(0),
        };
        assert!(!both.is_absent());
    }
}
