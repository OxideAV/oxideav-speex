//! Wideband high-band **fixed-codebook gain index** primitive (r269
//! scope) — the high-band counterpart of the r261 narrowband
//! [`crate::fixed_codebook_gain`] composition.
//!
//! ## Spec basis
//!
//! Per Speex Codec Manual §10.4 / Table 10.1 ("Bit allocation for
//! high-band in wideband mode"; mirrored in CELP companion §5.1), the
//! high band carries exactly **one** gain field per sub-frame — the
//! `Excitation gain` row — with per-mode widths:
//!
//! | high-band mode | 0 | 1 | 2 | 3 | 4 |
//! |---|--:|--:|--:|--:|--:|
//! | Excitation gain bits / sub-frame | 0 | 5 | 4 | 4 | 4 |
//!
//! Unlike the narrowband Table 9.1 layout there is **no frame-level
//! `OL Exc gain` factor and no per-sub-frame correction split**: the
//! §9.2 / companion §2.3 product structure
//! `fixed-codebook gain = g_frame × g_subf` does not appear in
//! Table 10.1's rows, so the high-band composition reduces to a single
//! per-sub-frame index. Since Manual §10.2 states there is *no pitch
//! prediction in the high band*, this gain is the only scaling factor
//! the high-band excitation `c[n]` (decoded by
//! [`crate::hb_innovation::decode_hb_subframe`] per §10.3) receives.
//!
//! ## Why an index-only typed primitive
//!
//! The raw index is already surfaced by the round-r160 high-band body
//! bit-reader as
//! [`crate::HighBandSubFrameIndices::excitation_gain_index`]. The
//! numeric gain magnitude is **gap-blocked**: the staged
//! `docs/audio/speex/tables/README.md` "Not extracted" subsection
//! records that the fixed-codebook gain quantisation is *computed*
//! (an open-loop scalar quantiser), not a static lookup array — there
//! is no staged gain table to consult and the quantiser curve itself
//! is not in the staged manual / companion. This module therefore
//! surfaces only the typed index algebra, matching the
//! r234 / r241 / r244 / r261 Q-format-agnostic design pattern: the
//! eventual magnitude pin commutes through with a single scaling step
//! once a docs round stages the quantiser specification.
//!
//! ## What this module DOES
//!
//! * [`HbExcitationGainIndex`] — typed wrapper over the per-sub-frame
//!   `Excitation gain` field, distinguishing the absent case (mode 0,
//!   0-bit budget) from the 5-bit (mode 1) and 4-bit (modes 2..=4)
//!   surface forms.
//! * [`hb_excitation_gain_indices`] — sub-frame batch helper returning
//!   `[HbExcitationGainIndex; 4]` over an entire high-band frame's
//!   [`crate::WidebandHighBandBody`].
//! * [`crate::WidebandHighBandBody::hb_excitation_gain_indices`] —
//!   convenience method wiring the resolution off the existing parsed
//!   body, mirroring r261's
//!   [`crate::NarrowbandFrameBody::fixed_codebook_gain_indices`].
//!
//! ## What this module DOES NOT do
//!
//! * No gain magnitude reconstruction (the documented "computed, not a
//!   lookup array" quantiser gap above).
//! * No Q-format pin — the index layer is Q-format-agnostic.
//! * No excitation scaling: multiplying the §10.3 high-band `c[n]` by
//!   the reconstructed gain is the downstream layer that consumes both
//!   this primitive and the eventual quantiser.
//! * No narrowband path — the narrowband `g_frame × g_subf` pair lives
//!   in [`crate::fixed_codebook_gain`] (r261).
//! * No reserved-high-rate coverage: modes 5..=7 have no Table 10.1
//!   column in the staged manual (the existing
//!   [`crate::WidebandSubmode::ReservedHighRate`] docs gap), so no
//!   gain budget can be resolved for them here either.

use crate::wideband::{WidebandHighBandBody, WidebandHighBandSubmode};
use core::fmt;

/// Number of high-band sub-frames per frame, as `usize` for array
/// typing (the existing
/// [`crate::HIGH_BAND_SUBFRAMES_PER_FRAME`] stays `u32` for the
/// bit-budget arithmetic in
/// [`WidebandHighBandSubmode::computed_total_bits`]).
const HB_SUBFRAMES: usize = crate::wideband::HIGH_BAND_SUBFRAMES_PER_FRAME as usize;

/// Width in bits of the per-sub-frame `Excitation gain` field for
/// high-band mode 1 (Table 10.1 column m1).
pub const HB_EXC_GAIN_BITS_MODE_1: u8 = 5;

/// Width in bits of the per-sub-frame `Excitation gain` field for
/// high-band modes 2..=4 (Table 10.1 columns m2 / m3 / m4).
pub const HB_EXC_GAIN_BITS_MODES_2_TO_4: u8 = 4;

/// Typed wrapper over the per-sub-frame high-band `Excitation gain`
/// index.
///
/// Distinguishes the silence case (mode 0 — no field transmitted) from
/// the two documented field widths. Per Table 10.1 the high band has
/// **no frame-level gain factor**, so this single index is the whole
/// fixed-codebook gain surface for one high-band sub-frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HbExcitationGainIndex {
    /// Mode 0 (silence / comfort-noise high band): no field
    /// transmitted, the gain is implicitly zero. The downstream
    /// magnitude reconstruction can short-circuit the excitation
    /// scaling without consulting any quantiser.
    Absent,
    /// Mode 1: a 5-bit index in `0..=31`. Mode 1 carries no
    /// excitation-VQ field (Table 10.1 `Excitation VQ` = 0), so this
    /// gain is the *only* per-sub-frame high-band payload.
    FiveBit(u8),
    /// Modes 2..=4: a 4-bit index in `0..=15`, scaling the §10.3
    /// excitation sub-vector decoded from the mode's `Excitation VQ`
    /// field.
    FourBit(u8),
}

impl HbExcitationGainIndex {
    /// Resolve the typed index from a raw `excitation_gain_index`
    /// field against the active sub-mode's
    /// [`WidebandHighBandSubmode::excitation_gain_bits`] budget.
    ///
    /// * 0-bit budget → [`Self::Absent`] (the parser stores `0` in the
    ///   absent field; this constructor never reads it).
    /// * 5-bit budget → [`Self::FiveBit`] carrying the raw index.
    /// * 4-bit budget → [`Self::FourBit`] carrying the raw index.
    ///
    /// Returns `None` if the sub-mode's budget is none of
    /// {0, 4, 5} — only possible with a hand-built non-conforming
    /// sub-mode literal (no Table 10.1 column uses another width).
    pub fn resolve(raw_index: u8, submode: &WidebandHighBandSubmode) -> Option<Self> {
        match submode.excitation_gain_bits {
            0 => Some(Self::Absent),
            HB_EXC_GAIN_BITS_MODE_1 => Some(Self::FiveBit(raw_index)),
            HB_EXC_GAIN_BITS_MODES_2_TO_4 => Some(Self::FourBit(raw_index)),
            _ => None,
        }
    }

    /// Resolve the typed index for one sub-frame slot (`0..4`) of a
    /// parsed [`WidebandHighBandBody`]. Returns `None` for an
    /// out-of-range slot or a non-conforming sub-mode budget.
    pub fn from_body(
        body: &WidebandHighBandBody,
        submode: &WidebandHighBandSubmode,
        subframe_index: usize,
    ) -> Option<Self> {
        if subframe_index >= HB_SUBFRAMES {
            return None;
        }
        Self::resolve(
            body.subframes[subframe_index].excitation_gain_index,
            submode,
        )
    }

    /// `true` if no gain field is transmitted (mode 0).
    pub fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }

    /// Bit budget (0, 4, or 5) of the surface form — the Table 10.1
    /// `Excitation gain` row value for the originating mode.
    pub fn bit_budget(&self) -> u8 {
        match self {
            Self::Absent => 0,
            Self::FiveBit(_) => HB_EXC_GAIN_BITS_MODE_1,
            Self::FourBit(_) => HB_EXC_GAIN_BITS_MODES_2_TO_4,
        }
    }

    /// Number of distinct quantiser entries the field can address when
    /// transmitted (`32` for the 5-bit form, `16` for the 4-bit form),
    /// else `None`.
    pub fn entries(&self) -> Option<u32> {
        match self {
            Self::Absent => None,
            Self::FiveBit(_) | Self::FourBit(_) => Some(1u32 << self.bit_budget()),
        }
    }

    /// Raw index when the field is present, else `None`. The index is
    /// in `0..=31` for [`Self::FiveBit`] and `0..=15` for
    /// [`Self::FourBit`].
    pub fn raw_index(&self) -> Option<u8> {
        match self {
            Self::Absent => None,
            Self::FiveBit(i) | Self::FourBit(i) => Some(*i),
        }
    }
}

impl fmt::Display for HbExcitationGainIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absent => f.write_str("absent (no Excitation gain field)"),
            Self::FiveBit(i) => write!(f, "5-bit Excitation gain index {}", i),
            Self::FourBit(i) => write!(f, "4-bit Excitation gain index {}", i),
        }
    }
}

/// Resolve the four per-sub-frame [`HbExcitationGainIndex`] values for
/// one parsed high-band frame. Returns `None` if the sub-mode's gain
/// budget is non-conforming (only possible with a hand-built sub-mode
/// literal).
pub fn hb_excitation_gain_indices(
    body: &WidebandHighBandBody,
    submode: &WidebandHighBandSubmode,
) -> Option<[HbExcitationGainIndex; HB_SUBFRAMES]> {
    let mut out = [HbExcitationGainIndex::Absent; HB_SUBFRAMES];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = HbExcitationGainIndex::from_body(body, submode, i)?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wideband::{HighBandSubFrameIndices, WIDEBAND_HIGH_BAND_SUBMODES};

    fn mk_body(gains: [u8; HB_SUBFRAMES]) -> WidebandHighBandBody {
        let mut subframes = [HighBandSubFrameIndices::default(); HB_SUBFRAMES];
        for (sf, g) in subframes.iter_mut().zip(gains) {
            sf.excitation_gain_index = g;
        }
        WidebandHighBandBody {
            lsp_index: 0,
            subframes,
        }
    }

    /// Mode 0 (silence high band) — every sub-frame resolves to
    /// `Absent` with a zero bit budget.
    #[test]
    fn mode0_silence_is_absent_everywhere() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[0];
        assert_eq!(submode.excitation_gain_bits, 0);
        let body = mk_body([0; HB_SUBFRAMES]);
        let resolved = hb_excitation_gain_indices(&body, &submode).expect("budget in spec");
        for (i, slot) in resolved.iter().enumerate() {
            assert!(slot.is_absent(), "sf {i} should be absent");
            assert_eq!(*slot, HbExcitationGainIndex::Absent);
            assert_eq!(slot.bit_budget(), 0);
            assert_eq!(slot.entries(), None);
            assert_eq!(slot.raw_index(), None);
        }
    }

    /// Mode 1 — the 5-bit gain-only column of Table 10.1 (its
    /// `Excitation VQ` row is 0; the gain is the whole sub-frame
    /// payload).
    #[test]
    fn mode1_five_bit_surface() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[1];
        assert_eq!(submode.excitation_gain_bits, HB_EXC_GAIN_BITS_MODE_1);
        assert_eq!(submode.excitation_vq_bits, 0);
        let body = mk_body([0, 13, 30, 31]);
        let resolved = hb_excitation_gain_indices(&body, &submode).expect("budget in spec");
        for (i, slot) in resolved.iter().enumerate() {
            assert!(!slot.is_absent());
            match slot {
                HbExcitationGainIndex::FiveBit(idx) => {
                    assert_eq!(*idx, [0u8, 13, 30, 31][i], "sf {i}");
                }
                other => panic!("expected FiveBit, got {:?}", other),
            }
            assert_eq!(slot.bit_budget(), 5);
            assert_eq!(slot.entries(), Some(32));
        }
    }

    /// Modes 2..=4 — the 4-bit gain columns of Table 10.1.
    #[test]
    fn modes_2_to_4_four_bit_surface() {
        for (mode, submode) in WIDEBAND_HIGH_BAND_SUBMODES.iter().enumerate().skip(2) {
            let submode = *submode;
            assert_eq!(
                submode.excitation_gain_bits, HB_EXC_GAIN_BITS_MODES_2_TO_4,
                "mode {mode}"
            );
            let body = mk_body([1, 7, 14, 15]);
            let resolved = hb_excitation_gain_indices(&body, &submode).expect("budget in spec");
            for (i, slot) in resolved.iter().enumerate() {
                match slot {
                    HbExcitationGainIndex::FourBit(idx) => {
                        assert_eq!(*idx, [1u8, 7, 14, 15][i], "mode {mode} sf {i}");
                    }
                    other => panic!("mode {mode}: expected FourBit, got {:?}", other),
                }
                assert_eq!(slot.bit_budget(), 4);
                assert_eq!(slot.entries(), Some(16));
            }
        }
    }

    /// Every documented high-band mode resolves with the Table 10.1
    /// `Excitation gain` row budget `0 / 5 / 4 / 4 / 4`, and the
    /// per-frame wire footprint is exactly `4 × budget` (no
    /// frame-level gain factor exists in Table 10.1, unlike the
    /// narrowband Table 9.1 `OL Exc gain` row).
    #[test]
    fn every_documented_mode_budget_matches_table_10_1() {
        let expected_bits = [0u8, 5, 4, 4, 4];
        for (mode, submode) in WIDEBAND_HIGH_BAND_SUBMODES.iter().enumerate() {
            assert_eq!(
                submode.excitation_gain_bits, expected_bits[mode],
                "mode {mode} Excitation gain bits"
            );
            let body = mk_body([0; HB_SUBFRAMES]);
            let resolved = hb_excitation_gain_indices(&body, submode).expect("documented mode");
            let frame_gain_bits: u32 = resolved.iter().map(|s| u32::from(s.bit_budget())).sum();
            assert_eq!(
                frame_gain_bits,
                4 * u32::from(expected_bits[mode]),
                "mode {mode}: gain footprint must be 4 × per-sub-frame budget \
                 with no frame-level term"
            );
        }
    }

    /// The 5-bit form covers the full `0..=31` index range.
    #[test]
    fn five_bit_field_covers_full_range() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[1];
        for idx in 0..32u8 {
            let resolved = HbExcitationGainIndex::resolve(idx, &submode).unwrap();
            assert_eq!(resolved, HbExcitationGainIndex::FiveBit(idx));
            assert_eq!(resolved.raw_index(), Some(idx));
        }
    }

    /// The 4-bit form covers the full `0..=15` index range.
    #[test]
    fn four_bit_field_covers_full_range() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
        for idx in 0..16u8 {
            let resolved = HbExcitationGainIndex::resolve(idx, &submode).unwrap();
            assert_eq!(resolved, HbExcitationGainIndex::FourBit(idx));
            assert_eq!(resolved.raw_index(), Some(idx));
        }
    }

    /// A hand-built non-conforming sub-mode (gain budget = 3 bits,
    /// which appears in no Table 10.1 column) is rejected.
    #[test]
    fn non_conforming_budget_rejected() {
        let mut submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
        submode.excitation_gain_bits = 3;
        assert!(HbExcitationGainIndex::resolve(0, &submode).is_none());
        let body = mk_body([0; HB_SUBFRAMES]);
        assert!(hb_excitation_gain_indices(&body, &submode).is_none());
        assert!(body.hb_excitation_gain_indices(&submode).is_none());
    }

    /// `from_body` returns `None` for an out-of-range sub-frame slot.
    #[test]
    fn out_of_range_subframe_slot_returns_none() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
        let body = mk_body([0; HB_SUBFRAMES]);
        assert!(HbExcitationGainIndex::from_body(&body, &submode, HB_SUBFRAMES).is_none());
        assert!(HbExcitationGainIndex::from_body(&body, &submode, 100).is_none());
    }

    /// `raw_index` helpers return the embedded integer for present
    /// fields and `None` for the absent variant.
    #[test]
    fn raw_index_helpers_match_embedded_values() {
        assert_eq!(HbExcitationGainIndex::Absent.raw_index(), None);
        assert_eq!(HbExcitationGainIndex::FiveBit(29).raw_index(), Some(29));
        assert_eq!(HbExcitationGainIndex::FourBit(11).raw_index(), Some(11));
    }

    /// Display impls render the human-readable surface.
    #[test]
    fn display_strings_pin_human_readable_form() {
        assert_eq!(
            format!("{}", HbExcitationGainIndex::Absent),
            "absent (no Excitation gain field)"
        );
        assert_eq!(
            format!("{}", HbExcitationGainIndex::FiveBit(31)),
            "5-bit Excitation gain index 31"
        );
        assert_eq!(
            format!("{}", HbExcitationGainIndex::FourBit(9)),
            "4-bit Excitation gain index 9"
        );
    }

    /// Batch helper matches per-position `from_body` calls.
    #[test]
    fn batch_matches_per_subframe_helper() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[3];
        let body = mk_body([2, 4, 8, 15]);
        let batch = hb_excitation_gain_indices(&body, &submode).unwrap();
        for (i, slot) in batch.iter().enumerate() {
            let single = HbExcitationGainIndex::from_body(&body, &submode, i).unwrap();
            assert_eq!(single, *slot, "sf {i}");
        }
        // And the convenience method on the body agrees with the free
        // function.
        assert_eq!(body.hb_excitation_gain_indices(&submode).unwrap(), batch);
    }

    /// `is_absent` flags exactly the mode-0 surface and nothing else.
    #[test]
    fn is_absent_tracks_mode_0_only() {
        assert!(HbExcitationGainIndex::Absent.is_absent());
        assert!(!HbExcitationGainIndex::FiveBit(0).is_absent());
        assert!(!HbExcitationGainIndex::FourBit(0).is_absent());
    }

    /// The width constants match the Table 10.1 row and the staged
    /// sub-mode table is the single source the resolver dispatches on.
    #[test]
    fn width_constants_match_table_10_1_row() {
        assert_eq!(HB_EXC_GAIN_BITS_MODE_1, 5);
        assert_eq!(HB_EXC_GAIN_BITS_MODES_2_TO_4, 4);
        assert_eq!(
            WIDEBAND_HIGH_BAND_SUBMODES[1].excitation_gain_bits,
            HB_EXC_GAIN_BITS_MODE_1
        );
        for submode in WIDEBAND_HIGH_BAND_SUBMODES.iter().skip(2) {
            assert_eq!(submode.excitation_gain_bits, HB_EXC_GAIN_BITS_MODES_2_TO_4);
        }
    }
}
