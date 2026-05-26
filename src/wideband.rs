//! Speex wideband (sub-band CELP) high-band sub-mode table + body
//! bit-reader.
//!
//! Per *The Speex Codec Manual* §10 "Speex wideband mode (sub-band
//! CELP)", a wideband frame is the concatenation of:
//!
//! ```text
//! +-------------------------------- wideband frame ----------+
//! |                                                          |
//! |   embedded narrowband bit-stream  (Table 9.1)            |
//! |                                                          |
//! |   high-band bit-stream            (Table 10.1)           |
//! |     - 1-bit wideband flag (= 1)                          |
//! |     - 3-bit high-band mode ID                            |
//! |     - 0 / 12 bits LSP (MSVQ)                             |
//! |     - 4 sub-frames × { excitation gain, excitation VQ }  |
//! |                                                          |
//! +----------------------------------------------------------+
//! ```
//!
//! Spec basis (in-tree clean-room sources, no external library
//! consulted):
//!
//! * `docs/audio/speex/speex-manual.pdf` §10 "Speex wideband mode
//!   (sub-band CELP)" — including §10.1..10.4. §10.4 states verbatim:
//!   *"For the wideband mode, the entire narrowband frame is packed
//!   before the high-band is encoded. The narrowband part of the
//!   bit-stream is as defined in table 9.1. The high-band follows, as
//!   described in table 10.1."*
//! * `docs/audio/speex/speex-manual.pdf` Table 10.1 "Bit allocation for
//!   high-band in wideband mode" — five columns (modes 0..=4) with rows
//!   `Wideband bit` / `Mode ID` (3-bit, not 4) / `LSP` (frame) /
//!   `Excitation gain` (sub-frame) / `Excitation VQ` (sub-frame) /
//!   `Total`.
//! * `docs/audio/speex/speex-manual.pdf` Table 10.2 "Quality versus
//!   bit-rate for the wideband encoder" — surfaces the nominal kbps for
//!   each high-band mode 0..=10 (the wideband composite numbers; Table
//!   10.1 itself only details the bit-allocation columns 0..=4).
//! * `docs/audio/speex/rfc5574-speex.txt` Table 2 "Mode vs. Bit-Rate
//!   for Wideband and Ultra-Wideband" — cross-references the nominal
//!   composite bit-rates and confirms the 0..=10 mode-ID range.
//!
//! Round-r160 scope: surface the per-column high-band bit budgets as a
//! typed [`WidebandHighBandSubmode`], stage Table 10.1 as the public
//! [`WIDEBAND_HIGH_BAND_SUBMODES`] array, and provide a
//! [`WidebandHighBandBody::parse`] routine that walks Table 10.1's
//! columns in the order documented in §10.4. **No** codebook lookup —
//! the high-band LSP MSVQ codebook + the high-band innovation codebook
//! both live in the libspeex `*_table.c` files and are #969-blocked
//! until staged under `docs/audio/speex/`. This round stops at the raw
//! bit-index layer, matching the precedent set by [`crate::narrowband_body`]
//! in round 3.
//!
//! ## Spec gaps noted (forwarded to follow-up rounds)
//!
//! * Table 10.1 only details five columns (modes 0..=4). Table 10.2
//!   names modes 0..=10 with associated bit-rates / qualities, but the
//!   per-field bit budgets for modes 5..=10 are not in the staged
//!   manual — the manual table stops at column 4. Modes 5..=10 are
//!   surfaced from [`WidebandHighBandSubmode::for_id`] as `None` and
//!   [`WidebandSubmode::for_id`] treats them as [`WidebandSubmode::ReservedHighRate`]
//!   so the dispatcher can distinguish "spec-documented mode" from
//!   "table-9.1-only mode" from "ID out of range".
//! * Ultra-wideband (mode 2 in the Ogg header) has no dedicated chapter
//!   in the staged manual; RFC 5574 §4 names the per-mode bit-rates
//!   but the bit-stream packing layout for the 32 kHz high-band is not
//!   in the staged spec. This round therefore covers wideband only;
//!   UWB framing is deferred to a follow-up round once the relevant
//!   §11 / Table 11.x material is staged.

use crate::bitreader::{BitError, BitReader};
use core::fmt;

/// The high-band frame-header prefix in the wideband mode is itself
/// `1 + 3 = 4` bits — *not* 5 like the narrowband frame header. The
/// high-band mode ID is only 3 bits wide because only modes 0..=10 are
/// defined for the high-band (the narrowband leading prefix uses 4
/// bits because it must also encode the §5.5 signalling slots 13/14/15
/// which the high-band stream never re-emits).
pub const HIGH_BAND_FRAME_PREFIX_BITS: u32 = 1 + 3;

/// Number of sub-frames per 20 ms wideband high-band frame. Matches the
/// narrowband structure since the QMF splits the 16 kHz input into two
/// 8 kHz half-bands and each half is processed by a narrowband-style
/// CELP at the same 20 ms / 4-sub-frame cadence.
pub const HIGH_BAND_SUBFRAMES_PER_FRAME: u32 = 4;

/// Errors returned from [`WidebandHighBandBody::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidebandBodyError {
    /// The bit-stream ran out mid-field. Carries the underlying
    /// [`BitError`] for diagnostics.
    Underflow(BitError),
    /// The high-band mode ID falls in `5..=7` (the 3-bit field can
    /// encode `0..=7` but only modes `0..=4` have a bit allocation
    /// in Table 10.1).
    ReservedHighRate(u8),
}

impl fmt::Display for WidebandBodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WidebandBodyError::Underflow(e) => {
                write!(f, "wideband body underflow: {}", e)
            }
            WidebandBodyError::ReservedHighRate(id) => write!(
                f,
                "wideband high-band mode {} is outside Table 10.1's documented range 0..=4",
                id
            ),
        }
    }
}

impl std::error::Error for WidebandBodyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WidebandBodyError::Underflow(e) => Some(e),
            WidebandBodyError::ReservedHighRate(_) => None,
        }
    }
}

impl From<BitError> for WidebandBodyError {
    fn from(e: BitError) -> Self {
        WidebandBodyError::Underflow(e)
    }
}

/// Per-column record from Table 10.1, "Bit allocation for high-band in
/// wideband mode".
///
/// Field bit counts are stored as `u8` since the widest documented value
/// is 80 (Excitation VQ for mode 4). `total_bits` mirrors the table's
/// `Total` row exactly so the table is auditable from a single struct
/// literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WidebandHighBandSubmode {
    /// High-band mode ID (3-bit, 0..=4 as listed in Table 10.1).
    pub mode_id: u8,

    /// Frame-level: LSP MSVQ width.
    /// Per §10.1: *"we use only 12 bits to encode the high-band LSP's
    /// using a multi-stage vector quantizer (MSVQ). The first level
    /// quantizes the 10 coefficients with 6 bits and the error is then
    /// quantized using 6 bits, too."* Always `12` for modes 1..=4 and
    /// `0` for mode 0 (silence / comfort-noise high-band).
    pub lsp_bits: u8,

    /// Sub-frame: excitation gain bits (`5` for mode 1, `4` for
    /// modes 2..=4, `0` for mode 0).
    pub excitation_gain_bits: u8,

    /// Sub-frame: excitation VQ bits (`0` / `0` / `20` / `40` / `80`
    /// for modes 0..=4 — note mode 1 has no innovation VQ).
    pub excitation_vq_bits: u8,

    /// `Total` row from Table 10.1, in bits per 20 ms high-band frame,
    /// **including** the 1-bit wideband flag and 3-bit mode-ID prefix.
    pub total_bits: u16,
}

impl WidebandHighBandSubmode {
    /// Returns the high-band sub-mode record for `mode_id`, or `None`
    /// if the ID is not one of the spec-documented columns (`0..=4`).
    /// IDs in `5..=7` are encodable in the 3-bit field but Table 10.1
    /// does not list them — see [`WidebandSubmode::for_id`] for the
    /// disposition.
    pub const fn for_id(mode_id: u8) -> Option<Self> {
        let idx = mode_id as usize;
        if idx >= WIDEBAND_HIGH_BAND_SUBMODES.len() {
            return None;
        }
        Some(WIDEBAND_HIGH_BAND_SUBMODES[idx])
    }

    /// Re-derive the table's `Total` row from the individual field
    /// counts as an audit-time consistency check.
    pub fn computed_total_bits(&self) -> u32 {
        // 1 wideband flag + 3 mode ID + LSP + 4 * (gain + VQ).
        let per_subframe =
            u32::from(self.excitation_gain_bits) + u32::from(self.excitation_vq_bits);
        HIGH_BAND_FRAME_PREFIX_BITS
            + u32::from(self.lsp_bits)
            + HIGH_BAND_SUBFRAMES_PER_FRAME * per_subframe
    }
}

/// All five documented wideband high-band sub-modes, indexed by
/// `mode_id`. Each entry mirrors one column of Table 10.1 verbatim;
/// [`WidebandHighBandSubmode::computed_total_bits`] re-derives `total_bits`
/// from the field breakdown (verified by the round-r160 self-consistency
/// test below).
pub const WIDEBAND_HIGH_BAND_SUBMODES: [WidebandHighBandSubmode; 5] = [
    // Mode 0 — silence / comfort-noise high-band (4 bits / 20 ms total).
    WidebandHighBandSubmode {
        mode_id: 0,
        lsp_bits: 0,
        excitation_gain_bits: 0,
        excitation_vq_bits: 0,
        total_bits: 4,
    },
    // Mode 1 — 5,750 bps composite (36 high-band bits / 20 ms).
    WidebandHighBandSubmode {
        mode_id: 1,
        lsp_bits: 12,
        excitation_gain_bits: 5,
        excitation_vq_bits: 0,
        total_bits: 36,
    },
    // Mode 2 — 7,750 bps composite (112 high-band bits / 20 ms).
    WidebandHighBandSubmode {
        mode_id: 2,
        lsp_bits: 12,
        excitation_gain_bits: 4,
        excitation_vq_bits: 20,
        total_bits: 112,
    },
    // Mode 3 — 9,800 bps composite (192 high-band bits / 20 ms).
    WidebandHighBandSubmode {
        mode_id: 3,
        lsp_bits: 12,
        excitation_gain_bits: 4,
        excitation_vq_bits: 40,
        total_bits: 192,
    },
    // Mode 4 — 12,800 bps composite (352 high-band bits / 20 ms).
    WidebandHighBandSubmode {
        mode_id: 4,
        lsp_bits: 12,
        excitation_gain_bits: 4,
        excitation_vq_bits: 80,
        total_bits: 352,
    },
];

/// Disposition of the 3-bit high-band mode ID. Distinguishes a
/// fully-documented sub-mode from a higher-rate ID that the staged
/// manual does not detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidebandSubmode {
    /// One of the five high-band sub-modes documented in Table 10.1.
    Documented(WidebandHighBandSubmode),
    /// Mode IDs 5, 6, 7 — encodable in the 3-bit field but Table 10.1
    /// stops at column 4. Table 10.2 lists modes 5..=10 with composite
    /// bit-rates but does not detail the per-field bit allocation; the
    /// missing rows are tracked in the module-level "Spec gaps noted"
    /// list and resolved when a follow-up docs round stages them.
    ReservedHighRate(u8),
}

impl WidebandSubmode {
    /// Map a 3-bit high-band mode ID to either a documented sub-mode
    /// or the reserved-high-rate placeholder. Returns `None` if the ID
    /// is out of the 3-bit range (which should never happen for a
    /// correctly-read field, but defends against a caller that hands a
    /// pre-shifted value).
    pub fn for_id(mode_id: u8) -> Option<Self> {
        match mode_id {
            0..=4 => WidebandHighBandSubmode::for_id(mode_id).map(WidebandSubmode::Documented),
            5..=7 => Some(WidebandSubmode::ReservedHighRate(mode_id)),
            _ => None,
        }
    }

    /// Convenience predicate for the fully-documented sub-modes.
    pub fn is_documented(&self) -> bool {
        matches!(self, WidebandSubmode::Documented(_))
    }
}

/// Per-sub-frame raw bit-indices for a wideband high-band frame, in the
/// order they were unpacked from the bit-stream
/// (`excitation_gain`, `excitation_vq` — see Table 10.1).
///
/// Absent fields store `0` and the corresponding "_bits" field on the
/// originating [`WidebandHighBandSubmode`] is also `0`. Codebook lookup
/// is deferred until the high-band LSP MSVQ + innovation tables are
/// staged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HighBandSubFrameIndices {
    /// 0/4/5-bit excitation-gain index. `0` if absent.
    pub excitation_gain_index: u8,
    /// Raw excitation-VQ bits, treated as a single opaque value of up
    /// to 80 bits per sub-frame (mode 4). Stored as `u128` for
    /// headroom; the widest documented value (80) fits in `u128` with
    /// room to spare.
    pub excitation_vq_index: u128,
}

/// Parsed wideband high-band frame body (after the 4-bit prefix).
///
/// Contains the frame-level LSP MSVQ index plus four sub-frame
/// excitation records. The 1-bit wideband flag + 3-bit mode ID prefix
/// itself is **not** stored here — it is parsed separately by
/// [`WidebandHighBandFrameHeader::parse`] and the sub-mode is passed
/// to [`WidebandHighBandBody::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WidebandHighBandBody {
    /// Raw 12-bit LSP MSVQ index (top 6 bits = level-1 codebook,
    /// low 6 bits = level-2 codebook — splitting deferred to codebook
    /// lookup once tables are staged). `0` when the sub-mode skips
    /// LSP transmission (mode 0).
    pub lsp_index: u16,
    /// Per-sub-frame indices for sub-frames 0..=3.
    pub subframes: [HighBandSubFrameIndices; HIGH_BAND_SUBFRAMES_PER_FRAME as usize],
}

impl WidebandHighBandBody {
    /// Consume the high-band body for the given resolved `submode` from
    /// a [`BitReader`] that has already had its 4-bit prefix stripped
    /// by [`WidebandHighBandFrameHeader::parse`]. Returns the parsed
    /// indices and leaves the cursor positioned immediately after the
    /// last excitation-VQ field of sub-frame 3.
    pub fn parse(
        reader: &mut BitReader<'_>,
        submode: &WidebandHighBandSubmode,
    ) -> Result<Self, WidebandBodyError> {
        // Frame-level field, in Table 10.1 order.
        let lsp_index = reader.read(u32::from(submode.lsp_bits))? as u16;

        // Sub-frame fields, four times, in Table 10.1 order.
        let mut subframes =
            [HighBandSubFrameIndices::default(); HIGH_BAND_SUBFRAMES_PER_FRAME as usize];
        for sf in subframes.iter_mut() {
            let excitation_gain_index = reader.read(u32::from(submode.excitation_gain_bits))? as u8;
            let excitation_vq_index = read_wide(reader, u32::from(submode.excitation_vq_bits))?;
            *sf = HighBandSubFrameIndices {
                excitation_gain_index,
                excitation_vq_index,
            };
        }

        Ok(Self {
            lsp_index,
            subframes,
        })
    }

    /// Per-frame bit count consumed by this body (excludes the 4-bit
    /// prefix; matches the sub-mode's `total_bits - 4`).
    pub fn body_bits(submode: &WidebandHighBandSubmode) -> u32 {
        submode.computed_total_bits() - HIGH_BAND_FRAME_PREFIX_BITS
    }
}

/// Parsed wideband high-band frame header (the 4-bit prefix: 1-bit
/// wideband flag + 3-bit mode ID).
///
/// The wideband flag MUST be `1` to indicate that high-band data
/// follows — a `0` here means the next frame in the packet is a fresh
/// narrowband-only frame (§10.4: *"if more than one frame is packed in
/// the same packet, the decoder will need to skip the high-band parts
/// in order to sync with the bit-stream"*).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WidebandHighBandFrameHeader {
    /// 1-bit wideband flag. Must be `true` to indicate a high-band
    /// payload follows. A `false` value means this is not actually a
    /// high-band frame and the caller should rewind the 4-bit read.
    pub wideband: bool,
    /// 3-bit raw high-band mode ID.
    pub mode_id: u8,
    /// Resolved sub-mode (documented or reserved-high-rate).
    pub submode: WidebandSubmode,
}

impl WidebandHighBandFrameHeader {
    /// Parse the 4-bit high-band frame prefix from a [`BitReader`]
    /// positioned immediately after the narrowband portion of a
    /// wideband packet. The reader is left at the first bit of the
    /// high-band body (or, if `submode` is
    /// [`WidebandSubmode::ReservedHighRate`], the caller decides how
    /// to proceed since the per-field bit budget is not in the staged
    /// table).
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, WidebandBodyError> {
        let wideband_bit = reader.read_bit()?;
        let mode_id = reader.read(3)? as u8;
        let submode =
            WidebandSubmode::for_id(mode_id).ok_or(WidebandBodyError::ReservedHighRate(mode_id))?;
        Ok(Self {
            wideband: wideband_bit == 1,
            mode_id,
            submode,
        })
    }
}

/// Read up to 128 bits as a `u128`, chunked through the 32-bit
/// [`BitReader::read`] primitive. The widest Table 10.1 excitation-VQ
/// value is 80 bits per sub-frame (mode 4); `u128` covers the full
/// documented range with headroom.
fn read_wide(reader: &mut BitReader<'_>, n: u32) -> Result<u128, BitError> {
    debug_assert!(n <= 128, "high-band excitation VQ exceeds u128 capacity");
    let mut remaining = n;
    let mut acc: u128 = 0;
    while remaining > 0 {
        let chunk = remaining.min(32);
        let v = reader.read(chunk)? as u128;
        acc = (acc << chunk) | v;
        remaining -= chunk;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Table 10.1 structural sanity ----

    #[test]
    fn table_10_1_has_five_documented_columns() {
        assert_eq!(WIDEBAND_HIGH_BAND_SUBMODES.len(), 5);
    }

    #[test]
    fn table_10_1_row_mode_ids_match_index() {
        for (i, row) in WIDEBAND_HIGH_BAND_SUBMODES.iter().enumerate() {
            assert_eq!(
                row.mode_id as usize, i,
                "row {} has mode_id {}",
                i, row.mode_id
            );
        }
    }

    #[test]
    fn table_10_1_totals_match_field_breakdown() {
        // Every column of Table 10.1 must satisfy
        //   1 (wideband flag) + 3 (mode ID) + LSP + 4 * (gain + VQ) == Total
        for s in WIDEBAND_HIGH_BAND_SUBMODES {
            let computed = s.computed_total_bits();
            assert_eq!(
                computed,
                u32::from(s.total_bits),
                "mode {} total: spec {}, computed {}",
                s.mode_id,
                s.total_bits,
                computed,
            );
        }
    }

    #[test]
    fn table_10_1_totals_are_spec_values() {
        // Hard-coded spot-check: Table 10.1's Total row, modes 0..=4.
        let expected = [4u16, 36, 112, 192, 352];
        for (i, want) in expected.iter().enumerate() {
            let got = WIDEBAND_HIGH_BAND_SUBMODES[i].total_bits;
            assert_eq!(got, *want, "mode {} total mismatch", i);
        }
    }

    #[test]
    fn lsp_bits_are_zero_for_silence_and_twelve_for_active_modes() {
        // §10.1: 12-bit MSVQ for all active modes; mode 0 (silence) skips
        // the LSP frame field entirely.
        assert_eq!(WIDEBAND_HIGH_BAND_SUBMODES[0].lsp_bits, 0);
        for s in &WIDEBAND_HIGH_BAND_SUBMODES[1..] {
            assert_eq!(s.lsp_bits, 12, "mode {} LSP bits", s.mode_id);
        }
    }

    #[test]
    fn excitation_gain_bits_match_table_10_1() {
        let expected = [0u8, 5, 4, 4, 4];
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(
                WIDEBAND_HIGH_BAND_SUBMODES[i].excitation_gain_bits, *want,
                "mode {} excitation gain bits",
                i
            );
        }
    }

    #[test]
    fn excitation_vq_bits_match_table_10_1() {
        let expected = [0u8, 0, 20, 40, 80];
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(
                WIDEBAND_HIGH_BAND_SUBMODES[i].excitation_vq_bits, *want,
                "mode {} excitation VQ bits",
                i
            );
        }
    }

    #[test]
    fn for_id_covers_documented_modes_0_through_4() {
        for id in 0u8..=4 {
            let s = WidebandHighBandSubmode::for_id(id).expect("documented");
            assert_eq!(s.mode_id, id);
        }
    }

    #[test]
    fn for_id_rejects_undocumented_modes_above_4() {
        for id in 5u8..=255 {
            assert!(
                WidebandHighBandSubmode::for_id(id).is_none(),
                "mode {} should be undocumented in Table 10.1",
                id
            );
        }
    }

    // ---- WidebandSubmode dispatch ----

    #[test]
    fn submode_dispatch_for_documented_range() {
        for id in 0u8..=4 {
            match WidebandSubmode::for_id(id) {
                Some(WidebandSubmode::Documented(s)) => assert_eq!(s.mode_id, id),
                other => panic!("mode {} should be Documented, got {:?}", id, other),
            }
        }
    }

    #[test]
    fn submode_dispatch_for_reserved_high_rate() {
        for id in 5u8..=7 {
            match WidebandSubmode::for_id(id) {
                Some(WidebandSubmode::ReservedHighRate(got)) => assert_eq!(got, id),
                other => panic!("mode {} should be ReservedHighRate, got {:?}", id, other),
            }
        }
    }

    #[test]
    fn submode_dispatch_rejects_out_of_three_bit_range() {
        for id in 8u8..=255 {
            assert!(
                WidebandSubmode::for_id(id).is_none(),
                "mode {} out of 3-bit range",
                id
            );
        }
    }

    #[test]
    fn is_documented_predicate() {
        assert!(WidebandSubmode::for_id(0).unwrap().is_documented());
        assert!(WidebandSubmode::for_id(4).unwrap().is_documented());
        assert!(!WidebandSubmode::for_id(5).unwrap().is_documented());
        assert!(!WidebandSubmode::for_id(7).unwrap().is_documented());
    }

    // ---- Body parser ----

    /// Build a synthetic wideband high-band frame whose bits are all
    /// `1`, with the 4-bit prefix forcing the supplied `mode`. Useful
    /// because every read at the maximum field width returns
    /// `(1 << n) - 1`.
    fn all_ones_hb_frame(mode: u8) -> Vec<u8> {
        let submode = WidebandHighBandSubmode::for_id(mode).expect("documented mode");
        let total_bits = u32::from(submode.total_bits);
        let total_bytes = total_bits.div_ceil(8);
        let mut buf = vec![0xFFu8; total_bytes as usize];
        // Top bit = wideband=1, next 3 bits = mode_id, low 4 bits of
        // the first byte remain all 1s (the body bits keep their `0xFF`
        // pattern).
        let prefix_byte = (1u8 << 7) | ((mode & 0x07) << 4) | 0b0000_1111;
        buf[0] = prefix_byte;
        buf
    }

    #[test]
    fn silence_high_band_has_empty_body() {
        // Mode 0: 4-bit prefix + zero-bit body. After parsing the prefix
        // from a 1-byte buffer the body parser should not consume any
        // further bits.
        // wb=1, mode=0, rest zero — packed MSB-first into a single byte.
        let buf = [0b1000_0000_u8];
        let mut r = BitReader::new(&buf);
        let h = WidebandHighBandFrameHeader::parse(&mut r).expect("header parses");
        assert!(h.wideband);
        assert_eq!(h.mode_id, 0);
        let s = match h.submode {
            WidebandSubmode::Documented(s) => s,
            other => panic!("expected Documented, got {:?}", other),
        };
        let before = r.consumed_bits();
        let body = WidebandHighBandBody::parse(&mut r, &s).expect("body parses");
        assert_eq!(r.consumed_bits(), before, "body must consume zero bits");
        assert_eq!(body.lsp_index, 0);
        for sf in &body.subframes {
            assert_eq!(sf.excitation_gain_index, 0);
            assert_eq!(sf.excitation_vq_index, 0);
        }
    }

    #[test]
    fn body_bits_matches_table_total_minus_prefix() {
        for s in WIDEBAND_HIGH_BAND_SUBMODES {
            let expected = u32::from(s.total_bits) - HIGH_BAND_FRAME_PREFIX_BITS;
            assert_eq!(
                WidebandHighBandBody::body_bits(&s),
                expected,
                "mode {} body_bits mismatch",
                s.mode_id
            );
        }
    }

    #[test]
    fn parse_consumes_exactly_total_bits_for_every_documented_mode() {
        for mode in 0u8..=4 {
            let buf = all_ones_hb_frame(mode);
            let mut r = BitReader::new(&buf);
            let h = WidebandHighBandFrameHeader::parse(&mut r).expect("header");
            let s = match h.submode {
                WidebandSubmode::Documented(s) => s,
                _ => unreachable!(),
            };
            let _ = WidebandHighBandBody::parse(&mut r, &s).expect("body");
            assert_eq!(
                r.consumed_bits(),
                u32::from(s.total_bits),
                "mode {} consumed wrong total",
                mode
            );
        }
    }

    #[test]
    fn mode_4_field_widths_match_table_10_1() {
        // Mode 4 (highest documented) has all the widest fields filled.
        let buf = all_ones_hb_frame(4);
        let mut r = BitReader::new(&buf);
        let h = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        let s = match h.submode {
            WidebandSubmode::Documented(s) => s,
            _ => unreachable!(),
        };
        assert_eq!(s.mode_id, 4);
        let body = WidebandHighBandBody::parse(&mut r, &s).unwrap();

        // LSP: 12 bits all 1 → 4095.
        assert_eq!(body.lsp_index, 0x0FFF);
        for sf in &body.subframes {
            // Excitation gain: 4 bits → 15.
            assert_eq!(sf.excitation_gain_index, 0b1111);
            // Excitation VQ: 80 bits → (1 << 80) - 1.
            assert_eq!(sf.excitation_vq_index, (1u128 << 80) - 1);
        }
    }

    #[test]
    fn mode_1_has_no_innovation_vq() {
        // Mode 1: gain=5, VQ=0. Per Table 10.1 the innovation codebook
        // is empty for this mode — only the gain conveys the high-band.
        let s = WidebandHighBandSubmode::for_id(1).unwrap();
        assert_eq!(s.excitation_vq_bits, 0);
        assert_eq!(s.excitation_gain_bits, 5);
        // Synthesise an all-ones frame and confirm every sub-frame's
        // innovation VQ index is parsed as 0 (zero-width read).
        let buf = all_ones_hb_frame(1);
        let mut r = BitReader::new(&buf);
        let _h = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        let body = WidebandHighBandBody::parse(&mut r, &s).unwrap();
        for sf in &body.subframes {
            assert_eq!(sf.excitation_gain_index, 0b11111); // 5 bits = 31.
            assert_eq!(sf.excitation_vq_index, 0);
        }
    }

    #[test]
    fn underflow_diagnosed_for_truncated_high_band() {
        // Mode 4 needs 352 bits = 44 bytes. Give it 4 → must underflow.
        let mut buf = vec![0xFFu8; 4];
        buf[0] = (1u8 << 7) | ((4u8 & 0x07) << 4) | 0b0000_1111;
        let mut r = BitReader::new(&buf);
        let h = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        let s = match h.submode {
            WidebandSubmode::Documented(s) => s,
            _ => unreachable!(),
        };
        match WidebandHighBandBody::parse(&mut r, &s) {
            Err(WidebandBodyError::Underflow(_)) => {}
            other => panic!("expected Underflow, got {:?}", other),
        }
    }

    #[test]
    fn header_parser_propagates_reserved_high_rate() {
        // 3-bit mode field set to 5 (encodable but undocumented).
        let buf = [(1u8 << 7) | ((5u8 & 0x07) << 4)];
        let mut r = BitReader::new(&buf);
        match WidebandHighBandFrameHeader::parse(&mut r) {
            // for_id returns Some(ReservedHighRate(5)) so the header
            // *does* parse; the error path only fires if the 3-bit
            // field somehow exceeds 7, which the read(3) guarantees
            // can't happen. So expect a successful parse with the
            // reserved-high-rate disposition.
            Ok(h) => {
                assert_eq!(h.mode_id, 5);
                assert_eq!(h.submode, WidebandSubmode::ReservedHighRate(5));
            }
            other => panic!("expected Ok(ReservedHighRate(5)), got {:?}", other),
        }
    }

    #[test]
    fn header_parser_rejects_short_input() {
        let mut r = BitReader::new(&[]);
        match WidebandHighBandFrameHeader::parse(&mut r) {
            Err(WidebandBodyError::Underflow(_)) => {}
            other => panic!("expected Underflow, got {:?}", other),
        }
    }

    #[test]
    fn read_wide_handles_eighty_bit_fields() {
        // The widest documented field is 80 bits; verify read_wide
        // returns the correct value when fed a known pattern.
        let mut bytes = [0xFFu8; 10];
        // Place a known pattern in the top 80 bits = all ones, then
        // confirm that read_wide(80) consumes them as (1 << 80) - 1.
        let mut r = BitReader::new(&bytes);
        let v = read_wide(&mut r, 80).unwrap();
        assert_eq!(v, (1u128 << 80) - 1);

        // Mixed pattern: top 32 = 0xCAFE_BABE, next 32 = 0xDEAD_BEEF,
        // next 16 = 0x1234. Total 80 bits.
        bytes = [0xCA, 0xFE, 0xBA, 0xBE, 0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34];
        let mut r = BitReader::new(&bytes);
        let v = read_wide(&mut r, 80).unwrap();
        let expected: u128 = (0xCAFE_BABE_u128 << 48) | (0xDEAD_BEEF_u128 << 16) | 0x1234_u128;
        assert_eq!(v, expected);
    }

    #[test]
    fn wideband_flag_false_is_surfaced() {
        // A 4-bit prefix with the wideband flag clear (= 0) is encodable
        // but conceptually means "no high-band data follows" per §10.4.
        // Verify the header parser surfaces it without erroring (the
        // caller decides whether to honour it).
        // wb=0, mode=0 — packed MSB-first into a single byte.
        let buf = [0b0000_0000_u8];
        let mut r = BitReader::new(&buf);
        let h = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        assert!(!h.wideband);
        assert_eq!(h.mode_id, 0);
    }
}
