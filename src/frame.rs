//! Speex narrowband frame-header parser.
//!
//! Every Speex frame begins with a fixed 5-bit prefix:
//!
//! ```text
//! +-+-+-+-+-+
//! |W| MODE  |
//! +-+-+-+-+-+
//!   |    \-- 4-bit mode ID (Table 9.1 column / Table 5.1 / terminator)
//!   \------- 1-bit wideband flag (§10.4; for a pure-narrowband stream
//!            this is 0; for a wideband stream the narrowband portion
//!            also carries it as 0 and the high-band portion has it as
//!            1 — per §10.4: "the entire narrowband frame is packed
//!            before the high-band is encoded").
//! ```
//!
//! Spec basis:
//! * §9.3 "Bit allocation" — *"Each frame starts with the mode ID
//!   encoded with 4 bits which allows a range from 0 to 15"*. Table
//!   9.1 lists the 1-bit wideband flag immediately above the 4-bit
//!   mode ID, in the order the bit-stream is packed.
//! * §5.5 "Packing and in-band signalling" — mode 13 (custom in-band),
//!   mode 14 (in-band signalling), mode 15 (terminator) are not CELP
//!   frames and carry their own short follow-on fields.
//! * §10.4 "Bit allocation" — wideband flag distinguishes the
//!   narrowband part of a frame from the high-band part packed after
//!   it.
//!
//! Round-2 scope: surface the leading 5 bits as a typed
//! [`NarrowbandFrameHeader`] + dispatch the mode ID through
//! [`crate::submode::Submode`]. No CELP body parsing yet.

use crate::bitreader::{BitError, BitReader, BitWriter};
use crate::submode::Submode;
use core::fmt;

/// Number of bits consumed by the leading wideband-flag + mode-ID
/// prefix at the start of every Speex frame.
pub const NARROWBAND_FRAME_PREFIX_BITS: u32 = 5;

/// Errors produced when parsing the narrowband frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    /// The packet was shorter than 5 bits — couldn't even decode the
    /// leading wideband-flag + mode-ID prefix.
    Underflow(BitError),
    /// The mode ID falls in the spec's "reserved" range (9..=12), for
    /// which no behaviour is defined. Carries the raw ID for
    /// diagnostics.
    ReservedMode(u8),
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameError::Underflow(e) => write!(f, "frame header underflow: {}", e),
            FrameError::ReservedMode(id) => write!(
                f,
                "frame header: mode ID {} is in the spec's reserved range 9..=12",
                id
            ),
        }
    }
}

impl std::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FrameError::Underflow(e) => Some(e),
            FrameError::ReservedMode(_) => None,
        }
    }
}

impl From<BitError> for FrameError {
    fn from(e: BitError) -> Self {
        FrameError::Underflow(e)
    }
}

/// Parsed narrowband frame header (the leading 5 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NarrowbandFrameHeader {
    /// 1-bit wideband flag. `false` for a narrowband-only stream or
    /// the narrowband portion of a wideband stream; `true` for the
    /// high-band portion of a wideband stream (§10.4).
    pub wideband: bool,
    /// 4-bit raw mode ID.
    pub mode_id: u8,
    /// Resolved sub-mode (regular CELP or §5.5 signalling slot).
    pub submode: Submode,
}

impl NarrowbandFrameHeader {
    /// Parse the leading 5 bits of a Speex frame from a fresh
    /// [`BitReader`]. Returns the parsed header and leaves the reader
    /// positioned at the first bit *after* the mode ID, ready for the
    /// per-sub-mode body to be consumed.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, FrameError> {
        let wideband_bit = reader.read_bit()?;
        let mode_id = reader.read(4)? as u8;
        let submode = Submode::for_id(mode_id).ok_or(FrameError::ReservedMode(mode_id))?;
        Ok(Self {
            wideband: wideband_bit == 1,
            mode_id,
            submode,
        })
    }

    /// Convenience: parse the prefix directly from a byte slice.
    pub fn parse_bytes(buf: &[u8]) -> Result<(Self, BitReader<'_>), FrameError> {
        let mut r = BitReader::new(buf);
        let h = Self::parse(&mut r)?;
        Ok((h, r))
    }

    /// Build a header value from a (wideband, mode_id) pair, dispatching
    /// the mode ID through [`Submode::for_id`]. Returns
    /// [`FrameError::ReservedMode`] for IDs in `9..=12`, matching the
    /// rejection [`Self::parse`] would issue if it read the same bits
    /// off the wire.
    ///
    /// This is the constructor an encoder calls before [`Self::write`]
    /// — round 2's [`Self::parse`] is the inverse mapping for the
    /// reader side. The two are designed to round-trip:
    /// `parse(write(header))` recovers `header`.
    pub fn new(wideband: bool, mode_id: u8) -> Result<Self, FrameError> {
        let submode = Submode::for_id(mode_id).ok_or(FrameError::ReservedMode(mode_id))?;
        Ok(Self {
            wideband,
            mode_id,
            submode,
        })
    }

    /// Serialise the 5-bit Speex frame prefix (1-bit wideband flag +
    /// 4-bit mode ID, MSB-first) to a [`BitWriter`]. Inverse operation
    /// of [`Self::parse`]: a buffer built by `write(header)` and then
    /// parsed by [`Self::parse`] yields a header equal to the original.
    ///
    /// `mode_id` is masked to its low 4 bits before emission so the
    /// caller cannot inadvertently emit a 5-bit value. The wideband
    /// flag is the single high bit of the prefix.
    pub fn write(&self, writer: &mut BitWriter) -> Result<(), FrameError> {
        writer.write(u32::from(self.wideband as u8), 1)?;
        writer.write(u32::from(self.mode_id) & 0x0F, 4)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::submode::{NarrowbandSubmode, Submode};

    /// Build a single byte whose top 5 bits encode (wideband_flag,
    /// mode_id) per the §9.3 packing order.
    fn prefix_byte(wideband: bool, mode_id: u8) -> u8 {
        assert!(mode_id < 16);
        let w = if wideband { 1u8 } else { 0u8 };
        // Top bit = wideband, next 4 = mode_id, low 3 = filler zeros.
        (w << 7) | ((mode_id & 0x0F) << 3)
    }

    #[test]
    fn parses_mode_3_narrowband_frame() {
        // Mode 3 (8 kbps): wideband=0, mode=3.
        let buf = [prefix_byte(false, 3)];
        let (h, r) = NarrowbandFrameHeader::parse_bytes(&buf).expect("must parse");
        assert!(!h.wideband);
        assert_eq!(h.mode_id, 3);
        match h.submode {
            Submode::Celp(s) => assert_eq!(s.mode_id, 3),
            other => panic!("expected CELP sub-mode, got {:?}", other),
        }
        // The reader must have consumed exactly NARROWBAND_FRAME_PREFIX_BITS.
        assert_eq!(r.consumed_bits(), NARROWBAND_FRAME_PREFIX_BITS);
        // …and not have moved further.
        assert_eq!(r.remaining_bits(), 8 - NARROWBAND_FRAME_PREFIX_BITS);
    }

    #[test]
    fn wideband_flag_is_preserved() {
        // Mode 0, wideband flag set.
        let buf = [prefix_byte(true, 0)];
        let (h, _) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        assert!(h.wideband);
        assert_eq!(h.mode_id, 0);
    }

    #[test]
    fn terminator_dispatches_to_special_slot() {
        let buf = [prefix_byte(false, 15)];
        let (h, _) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        assert_eq!(h.mode_id, 15);
        assert_eq!(h.submode, Submode::Terminator);
    }

    #[test]
    fn inband_signalling_dispatches_to_special_slot() {
        let buf = [prefix_byte(false, 14)];
        let (h, _) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        assert_eq!(h.submode, Submode::InbandSignalling);
    }

    #[test]
    fn custom_inband_dispatches_to_special_slot() {
        let buf = [prefix_byte(false, 13)];
        let (h, _) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        assert_eq!(h.submode, Submode::CustomInband);
    }

    #[test]
    fn reserved_modes_9_through_12_are_rejected() {
        for id in 9u8..=12 {
            let buf = [prefix_byte(false, id)];
            match NarrowbandFrameHeader::parse_bytes(&buf) {
                Err(FrameError::ReservedMode(got)) => assert_eq!(got, id),
                other => panic!(
                    "mode {} should be rejected as reserved, got {:?}",
                    id, other
                ),
            }
        }
    }

    #[test]
    fn empty_buffer_underflows() {
        match NarrowbandFrameHeader::parse_bytes(&[]) {
            Err(FrameError::Underflow(_)) => {}
            other => panic!("expected Underflow, got {:?}", other),
        }
    }

    #[test]
    fn modes_0_through_8_resolve_to_celp() {
        for id in 0u8..=8 {
            let buf = [prefix_byte(false, id)];
            let (h, _) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
            match h.submode {
                Submode::Celp(NarrowbandSubmode { mode_id, .. }) => {
                    assert_eq!(mode_id, id);
                }
                other => panic!("mode {} should resolve to CELP, got {:?}", id, other),
            }
        }
    }

    #[test]
    fn reader_cursor_is_at_bit_5_after_parse() {
        // Stuff some additional bits into the byte; after parsing the
        // header, the next read(3) should observe exactly the bottom
        // 3 bits of the byte.
        // wideband=1, mode=0b0101 → top 5 bits = 1_0101; bottom 3
        // bits = 0b110 (decimal 6).
        let byte = (1u8 << 7) | (0b0101 << 3) | 0b110;
        let buf = [byte];
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        assert!(h.wideband);
        assert_eq!(h.mode_id, 0b0101);
        assert_eq!(r.read(3).unwrap(), 0b110);
    }

    // ---- Round-187 round-trip: NarrowbandFrameHeader::write ----

    #[test]
    fn new_constructs_header_for_celp_mode() {
        let h = NarrowbandFrameHeader::new(false, 5).expect("mode 5 is valid CELP");
        assert!(!h.wideband);
        assert_eq!(h.mode_id, 5);
        match h.submode {
            Submode::Celp(s) => assert_eq!(s.mode_id, 5),
            other => panic!("expected CELP, got {:?}", other),
        }
    }

    #[test]
    fn new_rejects_reserved_mode_ids() {
        for id in 9u8..=12 {
            match NarrowbandFrameHeader::new(false, id) {
                Err(FrameError::ReservedMode(got)) => assert_eq!(got, id),
                other => panic!("mode {} should be reserved, got {:?}", id, other),
            }
        }
    }

    #[test]
    fn new_dispatches_signalling_slots() {
        assert_eq!(
            NarrowbandFrameHeader::new(false, 13).unwrap().submode,
            Submode::CustomInband
        );
        assert_eq!(
            NarrowbandFrameHeader::new(false, 14).unwrap().submode,
            Submode::InbandSignalling
        );
        assert_eq!(
            NarrowbandFrameHeader::new(false, 15).unwrap().submode,
            Submode::Terminator
        );
    }

    #[test]
    fn write_then_parse_round_trips_all_celp_modes() {
        use crate::bitreader::BitWriter;
        for id in 0u8..=8 {
            for wb in [false, true] {
                let h = NarrowbandFrameHeader::new(wb, id).unwrap();
                let mut w = BitWriter::new();
                h.write(&mut w).unwrap();
                w.pad_to_byte().unwrap();
                let bytes = w.into_bytes();
                let (round, _) = NarrowbandFrameHeader::parse_bytes(&bytes).unwrap();
                assert_eq!(round, h, "round-trip failed for (wb={}, id={})", wb, id);
            }
        }
    }

    #[test]
    fn write_then_parse_round_trips_signalling_slots() {
        use crate::bitreader::BitWriter;
        for id in [13u8, 14, 15] {
            let h = NarrowbandFrameHeader::new(false, id).unwrap();
            let mut w = BitWriter::new();
            h.write(&mut w).unwrap();
            w.pad_to_byte().unwrap();
            let bytes = w.into_bytes();
            let (round, _) = NarrowbandFrameHeader::parse_bytes(&bytes).unwrap();
            assert_eq!(round.mode_id, id);
            assert_eq!(round.submode, h.submode);
        }
    }

    #[test]
    fn write_emits_exactly_five_bits() {
        use crate::bitreader::BitWriter;
        let h = NarrowbandFrameHeader::new(true, 7).unwrap();
        let mut w = BitWriter::new();
        h.write(&mut w).unwrap();
        assert_eq!(w.bits_written(), NARROWBAND_FRAME_PREFIX_BITS);
    }

    #[test]
    fn write_emits_high_bit_for_wideband_flag() {
        use crate::bitreader::BitWriter;
        let h = NarrowbandFrameHeader::new(true, 0).unwrap();
        let mut w = BitWriter::new();
        h.write(&mut w).unwrap();
        w.pad_to_byte().unwrap();
        // wb=1, mode=0 → bits "1_0000_000" = 0b1000_0000 = 0x80
        assert_eq!(w.as_bytes(), &[0x80]);
    }

    #[test]
    fn write_masks_mode_id_to_four_bits() {
        // mode_id high bits above 0x0F should be ignored on emission.
        // (We can't actually build such a header via `new`, but the
        // contract holds for any value of the field.)
        use crate::bitreader::BitWriter;
        let h = NarrowbandFrameHeader {
            wideband: false,
            mode_id: 0xFF, // synthetic — would never arise from `new`
            submode: Submode::Terminator,
        };
        let mut w = BitWriter::new();
        h.write(&mut w).unwrap();
        w.pad_to_byte().unwrap();
        // Only the low 4 bits land: 0_1111_000 = 0b0111_1000 = 0x78
        assert_eq!(w.as_bytes(), &[0x78]);
    }
}
