//! Speex §5.5 in-band signalling parser — modes 13, 14, 15 bodies.
//!
//! The round-2 [`crate::frame::NarrowbandFrameHeader`] parser dispatches
//! the leading 5-bit prefix's 4-bit mode ID into one of three
//! reserved-for-signalling slots when the ID is not in the regular CELP
//! range:
//!
//! * **Mode 13** — custom in-band message: the encoder emits a 5-bit
//!   `size_bytes` field, followed by `size_bytes * 8` bits of opaque
//!   payload.
//! * **Mode 14** — in-band signalling: the encoder emits a 4-bit
//!   `code` field naming one of the sixteen entries in Table 5.1; the
//!   code's row dictates how many payload bits follow.
//! * **Mode 15** — terminator pseudo-frame: emits no further bits.
//!
//! Round 3 stopped at the header dispatch — the bodies were never read.
//! This module is the round-4 next step: walk the §5.5 prose verbatim
//! to consume the signalling body bits from the [`crate::bitreader::BitReader`]
//! the round-2 parser hands back, leaving the cursor immediately after
//! the message so the next frame in the packet can be parsed by
//! [`crate::frame::NarrowbandFrameHeader::parse`] without a manual
//! re-sync.
//!
//! Spec basis:
//! * *The Speex Codec Manual* (Jean-Marc Valin, December 2007),
//!   `docs/audio/speex/speex-manual.pdf` §5.5 "Packing and in-band
//!   signalling" + Table 5.1 "In-band signalling codes". The table
//!   lists sixteen 4-bit codes (0..=15) with per-code payload widths
//!   of 1, 4, 8, 16, 32, or 64 bits, and a one-line description of
//!   each code's intended request. The final paragraph of §5.5
//!   describes the mode 13 custom-message size field as 5 bits in
//!   bytes.
//!
//! Clean-room note: this module is built from §5.5 + Table 5.1 only.
//! No external library source is consulted; the payload widths are
//! transcribed straight from the manual table column "Size (bits)".

use crate::bitreader::{BitError, BitReader};
use core::fmt;

/// Per *The Speex Codec Manual* §5.5, the custom in-band message
/// length field that follows mode 13's 5-bit prefix is itself a 5-bit
/// integer in bytes — so a custom in-band message is bounded to at
/// most 31 bytes (2^5 − 1) of payload (the manual: *"The size of the
/// message in bytes is encoded with 5 bits, so that the decoder can
/// skip it if it doesn't know how to interpret it."*).
pub const CUSTOM_INBAND_SIZE_BITS: u32 = 5;

/// The maximum custom-in-band payload length in bytes (the upper bound
/// of the 5-bit size field).
pub const CUSTOM_INBAND_MAX_BYTES: u32 = 31;

/// The mode-14 (in-band signalling) message-type code is itself a
/// 4-bit field — exactly enough to index Table 5.1's sixteen rows.
pub const INBAND_CODE_BITS: u32 = 4;

/// Errors produced while parsing a §5.5 signalling body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignallingError {
    /// The bit-reader ran out of bits mid-field.
    Underflow(BitError),
}

impl fmt::Display for SignallingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignallingError::Underflow(e) => write!(f, "signalling body underflow: {}", e),
        }
    }
}

impl std::error::Error for SignallingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SignallingError::Underflow(e) => Some(e),
        }
    }
}

impl From<BitError> for SignallingError {
    fn from(e: BitError) -> Self {
        SignallingError::Underflow(e)
    }
}

/// Table 5.1 row for a single in-band signalling code, transcribed
/// verbatim from §5.5 of *The Speex Codec Manual*.
///
/// `payload_bits` is the table's "Size (bits)" column. `kind` carries
/// the semantic intent of the code as documented in the table's
/// "Content" column. Unknown codes still parse cleanly — the decoder
/// is required by the manual to *"comply or ignore"* and "By default,
/// all in-band messages are ignored."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InbandCodeSpec {
    /// Code value, 0..=15 (4-bit field).
    pub code: u8,
    /// Number of payload bits that follow the code per Table 5.1.
    pub payload_bits: u32,
    /// Categorised intent of the code.
    pub kind: InbandKind,
}

/// Semantic category for each Table 5.1 code.
///
/// The table's "Content" column is reproduced as English doc-strings;
/// the variants below carve out the categories the spec distinguishes
/// (per-row distinct semantics — every code does something different,
/// so each row is its own variant). Reserved codes (11, 13, 14, 15)
/// surface as [`InbandKind::Reserved`] with their declared
/// payload width preserved so the decoder can still advance the
/// bit-stream cursor past them per the §5.5 "by default ignore" rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InbandKind {
    /// Code 0 — perceptual-enhancement on/off (1-bit `value`,
    /// 0 = off, 1 = on).
    PerceptualEnhancement,
    /// Code 1 — request encoder to be less aggressive due to packet
    /// loss (1-bit `value`, 1 = engage).
    LessAggressive,
    /// Code 2 — request encoder to switch to mode N (4-bit `value`).
    SwitchMode,
    /// Code 3 — request encoder to switch to mode N for the low
    /// (narrowband) band of a wideband stream (4-bit `value`).
    SwitchModeLowBand,
    /// Code 4 — request encoder to switch to mode N for the
    /// high-band of a wideband stream (4-bit `value`).
    SwitchModeHighBand,
    /// Code 5 — request encoder to switch to quality N for VBR
    /// (4-bit `value`).
    SwitchQualityVbr,
    /// Code 6 — request acknowledge (4-bit `value`,
    /// 0 = no acks, 1 = ack every packet, 2 = ack only in-band data).
    RequestAcknowledge,
    /// Code 7 — request encoder to set rate mode: 0 = CBR, 1 = VAD,
    /// 3 = DTX, 5 = VBR, 7 = VBR+DTX (4-bit `value` carrying that
    /// bitmask).
    SetRateMode,
    /// Code 8 — transmit an 8-bit character to the other end.
    TransmitCharacter,
    /// Code 9 — intensity-stereo information (8-bit `value`).
    IntensityStereo,
    /// Code 10 — announce maximum acceptable bit-rate (16-bit
    /// `value`, in bytes/second).
    AnnounceMaxBitrate,
    /// Code 12 — acknowledge receiving packet N (32-bit `value`).
    AcknowledgePacket,
    /// Codes 11, 13, 14, 15 — reserved by the spec; the payload
    /// width is given in Table 5.1 so the bit-stream cursor can
    /// still be advanced.
    Reserved,
}

/// Table 5.1 itself, indexed by code value (0..=15). The
/// `(payload_bits, kind)` of every row is taken straight from the
/// manual:
///
/// | Code | Size (bits) | Content                                                                  |
/// | ---- | ----------- | ------------------------------------------------------------------------ |
/// | 0    | 1           | Asks decoder to set perceptual enhancement off (0) or on (1)             |
/// | 1    | 1           | Asks (if 1) the encoder to be less "aggressive" due to high packet loss  |
/// | 2    | 4           | Asks encoder to switch to mode N                                         |
/// | 3    | 4           | Asks encoder to switch to mode N for low-band                            |
/// | 4    | 4           | Asks encoder to switch to mode N for high-band                           |
/// | 5    | 4           | Asks encoder to switch to quality N for VBR                              |
/// | 6    | 4           | Request acknowledge (0=no, 1=all, 2=only for in-band data)               |
/// | 7    | 4           | Asks encoder to set CBR (0), VAD(1), DTX(3), VBR(5), VBR+DTX(7)          |
/// | 8    | 8           | Transmit (8-bit) character to the other end                              |
/// | 9    | 8           | Intensity stereo information                                             |
/// | 10   | 16          | Announce maximum bit-rate acceptable (N in bytes/second)                 |
/// | 11   | 16          | reserved                                                                 |
/// | 12   | 32          | Acknowledge receiving packet N                                           |
/// | 13   | 32          | reserved                                                                 |
/// | 14   | 64          | reserved                                                                 |
/// | 15   | 64          | reserved                                                                 |
pub const INBAND_TABLE_5_1: [InbandCodeSpec; 16] = [
    InbandCodeSpec {
        code: 0,
        payload_bits: 1,
        kind: InbandKind::PerceptualEnhancement,
    },
    InbandCodeSpec {
        code: 1,
        payload_bits: 1,
        kind: InbandKind::LessAggressive,
    },
    InbandCodeSpec {
        code: 2,
        payload_bits: 4,
        kind: InbandKind::SwitchMode,
    },
    InbandCodeSpec {
        code: 3,
        payload_bits: 4,
        kind: InbandKind::SwitchModeLowBand,
    },
    InbandCodeSpec {
        code: 4,
        payload_bits: 4,
        kind: InbandKind::SwitchModeHighBand,
    },
    InbandCodeSpec {
        code: 5,
        payload_bits: 4,
        kind: InbandKind::SwitchQualityVbr,
    },
    InbandCodeSpec {
        code: 6,
        payload_bits: 4,
        kind: InbandKind::RequestAcknowledge,
    },
    InbandCodeSpec {
        code: 7,
        payload_bits: 4,
        kind: InbandKind::SetRateMode,
    },
    InbandCodeSpec {
        code: 8,
        payload_bits: 8,
        kind: InbandKind::TransmitCharacter,
    },
    InbandCodeSpec {
        code: 9,
        payload_bits: 8,
        kind: InbandKind::IntensityStereo,
    },
    InbandCodeSpec {
        code: 10,
        payload_bits: 16,
        kind: InbandKind::AnnounceMaxBitrate,
    },
    InbandCodeSpec {
        code: 11,
        payload_bits: 16,
        kind: InbandKind::Reserved,
    },
    InbandCodeSpec {
        code: 12,
        payload_bits: 32,
        kind: InbandKind::AcknowledgePacket,
    },
    InbandCodeSpec {
        code: 13,
        payload_bits: 32,
        kind: InbandKind::Reserved,
    },
    InbandCodeSpec {
        code: 14,
        payload_bits: 64,
        kind: InbandKind::Reserved,
    },
    InbandCodeSpec {
        code: 15,
        payload_bits: 64,
        kind: InbandKind::Reserved,
    },
];

/// Look up a row of Table 5.1 by its 4-bit code.
///
/// The code field is itself only 4 bits, so every legal in-band
/// signalling code maps to one of the sixteen rows — there is no
/// "unknown code" case for a well-formed bit-stream.
pub const fn inband_code_spec(code: u8) -> InbandCodeSpec {
    INBAND_TABLE_5_1[(code & 0x0F) as usize]
}

/// A parsed mode-14 in-band signalling message.
///
/// The payload is stored as a `u64` because the widest Table 5.1
/// payload is 64 bits (reserved codes 14 and 15). When the spec'd
/// payload width is smaller, the value is zero-extended into the
/// low-order bits of `payload`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InbandMessage {
    /// Table 5.1 row that produced this message (carries `code`,
    /// `payload_bits`, and `kind`).
    pub spec: InbandCodeSpec,
    /// The payload bits, MSB-first into the low end of a `u64`.
    pub payload: u64,
}

impl InbandMessage {
    /// Parse a mode-14 in-band signalling message from a [`BitReader`]
    /// that is positioned **immediately after** the 5-bit Speex frame
    /// prefix (the round-2 [`crate::frame::NarrowbandFrameHeader`]
    /// guarantees this cursor position when the dispatched
    /// [`crate::submode::Submode`] is
    /// [`crate::submode::Submode::InbandSignalling`]).
    ///
    /// Consumes the 4-bit `code` + the payload of width
    /// `inband_code_spec(code).payload_bits`, leaving the cursor
    /// positioned at the first bit of the next frame in the packet.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, SignallingError> {
        let code = reader.read(INBAND_CODE_BITS)? as u8;
        let spec = inband_code_spec(code);
        // Payload widths in Table 5.1 are at most 64 bits; the
        // BitReader returns u32, so a 64-bit payload needs to be read
        // in two halves to avoid the BitReader::TooWide guard.
        let payload = if spec.payload_bits <= 32 {
            reader.read(spec.payload_bits)? as u64
        } else {
            let hi_bits = spec.payload_bits - 32;
            let hi = reader.read(hi_bits)? as u64;
            let lo = reader.read(32)? as u64;
            (hi << 32) | lo
        };
        Ok(Self { spec, payload })
    }
}

/// A parsed mode-13 custom in-band message.
///
/// Mode 13's prefix is followed by a 5-bit byte-count field (`size_bytes`)
/// and `size_bytes * 8` payload bits whose interpretation is reserved
/// to the application that defined the custom code. The decoder is
/// required to be able to skip the payload — that's the entire point
/// of the size field — but does not parse its contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CustomInbandMessage {
    /// Payload size in bytes, from the 5-bit `size_bytes` field.
    pub size_bytes: u8,
}

impl CustomInbandMessage {
    /// Parse the 5-bit `size_bytes` field for a mode-13 message from a
    /// [`BitReader`] positioned immediately after the 5-bit frame
    /// prefix, and advance the cursor past the `size_bytes * 8`
    /// payload bits. The opaque payload itself is **not** returned —
    /// only the byte-count, since the spec defines no on-wire layout
    /// for the body beyond "skip this many bytes".
    ///
    /// Returns [`SignallingError::Underflow`] if the bit-stream is
    /// shorter than the declared payload.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, SignallingError> {
        let size_bytes = reader.read(CUSTOM_INBAND_SIZE_BITS)? as u8;
        // The payload is opaque, but the decoder still has to advance
        // its cursor past the declared bytes so subsequent frames in
        // the same packet realign. Read+discard.
        let payload_bits = u32::from(size_bytes) * 8;
        // BitReader::read caps n at 32; iterate to drain larger fields.
        let mut remaining = payload_bits;
        while remaining > 0 {
            let chunk = remaining.min(32);
            let _ = reader.read(chunk)?;
            remaining -= chunk;
        }
        Ok(Self { size_bytes })
    }

    /// Number of payload bits the parser advanced past.
    pub fn payload_bits(&self) -> u32 {
        u32::from(self.size_bytes) * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Table 5.1 structural sanity ----

    #[test]
    fn table_5_1_has_sixteen_rows() {
        assert_eq!(INBAND_TABLE_5_1.len(), 16);
    }

    #[test]
    fn table_5_1_row_codes_match_index() {
        for (i, row) in INBAND_TABLE_5_1.iter().enumerate() {
            assert_eq!(row.code as usize, i, "row {} has code {}", i, row.code);
        }
    }

    #[test]
    fn table_5_1_payload_widths_match_manual() {
        // Transcribed verbatim from §5.5's "Size (bits)" column.
        let expected = [1, 1, 4, 4, 4, 4, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64];
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(
                INBAND_TABLE_5_1[i].payload_bits, *want,
                "code {} payload width: spec {}, table {}",
                i, want, INBAND_TABLE_5_1[i].payload_bits
            );
        }
    }

    #[test]
    fn inband_code_spec_lookup_round_trips() {
        for code in 0u8..16 {
            let spec = inband_code_spec(code);
            assert_eq!(spec.code, code);
        }
    }

    #[test]
    fn reserved_rows_are_codes_11_13_14_15() {
        // Per §5.5: rows 11 / 13 / 14 / 15 are labelled "reserved".
        for code in 0u8..16 {
            let spec = inband_code_spec(code);
            let is_reserved = matches!(code, 11 | 13 | 14 | 15);
            assert_eq!(
                spec.kind == InbandKind::Reserved,
                is_reserved,
                "code {} reserved-ness mismatch (kind={:?})",
                code,
                spec.kind
            );
        }
    }

    // ---- Mode 14 (in-band signalling) parser ----

    /// Build a single u8 carrying a 4-bit code followed by `payload_bits`
    /// of payload (zero-padded toward the byte's LSB to match Speex's
    /// MSB-first packing convention).
    fn pack_inband_byte(code: u8, payload_bits: u32, payload: u64) -> Vec<u8> {
        // Total bits = 4 (code) + payload_bits. Pad to a whole number
        // of bytes with trailing zeros.
        let total = 4 + payload_bits;
        let mut acc: u128 = (u128::from(code) & 0x0F) << payload_bits as u128;
        acc |= u128::from(payload);
        let nbytes = total.div_ceil(8) as usize;
        // Left-pad inside the high-order bits so the very first bit of
        // the byte slice is the MSB of `code` (matching the BitReader).
        let shift = (nbytes as u32) * 8 - total;
        acc <<= shift as u128;
        let mut out = vec![0u8; nbytes];
        for (i, byte) in out.iter_mut().enumerate() {
            let pos = (nbytes - 1 - i) * 8;
            *byte = ((acc >> pos) & 0xFF) as u8;
        }
        out
    }

    #[test]
    fn parses_perceptual_enhancement_code_0_value_1() {
        // Code 0, payload bit = 1 ("enhancement on").
        let buf = pack_inband_byte(0, 1, 1);
        let mut r = BitReader::new(&buf);
        let msg = InbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.spec.code, 0);
        assert_eq!(msg.spec.kind, InbandKind::PerceptualEnhancement);
        assert_eq!(msg.spec.payload_bits, 1);
        assert_eq!(msg.payload, 1);
        // 4 (code) + 1 (payload) = 5 bits consumed.
        assert_eq!(r.consumed_bits(), 5);
    }

    #[test]
    fn parses_switch_mode_code_2_value_7() {
        // Code 2, 4-bit payload = 7 (request mode 7).
        let buf = pack_inband_byte(2, 4, 7);
        let mut r = BitReader::new(&buf);
        let msg = InbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.spec.code, 2);
        assert_eq!(msg.spec.kind, InbandKind::SwitchMode);
        assert_eq!(msg.payload, 7);
        // 4 + 4 = 8 bits.
        assert_eq!(r.consumed_bits(), 8);
    }

    #[test]
    fn parses_transmit_character_code_8_value_capital_a() {
        // Code 8, 8-bit payload = 0x41 ('A').
        let buf = pack_inband_byte(8, 8, 0x41);
        let mut r = BitReader::new(&buf);
        let msg = InbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.spec.code, 8);
        assert_eq!(msg.spec.kind, InbandKind::TransmitCharacter);
        assert_eq!(msg.payload, 0x41);
        assert_eq!(r.consumed_bits(), 12);
    }

    #[test]
    fn parses_max_bitrate_code_10_value_64000() {
        // Code 10, 16-bit payload = 64_000 bytes/sec advertisement.
        let buf = pack_inband_byte(10, 16, 64_000);
        let mut r = BitReader::new(&buf);
        let msg = InbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.spec.code, 10);
        assert_eq!(msg.spec.kind, InbandKind::AnnounceMaxBitrate);
        assert_eq!(msg.payload, 64_000);
        assert_eq!(r.consumed_bits(), 20);
    }

    #[test]
    fn parses_packet_ack_code_12_value_u32_max() {
        // Code 12, 32-bit payload = 0xDEAD_BEEF (some packet sequence id).
        let buf = pack_inband_byte(12, 32, 0xDEAD_BEEF);
        let mut r = BitReader::new(&buf);
        let msg = InbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.spec.code, 12);
        assert_eq!(msg.spec.kind, InbandKind::AcknowledgePacket);
        assert_eq!(msg.payload, 0xDEAD_BEEF);
        // 4 + 32 = 36 bits.
        assert_eq!(r.consumed_bits(), 36);
    }

    #[test]
    fn parses_reserved_code_14_64_bit_payload() {
        // Code 14, 64-bit payload — exercises the >32-bit split path.
        let payload: u64 = 0x0123_4567_89AB_CDEF;
        let buf = pack_inband_byte(14, 64, payload);
        let mut r = BitReader::new(&buf);
        let msg = InbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.spec.code, 14);
        assert_eq!(msg.spec.kind, InbandKind::Reserved);
        assert_eq!(msg.payload, payload);
        // 4 + 64 = 68 bits.
        assert_eq!(r.consumed_bits(), 68);
    }

    #[test]
    fn parses_reserved_code_15_64_bit_payload_all_ones() {
        // Boundary check: u64::MAX through the wide path.
        let payload: u64 = u64::MAX;
        let buf = pack_inband_byte(15, 64, payload);
        let mut r = BitReader::new(&buf);
        let msg = InbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.spec.code, 15);
        assert_eq!(msg.payload, payload);
        assert_eq!(r.consumed_bits(), 68);
    }

    #[test]
    fn parses_request_acknowledge_code_6_value_2() {
        // Code 6, 4-bit value = 2 ("ack only for in-band data").
        let buf = pack_inband_byte(6, 4, 2);
        let mut r = BitReader::new(&buf);
        let msg = InbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.spec.kind, InbandKind::RequestAcknowledge);
        assert_eq!(msg.payload, 2);
    }

    #[test]
    fn parses_rate_mode_code_7_dtx_value_3() {
        // Code 7, 4-bit value = 3 (DTX).
        let buf = pack_inband_byte(7, 4, 3);
        let mut r = BitReader::new(&buf);
        let msg = InbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.spec.kind, InbandKind::SetRateMode);
        assert_eq!(msg.payload, 3);
    }

    #[test]
    fn truncated_code_field_underflows() {
        // Only 3 bits available — not enough for the 4-bit code.
        let mut r = BitReader::new(&[]);
        // Drain initial capacity to leave 0 bits.
        match InbandMessage::parse(&mut r) {
            Err(SignallingError::Underflow(_)) => {}
            other => panic!("expected Underflow, got {:?}", other),
        }
    }

    #[test]
    fn truncated_payload_underflows() {
        // Code 12 needs 32 payload bits; give it 16 → must error.
        let buf = pack_inband_byte(12, 16, 0xCAFE);
        // Truncate to drop the bytes that would carry the missing
        // 16 bits.
        let trimmed = &buf[..buf.len() - 2];
        let mut r = BitReader::new(trimmed);
        match InbandMessage::parse(&mut r) {
            Err(SignallingError::Underflow(_)) => {}
            other => panic!("expected Underflow, got {:?}", other),
        }
    }

    // ---- Mode 13 (custom in-band) parser ----

    #[test]
    fn custom_inband_size_zero_consumes_only_size_field() {
        // size_bytes = 0 → no payload bits to skip past the 5-bit
        // size field.
        let buf = [0b0000_0000u8];
        let mut r = BitReader::new(&buf);
        let msg = CustomInbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.size_bytes, 0);
        assert_eq!(msg.payload_bits(), 0);
        assert_eq!(r.consumed_bits(), CUSTOM_INBAND_SIZE_BITS);
    }

    #[test]
    fn custom_inband_size_one_skips_eight_bits() {
        // size_bytes = 1 → skip 8 payload bits after the 5-bit size.
        // Total bits to consume = 5 + 8 = 13. Two bytes suffice
        // (16 bits).
        // Top 5 bits = 00001 (size=1), next 8 = anything (payload),
        // remaining 3 = ignored.
        let buf = [0b0000_1101u8, 0b1010_0000u8];
        let mut r = BitReader::new(&buf);
        let msg = CustomInbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.size_bytes, 1);
        assert_eq!(msg.payload_bits(), 8);
        assert_eq!(r.consumed_bits(), 5 + 8);
    }

    #[test]
    fn custom_inband_size_max_consumes_full_payload() {
        // size_bytes = 31 (5-bit max) → 31 * 8 = 248 payload bits
        // after the 5-bit size, total 253 bits. Provide 32 bytes
        // (256 bits) so there's room.
        let mut buf = vec![0u8; 32];
        // Top 5 bits = 11111 = 31. Fill the rest with arbitrary
        // pattern.
        buf[0] = 0b1111_1000;
        // Remainder is opaque — the parser doesn't care what's in it.
        let mut r = BitReader::new(&buf);
        let msg = CustomInbandMessage::parse(&mut r).expect("must parse");
        assert_eq!(msg.size_bytes, 31);
        assert_eq!(msg.payload_bits(), 248);
        assert_eq!(r.consumed_bits(), 253);
    }

    #[test]
    fn custom_inband_truncated_payload_underflows() {
        // size_bytes = 4 → 32 payload bits required; only give 16.
        // Top 5 bits = 00100. Buffer is 3 bytes (24 bits) — that's
        // 5 (size) + 19 < 5 + 32.
        let buf = [0b0010_0000u8, 0u8, 0u8];
        let mut r = BitReader::new(&buf);
        match CustomInbandMessage::parse(&mut r) {
            Err(SignallingError::Underflow(_)) => {}
            other => panic!("expected Underflow, got {:?}", other),
        }
    }

    // ---- Round-trip through the round-2 frame header dispatcher ----

    #[test]
    fn end_to_end_inband_signalling_after_frame_prefix() {
        use crate::frame::NarrowbandFrameHeader;
        use crate::submode::Submode;

        // Build a packet: wideband=0, mode=14 (in-band signalling),
        // followed by code=8 (transmit character), payload=0x42 ('B').
        // Prefix is 5 bits: 0_1110 = 0b01110.
        // Then 4 (code) + 8 (payload) = 12 bits: 1000_01000010.
        // Total = 17 bits. Pack MSB-first into 3 bytes (24 bits),
        // pad trailing with zeros.
        // 01110 1000 01000010 000_0000 = 0b0111_0100, 0b0010_0001, 0b0000_0000
        // wait let me recompute:
        //   bits 0..=4  = 0 1 1 1 0   (prefix: wb=0, mode=14)
        //   bits 5..=8  = 1 0 0 0     (code = 8)
        //   bits 9..=16 = 0 1 0 0 0 0 1 0  (payload 'B' = 0x42)
        //   bits 17..=23 = 0 0 0 0 0 0 0   (padding)
        // Concatenated: 01110 1000 01000010 0000000
        //             = 0b0_1110_1000  0b0100_0010  0b0000_000_
        //             = 0xE8 0x42 0x00 (with low bit of last byte = 0)
        // Let me verify by streaming:
        //   byte0 bits MSB->LSB: 0 1 1 1 0 1 0 0  = 0b0111_0100 = 0x74
        //   byte1 bits MSB->LSB: 0 0 1 0 0 0 0 1  = 0b0010_0001 = 0x21
        //   byte2 bits MSB->LSB: 0 0 0 0 0 0 0 0  = 0x00
        // Bits 0..=4 = byte0[7..=3] = 0,1,1,1,0  -> wb=0, mode=0b1110=14 ✓
        // Bits 5..=8 = byte0[2..=0] + byte1[7]
        //            = 1,0,0,0 -> code=0b1000=8 ✓
        // Bits 9..=16 = byte1[6..=0] + byte2[7]
        //             = 0,1,0,0,0,0,1,0 -> payload=0b01000010=0x42 ✓
        let buf = [0x74, 0x21, 0x00];

        let mut r = BitReader::new(&buf);
        let hdr = NarrowbandFrameHeader::parse(&mut r).expect("header parses");
        assert_eq!(hdr.mode_id, 14);
        assert_eq!(hdr.submode, Submode::InbandSignalling);

        let msg = InbandMessage::parse(&mut r).expect("inband body parses");
        assert_eq!(msg.spec.code, 8);
        assert_eq!(msg.spec.kind, InbandKind::TransmitCharacter);
        assert_eq!(msg.payload, 0x42);
        // 5 (prefix) + 4 (code) + 8 (payload) = 17 bits consumed.
        assert_eq!(r.consumed_bits(), 17);
    }

    #[test]
    fn end_to_end_custom_inband_after_frame_prefix() {
        use crate::frame::NarrowbandFrameHeader;
        use crate::submode::Submode;

        // Packet: wideband=0, mode=13 (custom in-band), size_bytes=0
        // (so no payload to skip).
        // Prefix bits 0..=4 = 0,1,1,0,1 -> wb=0, mode=0b1101=13 ✓
        // Size bits 5..=9 = 0,0,0,0,0 -> size_bytes=0 ✓
        // Pack: 01101 00000 + 6 padding bits = 0b0110_1000  0b0000_0000
        let buf = [0b0110_1000u8, 0b0000_0000u8];

        let mut r = BitReader::new(&buf);
        let hdr = NarrowbandFrameHeader::parse(&mut r).expect("header parses");
        assert_eq!(hdr.mode_id, 13);
        assert_eq!(hdr.submode, Submode::CustomInband);

        let msg = CustomInbandMessage::parse(&mut r).expect("custom in-band parses");
        assert_eq!(msg.size_bytes, 0);
        assert_eq!(msg.payload_bits(), 0);
        // 5 (prefix) + 5 (size) + 0 (payload) = 10 bits.
        assert_eq!(r.consumed_bits(), 10);
    }
}
