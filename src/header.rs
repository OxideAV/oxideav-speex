//! Ogg/Speex stream-header packet parser.
//!
//! The Speex bitstream, when carried in an Ogg container, begins with a
//! fixed-layout header packet that identifies the stream and declares the
//! mode (narrowband / wideband / ultra-wideband), sampling rate, channel
//! count, frame size, and a handful of negotiation fields. The layout is
//! specified verbatim in *The Speex Codec Manual* §7.3 "Ogg file format",
//! Table 7.1, which the parser in this module follows byte-for-byte.
//!
//! Round-1 scope: this is a **header walker only**. It validates the
//! magic, decodes every field as a little-endian integer per the spec's
//! "All integer fields in the headers are stored as little-endian"
//! sentence, and surfaces the fields as a Rust struct. No frame decode
//! is wired up yet — that is round 2+.
//!
//! Spec references (in-tree clean-room sources, no external library
//! consulted):
//! * `docs/audio/speex/speex-manual.pdf` §7.3 "Ogg file format" + Table
//!   7.1 "Ogg/Speex header packet" (Jean-Marc Valin, December 2007).
//! * `docs/audio/speex/rfc5574-speex.txt` Tables 1 & 2 for the
//!   `mode` ↔ sampling-rate mapping cross-check.

use core::fmt;

/// The 8-byte magic that identifies a Speex stream header packet.
///
/// Per §7.3: *"The speex_string field must contain the 'Speex   ' (with
/// 3 trailing spaces), which identifies the bit-stream."* The trailing
/// spaces are part of the magic and MUST be present.
pub const SPEEX_MAGIC: &[u8; 8] = b"Speex   ";

/// Fixed serialized length of the Speex header packet, in bytes.
///
/// Derived from Table 7.1: `speex_string` (8) + `speex_version` (20)
/// + 13 little-endian `int` fields × 4 bytes each = **80 bytes**.
pub const SPEEX_HEADER_LEN: usize = 8 + 20 + 13 * 4;

/// Length of the `speex_string` magic field, in bytes.
pub const SPEEX_STRING_LEN: usize = 8;

/// Length of the `speex_version` field, in bytes.
pub const SPEEX_VERSION_LEN: usize = 20;

/// Narrowband mode identifier (8 kHz nominal).
///
/// Cross-referenced against RFC 5574 §3 ("either narrowband (nominal 8
/// kHz), wideband (nominal 16 kHz), or ultra-wideband (nominal 32 kHz)")
/// and the manual's §2.1 sampling-rate discussion.
pub const SPEEX_MODE_NARROWBAND: u32 = 0;

/// Wideband mode identifier (16 kHz nominal).
pub const SPEEX_MODE_WIDEBAND: u32 = 1;

/// Ultra-wideband mode identifier (32 kHz nominal).
pub const SPEEX_MODE_ULTRAWIDEBAND: u32 = 2;

/// Errors produced when parsing an Ogg/Speex stream header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderError {
    /// The input buffer is shorter than [`SPEEX_HEADER_LEN`].
    ///
    /// Carries the observed length; the required length is the constant.
    TooShort(usize),
    /// The `speex_string` magic did not equal `b"Speex   "`.
    BadMagic([u8; SPEEX_STRING_LEN]),
}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeaderError::TooShort(n) => write!(
                f,
                "speex header: input is {} bytes, need {}",
                n, SPEEX_HEADER_LEN
            ),
            HeaderError::BadMagic(m) => write!(
                f,
                "speex header: bad magic {:02x?}, want {:02x?}",
                m, SPEEX_MAGIC
            ),
        }
    }
}

impl std::error::Error for HeaderError {}

/// Parsed Ogg/Speex stream-header packet (Table 7.1).
///
/// Field names and order are taken verbatim from the spec table to keep
/// the mapping audit-trivial. Numeric fields are stored as `u32` since
/// the spec only specifies "int" (32-bit) without signedness — every
/// known field has non-negative semantics, but we preserve the raw
/// little-endian bit pattern so that downstream code may interpret bits
/// however the spec dictates per field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeexHeader {
    /// Bitstream magic. Exactly `b"Speex   "` (8 bytes, 3 trailing spaces).
    pub speex_string: [u8; SPEEX_STRING_LEN],

    /// Encoder version string. The spec says: *"the next field
    /// speex_version contains the version of Speex that encoded the
    /// file."* Padded with NUL bytes; not required to be NUL-terminated.
    pub speex_version: [u8; SPEEX_VERSION_LEN],

    /// Numeric form of the encoder version.
    pub speex_version_id: u32,

    /// Self-declared header size in bytes (informational — the parser
    /// uses [`SPEEX_HEADER_LEN`] as the authoritative length).
    pub header_size: u32,

    /// Sampling rate in Hz. Per RFC 5574 §3 and the manual, one of
    /// 8 000, 16 000, or 32 000.
    pub rate: u32,

    /// Mode identifier — see [`SPEEX_MODE_NARROWBAND`],
    /// [`SPEEX_MODE_WIDEBAND`], [`SPEEX_MODE_ULTRAWIDEBAND`].
    pub mode: u32,

    /// Mode bitstream version. Allows the decoder to refuse streams
    /// produced by an encoder revision it doesn't understand.
    pub mode_bitstream_version: u32,

    /// Number of channels. Spec restricts to 1 (mono); RFC 5574 §3
    /// also says *"This specification defines only single channel
    /// audio (mono)"* but we surface the field as the encoder wrote it.
    pub nb_channels: u32,

    /// Nominal bit-rate in bit/s, or 0 for VBR streams.
    pub bitrate: u32,

    /// Frame size in samples per channel (160 for narrowband, 320 for
    /// wideband, 640 for ultra-wideband per manual §2.1).
    pub frame_size: u32,

    /// VBR flag. 1 = variable bit-rate, 0 = constant.
    pub vbr: u32,

    /// Number of Speex frames packed into each Ogg packet.
    pub frames_per_packet: u32,

    /// Number of extra-header packets following this one before the
    /// audio packets start (always 0 in baseline streams).
    pub extra_headers: u32,

    /// Reserved field 1 — MUST be ignored on read; preserved verbatim
    /// so a future re-encode can round-trip the exact byte pattern.
    pub reserved1: u32,

    /// Reserved field 2 — MUST be ignored on read; preserved verbatim.
    pub reserved2: u32,
}

impl SpeexHeader {
    /// Parse a fixed-layout Ogg/Speex header packet.
    ///
    /// Returns `Err(HeaderError::TooShort(_))` if the input is shorter
    /// than [`SPEEX_HEADER_LEN`], and `Err(HeaderError::BadMagic(_))` if
    /// the leading 8 bytes don't equal [`SPEEX_MAGIC`]. Trailing bytes
    /// past [`SPEEX_HEADER_LEN`] are ignored (the Ogg page may carry
    /// padding past the header struct).
    pub fn parse(buf: &[u8]) -> Result<Self, HeaderError> {
        if buf.len() < SPEEX_HEADER_LEN {
            return Err(HeaderError::TooShort(buf.len()));
        }

        let mut speex_string = [0u8; SPEEX_STRING_LEN];
        speex_string.copy_from_slice(&buf[0..SPEEX_STRING_LEN]);
        if &speex_string != SPEEX_MAGIC {
            return Err(HeaderError::BadMagic(speex_string));
        }

        let mut speex_version = [0u8; SPEEX_VERSION_LEN];
        speex_version.copy_from_slice(&buf[SPEEX_STRING_LEN..SPEEX_STRING_LEN + SPEEX_VERSION_LEN]);

        // After the two char[] fields, thirteen consecutive little-endian
        // 32-bit integers follow in the order given by Table 7.1.
        let mut off = SPEEX_STRING_LEN + SPEEX_VERSION_LEN;
        let mut read_u32_le = || -> u32 {
            let v = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
            off += 4;
            v
        };

        let speex_version_id = read_u32_le();
        let header_size = read_u32_le();
        let rate = read_u32_le();
        let mode = read_u32_le();
        let mode_bitstream_version = read_u32_le();
        let nb_channels = read_u32_le();
        let bitrate = read_u32_le();
        let frame_size = read_u32_le();
        let vbr = read_u32_le();
        let frames_per_packet = read_u32_le();
        let extra_headers = read_u32_le();
        let reserved1 = read_u32_le();
        let reserved2 = read_u32_le();

        Ok(Self {
            speex_string,
            speex_version,
            speex_version_id,
            header_size,
            rate,
            mode,
            mode_bitstream_version,
            nb_channels,
            bitrate,
            frame_size,
            vbr,
            frames_per_packet,
            extra_headers,
            reserved1,
            reserved2,
        })
    }

    /// Serialise the header to its fixed Table 7.1 wire layout — the
    /// exact inverse of [`SpeexHeader::parse`] (8-byte magic, 20-byte
    /// version string, thirteen little-endian 32-bit fields, 80 bytes
    /// total). `parse(write_bytes(h)) == h` for every header whose
    /// `speex_string` is the [`SPEEX_MAGIC`]; the stored `speex_string`
    /// is written verbatim, so a hand-built header with a wrong magic
    /// round-trips to the same parse error a foreign stream would get.
    pub fn write_bytes(&self) -> [u8; SPEEX_HEADER_LEN] {
        let mut out = [0u8; SPEEX_HEADER_LEN];
        out[0..SPEEX_STRING_LEN].copy_from_slice(&self.speex_string);
        out[SPEEX_STRING_LEN..SPEEX_STRING_LEN + SPEEX_VERSION_LEN]
            .copy_from_slice(&self.speex_version);
        let mut off = SPEEX_STRING_LEN + SPEEX_VERSION_LEN;
        for v in [
            self.speex_version_id,
            self.header_size,
            self.rate,
            self.mode,
            self.mode_bitstream_version,
            self.nb_channels,
            self.bitrate,
            self.frame_size,
            self.vbr,
            self.frames_per_packet,
            self.extra_headers,
            self.reserved1,
            self.reserved2,
        ] {
            out[off..off + 4].copy_from_slice(&v.to_le_bytes());
            off += 4;
        }
        out
    }

    /// Return the `speex_version` field as a `&str` trimmed of trailing
    /// NUL padding. Returns `None` if the field is not valid UTF-8.
    pub fn version_str(&self) -> Option<&str> {
        let end = self
            .speex_version
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(SPEEX_VERSION_LEN);
        core::str::from_utf8(&self.speex_version[..end]).ok()
    }

    /// Convenience: returns `true` when the `mode` field matches one of
    /// the three modes documented in the spec (NB/WB/UWB).
    pub fn is_known_mode(&self) -> bool {
        matches!(
            self.mode,
            SPEEX_MODE_NARROWBAND | SPEEX_MODE_WIDEBAND | SPEEX_MODE_ULTRAWIDEBAND
        )
    }

    /// `true` for a narrowband-mode stream (`mode == 0`, 8 kHz).
    pub fn is_narrowband(&self) -> bool {
        self.mode == SPEEX_MODE_NARROWBAND
    }

    /// `true` for a wideband-mode stream (`mode == 1`, 16 kHz).
    pub fn is_wideband(&self) -> bool {
        self.mode == SPEEX_MODE_WIDEBAND
    }

    /// `true` for an ultra-wideband-mode stream (`mode == 2`, 32 kHz).
    pub fn is_ultrawideband(&self) -> bool {
        self.mode == SPEEX_MODE_ULTRAWIDEBAND
    }

    /// The **canonical** output sampling rate the `mode` class implies,
    /// in Hz — narrowband `8000`, wideband `16000`, ultra-wideband
    /// `32000` (manual §2.2 "Embedded wideband structure": each sub-band
    /// layer doubles the rate; §7.3). Returns `None` for an unknown mode.
    ///
    /// This is derived from the *mode class*, independent of the
    /// self-declared [`SpeexHeader::rate`] field, so a consumer can
    /// cross-check the two (a conformant header has `rate ==
    /// mode_sampling_rate_hz()`) or default the playback rate from the
    /// mode alone. It matches the per-frame full-rate output the
    /// [`crate::SpeexDecoder`] produces ([`crate::DecodedFrame::sample_rate_hz`])
    /// for the narrowband / wideband decode paths.
    pub fn mode_sampling_rate_hz(&self) -> Option<u32> {
        match self.mode {
            SPEEX_MODE_NARROWBAND => Some(8_000),
            SPEEX_MODE_WIDEBAND => Some(16_000),
            SPEEX_MODE_ULTRAWIDEBAND => Some(32_000),
            _ => None,
        }
    }

    /// `true` when the self-declared [`SpeexHeader::rate`] field agrees
    /// with the rate the `mode` class implies
    /// ([`SpeexHeader::mode_sampling_rate_hz`]). Always `false` for an
    /// unknown mode (no canonical rate to compare against).
    pub fn rate_matches_mode(&self) -> bool {
        self.mode_sampling_rate_hz() == Some(self.rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic header buffer with the given mode / rate /
    /// frame_size and otherwise reasonable defaults.
    fn synth_header(rate: u32, mode: u32, frame_size: u32, channels: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(SPEEX_HEADER_LEN);
        v.extend_from_slice(SPEEX_MAGIC);
        // speex_version: "speex-1.2beta3" padded with NULs.
        let ver = b"speex-1.2beta3";
        v.extend_from_slice(ver);
        v.resize(SPEEX_STRING_LEN + SPEEX_VERSION_LEN, 0);
        // 14 LE int32 fields:
        for x in [
            1u32,       // speex_version_id
            80u32,      // header_size
            rate,       // rate
            mode,       // mode
            4u32,       // mode_bitstream_version
            channels,   // nb_channels
            0u32,       // bitrate (0 = VBR or unspecified)
            frame_size, // frame_size
            0u32,       // vbr
            1u32,       // frames_per_packet
            0u32,       // extra_headers
            0u32,       // reserved1
            0u32,       // reserved2
        ] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        debug_assert_eq!(v.len(), SPEEX_HEADER_LEN);
        v
    }

    #[test]
    fn parses_narrowband_header() {
        // Per manual §2.1, narrowband uses 8 kHz / 160-sample frames.
        let buf = synth_header(8000, SPEEX_MODE_NARROWBAND, 160, 1);
        let h = SpeexHeader::parse(&buf).expect("must parse");
        assert_eq!(&h.speex_string, SPEEX_MAGIC);
        assert_eq!(h.rate, 8000);
        assert_eq!(h.mode, SPEEX_MODE_NARROWBAND);
        assert_eq!(h.frame_size, 160);
        assert_eq!(h.nb_channels, 1);
        assert_eq!(h.frames_per_packet, 1);
        assert!(h.is_known_mode());
        assert_eq!(h.version_str(), Some("speex-1.2beta3"));
    }

    #[test]
    fn parses_wideband_header() {
        // Wideband: 16 kHz / 320-sample frames.
        let buf = synth_header(16000, SPEEX_MODE_WIDEBAND, 320, 1);
        let h = SpeexHeader::parse(&buf).expect("must parse");
        assert_eq!(h.rate, 16000);
        assert_eq!(h.mode, SPEEX_MODE_WIDEBAND);
        assert_eq!(h.frame_size, 320);
        assert!(h.is_known_mode());
    }

    #[test]
    fn parses_ultrawideband_header() {
        // Ultra-wideband: 32 kHz / 640-sample frames.
        let buf = synth_header(32000, SPEEX_MODE_ULTRAWIDEBAND, 640, 1);
        let h = SpeexHeader::parse(&buf).expect("must parse");
        assert_eq!(h.rate, 32000);
        assert_eq!(h.mode, SPEEX_MODE_ULTRAWIDEBAND);
        assert_eq!(h.frame_size, 640);
        assert!(h.is_known_mode());
    }

    #[test]
    fn rejects_short_buffer() {
        let buf = vec![0u8; SPEEX_HEADER_LEN - 1];
        match SpeexHeader::parse(&buf) {
            Err(HeaderError::TooShort(n)) => assert_eq!(n, SPEEX_HEADER_LEN - 1),
            other => panic!("expected TooShort, got {:?}", other),
        }
    }

    #[test]
    fn rejects_bad_magic() {
        // Replace the first byte with a wrong character; the rest of
        // the buffer is irrelevant because magic check fires first.
        let mut buf = synth_header(8000, SPEEX_MODE_NARROWBAND, 160, 1);
        buf[0] = b'X';
        match SpeexHeader::parse(&buf) {
            Err(HeaderError::BadMagic(m)) => assert_eq!(m[0], b'X'),
            other => panic!("expected BadMagic, got {:?}", other),
        }
    }

    #[test]
    fn little_endian_field_order_is_table_7_1() {
        // Hand-craft a buffer with each int32 field set to its 1-indexed
        // position so we can confirm the parser maps slot -> field
        // exactly as Table 7.1 dictates. Byte order: LE.
        let mut buf = Vec::with_capacity(SPEEX_HEADER_LEN);
        buf.extend_from_slice(SPEEX_MAGIC);
        buf.extend_from_slice(&[0u8; SPEEX_VERSION_LEN]);
        for slot in 1u32..=13 {
            buf.extend_from_slice(&slot.to_le_bytes());
        }
        assert_eq!(buf.len(), SPEEX_HEADER_LEN);

        let h = SpeexHeader::parse(&buf).expect("must parse");
        // Table 7.1 ordering: slot N (1-indexed) maps onto the Nth
        // int32 field. The parser must consume all 13 fields.
        assert_eq!(h.speex_version_id, 1);
        assert_eq!(h.header_size, 2);
        assert_eq!(h.rate, 3);
        assert_eq!(h.mode, 4);
        assert_eq!(h.mode_bitstream_version, 5);
        assert_eq!(h.nb_channels, 6);
        assert_eq!(h.bitrate, 7);
        assert_eq!(h.frame_size, 8);
        assert_eq!(h.vbr, 9);
        assert_eq!(h.frames_per_packet, 10);
        assert_eq!(h.extra_headers, 11);
        assert_eq!(h.reserved1, 12);
        assert_eq!(h.reserved2, 13);
        // Sanity-check the header length math against Table 7.1.
        assert_eq!(SPEEX_HEADER_LEN, 8 + 20 + 13 * 4);
    }

    #[test]
    fn ignores_trailing_bytes() {
        // The Ogg page may carry padding past the 84-byte struct; the
        // parser should accept any input >= SPEEX_HEADER_LEN.
        let mut buf = synth_header(8000, SPEEX_MODE_NARROWBAND, 160, 1);
        buf.extend_from_slice(&[0xAB; 32]);
        let h = SpeexHeader::parse(&buf).expect("must parse");
        assert_eq!(h.rate, 8000);
    }

    #[test]
    fn mode_class_predicates_and_canonical_rate() {
        let nb = SpeexHeader::parse(&synth_header(8000, SPEEX_MODE_NARROWBAND, 160, 1)).unwrap();
        assert!(nb.is_narrowband() && !nb.is_wideband() && !nb.is_ultrawideband());
        assert_eq!(nb.mode_sampling_rate_hz(), Some(8_000));
        assert!(nb.rate_matches_mode());

        let wb = SpeexHeader::parse(&synth_header(16000, SPEEX_MODE_WIDEBAND, 320, 1)).unwrap();
        assert!(wb.is_wideband() && !wb.is_narrowband() && !wb.is_ultrawideband());
        assert_eq!(wb.mode_sampling_rate_hz(), Some(16_000));
        assert!(wb.rate_matches_mode());

        let uwb =
            SpeexHeader::parse(&synth_header(32000, SPEEX_MODE_ULTRAWIDEBAND, 640, 1)).unwrap();
        assert!(uwb.is_ultrawideband() && !uwb.is_narrowband() && !uwb.is_wideband());
        assert_eq!(uwb.mode_sampling_rate_hz(), Some(32_000));
        assert!(uwb.rate_matches_mode());
    }

    #[test]
    fn unknown_mode_has_no_canonical_rate() {
        // mode 7 is not one of the three documented classes.
        let h = SpeexHeader::parse(&synth_header(8000, 7, 160, 1)).unwrap();
        assert!(!h.is_known_mode());
        assert_eq!(h.mode_sampling_rate_hz(), None);
        assert!(!h.rate_matches_mode());
    }

    #[test]
    fn rate_field_disagreeing_with_mode_is_flagged() {
        // A header whose self-declared rate contradicts its mode class
        // (16 kHz rate but narrowband mode) is detected by rate_matches_mode.
        let h = SpeexHeader::parse(&synth_header(16000, SPEEX_MODE_NARROWBAND, 160, 1)).unwrap();
        assert!(h.is_narrowband());
        assert_eq!(h.mode_sampling_rate_hz(), Some(8_000));
        assert!(!h.rate_matches_mode(), "rate 16000 contradicts NB mode");
    }

    #[test]
    fn write_bytes_is_the_exact_parse_inverse() {
        // Byte-for-byte round trip on every rate class (write ∘ parse =
        // identity on the 80-byte header, parse ∘ write = identity on
        // the struct).
        for (rate, mode, frame_size) in [
            (8_000u32, SPEEX_MODE_NARROWBAND, 160u32),
            (16_000, SPEEX_MODE_WIDEBAND, 320),
            (32_000, SPEEX_MODE_ULTRAWIDEBAND, 640),
        ] {
            let buf = synth_header(rate, mode, frame_size, 1);
            let h = SpeexHeader::parse(&buf).unwrap();
            let out = h.write_bytes();
            assert_eq!(&out[..], &buf[..SPEEX_HEADER_LEN], "mode {mode}");
            assert_eq!(SpeexHeader::parse(&out).unwrap(), h, "mode {mode}");
        }
    }
}
