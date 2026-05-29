//! Minimal MSB-first bit reader + writer for the Speex frame header.
//!
//! Speex packs every bit-stream field big-endian-within-byte (the
//! oldest bit appears in the most significant bit of the first byte of
//! the packet). The packing convention is stated in *The Speex Codec
//! Manual* §9.3 ("the parameters are listed in the table in the order
//! they are packed in the bit-stream") and observed throughout the
//! table layouts, where multi-bit codes such as the 4-bit mode ID are
//! written from MSB toward LSB as a single binary integer.
//!
//! Round-2 scope only needed to read fields up to the narrowband frame
//! header — a handful of small unsigned values — so the [`BitReader`]
//! implementation here is deliberately small and free of any general
//! bit-stream machinery: read a single bit, read up to 32 bits, count
//! consumed bits, surface `BitError::Underflow` when the requested span
//! exceeds the buffer.
//!
//! Round 179 (this commit) adds [`BitWriter`], the MSB-first companion
//! that lays bits down in the same big-endian-within-byte convention.
//! It is the symmetric operation of [`BitReader::read`] / [`read_bit`]:
//! a `BitWriter` started from an empty `Vec<u8>` and fed the same
//! `(value, n)` arguments a `BitReader` would emit produces a buffer
//! the `BitReader` then re-reads to the original values. This is the
//! groundwork an encoder needs and immediately retires the
//! cfg-test-only `BitPacker` helper that the `packet` module had been
//! using to assemble synthetic packets.
//!
//! No external library source consulted; the writer's behaviour is
//! defined entirely by the reader's symmetry contract — `BitReader`
//! reads MSB-first within each byte, so `BitWriter` writes MSB-first
//! within each byte.

use core::fmt;

/// Error type for [`BitReader::read`] and [`BitReader::read_bit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitError {
    /// The reader was asked for more bits than the backing slice held.
    ///
    /// Carries the number of bits still requested at the point the
    /// reader ran out, so a caller can correlate the failure to the
    /// field it was decoding.
    Underflow { requested: u32, remaining: u32 },
    /// `read` was called with `n > 32`. The reader returns `u32` so
    /// values wider than 32 bits would silently overflow.
    TooWide(u32),
}

impl fmt::Display for BitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BitError::Underflow {
                requested,
                remaining,
            } => write!(
                f,
                "bit reader underflow: requested {} more bits, only {} remain",
                requested, remaining
            ),
            BitError::TooWide(n) => {
                write!(f, "bit reader cannot read {} bits at once into a u32", n)
            }
        }
    }
}

impl std::error::Error for BitError {}

/// MSB-first bit cursor over a byte slice.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    buf: &'a [u8],
    /// Bit offset from the start of `buf`, counting MSB-first within
    /// each byte (so `pos = 0` is the MSB of `buf[0]`).
    pos: u32,
}

impl<'a> BitReader<'a> {
    /// Wrap a byte slice in a fresh cursor at bit offset 0.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Total number of bits the underlying buffer can supply.
    pub fn capacity_bits(&self) -> u32 {
        // 32 bits suffices: a single Speex packet has never been
        // anywhere near 2^29 bytes long.
        (self.buf.len() as u32).saturating_mul(8)
    }

    /// Number of bits consumed since construction.
    pub fn consumed_bits(&self) -> u32 {
        self.pos
    }

    /// Number of bits remaining ahead of the cursor.
    pub fn remaining_bits(&self) -> u32 {
        self.capacity_bits().saturating_sub(self.pos)
    }

    /// Read a single bit; returns `0` or `1`.
    pub fn read_bit(&mut self) -> Result<u8, BitError> {
        if self.remaining_bits() == 0 {
            return Err(BitError::Underflow {
                requested: 1,
                remaining: 0,
            });
        }
        let byte = self.buf[(self.pos / 8) as usize];
        let bit_in_byte = 7 - (self.pos % 8); // MSB-first
        self.pos += 1;
        Ok((byte >> bit_in_byte) & 1)
    }

    /// Read `n` MSB-first bits as a `u32`. `n` must be in `0..=32`.
    pub fn read(&mut self, n: u32) -> Result<u32, BitError> {
        if n > 32 {
            return Err(BitError::TooWide(n));
        }
        if n == 0 {
            return Ok(0);
        }
        if self.remaining_bits() < n {
            return Err(BitError::Underflow {
                requested: n,
                remaining: self.remaining_bits(),
            });
        }
        let mut acc: u32 = 0;
        for _ in 0..n {
            acc = (acc << 1) | (self.read_bit()? as u32);
        }
        Ok(acc)
    }
}

/// MSB-first bit sink — the symmetric companion to [`BitReader`].
///
/// Maintains an internal `Vec<u8>` that grows one bit at a time. Each
/// `write` / `write_bit` call deposits bits into the next available
/// slot, where "next available" means the bit immediately to the right
/// of the previously-written bit within the same byte (MSB-first), or
/// the MSB of a freshly-appended byte when the current byte is full.
///
/// The contract is: feeding a buffer the same `(value, n)` pairs a
/// `BitReader` would emit from a matching buffer produces a sequence of
/// bytes a `BitReader` would then round-trip back to the original
/// values, in the same order. The round-trip invariant is asserted by
/// the [`bitreader_bitwriter_round_trip`] test below.
///
/// When the last write does not land on a byte boundary, the trailing
/// bits inside the final byte are left as `0` — matching the §5.5
/// convention that the encoder pads the last byte of a packet with the
/// 5-bit mode-15 terminator's all-`1`s prefix or with zero bits after
/// the terminator until byte alignment is restored.
#[derive(Debug, Clone, Default)]
pub struct BitWriter {
    buf: Vec<u8>,
    /// Bit offset from the start of `buf`, counting MSB-first within
    /// each byte (so `bits = 0` is "no bits emitted yet" and the next
    /// `write_bit` lands in the MSB of `buf[0]`).
    bits: u32,
}

impl BitWriter {
    /// Construct a fresh empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a writer with at least `capacity_bytes` of pre-reserved
    /// backing buffer. Useful when the caller knows the final size up
    /// front (e.g. the encoder has already computed `Total / 8` from
    /// the sub-mode table).
    pub fn with_capacity(capacity_bytes: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity_bytes),
            bits: 0,
        }
    }

    /// Total number of bits emitted since construction.
    pub fn bits_written(&self) -> u32 {
        self.bits
    }

    /// Number of bits remaining inside the current (last) byte before
    /// the next write triggers a fresh `buf.push(0)`. Returns `0` when
    /// the writer has never written anything (no current byte yet) or
    /// the current byte is exactly full.
    pub fn bits_left_in_last_byte(&self) -> u32 {
        if self.bits == 0 {
            0
        } else {
            (8 - (self.bits % 8)) % 8
        }
    }

    /// `true` when the cursor is exactly on a byte boundary (the next
    /// write will start a fresh byte).
    pub fn is_byte_aligned(&self) -> bool {
        self.bits % 8 == 0
    }

    /// Write a single bit. `bit != 0` writes a `1`; `bit == 0` writes a
    /// `0`.
    pub fn write_bit(&mut self, bit: u8) -> Result<(), BitError> {
        let byte_idx = (self.bits / 8) as usize;
        if byte_idx == self.buf.len() {
            self.buf.push(0);
        }
        let bit_in_byte = 7 - (self.bits % 8);
        if bit & 1 == 1 {
            self.buf[byte_idx] |= 1 << bit_in_byte;
        }
        self.bits += 1;
        Ok(())
    }

    /// Write the low `n` bits of `value` MSB-first. `n` must be in
    /// `0..=32`; `n == 0` is a no-op. Higher-order bits of `value` are
    /// silently ignored — the writer's contract is "the low `n` bits
    /// land in the bit-stream, MSB-first". A caller that wants a hard
    /// width check should mask its argument before calling.
    pub fn write(&mut self, value: u32, n: u32) -> Result<(), BitError> {
        if n > 32 {
            return Err(BitError::TooWide(n));
        }
        if n == 0 {
            return Ok(());
        }
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.write_bit(bit)?;
        }
        Ok(())
    }

    /// Pad with `0` bits until the cursor reaches the next byte
    /// boundary. After calling, `is_byte_aligned()` is `true` and
    /// further writes start a fresh byte. No-op when already aligned.
    ///
    /// The Speex §5.5 packing convention pads the last byte of a packet
    /// with bits inserted automatically by the encoder around the
    /// mode-15 terminator. This helper is the writer-side equivalent
    /// for callers assembling a packet by hand.
    pub fn pad_to_byte(&mut self) -> Result<(), BitError> {
        let left = self.bits_left_in_last_byte();
        for _ in 0..left {
            self.write_bit(0)?;
        }
        Ok(())
    }

    /// Take ownership of the backing byte buffer.
    ///
    /// The returned vector reflects every bit that was written;
    /// trailing bits inside the final byte that weren't filled by an
    /// explicit write are zero. A caller that needs a known byte total
    /// should call [`Self::pad_to_byte`] before `into_bytes`.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Borrow the backing byte buffer without consuming the writer.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_single_bits_msb_first() {
        // 0b1010_0101 == 0xA5
        let mut r = BitReader::new(&[0xA5]);
        let bits: Vec<u8> = (0..8).map(|_| r.read_bit().unwrap()).collect();
        assert_eq!(bits, vec![1, 0, 1, 0, 0, 1, 0, 1]);
        assert_eq!(r.remaining_bits(), 0);
    }

    #[test]
    fn reads_multibit_msb_first() {
        // 0b1010_0101 0b1100_0011
        let mut r = BitReader::new(&[0xA5, 0xC3]);
        assert_eq!(r.read(4).unwrap(), 0b1010);
        assert_eq!(r.read(4).unwrap(), 0b0101);
        assert_eq!(r.read(8).unwrap(), 0xC3);
        assert_eq!(r.consumed_bits(), 16);
    }

    #[test]
    fn straddles_byte_boundary() {
        // 0b1010_0101 0b1100_0011 — read 6 then 10 should give
        // top-6 of byte1 (0b101001) and next 10 bits (low 2 of byte1
        // + 8 of byte2) == 0b01_1100_0011 == 0x1C3.
        let mut r = BitReader::new(&[0xA5, 0xC3]);
        assert_eq!(r.read(6).unwrap(), 0b101001);
        assert_eq!(r.read(10).unwrap(), 0b0111000011);
    }

    #[test]
    fn underflow_is_diagnosed() {
        let mut r = BitReader::new(&[0xFF]);
        assert!(r.read(7).is_ok());
        match r.read(2) {
            Err(BitError::Underflow {
                requested,
                remaining,
            }) => {
                assert_eq!(requested, 2);
                assert_eq!(remaining, 1);
            }
            other => panic!("expected Underflow, got {:?}", other),
        }
    }

    #[test]
    fn read_zero_bits_yields_zero() {
        let mut r = BitReader::new(&[]);
        assert_eq!(r.read(0).unwrap(), 0);
        assert_eq!(r.consumed_bits(), 0);
    }

    #[test]
    fn read_too_wide_is_diagnosed() {
        let mut r = BitReader::new(&[0u8; 8]);
        match r.read(33) {
            Err(BitError::TooWide(33)) => {}
            other => panic!("expected TooWide(33), got {:?}", other),
        }
    }

    #[test]
    fn full_u32_round_trips() {
        let val: u32 = 0xDEAD_BEEF;
        let bytes = val.to_be_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(32).unwrap(), val);
    }

    // ---- BitWriter tests ----

    #[test]
    fn writer_starts_empty() {
        let w = BitWriter::new();
        assert_eq!(w.bits_written(), 0);
        assert!(w.as_bytes().is_empty());
        assert!(w.is_byte_aligned());
        assert_eq!(w.bits_left_in_last_byte(), 0);
    }

    #[test]
    fn writes_single_bits_msb_first() {
        // Write 1, 0, 1, 0, 0, 1, 0, 1 → byte should be 0b1010_0101 == 0xA5.
        let mut w = BitWriter::new();
        for &b in &[1u8, 0, 1, 0, 0, 1, 0, 1] {
            w.write_bit(b).unwrap();
        }
        assert_eq!(w.as_bytes(), &[0xA5]);
        assert_eq!(w.bits_written(), 8);
        assert!(w.is_byte_aligned());
    }

    #[test]
    fn writes_multibit_msb_first() {
        // Write 0b1010 then 0b0101 then 0xC3 — should pack as
        //   byte 0 = 0xA5, byte 1 = 0xC3 — i.e. mirror the
        //   reader's `reads_multibit_msb_first` exactly.
        let mut w = BitWriter::new();
        w.write(0b1010, 4).unwrap();
        w.write(0b0101, 4).unwrap();
        w.write(0xC3, 8).unwrap();
        assert_eq!(w.as_bytes(), &[0xA5, 0xC3]);
        assert_eq!(w.bits_written(), 16);
    }

    #[test]
    fn writer_straddles_byte_boundary() {
        // Symmetric of the reader's straddle test: write 6 bits of
        // 0b101001 then 10 bits of 0b0111000011 should land as
        // 0xA5, 0xC3 — same bytes the reader splits the other way.
        let mut w = BitWriter::new();
        w.write(0b101001, 6).unwrap();
        w.write(0b0111000011, 10).unwrap();
        assert_eq!(w.as_bytes(), &[0xA5, 0xC3]);
    }

    #[test]
    fn writer_zero_width_is_noop() {
        let mut w = BitWriter::new();
        w.write(0xDEAD_BEEF, 0).unwrap();
        assert_eq!(w.bits_written(), 0);
        assert!(w.as_bytes().is_empty());
    }

    #[test]
    fn writer_too_wide_is_diagnosed() {
        let mut w = BitWriter::new();
        match w.write(0, 33) {
            Err(BitError::TooWide(33)) => {}
            other => panic!("expected TooWide(33), got {:?}", other),
        }
    }

    #[test]
    fn writer_full_u32() {
        let val: u32 = 0xDEAD_BEEF;
        let mut w = BitWriter::new();
        w.write(val, 32).unwrap();
        assert_eq!(w.as_bytes(), &val.to_be_bytes());
        assert_eq!(w.bits_written(), 32);
    }

    #[test]
    fn writer_pad_to_byte_rounds_up_with_zeros() {
        // Five bits = 0b1_1111 → byte 0b1111_1000 = 0xF8 after padding.
        let mut w = BitWriter::new();
        w.write(0b11111, 5).unwrap();
        assert_eq!(w.bits_left_in_last_byte(), 3);
        assert!(!w.is_byte_aligned());
        w.pad_to_byte().unwrap();
        assert!(w.is_byte_aligned());
        assert_eq!(w.as_bytes(), &[0xF8]);
        assert_eq!(w.bits_written(), 8);
        // Pad on an already-aligned writer is a no-op.
        w.pad_to_byte().unwrap();
        assert_eq!(w.bits_written(), 8);
    }

    #[test]
    fn writer_ignores_high_bits_above_n() {
        // value = 0xFF, n = 3 → only the low 3 bits (0b111) land.
        let mut w = BitWriter::new();
        w.write(0xFF, 3).unwrap();
        w.pad_to_byte().unwrap();
        assert_eq!(w.as_bytes(), &[0b1110_0000]);
    }

    #[test]
    fn bits_left_in_last_byte_tracks_cursor() {
        let mut w = BitWriter::new();
        assert_eq!(w.bits_left_in_last_byte(), 0);
        w.write_bit(1).unwrap();
        assert_eq!(w.bits_left_in_last_byte(), 7);
        w.write(0, 4).unwrap();
        assert_eq!(w.bits_left_in_last_byte(), 3);
        w.write(0, 3).unwrap();
        assert!(w.is_byte_aligned());
        assert_eq!(w.bits_left_in_last_byte(), 0);
    }

    #[test]
    fn with_capacity_preallocates_buffer() {
        // No public way to query Vec capacity through a slice handle;
        // verify the constructor does not yet hold any bits and the
        // backing buffer is empty until something is written.
        let mut w = BitWriter::with_capacity(16);
        assert_eq!(w.bits_written(), 0);
        assert!(w.as_bytes().is_empty());
        // Write exactly 16 bytes' worth and confirm length matches.
        for _ in 0..16 {
            w.write(0xAB, 8).unwrap();
        }
        assert_eq!(w.as_bytes().len(), 16);
    }

    // ---- Round-trip invariants ----

    #[test]
    fn bitreader_bitwriter_round_trip_short_sequence() {
        // The defining contract: feed the writer the same (value, n)
        // triples the reader would emit from a buffer, and the reader
        // re-parsing the writer's output recovers the same values.
        let plan: &[(u32, u32)] = &[
            (1, 1),      // wideband flag
            (0b0101, 4), // mode_id
            (0xA5, 8),
            (0b1101, 4),
            (0, 0), // zero-width should not advance
            (0xDEAD_BEEF, 32),
        ];
        let mut w = BitWriter::new();
        for (v, n) in plan.iter().copied() {
            w.write(v, n).unwrap();
        }
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        for (v, n) in plan.iter().copied() {
            assert_eq!(r.read(n).unwrap(), v, "mismatch at width {}", n);
        }
    }

    #[test]
    fn bitreader_bitwriter_round_trip_per_bit() {
        // Per-bit round trip: writing bits 1, 1, 0, 1, 1, 0, 0, 1 and
        // reading them back yields the same sequence. This is the
        // tightest possible MSB-first contract.
        let pattern = [1u8, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1];
        let mut w = BitWriter::new();
        for &b in &pattern {
            w.write_bit(b).unwrap();
        }
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        for &expected in &pattern {
            assert_eq!(r.read_bit().unwrap(), expected);
        }
    }

    #[test]
    fn bitreader_bitwriter_round_trip_long_random_pattern() {
        // 256 small values of varied widths — round-trip every (v, n)
        // pair through writer then reader. Uses a deterministic LCG so
        // the test is reproducible without `rand`.
        let mut state: u32 = 0xC0DE_BAD1;
        let mut plan = Vec::new();
        for _ in 0..256 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let n = (state.rotate_right(13) % 17) + 1; // 1..=17 bits
            let value = (state >> (32 - n.min(32))).reverse_bits() & ((1u32 << n) - 1);
            plan.push((value, n));
        }
        let mut w = BitWriter::new();
        for (v, n) in plan.iter().copied() {
            w.write(v, n).unwrap();
        }
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        for (v, n) in plan.iter().copied() {
            assert_eq!(r.read(n).unwrap(), v, "round-trip mismatch at width {}", n);
        }
    }

    #[test]
    fn writer_then_reader_consumes_exact_bits_written() {
        let mut w = BitWriter::new();
        w.write(0b10110, 5).unwrap();
        w.write(0b1010_1010, 8).unwrap();
        let total = w.bits_written();
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(5).unwrap(), 0b10110);
        assert_eq!(r.read(8).unwrap(), 0b1010_1010);
        assert_eq!(r.consumed_bits(), total);
    }
}
