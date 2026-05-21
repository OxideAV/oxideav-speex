//! Minimal MSB-first bit reader for the Speex frame header.
//!
//! Speex packs every bit-stream field big-endian-within-byte (the
//! oldest bit appears in the most significant bit of the first byte of
//! the packet). The packing convention is stated in *The Speex Codec
//! Manual* §9.3 ("the parameters are listed in the table in the order
//! they are packed in the bit-stream") and observed throughout the
//! table layouts, where multi-bit codes such as the 4-bit mode ID are
//! written from MSB toward LSB as a single binary integer.
//!
//! Round-2 scope only needs to read fields up to the narrowband frame
//! header — a handful of small unsigned values — so the implementation
//! here is deliberately small and free of any general bit-stream
//! machinery: read a single bit, read up to 32 bits, count consumed
//! bits, surface `BitError::Underflow` when the requested span exceeds
//! the buffer.

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
}
