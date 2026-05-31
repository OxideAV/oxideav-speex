//! Narrowband CELP frame-body bit-reader (round 3 scope).
//!
//! Given a [`BitReader`] positioned immediately after the 5-bit Speex
//! frame prefix (the 1-bit wideband flag + 4-bit mode ID parsed by
//! [`crate::frame::NarrowbandFrameHeader`]) and the resolved
//! [`NarrowbandSubmode`] for that mode ID, this module consumes the
//! per-field bit budget recorded in *The Speex Codec Manual*
//! Table 9.1 ("Bit allocation for narrowband modes") and surfaces every
//! field as a raw bit-index. No codebook lookup, no LSP→LPC conversion,
//! no synthesis filtering yet — those rely on companion tables (see
//! "Spec gap" at the bottom of the module docs) which are not staged in
//! `docs/audio/speex/` at the time of this round.
//!
//! ## Bit-stream order (§9.3 + Table 9.1)
//!
//! The manual is explicit about the packing order: *"The parameters
//! are listed in the table in the order they are packed in the
//! bit-stream. All frame-based parameters are packed before sub-frame
//! parameters. The parameters for a certain sub-frame are all packed
//! before the following sub-frame is packed."*
//!
//! After the 5-bit prefix, then, the layout is:
//!
//! ```text
//! +---------------- frame-level ----------------+
//! | LSP index            (0 / 18 / 30 bits)     |
//! | OL pitch period      (0 / 7 bits)           |
//! | OL pitch gain        (0 / 4 bits)           |
//! | OL excitation gain   (0 / 5 bits)           |
//! +---------------------------------------------+
//! +---------- sub-frame 0 (5 ms / 40 samples) --+
//! | Fine pitch period    (0 / 7 bits)           |
//! | Pitch-gain VQ index  (0 / 5 / 7 bits)       |
//! | Innovation gain      (0 / 1 / 3 bits)       |
//! | Innovation VQ index  (0..=96 bits)          |
//! +---------------------------------------------+
//! …repeated for sub-frames 1, 2, 3…
//! ```
//!
//! The exact bit counts come straight from the sub-mode's
//! [`NarrowbandSubmode`] record built in round 2; this module is the
//! routine that *uses* those counts to walk the bit-stream.
//!
//! ## Pitch period range (§9.2)
//!
//! The manual notes: *"In most modes, the pitch period is encoded with
//! 7 bits in the [17, 144] range"*. The 7-bit field therefore encodes
//! `pitch_period - 17`, i.e. a value in `0..=127`. This module surfaces
//! the raw 7-bit value as [`NarrowbandSubFrameIndices::pitch_index`]
//! and additionally exposes the de-biased period as
//! [`NarrowbandSubFrameIndices::pitch_period`] when the sub-mode
//! actually carries a fine pitch field.
//!
//! ## Silence (mode 0)
//!
//! Mode 0 (250 bps) has a zero-bit body — the 5-bit prefix is the
//! entire frame. [`NarrowbandFrameBody::parse`] still walks the
//! sub-frame loop in this case but every bit-read is a no-op (`read(0)`
//! per the round-2 [`BitReader`] guarantee returns `Ok(0)` without
//! advancing the cursor).
//!
//! ## Spec gap (forwarded to round 4+)
//!
//! Mapping `lsp_index` → ten LSP coefficients, and `innovation_vq_index`
//! → a 40-sample sub-vector innovation signal, both require the Speex
//! CELP **companion table** files (LSP VQ codebooks, innovation
//! codebooks, pitch-gain VQ codebooks). Those tables are not yet staged
//! under `docs/audio/speex/` for clean-room transcription (tracked as
//! #969). Until they are, this module stops at the raw bit-index layer.

use crate::bitreader::{BitError, BitReader};
use crate::codebooks::NB_LSP_ORDER;
use crate::lsp::{reconstruct_q10, NbLspStages};
use crate::lsp_interp::NbSubFrameLsp;
use crate::submode::{LspQuant, NarrowbandSubmode, PitchGainQuant};
use core::fmt;

/// Pitch-period bias from *The Speex Codec Manual* §9.2: *"the pitch
/// period is encoded with 7 bits in the [17, 144] range"*. The 7-bit
/// field therefore stores `period - PITCH_PERIOD_MIN`.
pub const PITCH_PERIOD_MIN: u16 = 17;

/// Inclusive upper bound for the pitch period (from §9.2 [17, 144]).
pub const PITCH_PERIOD_MAX: u16 = 144;

/// Errors returned from [`NarrowbandFrameBody::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrowbandBodyError {
    /// The bit-stream ran out while consuming one of the per-field
    /// budgets. Carries the underlying [`BitError`] for diagnostics.
    Underflow(BitError),
}

impl fmt::Display for NarrowbandBodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NarrowbandBodyError::Underflow(e) => {
                write!(f, "narrowband body underflow: {}", e)
            }
        }
    }
}

impl std::error::Error for NarrowbandBodyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NarrowbandBodyError::Underflow(e) => Some(e),
        }
    }
}

impl From<BitError> for NarrowbandBodyError {
    fn from(e: BitError) -> Self {
        NarrowbandBodyError::Underflow(e)
    }
}

/// Per-sub-frame raw bit-indices, in the order they were unpacked from
/// the bit-stream (`fine_pitch`, `pitch_gain`, `innovation_gain`,
/// `innovation_vq` — see module docs).
///
/// Width of each field depends on the active sub-mode; absent fields
/// store `0` and the corresponding "_bits" field on the originating
/// [`NarrowbandSubmode`] is also `0`. The struct intentionally records
/// only the index integers; codebook lookup is deferred until the
/// companion-table docs gap closes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NarrowbandSubFrameIndices {
    /// Raw 7-bit fine-pitch field, or `0` when the sub-mode skips it.
    /// Per §9.2 the actual period is `pitch_index + 17` when the field
    /// is present; see [`Self::pitch_period`].
    pub pitch_index: u8,
    /// 5- or 7-bit pitch-gain VQ codebook index. `0` if absent.
    pub pitch_gain_index: u8,
    /// 0/1/3-bit innovation-gain correction. `0` if absent.
    pub innovation_gain_index: u8,
    /// Raw innovation-VQ bits, treated as a single opaque value of up
    /// to 96 bits per sub-frame. Decoded into a sub-vector index list
    /// once the innovation codebook tables are staged.
    pub innovation_vq_index: u128,
}

impl NarrowbandSubFrameIndices {
    /// De-biased pitch period per §9.2 (`pitch_index + 17`) when the
    /// sub-mode carries a fine pitch field. Returns `None` if the
    /// `fine_pitch_bits` budget for the sub-mode is `0` (i.e. the
    /// fine pitch was not transmitted for this sub-frame).
    pub fn pitch_period(&self, submode: &NarrowbandSubmode) -> Option<u16> {
        if submode.fine_pitch_bits == 0 {
            None
        } else {
            Some(u16::from(self.pitch_index) + PITCH_PERIOD_MIN)
        }
    }
}

/// Frame-level + four sub-frame raw bit-indices, in the order they
/// were unpacked from the bit-stream per Table 9.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NarrowbandFrameBody {
    /// Raw LSP VQ index — 0 / 18 / 30 bits per [`NarrowbandSubmode::lsp`].
    /// Stored as `u32` to fit the widest documented case (30 bits).
    pub lsp_index: u32,
    /// 7-bit open-loop pitch period, or `0` when the sub-mode skips it.
    /// When present, the actual period is `ol_pitch_index + 17` (§9.2).
    pub ol_pitch_index: u8,
    /// 4-bit open-loop pitch-gain index, or `0` when absent.
    pub ol_pitch_gain_index: u8,
    /// 5-bit open-loop excitation-gain index, or `0` for silence.
    pub ol_exc_gain_index: u8,
    /// Per-sub-frame indices for sub-frames 0..=3.
    pub subframes: [NarrowbandSubFrameIndices; NarrowbandSubmode::SUBFRAMES_PER_FRAME as usize],
}

impl NarrowbandFrameBody {
    /// Consume the frame body for the given resolved `submode` from a
    /// [`BitReader`] that has already had its 5-bit frame prefix
    /// stripped. Returns the parsed indices and leaves the cursor
    /// positioned immediately after the last innovation-VQ field of
    /// sub-frame 3.
    pub fn parse(
        reader: &mut BitReader<'_>,
        submode: &NarrowbandSubmode,
    ) -> Result<Self, NarrowbandBodyError> {
        // Frame-level fields, in Table 9.1 order.
        let lsp_bits = match submode.lsp {
            LspQuant::None => 0,
            LspQuant::Bits18 => 18,
            LspQuant::Bits30 => 30,
        };
        let lsp_index = reader.read(lsp_bits)?;
        let ol_pitch_index = reader.read(u32::from(submode.ol_pitch_bits))? as u8;
        let ol_pitch_gain_index = reader.read(u32::from(submode.ol_pitch_gain_bits))? as u8;
        let ol_exc_gain_index = reader.read(u32::from(submode.ol_exc_gain_bits))? as u8;

        // Sub-frame fields, four times, in Table 9.1 order.
        let pitch_gain_bits = match submode.pitch_gain {
            PitchGainQuant::None => 0,
            PitchGainQuant::Vq5Bit => 5,
            PitchGainQuant::Vq7Bit => 7,
        };
        let mut subframes =
            [NarrowbandSubFrameIndices::default(); NarrowbandSubmode::SUBFRAMES_PER_FRAME as usize];
        for sf in subframes.iter_mut() {
            let pitch_index = reader.read(u32::from(submode.fine_pitch_bits))? as u8;
            let pitch_gain_index = reader.read(pitch_gain_bits)? as u8;
            let innovation_gain_index = reader.read(u32::from(submode.innovation_gain_bits))? as u8;
            let innovation_vq_index = read_wide(reader, u32::from(submode.innovation_vq_bits))?;
            *sf = NarrowbandSubFrameIndices {
                pitch_index,
                pitch_gain_index,
                innovation_gain_index,
                innovation_vq_index,
            };
        }

        Ok(Self {
            lsp_index,
            ol_pitch_index,
            ol_pitch_gain_index,
            ol_exc_gain_index,
            subframes,
        })
    }

    /// Per-frame bit count consumed by this body (excludes the 5-bit
    /// prefix; matches the sub-mode's `total_bits - 5`).
    pub fn body_bits(submode: &NarrowbandSubmode) -> u32 {
        submode.computed_total_bits() - crate::frame::NARROWBAND_FRAME_PREFIX_BITS
    }

    /// Resolve the raw [`Self::lsp_index`] into per-stage 6-bit
    /// codebook indices using the documented stage-ordering
    /// convention (see [`crate::lsp`] module docs). Returns `None`
    /// for sub-modes that do not transmit an LSP field (mode 0 —
    /// silence).
    pub fn lsp_stages(&self, submode: &NarrowbandSubmode) -> Option<NbLspStages> {
        NbLspStages::from_packed(self.lsp_index, submode.lsp)
    }

    /// Reconstruct the ten narrowband LSP frequency coefficients in
    /// Q[`crate::lsp::NB_LSP_OUTPUT_Q`] fixed-point from the parsed
    /// `lsp_index` field. Returns `None` for sub-modes that do not
    /// transmit an LSP field (mode 0 — silence).
    ///
    /// The reconstruction is the typed lookup + per-stage summation
    /// described in [`crate::lsp`]; sub-frame interpolation and
    /// LSP→LPC conversion stay deferred.
    pub fn reconstructed_lsp_q10(
        &self,
        submode: &NarrowbandSubmode,
    ) -> Option<[i32; NB_LSP_ORDER]> {
        let stages = self.lsp_stages(submode)?;
        reconstruct_q10(stages)
    }

    /// Compose the r194 [`Self::reconstructed_lsp_q10`] with the r200
    /// sub-frame LSP interpolation (`crate::lsp_interp`) to produce
    /// one ten-coefficient LSP vector per sub-frame in
    /// Q[`crate::NB_LSP_INTERP_OUTPUT_Q`] = Q12 fixed-point.
    ///
    /// `prev_q10` is the **previous frame's** reconstructed Q10 LSPs
    /// (caller-managed across frames). For the first frame of a
    /// stream, see [`NbSubFrameLsp::first_frame`] — pass the current
    /// frame's LSPs as `prev_q10` to obtain the no-transient
    /// initialisation.
    ///
    /// Returns `None` for sub-modes that transmit no LSP field
    /// (mode 0 — silence; the manual prescribes no LSP update in
    /// that case, so the caller must use its own LSP state).
    pub fn interpolated_lsp_q12(
        &self,
        submode: &NarrowbandSubmode,
        prev_q10: &[i32; NB_LSP_ORDER],
    ) -> Option<NbSubFrameLsp> {
        let curr = self.reconstructed_lsp_q10(submode)?;
        Some(NbSubFrameLsp::new(prev_q10, &curr))
    }
}

/// Read up to 128 bits as a `u128`, chunked through the 32-bit
/// [`BitReader::read`] primitive. Speex narrowband's widest innovation
/// VQ field is 96 bits per sub-frame (mode 7, Table 9.1), so this
/// covers the entire spec range with headroom.
fn read_wide(reader: &mut BitReader<'_>, n: u32) -> Result<u128, BitError> {
    debug_assert!(n <= 128, "innovation VQ exceeds u128 capacity");
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
    use crate::frame::{NarrowbandFrameHeader, NARROWBAND_FRAME_PREFIX_BITS};
    use crate::submode::{NarrowbandSubmode, Submode, NARROWBAND_SUBMODES};

    /// Build a synthetic CELP frame body whose bits are all `1`,
    /// prefixed by a 5-bit header (wideband=0, mode_id=`mode`). Useful
    /// because every read at the maximum field width returns
    /// `(1 << n) - 1`, which makes round-trip math easy to assert.
    fn all_ones_frame(mode: u8) -> Vec<u8> {
        let submode = NarrowbandSubmode::for_id(mode).expect("valid mode");
        let total_bits = u32::from(submode.total_bits);
        let total_bytes = total_bits.div_ceil(8);
        let mut buf = vec![0xFFu8; total_bytes as usize];
        // Zero the wideband flag (high bit) and write mode_id in the
        // next 4 bits; leave everything below the 5-bit prefix as 1s.
        // First byte layout: [w=0][m3 m2 m1 m0][111] then 0xFF tail.
        let prefix_byte = (mode & 0x0F) << 3 | 0b0000_0111;
        buf[0] = prefix_byte;
        buf
    }

    #[test]
    fn silence_mode_has_empty_body() {
        // Mode 0: 5-bit prefix + zero-bit body. After parsing the
        // header from a 1-byte buffer (0b0_0000_xxx), the body parser
        // should not consume any further bits.
        let buf = [0u8; 1];
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        let s = match h.submode {
            Submode::Celp(s) => s,
            other => panic!("expected CELP, got {:?}", other),
        };
        assert_eq!(s.mode_id, 0);
        let before = r.consumed_bits();
        let body = NarrowbandFrameBody::parse(&mut r, &s).expect("must parse");
        assert_eq!(r.consumed_bits(), before, "body must consume zero bits");
        assert_eq!(body.lsp_index, 0);
        assert_eq!(body.ol_pitch_index, 0);
        for sf in &body.subframes {
            assert_eq!(sf.pitch_index, 0);
            assert_eq!(sf.pitch_gain_index, 0);
            assert_eq!(sf.innovation_gain_index, 0);
            assert_eq!(sf.innovation_vq_index, 0);
        }
    }

    #[test]
    fn body_bits_matches_table_total_minus_prefix() {
        // For every documented sub-mode, body_bits should equal
        // total_bits - 5 (the wideband+mode_id prefix is consumed by
        // the header parser, not the body).
        for s in NARROWBAND_SUBMODES {
            let expected = u32::from(s.total_bits) - NARROWBAND_FRAME_PREFIX_BITS;
            assert_eq!(
                NarrowbandFrameBody::body_bits(&s),
                expected,
                "mode {} body_bits mismatch",
                s.mode_id
            );
        }
    }

    #[test]
    fn parse_consumes_exactly_body_bits_for_every_mode() {
        // Cursor delta after a successful parse should equal the
        // sub-mode's `body_bits` for every documented column.
        for mode in 0u8..=8 {
            let buf = all_ones_frame(mode);
            let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
            let s = match h.submode {
                Submode::Celp(s) => s,
                _ => unreachable!(),
            };
            let before = r.consumed_bits();
            let _body = NarrowbandFrameBody::parse(&mut r, &s).expect("must parse");
            assert_eq!(
                r.consumed_bits() - before,
                NarrowbandFrameBody::body_bits(&s),
                "mode {} consumed wrong number of bits",
                mode
            );
        }
    }

    #[test]
    fn mode_5_field_widths_match_table_9_1() {
        // Mode 5 (15 kbps) has all the widest fields filled in. Build
        // a frame body where every bit is 1, then assert each parsed
        // index equals (1 << width) - 1.
        let buf = all_ones_frame(5);
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        let s = match h.submode {
            Submode::Celp(s) => s,
            _ => unreachable!(),
        };
        assert_eq!(s.mode_id, 5);
        let body = NarrowbandFrameBody::parse(&mut r, &s).unwrap();

        // LSP: 30 bits all 1 → 2^30 - 1.
        assert_eq!(body.lsp_index, (1u32 << 30) - 1);
        // OL pitch: 0 bits in mode 5 (fine pitch is per-subframe here).
        assert_eq!(body.ol_pitch_index, 0);
        // OL pitch gain: 0 bits in mode 5.
        assert_eq!(body.ol_pitch_gain_index, 0);
        // OL excitation gain: 5 bits → 0b11111 = 31.
        assert_eq!(body.ol_exc_gain_index, 0b11111);

        for sf in &body.subframes {
            // Fine pitch: 7 bits → 0b1111111 = 127.
            assert_eq!(sf.pitch_index, 0b111_1111);
            // Pitch gain: 7-bit VQ → 127.
            assert_eq!(sf.pitch_gain_index, 0b111_1111);
            // Innovation gain: 3 bits → 7.
            assert_eq!(sf.innovation_gain_index, 7);
            // Innovation VQ: 48 bits → (1 << 48) - 1.
            assert_eq!(sf.innovation_vq_index, (1u128 << 48) - 1);
        }
    }

    #[test]
    fn mode_7_innovation_vq_uses_full_96_bits_per_subframe() {
        // Mode 7 has the widest innovation VQ at 96 bits/sub-frame.
        let buf = all_ones_frame(7);
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        let s = match h.submode {
            Submode::Celp(s) => s,
            _ => unreachable!(),
        };
        let body = NarrowbandFrameBody::parse(&mut r, &s).unwrap();
        for sf in &body.subframes {
            assert_eq!(sf.innovation_vq_index, (1u128 << 96) - 1);
        }
    }

    #[test]
    fn pitch_period_de_biases_when_field_present() {
        // §9.2: "pitch period is encoded with 7 bits in the [17, 144] range"
        let buf = all_ones_frame(3);
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        let s = match h.submode {
            Submode::Celp(s) => s,
            _ => unreachable!(),
        };
        assert!(s.fine_pitch_bits == 7, "mode 3 has 7-bit fine pitch");
        let body = NarrowbandFrameBody::parse(&mut r, &s).unwrap();
        let p = body.subframes[0].pitch_period(&s).expect("present");
        // All-ones index = 127 → period = 127 + 17 = 144 (table max).
        assert_eq!(p, PITCH_PERIOD_MAX);
    }

    #[test]
    fn pitch_period_is_none_when_field_absent() {
        // Mode 1 (2.15 kbps): no fine pitch field per sub-frame
        // (Table 9.1 row "Fine pitch" = 0).
        let s = NarrowbandSubmode::for_id(1).unwrap();
        assert_eq!(s.fine_pitch_bits, 0);
        let sf = NarrowbandSubFrameIndices::default();
        assert_eq!(sf.pitch_period(&s), None);
    }

    #[test]
    fn ol_pitch_bias_documentation_matches_constants() {
        // Sanity check: the public constants from §9.2.
        assert_eq!(PITCH_PERIOD_MIN, 17);
        assert_eq!(PITCH_PERIOD_MAX, 144);
        // The 7-bit field can index [0, 127]; min+127 == max.
        assert_eq!(PITCH_PERIOD_MIN + 127, PITCH_PERIOD_MAX);
    }

    #[test]
    fn underflow_diagnosed_for_truncated_frame() {
        // Mode 5 (15 kbps) needs 300 bits = 38 bytes (37.5 rounded up).
        // Hand it 4 bytes → must underflow somewhere.
        let mut buf = vec![0xFFu8; 4];
        buf[0] = (5u8 & 0x0F) << 3 | 0b0000_0111;
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        let s = match h.submode {
            Submode::Celp(s) => s,
            _ => unreachable!(),
        };
        match NarrowbandFrameBody::parse(&mut r, &s) {
            Err(NarrowbandBodyError::Underflow(_)) => {}
            other => panic!("expected Underflow, got {:?}", other),
        }
    }

    #[test]
    fn parse_preserves_bit_alignment_across_byte_boundaries() {
        // Build a mode-3 frame body where every byte is a different
        // pattern; assert the parser stays bit-aligned by reading the
        // next 0 bits (should not advance) and then asserting cursor
        // position == sum of field widths.
        let s = NarrowbandSubmode::for_id(3).unwrap();
        let total_bytes = u32::from(s.total_bits).div_ceil(8) as usize;
        let mut buf = vec![0u8; total_bytes];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        // Force mode_id=3, wideband=0 in the high 5 bits of byte 0.
        buf[0] = (buf[0] & 0b0000_0111) | ((3u8 & 0x0F) << 3);

        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        let sub = match h.submode {
            Submode::Celp(s) => s,
            _ => unreachable!(),
        };
        let body = NarrowbandFrameBody::parse(&mut r, &sub).unwrap();
        assert_eq!(
            r.consumed_bits(),
            u32::from(sub.total_bits),
            "cursor should be at total_bits after frame parse"
        );
        // Make sure pitch_period de-biases correctly when present.
        let p = body.subframes[0].pitch_period(&sub).unwrap();
        assert!(
            (PITCH_PERIOD_MIN..=PITCH_PERIOD_MAX).contains(&p),
            "pitch period {} outside [17, 144]",
            p
        );
    }

    #[test]
    fn read_wide_chunks_correctly_at_32_bit_boundary() {
        // Two-byte buffer 0xDE 0xAD then 0xBE 0xEF = 0xDEAD_BEEF when
        // read as 32 bits, but read_wide(33) should pull one extra bit
        // from the next byte.
        let bytes = [0xDE, 0xAD, 0xBE, 0xEF, 0x80];
        let mut r = BitReader::new(&bytes);
        let v = read_wide(&mut r, 33).unwrap();
        // Expected: top-32 == 0xDEAD_BEEF, next bit = 1.
        assert_eq!(v, (0xDEAD_BEEFu128 << 1) | 1);
    }
}
