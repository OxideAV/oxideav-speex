//! Narrowband CELP frame packing — the encode-direction inverse of
//! [`crate::narrowband_body::NarrowbandFrameBody::parse`] (round r382
//! scope).
//!
//! Given a fully-populated [`NarrowbandFrameBody`] (the quantised
//! indices produced by the LSP / pitch / innovation encoders) and its
//! resolved [`NarrowbandSubmode`], this module writes the frame onto the
//! wire in the exact Table 9.1 field order the parser consumes: the
//! 5-bit frame prefix (1-bit wideband flag + 4-bit mode ID, §9.3), then
//! the frame-level fields (LSP, OL pitch, OL pitch gain, OL exc gain),
//! then four sub-frames of `fine pitch || pitch gain || innovation gain
//! || innovation VQ` (MAN §9.3: "All frame-based parameters are packed
//! before sub-frame parameters. The parameters for a certain sub-frame
//! are all packed before the following sub-frame is packed.").
//!
//! The round-trip is exact by construction: `parse(write(body)) == body`
//! for any body whose index fields fit their sub-mode's bit budgets.

use crate::bitreader::{BitError, BitWriter};
use crate::frame::{FrameError, NarrowbandFrameHeader};
use crate::narrowband_body::NarrowbandFrameBody;
use crate::submode::{LspQuant, NarrowbandSubmode, PitchGainQuant};

/// Field bit-widths for a narrowband sub-mode, mirroring the parser's
/// per-field reads.
struct FieldWidths {
    lsp: u32,
    ol_pitch: u32,
    ol_pitch_gain: u32,
    ol_exc_gain: u32,
    fine_pitch: u32,
    pitch_gain: u32,
    innovation_gain: u32,
    innovation_vq: u32,
}

impl FieldWidths {
    fn for_submode(submode: &NarrowbandSubmode) -> Self {
        Self {
            lsp: match submode.lsp {
                LspQuant::None => 0,
                LspQuant::Bits18 => 18,
                LspQuant::Bits30 => 30,
            },
            ol_pitch: u32::from(submode.ol_pitch_bits),
            ol_pitch_gain: u32::from(submode.ol_pitch_gain_bits),
            ol_exc_gain: u32::from(submode.ol_exc_gain_bits),
            fine_pitch: u32::from(submode.fine_pitch_bits),
            pitch_gain: match submode.pitch_gain {
                PitchGainQuant::None => 0,
                PitchGainQuant::Vq5Bit => 5,
                PitchGainQuant::Vq7Bit => 7,
            },
            innovation_gain: u32::from(submode.innovation_gain_bits),
            innovation_vq: u32::from(submode.innovation_vq_bits),
        }
    }
}

/// Write a `u128` value as `n` MSB-first bits — the exact inverse of the
/// parser's `read_wide` chunked read. Shared with the high-band frame
/// writer ([`crate::hb_encode`]), whose excitation-VQ fields are up to
/// 80 bits wide.
pub(crate) fn write_wide(writer: &mut BitWriter, value: u128, n: u32) -> Result<(), BitError> {
    debug_assert!(n <= 128, "innovation VQ exceeds u128 capacity");
    let mut remaining = n;
    while remaining > 0 {
        let chunk = remaining.min(32);
        let low = remaining - chunk;
        let mask: u128 = if chunk == 128 {
            u128::MAX
        } else {
            (1u128 << chunk) - 1
        };
        let v = ((value >> low) & mask) as u32;
        writer.write(v, chunk)?;
        remaining = low;
    }
    Ok(())
}

/// Write the frame body (everything after the 5-bit prefix) for
/// `submode` into `writer`, in Table 9.1 field order.
pub fn write_narrowband_body(
    writer: &mut BitWriter,
    body: &NarrowbandFrameBody,
    submode: &NarrowbandSubmode,
) -> Result<(), BitError> {
    let w = FieldWidths::for_submode(submode);

    // Frame-level fields.
    writer.write(body.lsp_index, w.lsp)?;
    writer.write(u32::from(body.ol_pitch_index), w.ol_pitch)?;
    writer.write(u32::from(body.ol_pitch_gain_index), w.ol_pitch_gain)?;
    writer.write(u32::from(body.ol_exc_gain_index), w.ol_exc_gain)?;

    // Sub-frame fields, four times, in Table 9.1 order.
    for sf in &body.subframes {
        writer.write(u32::from(sf.pitch_index), w.fine_pitch)?;
        writer.write(u32::from(sf.pitch_gain_index), w.pitch_gain)?;
        writer.write(u32::from(sf.innovation_gain_index), w.innovation_gain)?;
        write_wide(writer, sf.innovation_vq_index, w.innovation_vq)?;
    }
    Ok(())
}

/// Encode a complete narrowband frame — the 5-bit prefix (wideband flag
/// `0` + `mode_id`) followed by the body — into a fresh byte buffer.
///
/// The returned buffer is *not* byte-padded or terminated; use
/// [`crate::BitWriter::pad_to_byte`] / the terminator code (mode 15,
/// §5.5) at the packet level when concatenating frames. `mode_id` is
/// taken from `submode.mode_id`.
pub fn encode_narrowband_frame(
    body: &NarrowbandFrameBody,
    submode: &NarrowbandSubmode,
) -> Result<BitWriter, FrameError> {
    let mut writer = BitWriter::new();
    let header = NarrowbandFrameHeader::new(false, submode.mode_id)?;
    header.write(&mut writer)?;
    write_narrowband_body(&mut writer, body, submode)?;
    Ok(writer)
}

/// Close a multi-frame Speex packet: append the 5-bit mode-15
/// **terminator** pseudo-frame (§5.5: *"it is possible to include a
/// terminator code. That terminator consists of the code 15 (decimal)
/// encoded with 5 bits"*) and zero-pad to the byte boundary.
///
/// The packet walker ([`crate::PacketFrames`]) stops cleanly at the
/// terminator, so the padding bits are never misparsed as another
/// frame regardless of how many bits the frames left in the last byte.
pub fn write_packet_terminator(writer: &mut BitWriter) -> Result<(), FrameError> {
    let terminator = NarrowbandFrameHeader::new(false, 15)?;
    terminator.write(writer)?;
    writer.pad_to_byte()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::frame::NARROWBAND_FRAME_PREFIX_BITS;
    use crate::narrowband_body::NarrowbandSubFrameIndices;
    use crate::submode::NARROWBAND_SUBMODES;

    fn sample_body(submode: &NarrowbandSubmode) -> NarrowbandFrameBody {
        let w = FieldWidths::for_submode(submode);
        let cap = |v: u32, bits: u32| -> u32 {
            if bits == 0 {
                0
            } else if bits >= 32 {
                v
            } else {
                v & ((1 << bits) - 1)
            }
        };
        let cap128 = |v: u128, bits: u32| -> u128 {
            if bits == 0 {
                0
            } else if bits >= 128 {
                v
            } else {
                v & ((1u128 << bits) - 1)
            }
        };
        let mut subframes = [NarrowbandSubFrameIndices::default(); 4];
        for (i, sf) in subframes.iter_mut().enumerate() {
            let base = i as u32 + 1;
            *sf = NarrowbandSubFrameIndices {
                pitch_index: cap(base * 7, w.fine_pitch) as u8,
                pitch_gain_index: cap(base * 3, w.pitch_gain) as u8,
                innovation_gain_index: cap(base, w.innovation_gain) as u8,
                innovation_vq_index: cap128(
                    (base as u128) * 0x1234_5678_9abc_def0,
                    w.innovation_vq,
                ),
            };
        }
        NarrowbandFrameBody {
            lsp_index: cap(0x2_ABCD, w.lsp),
            ol_pitch_index: cap(42, w.ol_pitch) as u8,
            ol_pitch_gain_index: cap(9, w.ol_pitch_gain) as u8,
            ol_exc_gain_index: cap(21, w.ol_exc_gain) as u8,
            subframes,
        }
    }

    #[test]
    fn round_trips_all_documented_modes() {
        // Modes 2..=6 and 8 have fully-documented bodies; 0/1/7 have
        // special / undocumented pieces but their *bit layout* still
        // round-trips (the packer only touches raw index integers).
        for mode in [0u8, 1, 2, 3, 4, 5, 6, 7, 8] {
            let submode = NarrowbandSubmode::for_id(mode).unwrap();
            let body = sample_body(&submode);
            let writer = encode_narrowband_frame(&body, &submode).unwrap();
            let bytes = writer.into_bytes();

            let mut reader = BitReader::new(&bytes);
            let header = NarrowbandFrameHeader::parse(&mut reader).unwrap();
            assert_eq!(header.mode_id, mode, "mode {mode} header round-trip");
            let parsed = NarrowbandFrameBody::parse(&mut reader, &submode).unwrap();
            assert_eq!(parsed, body, "mode {mode} body round-trip");
        }
    }

    #[test]
    fn frame_bit_length_matches_submode_total() {
        for mode in [2u8, 3, 5, 6, 8] {
            let submode = NarrowbandSubmode::for_id(mode).unwrap();
            let body = sample_body(&submode);
            let writer = encode_narrowband_frame(&body, &submode).unwrap();
            assert_eq!(
                writer.bits_written(),
                u32::from(submode.total_bits),
                "mode {mode} total bits"
            );
        }
    }

    #[test]
    fn write_wide_inverts_read_wide_for_96_bits() {
        // Mode 7 uses a 96-bit innovation VQ field; the wide writer must
        // round-trip a 96-bit value through the parser's wide reader.
        let mut writer = BitWriter::new();
        let value: u128 = 0x0000_0000_abcd_1234_5678_9abc_def0_1111 & ((1u128 << 96) - 1);
        write_wide(&mut writer, value, 96).unwrap();
        writer.pad_to_byte().unwrap();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        // Re-read the 96 bits in the same chunked order the body parser
        // uses (32 + 32 + 32).
        let mut acc: u128 = 0;
        for _ in 0..3 {
            let v = reader.read(32).unwrap() as u128;
            acc = (acc << 32) | v;
        }
        assert_eq!(acc, value);
    }

    #[test]
    fn prefix_is_five_bits() {
        let submode = &NARROWBAND_SUBMODES[3];
        let body = sample_body(submode);
        let writer = encode_narrowband_frame(&body, submode).unwrap();
        // (NARROWBAND_SUBMODES yields &NarrowbandSubmode already)
        // Body bits + prefix = total.
        assert_eq!(
            writer.bits_written() - NARROWBAND_FRAME_PREFIX_BITS,
            NarrowbandFrameBody::body_bits(submode)
        );
    }
}
