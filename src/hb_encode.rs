//! Wideband high-band frame packing + wideband frame assembly — the
//! encode-direction inverse of [`crate::wideband::WidebandHighBandBody::parse`]
//! and the §10.4 embedded-frame concatenation (round r385 scope).
//!
//! Given a fully-populated [`WidebandHighBandBody`] (the quantised
//! indices produced by the high-band encoders) and its resolved
//! [`WidebandHighBandSubmode`], [`write_high_band_frame`] emits the
//! 4-bit high-band prefix (1-bit wideband flag = 1 + 3-bit mode ID)
//! followed by the body in the exact Table 10.1 field order the parser
//! consumes: the frame-level LSP MSVQ index, then four sub-frames of
//! `excitation gain || excitation VQ`.
//!
//! [`encode_wideband_frame`] assembles a complete wideband frame per
//! manual §10.4 (*"For the wideband mode, the entire narrowband frame
//! is packed before the high-band is encoded"*): the 5-bit narrowband
//! prefix with the **wideband flag set** (§9.3 — the flag announces the
//! high-band continuation), the Table 9.1 narrowband body, then the
//! high-band frame. The result parses back through the exact reader
//! chain [`crate::WidebandDecoder`] walks.
//!
//! The round-trip is exact by construction: `parse(write(body)) == body`
//! for any body whose index fields fit their sub-mode's bit budgets.

use crate::bitreader::{BitError, BitWriter};
use crate::frame::{FrameError, NarrowbandFrameHeader};
use crate::narrowband_body::NarrowbandFrameBody;
use crate::nb_encode::{write_narrowband_body, write_wide};
use crate::submode::NarrowbandSubmode;
use crate::wideband::{WidebandHighBandBody, WidebandHighBandSubmode};

/// Write the high-band frame body (everything after the 4-bit prefix)
/// for `submode` into `writer`, in Table 10.1 field order — the exact
/// inverse of [`WidebandHighBandBody::parse`].
pub fn write_high_band_body(
    writer: &mut BitWriter,
    body: &WidebandHighBandBody,
    submode: &WidebandHighBandSubmode,
) -> Result<(), BitError> {
    // Frame-level field, in Table 10.1 order.
    writer.write(u32::from(body.lsp_index), u32::from(submode.lsp_bits))?;

    // Sub-frame fields, four times, in Table 10.1 order.
    for sf in &body.subframes {
        writer.write(
            u32::from(sf.excitation_gain_index),
            u32::from(submode.excitation_gain_bits),
        )?;
        write_wide(
            writer,
            sf.excitation_vq_index,
            u32::from(submode.excitation_vq_bits),
        )?;
    }
    Ok(())
}

/// Write a complete high-band frame — the 4-bit prefix (wideband flag
/// `1` + 3-bit `mode_id`) followed by the body — into `writer`.
///
/// The wideband flag is always written as `1`: per §10.4 the flag
/// announces that high-band data follows (a `0` would mean the next
/// frame is a fresh narrowband-only frame).
pub fn write_high_band_frame(
    writer: &mut BitWriter,
    body: &WidebandHighBandBody,
    submode: &WidebandHighBandSubmode,
) -> Result<(), BitError> {
    writer.write_bit(1)?;
    writer.write(u32::from(submode.mode_id), 3)?;
    write_high_band_body(writer, body, submode)
}

/// Encode a complete **wideband frame** into a fresh byte buffer: the
/// embedded narrowband frame (5-bit prefix with the wideband flag
/// **set** + Table 9.1 body) followed by the high-band frame (4-bit
/// prefix + Table 10.1 body), per manual §10.4.
///
/// The returned writer is *not* byte-padded or terminated; use
/// [`crate::BitWriter::pad_to_byte`] / the terminator code (mode 15,
/// §5.5) at the packet level when concatenating frames.
pub fn encode_wideband_frame(
    nb_body: &NarrowbandFrameBody,
    nb_submode: &NarrowbandSubmode,
    hb_body: &WidebandHighBandBody,
    hb_submode: &WidebandHighBandSubmode,
) -> Result<BitWriter, FrameError> {
    let mut writer = BitWriter::new();
    let header = NarrowbandFrameHeader::new(false, nb_submode.mode_id)?;
    header.write(&mut writer)?;
    write_narrowband_body(&mut writer, nb_body, nb_submode)?;
    write_high_band_frame(&mut writer, hb_body, hb_submode)?;
    Ok(writer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::narrowband_body::NarrowbandSubFrameIndices;
    use crate::wideband::{
        HighBandSubFrameIndices, WidebandHighBandFrameHeader, WidebandSubmode,
        HIGH_BAND_FRAME_PREFIX_BITS, HIGH_BAND_SUBFRAMES_PER_FRAME, WIDEBAND_HIGH_BAND_SUBMODES,
    };

    /// A synthetic high-band body whose fields exercise every bit of the
    /// sub-mode's budgets.
    fn sample_hb_body(submode: &WidebandHighBandSubmode) -> WidebandHighBandBody {
        let cap = |v: u32, bits: u8| -> u32 {
            if bits == 0 {
                0
            } else {
                v & ((1u32 << bits) - 1)
            }
        };
        let cap128 = |v: u128, bits: u8| -> u128 {
            if bits == 0 {
                0
            } else {
                v & ((1u128 << bits) - 1)
            }
        };
        let mut subframes =
            [HighBandSubFrameIndices::default(); HIGH_BAND_SUBFRAMES_PER_FRAME as usize];
        for (i, sf) in subframes.iter_mut().enumerate() {
            let base = i as u32 + 1;
            *sf = HighBandSubFrameIndices {
                excitation_gain_index: cap(base * 5 + 3, submode.excitation_gain_bits) as u8,
                excitation_vq_index: cap128(
                    (base as u128) * 0xfedc_ba98_7654_3210_0f1e_2d3c,
                    submode.excitation_vq_bits,
                ),
            };
        }
        WidebandHighBandBody {
            lsp_index: if submode.lsp_bits == 0 { 0 } else { 0x0A55 },
            subframes,
        }
    }

    #[test]
    fn hb_frame_round_trips_all_documented_modes() {
        for mode in 0u8..=4 {
            let submode = WidebandHighBandSubmode::for_id(mode).unwrap();
            let body = sample_hb_body(&submode);
            let mut w = BitWriter::new();
            write_high_band_frame(&mut w, &body, &submode).unwrap();
            let bytes = w.into_bytes();

            let mut r = BitReader::new(&bytes);
            let header = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
            assert!(header.wideband, "mode {mode} wideband flag");
            assert_eq!(header.mode_id, mode, "mode {mode} header round-trip");
            let parsed = WidebandHighBandBody::parse(&mut r, &submode).unwrap();
            assert_eq!(parsed, body, "mode {mode} body round-trip");
        }
    }

    #[test]
    fn hb_frame_bit_length_matches_table_10_1_total() {
        for mode in 0u8..=4 {
            let submode = WidebandHighBandSubmode::for_id(mode).unwrap();
            let body = sample_hb_body(&submode);
            let mut w = BitWriter::new();
            write_high_band_frame(&mut w, &body, &submode).unwrap();
            assert_eq!(
                w.bits_written(),
                u32::from(submode.total_bits),
                "mode {mode} total bits"
            );
        }
    }

    #[test]
    fn hb_body_bits_matches_total_minus_prefix() {
        for mode in 0u8..=4 {
            let submode = WidebandHighBandSubmode::for_id(mode).unwrap();
            let body = sample_hb_body(&submode);
            let mut w = BitWriter::new();
            write_high_band_body(&mut w, &body, &submode).unwrap();
            assert_eq!(
                w.bits_written(),
                u32::from(submode.total_bits) - HIGH_BAND_FRAME_PREFIX_BITS,
                "mode {mode} body bits"
            );
        }
    }

    /// A synthetic narrowband body for the embedded low band (raw index
    /// integers only — the writer packs whatever fits the budgets).
    fn sample_nb_body(submode: &NarrowbandSubmode) -> NarrowbandFrameBody {
        let mut subframes = [NarrowbandSubFrameIndices::default(); 4];
        for (i, sf) in subframes.iter_mut().enumerate() {
            *sf = NarrowbandSubFrameIndices {
                pitch_index: (i as u8) * 9 + 1,
                pitch_gain_index: (i as u8) * 7 + 2,
                innovation_gain_index: i as u8,
                innovation_vq_index: (i as u128 + 1) * 0x0123_4567_89ab_cdef,
            };
        }
        // Mask fields to the sub-mode budgets by writing + re-parsing is
        // the test itself; keep raw values in range for the common modes.
        let _ = submode;
        NarrowbandFrameBody {
            lsp_index: 0x1_2345,
            ol_pitch_index: 33,
            ol_pitch_gain_index: 5,
            ol_exc_gain_index: 17,
            subframes,
        }
    }

    #[test]
    fn wideband_frame_round_trips_nb_and_hb_layers() {
        // NB mode 3 (18-bit LSP, per-sub-frame pitch) + HB mode 2.
        let nb_submode = NarrowbandSubmode::for_id(3).unwrap();
        let hb_submode = WidebandHighBandSubmode::for_id(2).unwrap();

        // Build an NB body that survives its own bit budgets exactly by
        // writing + reparsing once (masks high bits off wide fields).
        let raw_nb = sample_nb_body(&nb_submode);
        let mut w0 = BitWriter::new();
        write_narrowband_body(&mut w0, &raw_nb, &nb_submode).unwrap();
        let b0 = w0.into_bytes();
        let mut r0 = BitReader::new(&b0);
        let nb_body = NarrowbandFrameBody::parse(&mut r0, &nb_submode).unwrap();

        let hb_body = sample_hb_body(&hb_submode);

        let w = encode_wideband_frame(&nb_body, &nb_submode, &hb_body, &hb_submode).unwrap();
        let expected_bits = u32::from(nb_submode.total_bits) + u32::from(hb_submode.total_bits);
        assert_eq!(w.bits_written(), expected_bits, "wideband frame bits");
        let bytes = w.into_bytes();

        // Walk the exact reader chain the wideband decoder uses.
        let mut r = BitReader::new(&bytes);
        let nb_header = NarrowbandFrameHeader::parse(&mut r).unwrap();
        // r393 fixture-pinned grammar: flag 0 on the narrowband layer.
        assert!(!nb_header.wideband, "NB layer prefix carries flag 0");
        assert_eq!(nb_header.mode_id, 3);
        let nb_parsed = NarrowbandFrameBody::parse(&mut r, &nb_submode).unwrap();
        assert_eq!(nb_parsed, nb_body, "embedded NB body round-trip");

        let hb_header = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        assert!(hb_header.wideband);
        assert_eq!(hb_header.mode_id, 2);
        assert!(matches!(hb_header.submode, WidebandSubmode::Documented(_)));
        let hb_parsed = WidebandHighBandBody::parse(&mut r, &hb_submode).unwrap();
        assert_eq!(hb_parsed, hb_body, "HB body round-trip");
        assert_eq!(r.consumed_bits(), expected_bits);
    }

    #[test]
    fn silence_high_band_writes_only_the_prefix() {
        // HB mode 0: zero-bit body — the frame is exactly the 4-bit
        // prefix.
        let submode = WidebandHighBandSubmode::for_id(0).unwrap();
        let body = WidebandHighBandBody {
            lsp_index: 0,
            subframes: [HighBandSubFrameIndices::default(); HIGH_BAND_SUBFRAMES_PER_FRAME as usize],
        };
        let mut w = BitWriter::new();
        write_high_band_frame(&mut w, &body, &submode).unwrap();
        assert_eq!(w.bits_written(), HIGH_BAND_FRAME_PREFIX_BITS);
    }

    #[test]
    fn mode_4_eighty_bit_vq_fields_round_trip() {
        // Mode 4's 80-bit excitation-VQ fields exercise the wide writer
        // beyond 64 bits per field.
        let submode = WidebandHighBandSubmode::for_id(4).unwrap();
        let mut body = sample_hb_body(&submode);
        for (i, sf) in body.subframes.iter_mut().enumerate() {
            sf.excitation_vq_index =
                ((1u128 << 80) - 1) ^ ((i as u128) * 0x1111_2222_3333_4444_5555);
            sf.excitation_vq_index &= (1u128 << 80) - 1;
        }
        let mut w = BitWriter::new();
        write_high_band_frame(&mut w, &body, &submode).unwrap();
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let _ = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        let parsed = WidebandHighBandBody::parse(&mut r, &submode).unwrap();
        assert_eq!(parsed, body);
    }

    #[test]
    fn every_documented_hb_mode_is_wideband_decoder_walkable() {
        // encode_wideband_frame output must start-to-end match the layout
        // WidebandHighBandFrameHeader + body parsing expects for every
        // documented HB mode (0..=4) over an NB mode-5 low band.
        let nb_submode = NarrowbandSubmode::for_id(5).unwrap();
        let raw_nb = sample_nb_body(&nb_submode);
        let mut w0 = BitWriter::new();
        write_narrowband_body(&mut w0, &raw_nb, &nb_submode).unwrap();
        let b0 = w0.into_bytes();
        let mut r0 = BitReader::new(&b0);
        let nb_body = NarrowbandFrameBody::parse(&mut r0, &nb_submode).unwrap();

        for hb_mode in 0u8..=4 {
            let hb_submode = WIDEBAND_HIGH_BAND_SUBMODES[hb_mode as usize];
            let hb_body = sample_hb_body(&hb_submode);
            let w = encode_wideband_frame(&nb_body, &nb_submode, &hb_body, &hb_submode).unwrap();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let _ = NarrowbandFrameHeader::parse(&mut r).unwrap();
            let _ = NarrowbandFrameBody::parse(&mut r, &nb_submode).unwrap();
            let hdr = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
            assert_eq!(hdr.mode_id, hb_mode);
            let parsed = WidebandHighBandBody::parse(&mut r, &hb_submode).unwrap();
            assert_eq!(parsed, hb_body, "HB mode {hb_mode}");
        }
    }
}
