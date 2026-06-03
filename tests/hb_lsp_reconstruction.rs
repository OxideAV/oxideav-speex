//! Round r214 — integration test for high-band LSP MSVQ reconstruction.
//!
//! Builds synthetic wideband high-band bodies via the public
//! [`oxideav_speex::BitWriter`], parses them through the round-r160
//! [`oxideav_speex::WidebandHighBandBody::parse`], and verifies that
//! the new [`oxideav_speex::WidebandHighBandBody::reconstructed_lsp_q10`]
//! accessor yields the same vector as a direct
//! [`oxideav_speex::reconstruct_hb_lsp_q10`] call against the
//! synthesised per-stage indices.

use oxideav_speex::{
    reconstruct_hb_lsp_q10, BitReader, BitWriter, HbLspStages, WidebandHighBandBody,
    WidebandHighBandSubmode, HB_LPC_ORDER, HB_LSP_OUTPUT_Q, HB_LSP_STAGE_BITS,
    HB_LSP_STAGE_ENTRIES, HIGH_BAND_SUBFRAMES_PER_FRAME, WIDEBAND_HIGH_BAND_SUBMODES,
};

/// Construct the bit-stream body for high-band mode 2 (the smallest
/// mode that exercises every field that has non-zero width — LSP +
/// excitation gain + excitation VQ). 112 high-band bits total, of
/// which 4 are the prefix and 108 are the body.
fn synth_mode2_body(stage1: u8, stage2: u8) -> (Vec<u8>, WidebandHighBandSubmode) {
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
    let mut w = BitWriter::new();
    // 12-bit LSP MSVQ: stage1 in the top 6 bits, stage2 in the low 6.
    let packed_lsp = (u32::from(stage1) << HB_LSP_STAGE_BITS) | u32::from(stage2);
    w.write(packed_lsp, u32::from(submode.lsp_bits)).unwrap();
    // Four sub-frames of {excitation_gain, excitation_vq}.
    for sf in 0..HIGH_BAND_SUBFRAMES_PER_FRAME {
        w.write(
            u32::from(sf as u8) & 0xF,
            u32::from(submode.excitation_gain_bits),
        )
        .unwrap();
        w.write(0, u32::from(submode.excitation_vq_bits)).unwrap();
    }
    // Pad to byte boundary so the test buffer is well-formed.
    w.pad_to_byte().unwrap();
    (w.into_bytes(), submode)
}

#[test]
fn parse_then_reconstruct_matches_direct_path_for_synthesised_body() {
    let (bytes, submode) = synth_mode2_body(17, 42);
    let mut reader = BitReader::new(&bytes);
    let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();

    let via_body = body.reconstructed_lsp_q10(&submode).unwrap();
    let via_direct = reconstruct_hb_lsp_q10(HbLspStages {
        stage1: 17,
        stage2: 42,
    })
    .unwrap();
    assert_eq!(via_body, via_direct);
    assert_eq!(via_body.len(), HB_LPC_ORDER);
}

#[test]
fn reconstructed_lsp_is_none_for_silence_mode_0() {
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[0];
    // Mode 0 has zero-width fields, so an empty body is conforming.
    let bytes: [u8; 0] = [];
    let mut reader = BitReader::new(&bytes);
    let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();
    assert_eq!(body.lsp_index, 0);
    assert!(body.lsp_stages(&submode).is_none());
    assert!(body.reconstructed_lsp_q10(&submode).is_none());
}

/// Write `n` zero bits, splitting wide fields into 32-bit chunks so
/// the BitWriter's 32-bit-per-call width guard does not trip on the
/// 80-bit-per-sub-frame mode-4 excitation VQ.
fn write_zero_bits(w: &mut BitWriter, mut n: u32) {
    while n > 32 {
        w.write(0, 32).unwrap();
        n -= 32;
    }
    if n > 0 {
        w.write(0, n).unwrap();
    }
}

#[test]
fn reconstructed_lsp_round_trips_for_every_documented_mode() {
    // For each LSP-carrying mode (1..=4), synthesise a body with a
    // known (stage1, stage2) pair, parse, and check the
    // reconstruction matches the direct path.
    for (sm_idx, &submode) in WIDEBAND_HIGH_BAND_SUBMODES.iter().enumerate().skip(1) {
        let (s1, s2) = (3u8 * sm_idx as u8, 5u8 * sm_idx as u8);
        let mut w = BitWriter::new();
        let packed_lsp = (u32::from(s1) << HB_LSP_STAGE_BITS) | u32::from(s2);
        w.write(packed_lsp, u32::from(submode.lsp_bits)).unwrap();
        for sf in 0..HIGH_BAND_SUBFRAMES_PER_FRAME {
            w.write(
                u32::from(sf as u8) & 0xF,
                u32::from(submode.excitation_gain_bits),
            )
            .unwrap();
            write_zero_bits(&mut w, u32::from(submode.excitation_vq_bits));
        }
        w.pad_to_byte().unwrap();
        let bytes = w.into_bytes();
        let mut reader = BitReader::new(&bytes);
        let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();

        let via_body = body.reconstructed_lsp_q10(&submode).unwrap();
        let via_direct = reconstruct_hb_lsp_q10(HbLspStages {
            stage1: s1,
            stage2: s2,
        })
        .unwrap();
        assert_eq!(via_body, via_direct, "mode {} mismatch", sm_idx);
        assert_eq!(via_body.len(), HB_LPC_ORDER);
    }
}

#[test]
fn reconstructed_lsp_q_format_matches_constant() {
    // Sanity: the Q-format constant is in scope and matches r194.
    assert_eq!(HB_LSP_OUTPUT_Q, 10);
    // Reconstruction with any valid index pair must land inside the
    // documented Q10 dynamic range — entries are signed bytes and
    // the per-stage factors are 4 + 2 → max magnitude 127*4 + 127*2 = 762.
    let max_idx = (HB_LSP_STAGE_ENTRIES - 1) as u8;
    let v = reconstruct_hb_lsp_q10(HbLspStages {
        stage1: max_idx,
        stage2: max_idx,
    })
    .unwrap();
    for c in &v {
        assert!(c.abs() <= 762, "Q10 dynamic range exceeded");
    }
}
