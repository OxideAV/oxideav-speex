//! Round r302 — integration test for the wideband high-band LSP→LPC
//! conversion.
//!
//! Builds synthetic wideband high-band bodies via the public
//! [`oxideav_speex::BitWriter`], parses them through the round-r160
//! [`oxideav_speex::WidebandHighBandBody::parse`], and verifies that the
//! new [`oxideav_speex::WidebandHighBandBody::hb_lpc`] accessor composes
//! the r214 high-band LSP reconstruction with the order-8 LSP→LPC core
//! and matches a direct base-aware
//! [`oxideav_speex::lpc_from_hb_lsp_delta_q10`] call against the
//! reconstructed Q10 codebook-delta vector. This is the high-band
//! counterpart of the narrowband r286 LSP→LPC path.

// Round r347: `hb_lpc` now adds the pinned high-band LSP base vector
// (`LSP_LINEAR_HIGH`) before conversion, so the accessor tracks the
// base-aware `lpc_from_hb_lsp_delta_q10` (base + delta), not the
// delta-only `lpc_from_hb_lsp_q10`.
use oxideav_speex::{
    lpc_from_hb_lsp_delta_q10, reconstruct_hb_lsp_q10, BitReader, BitWriter, HbLspStages,
    WidebandHighBandBody, WidebandHighBandSubmode, HB_LPC_ORDER, HB_LSP_STAGE_BITS,
    HIGH_BAND_SUBFRAMES_PER_FRAME, WIDEBAND_HIGH_BAND_SUBMODES,
};

/// Write `n` zero bits, splitting wide fields into 32-bit chunks so the
/// BitWriter's 32-bit-per-call width guard does not trip on the
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

/// Assemble the bit-stream body for a documented high-band sub-mode with
/// a known per-stage LSP index pair (excitation gain + VQ fields zeroed).
fn synth_body(submode: &WidebandHighBandSubmode, stage1: u8, stage2: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    let packed_lsp = (u32::from(stage1) << HB_LSP_STAGE_BITS) | u32::from(stage2);
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
    w.into_bytes()
}

#[test]
fn hb_lpc_matches_direct_conversion_for_synthesised_body() {
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
    let bytes = synth_body(&submode, 17, 42);
    let mut reader = BitReader::new(&bytes);
    let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();

    let via_body = body.hb_lpc(&submode).unwrap();
    let lsp_q10 = reconstruct_hb_lsp_q10(HbLspStages {
        stage1: 17,
        stage2: 42,
    })
    .unwrap();
    let via_direct = lpc_from_hb_lsp_delta_q10(&lsp_q10);

    assert_eq!(via_body.len(), HB_LPC_ORDER);
    assert_eq!(via_body, via_direct);
    for &c in &via_body {
        assert!(c.is_finite());
    }
}

#[test]
fn hb_lpc_is_none_for_silence_mode_0() {
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[0];
    let bytes: [u8; 0] = [];
    let mut reader = BitReader::new(&bytes);
    let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();
    // Mode 0 carries no LSP field, so there is nothing to convert.
    assert!(body.hb_lpc(&submode).is_none());
}

#[test]
fn hb_lpc_round_trips_for_every_documented_mode() {
    for (sm_idx, &submode) in WIDEBAND_HIGH_BAND_SUBMODES.iter().enumerate().skip(1) {
        let (s1, s2) = (3u8 * sm_idx as u8, 5u8 * sm_idx as u8);
        let bytes = synth_body(&submode, s1, s2);
        let mut reader = BitReader::new(&bytes);
        let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();

        let via_body = body.hb_lpc(&submode).unwrap();
        let lsp_q10 = reconstruct_hb_lsp_q10(HbLspStages {
            stage1: s1,
            stage2: s2,
        })
        .unwrap();
        let via_direct = lpc_from_hb_lsp_delta_q10(&lsp_q10);

        assert_eq!(via_body, via_direct, "mode {} mismatch", sm_idx);
        assert_eq!(via_body.len(), HB_LPC_ORDER);
        for &c in &via_body {
            assert!(c.is_finite(), "mode {} produced non-finite coeff", sm_idx);
        }
    }
}

#[test]
fn hb_lpc_produces_distinct_filters_for_distinct_lsps() {
    // Two different LSP index pairs must (in general) produce different
    // LPC sets — confirming the conversion actually threads the
    // reconstructed coefficients through rather than emitting a constant.
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[2];

    let bytes_a = synth_body(&submode, 5, 9);
    let mut ra = BitReader::new(&bytes_a);
    let a = WidebandHighBandBody::parse(&mut ra, &submode)
        .unwrap()
        .hb_lpc(&submode)
        .unwrap();

    let bytes_b = synth_body(&submode, 40, 50);
    let mut rb = BitReader::new(&bytes_b);
    let b = WidebandHighBandBody::parse(&mut rb, &submode)
        .unwrap()
        .hb_lpc(&submode)
        .unwrap();

    assert_ne!(a, b, "distinct LSP indices collapsed to the same LPC set");
}
