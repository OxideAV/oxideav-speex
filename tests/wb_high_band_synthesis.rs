//! Round r340 — integration test for the wideband **high-band synthesis
//! chain**: build a synthetic high-band frame via the public
//! [`oxideav_speex::BitWriter`], parse it with the round-r160
//! [`oxideav_speex::WidebandHighBandBody::parse`], and drive the full
//! high-band branch
//! (LSP→LPC → gain-scaled excitation `e_hb[n] = g·c_hb[n]` → order-8
//! synthesis filter `1/A_hb(z)`) through
//! [`oxideav_speex::synthesise_high_band_frame`], asserting the chain
//! produces finite, range-bounded, **responsive** high-band PCM and that
//! the IIR filter history carries across frames.
//!
//! Spec basis: *The Speex Codec Manual* §10 (sub-band CELP), §10.1
//! (high-band LP "very similar to narrowband", order 8), §10.2 (no
//! high-band pitch), §10.3 (high-band excitation coded like narrowband),
//! Table 10.1. The QMF synthesis recombination is a recorded docs gap;
//! this test stops at the reconstructed high-band 8 kHz half-band signal.
//!
//! No external library source consulted — only the crate's public API
//! over the in-tree codebook tables.

use oxideav_speex::{
    synthesise_high_band_frame, BitReader, BitWriter, HbSynthesisFilter, SubBandLayer,
    UwbFrameLayout, WidebandHighBandBody, WidebandHighBandSubmode, HB_FRAME_SAMPLES,
    WIDEBAND_HIGH_BAND_SUBMODES,
};

/// Build a synthetic mode-2 high-band frame: 12-bit LSP MSVQ index,
/// then four sub-frames of {4-bit gain, 20-bit excitation VQ = 4 × 5-bit
/// `HbSv10_32` index}. Returns the byte buffer + the sub-mode.
fn synth_mode2_frame(
    lsp_index: u16,
    gains: [u8; 4],
    sub_indices: &[[u32; 4]; 4],
) -> (Vec<u8>, WidebandHighBandSubmode) {
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
    let mut w = BitWriter::new();
    w.write(u32::from(lsp_index), u32::from(submode.lsp_bits))
        .unwrap();
    for (sf, indices) in sub_indices.iter().enumerate() {
        w.write(
            u32::from(gains[sf]),
            u32::from(submode.excitation_gain_bits),
        )
        .unwrap();
        let mut packed: u32 = 0;
        for &idx in indices.iter() {
            packed = (packed << 5) | (idx & 0x1F);
        }
        w.write(packed, u32::from(submode.excitation_vq_bits))
            .unwrap();
    }
    w.pad_to_byte().unwrap();
    (w.into_bytes(), submode)
}

fn parse_frame(bytes: &[u8], submode: &WidebandHighBandSubmode) -> WidebandHighBandBody {
    let mut reader = BitReader::new(bytes);
    WidebandHighBandBody::parse(&mut reader, submode).expect("frame parses")
}

#[test]
fn full_high_band_chain_produces_finite_responsive_pcm() {
    let sub_indices = [
        [3u32, 17, 9, 31],
        [1, 2, 4, 8],
        [20, 5, 0, 12],
        [31, 7, 14, 2],
    ];
    // Non-trivial LSP + non-silent 4-bit gains.
    let (bytes, submode) = synth_mode2_frame(0x0ABC, [8, 10, 12, 6], &sub_indices);
    let body = parse_frame(&bytes, &submode);

    let mut filter = HbSynthesisFilter::new();
    let pcm = synthesise_high_band_frame(&body, &submode, &mut filter).expect("synthesis");

    assert_eq!(pcm.len(), HB_FRAME_SAMPLES, "160-sample half-band frame");
    assert!(pcm.iter().all(|v| v.is_finite()), "all samples finite");
    // Responsive: not all-zero, not a constant.
    assert!(pcm.iter().any(|&v| v != 0.0), "must be non-silent");
    let first = pcm[0];
    assert!(
        pcm.iter().any(|&v| v != first),
        "output must vary, not be constant"
    );
}

#[test]
fn silence_high_band_is_exactly_zero_at_stream_start() {
    // Mode 0: no LSP, no excitation — a fresh filter yields pure zero.
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[0];
    // Mode 0 body is just the (empty) frame; build via the writer with a
    // single padding byte.
    let mut w = BitWriter::new();
    w.pad_to_byte().unwrap();
    let bytes = w.into_bytes();
    let body = if bytes.is_empty() {
        // Zero-length body: construct the parsed body directly from an
        // empty reader (mode 0 consumes no bits).
        let mut r = BitReader::new(&[]);
        WidebandHighBandBody::parse(&mut r, &submode).expect("mode 0 empty body")
    } else {
        parse_frame(&bytes, &submode)
    };

    let mut filter = HbSynthesisFilter::new();
    let pcm = synthesise_high_band_frame(&body, &submode, &mut filter).unwrap();
    assert!(pcm.iter().all(|&v| v == 0.0), "silent high band is zero");
}

#[test]
fn history_carries_across_two_frames() {
    let sub_indices = [[1u32, 2, 3, 4]; 4];
    let (bytes, submode) = synth_mode2_frame(0x0555, [9, 9, 9, 9], &sub_indices);
    let body = parse_frame(&bytes, &submode);

    // One filter across two frames vs. a fresh filter for the second.
    let mut continuous = HbSynthesisFilter::new();
    let _frame0 = synthesise_high_band_frame(&body, &submode, &mut continuous).unwrap();
    let frame1_continuous = synthesise_high_band_frame(&body, &submode, &mut continuous).unwrap();

    let mut fresh = HbSynthesisFilter::new();
    let frame1_fresh = synthesise_high_band_frame(&body, &submode, &mut fresh).unwrap();

    // The continuous-filter second frame differs from a stream-start
    // frame of the same body, because its IIR history is live — unless
    // the body happens to be entirely silent (it is not here).
    assert!(
        frame1_continuous != frame1_fresh,
        "live IIR history must influence the second frame"
    );
}

#[test]
fn uwb_framing_recursion_ladder() {
    // The embedded scalable framing: narrowband → wideband → ultra-wideband
    // adds one high-band layer per step, doubling the reconstructed rate.
    assert_eq!(UwbFrameLayout::NARROWBAND.high_band_layer_count(), 0);
    assert_eq!(UwbFrameLayout::WIDEBAND.high_band_layer_count(), 1);
    assert_eq!(UwbFrameLayout::ULTRA_WIDEBAND.high_band_layer_count(), 2);

    assert_eq!(UwbFrameLayout::NARROWBAND.reconstructed_rate_hz(), 8_000);
    assert_eq!(UwbFrameLayout::WIDEBAND.reconstructed_rate_hz(), 16_000);
    assert_eq!(
        UwbFrameLayout::ULTRA_WIDEBAND.reconstructed_rate_hz(),
        32_000
    );

    assert_eq!(
        UwbFrameLayout::ULTRA_WIDEBAND.layers(),
        &[
            SubBandLayer::Narrowband,
            SubBandLayer::WidebandHighBand,
            SubBandLayer::UltraWidebandHighBand
        ]
    );
}
