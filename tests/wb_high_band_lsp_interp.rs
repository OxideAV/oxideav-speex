//! Round r350 — integration test for the wideband **high-band
//! per-sub-frame LSP interpolation** path.
//!
//! Drives [`oxideav_speex::synthesise_high_band_frame_interp`] — the
//! interpolating high-band synthesis entry that reconstructs a separate
//! order-8 LPC set per sub-frame from the §9.1/§10.1 four-way linear
//! interpolation of the previous and current frame's high-band LSP
//! vectors — and asserts the milestone properties:
//!
//! * the interpolated chain produces finite, responsive high-band PCM;
//! * the per-frame interpolation actually engages — when the previous
//!   frame's high-band LSP differs from the current frame's, the
//!   interpolated output differs from the stateless (first-frame /
//!   frame-level-LPC) output of [`synthesise_high_band_frame`];
//! * the prev-LSP state is threaded across frames (an LSP-carrying frame
//!   updates it; a silence frame leaves it untouched);
//! * a single interpolated frame at stream start (`prev = None`) matches
//!   the stateless entry exactly (the first-frame `prev = curr`
//!   convention).
//!
//! Spec basis: *The Speex Codec Manual* §10.1 — the high-band linear
//! prediction is "very similar to narrowband. The only difference is
//! that we use only 12 bits to encode the high-band LSP's", so the §9.1
//! per-sub-frame LSP interpolation applies verbatim to the high band.
//!
//! No external library source consulted — only the crate's public API
//! over the in-tree codebook tables.

use oxideav_speex::{
    synthesise_high_band_frame, synthesise_high_band_frame_interp, BitReader, BitWriter,
    HbSynthesisFilter, WidebandHighBandBody, WidebandHighBandSubmode, HB_FRAME_SAMPLES,
    WIDEBAND_HIGH_BAND_SUBMODES,
};

/// Build a synthetic mode-2 high-band frame (12-bit LSP MSVQ index, then
/// four sub-frames of {4-bit gain, 20-bit excitation VQ = 4 × 5-bit}).
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
fn interpolated_chain_produces_finite_responsive_pcm() {
    let sub_indices = [
        [3u32, 17, 9, 31],
        [1, 2, 4, 8],
        [20, 5, 0, 12],
        [31, 7, 14, 2],
    ];
    let (bytes, submode) = synth_mode2_frame(0x0ABC, [8, 10, 12, 6], &sub_indices);
    let body = parse_frame(&bytes, &submode);

    let mut filter = HbSynthesisFilter::new();
    let mut prev = None;
    let pcm =
        synthesise_high_band_frame_interp(&body, &submode, &mut filter, &mut prev).expect("synth");

    assert_eq!(pcm.len(), HB_FRAME_SAMPLES, "160-sample half-band frame");
    assert!(pcm.iter().all(|v| v.is_finite()), "all samples finite");
    assert!(pcm.iter().any(|&v| v != 0.0), "must be non-silent");
    let first = pcm[0];
    assert!(pcm.iter().any(|&v| v != first), "output must vary");
    // An LSP-carrying frame must update the prev-LSP slot.
    assert!(prev.is_some(), "LSP frame populates the prev-LSP state");
}

#[test]
fn first_frame_interp_matches_stateless_entry() {
    // At stream start (prev = None) the interpolating entry uses the
    // first-frame convention (prev = curr), so its output must equal the
    // stateless synthesise_high_band_frame for the same body + a fresh
    // filter.
    let sub_indices = [
        [5u32, 11, 1, 30],
        [2, 9, 17, 4],
        [8, 8, 8, 8],
        [0, 31, 16, 3],
    ];
    let (bytes, submode) = synth_mode2_frame(0x0742, [7, 11, 9, 13], &sub_indices);
    let body = parse_frame(&bytes, &submode);

    let mut f_interp = HbSynthesisFilter::new();
    let mut prev = None;
    let via_interp =
        synthesise_high_band_frame_interp(&body, &submode, &mut f_interp, &mut prev).unwrap();

    let mut f_stateless = HbSynthesisFilter::new();
    let via_stateless = synthesise_high_band_frame(&body, &submode, &mut f_stateless).unwrap();

    assert_eq!(
        via_interp, via_stateless,
        "first-frame interpolation must reproduce the stateless frame-level path"
    );
}

#[test]
fn interpolation_engages_when_lsp_changes_between_frames() {
    // Two frames with DIFFERENT LSP indices but identical excitation.
    // The second frame decoded with a live prev-LSP (interpolating from
    // frame-0's LSP) must differ from the same second frame decoded as a
    // stream-start frame (prev = curr), proving the per-sub-frame
    // interpolation actually depends on the previous frame's LSP.
    let sub_indices = [
        [10u32, 3, 21, 7],
        [5, 14, 2, 28],
        [9, 9, 9, 9],
        [1, 16, 30, 4],
    ];
    let (bytes0, submode) = synth_mode2_frame(0x0111, [9, 9, 9, 9], &sub_indices);
    let body0 = parse_frame(&bytes0, &submode);
    // A very different LSP for frame 1.
    let (bytes1, _) = synth_mode2_frame(0x0EEE, [9, 9, 9, 9], &sub_indices);
    let body1 = parse_frame(&bytes1, &submode);

    // Path A: continuous — frame 0 then frame 1, interpolating across.
    let mut f_a = HbSynthesisFilter::new();
    let mut prev_a = None;
    let _f0 = synthesise_high_band_frame_interp(&body0, &submode, &mut f_a, &mut prev_a).unwrap();
    let f1_continuous =
        synthesise_high_band_frame_interp(&body1, &submode, &mut f_a, &mut prev_a).unwrap();

    // Path B: frame 1 alone as a stream-start frame (prev = curr), same
    // fresh filter history as path A had at frame 1's start would NOT be
    // identical, so to isolate the LSP effect we replay frame 0 into B's
    // filter too, but reset the prev-LSP slot to None right before
    // frame 1 (forcing the first-frame convention for frame 1).
    let mut f_b = HbSynthesisFilter::new();
    let mut prev_b = None;
    let _f0b = synthesise_high_band_frame_interp(&body0, &submode, &mut f_b, &mut prev_b).unwrap();
    prev_b = None; // discard frame-0 LSP → frame 1 uses prev = curr
    let f1_first_frame =
        synthesise_high_band_frame_interp(&body1, &submode, &mut f_b, &mut prev_b).unwrap();

    // Filter histories are identical at frame 1's start (both replayed
    // frame 0 with the same body into a fresh filter), so any difference
    // is solely the per-sub-frame LSP interpolation depending on
    // frame 0's LSP vs. frame 1's own LSP.
    assert!(
        f1_continuous != f1_first_frame,
        "interpolating from the previous frame's LSP must change sub-frames 1..3"
    );
}

#[test]
fn silence_frame_preserves_prev_lsp_state() {
    // After an LSP-carrying frame, a high-band silence frame (mode 0, no
    // LSP field) must leave the prev-LSP slot untouched so the following
    // frame still interpolates from the last transmitted envelope.
    let sub_indices = [[4u32, 12, 1, 29]; 4];
    let (bytes, submode2) = synth_mode2_frame(0x0ABC, [8, 8, 8, 8], &sub_indices);
    let body2 = parse_frame(&bytes, &submode2);

    let mut filter = HbSynthesisFilter::new();
    let mut prev = None;
    synthesise_high_band_frame_interp(&body2, &submode2, &mut filter, &mut prev).unwrap();
    let after_lsp_frame = prev;
    assert!(after_lsp_frame.is_some(), "LSP frame sets the prev state");

    // Mode 0 silence frame: no LSP field, no excitation.
    let submode0 = WIDEBAND_HIGH_BAND_SUBMODES[0];
    let mut r = BitReader::new(&[]);
    let body0 = WidebandHighBandBody::parse(&mut r, &submode0).expect("mode 0 empty body");
    synthesise_high_band_frame_interp(&body0, &submode0, &mut filter, &mut prev).unwrap();

    assert_eq!(
        prev, after_lsp_frame,
        "a high-band silence frame must leave the prev-LSP state unchanged"
    );
}
