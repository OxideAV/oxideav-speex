//! Round r230 — integration tests for the wideband **high-band
//! innovation sub-vector dispatcher**.
//!
//! Builds synthetic wideband high-band bodies via the public
//! [`oxideav_speex::BitWriter`], parses them through the round-r160
//! [`oxideav_speex::WidebandHighBandBody::parse`], and verifies that
//! the new
//! [`oxideav_speex::WidebandHighBandBody::hb_innovation_sub_vector`]
//! accessor yields the expected 40-sample fixed-codebook sub-vector
//! against the directly-built dispatcher output, for the two
//! documented mode-2 / mode-3 bindings plus the silence + reserved
//! paths.
//!
//! Spec basis: Speex Codec Manual §10.3 (high-band excitation) plus
//! Table 10.1's per-sub-frame `excitation_vq_bits` widths plus the
//! staged `tables/README.md` two high-band codebook shapes.
//!
//! No external library source consulted; the test exercises only the
//! crate's public API surface over the in-tree codebook tables.

use oxideav_speex::{
    decode_hb_subframe, decode_hb_subframe_mode4_f32, hb_innovation_sub_vector, BitReader,
    BitWriter, HbInnovationCodebook, HbInnovationMapping, WidebandHighBandBody,
    WidebandHighBandSubmode, HB_SUBFRAME_SAMPLES, HIGH_BAND_SUBFRAMES_PER_FRAME,
    WIDEBAND_HIGH_BAND_SUBMODES,
};

/// Build a synthetic mode-2 high-band body whose four sub-frames carry
/// a known sequence of `HbSv10_32` indices, return the resulting
/// byte buffer + the sub-mode descriptor.
fn synth_mode2_body(sub_indices: &[[u32; 4]; 4]) -> (Vec<u8>, WidebandHighBandSubmode) {
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
    let mut w = BitWriter::new();
    // LSP MSVQ (12 bits) — value irrelevant to this test.
    w.write(0, u32::from(submode.lsp_bits)).unwrap();
    for indices in sub_indices.iter() {
        // excitation_gain (4 bits) — irrelevant value.
        w.write(0, u32::from(submode.excitation_gain_bits)).unwrap();
        // excitation_vq: pack the four 5-bit indices MSB-first into a
        // 20-bit field.
        let mut packed_excvq: u32 = 0;
        for &idx in indices.iter() {
            packed_excvq = (packed_excvq << 5) | (idx & 0x1F);
        }
        w.write(packed_excvq, u32::from(submode.excitation_vq_bits))
            .unwrap();
    }
    w.pad_to_byte().unwrap();
    (w.into_bytes(), submode)
}

/// Build a synthetic mode-3 high-band body whose four sub-frames each
/// carry five `HbSv8_128` lookups (index + sign) packed into a 40-bit
/// field.
fn synth_mode3_body(sub_indices: &[[(u32, bool); 5]; 4]) -> (Vec<u8>, WidebandHighBandSubmode) {
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[3];
    let mut w = BitWriter::new();
    w.write(0, u32::from(submode.lsp_bits)).unwrap();
    for slots in sub_indices.iter() {
        w.write(0, u32::from(submode.excitation_gain_bits)).unwrap();
        // Pack five 8-bit slots (7-bit index + 1-bit sign) MSB-first.
        // Total 40 bits — split into two 32-bit writes (top 32 bits +
        // bottom 8 bits) to fit through the writer's 32-bit width
        // guard.
        let mut packed: u64 = 0;
        for &(idx, sign) in slots.iter() {
            let slot = (u64::from(idx & 0x7F) << 1) | u64::from(sign as u8);
            packed = (packed << 8) | slot;
        }
        // packed now holds the 40 bits in its low 40 bits.
        let hi = ((packed >> 8) & 0xFFFF_FFFF) as u32; // top 32 bits of the 40
        let lo = (packed & 0xFF) as u32; // bottom 8 bits
        w.write(hi, 32).unwrap();
        w.write(lo, 8).unwrap();
    }
    w.pad_to_byte().unwrap();
    (w.into_bytes(), submode)
}

#[test]
fn parse_then_dispatch_matches_direct_path_for_mode_2_body() {
    let sub_indices = [
        [3u32, 17, 0, 31],
        [1, 2, 4, 8],
        [0, 0, 0, 0],
        [31, 31, 31, 31],
    ];
    let (bytes, submode) = synth_mode2_body(&sub_indices);
    let mut reader = BitReader::new(&bytes);
    let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();
    for (sf, indices) in sub_indices.iter().enumerate() {
        let via_body = body.hb_innovation_sub_vector(&submode, sf).unwrap();
        // Reconstruct the packed 20-bit excitation_vq_index for this sub-frame.
        let mut packed = 0u128;
        for &idx in indices.iter() {
            packed = (packed << 5) | u128::from(idx);
        }
        let via_direct = decode_hb_subframe(&submode, packed).unwrap();
        assert_eq!(via_body, via_direct, "sub-frame {sf} mismatch");
        assert_eq!(via_body.len(), HB_SUBFRAME_SAMPLES);
    }
}

#[test]
fn parse_then_dispatch_matches_direct_path_for_mode_3_body() {
    let sub_indices: [[(u32, bool); 5]; 4] = [
        [(0, false), (1, true), (2, false), (5, true), (127, false)],
        [
            (64, false),
            (65, false),
            (66, true),
            (67, true),
            (68, false),
        ],
        [(0, false), (0, false), (0, false), (0, false), (0, false)],
        [
            (127, true),
            (127, true),
            (127, true),
            (127, true),
            (127, true),
        ],
    ];
    let (bytes, submode) = synth_mode3_body(&sub_indices);
    let mut reader = BitReader::new(&bytes);
    let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();
    for (sf, slots) in sub_indices.iter().enumerate() {
        let via_body = body.hb_innovation_sub_vector(&submode, sf).unwrap();
        let mut packed = 0u128;
        for &(idx, sign) in slots.iter() {
            let slot = (u128::from(idx) << 1) | u128::from(sign as u8);
            packed = (packed << 8) | slot;
        }
        let via_direct = decode_hb_subframe(&submode, packed).unwrap();
        assert_eq!(via_body, via_direct, "sub-frame {sf} mismatch");
        assert_eq!(via_body.len(), HB_SUBFRAME_SAMPLES);
    }
}

#[test]
fn silence_modes_return_all_zero_sub_vector() {
    for mode_id in [0u8, 1] {
        let submode = WidebandHighBandSubmode::for_id(mode_id).unwrap();
        // Build a conforming body (just the header was already consumed
        // upstream; modes 0/1 have either 0 or only gain bits, no VQ field).
        let mut w = BitWriter::new();
        if submode.lsp_bits > 0 {
            w.write(0, u32::from(submode.lsp_bits)).unwrap();
        }
        for _ in 0..HIGH_BAND_SUBFRAMES_PER_FRAME {
            if submode.excitation_gain_bits > 0 {
                w.write(0, u32::from(submode.excitation_gain_bits)).unwrap();
            }
            // No excitation VQ bits for modes 0 / 1.
            assert_eq!(submode.excitation_vq_bits, 0);
        }
        w.pad_to_byte().unwrap();
        let bytes = w.into_bytes();
        let mut reader = BitReader::new(&bytes);
        let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();
        for sf in 0..HIGH_BAND_SUBFRAMES_PER_FRAME as usize {
            let v = body.hb_innovation_sub_vector(&submode, sf).unwrap();
            assert_eq!(v.len(), HB_SUBFRAME_SAMPLES);
            assert!(
                v.iter().all(|&x| x == 0),
                "mode {mode_id} sub-frame {sf} not all-zero"
            );
        }
    }
}

#[test]
fn mode_4_dispatcher_decodes_the_two_stage_binding() {
    // Mode 4: 80 bits per sub-frame; build a conforming body and
    // confirm the dispatcher decodes it (r450 — the two-stage binding
    // is pinned; an all-zero field decodes to twice codebook row 0 at
    // stage weight 0.4, rounded on this i16 surface) consistently
    // across all four sub-frames.
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[4];
    let mut w = BitWriter::new();
    w.write(0, u32::from(submode.lsp_bits)).unwrap();
    for _ in 0..HIGH_BAND_SUBFRAMES_PER_FRAME {
        w.write(0, u32::from(submode.excitation_gain_bits)).unwrap();
        // 80 bits of zeros, split as two 32-bit + one 16-bit writes.
        w.write(0, 32).unwrap();
        w.write(0, 32).unwrap();
        w.write(0, 16).unwrap();
    }
    w.pad_to_byte().unwrap();
    let bytes = w.into_bytes();
    let mut reader = BitReader::new(&bytes);
    let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();
    for sf in 0..HIGH_BAND_SUBFRAMES_PER_FRAME as usize {
        let v = body
            .hb_innovation_sub_vector(&submode, sf)
            .expect("mode 4 decodes (r450)");
        let f = decode_hb_subframe_mode4_f32(0);
        for (n, (&iv, &fv)) in v.iter().zip(f.iter()).enumerate() {
            assert_eq!(i32::from(iv), f64::from(fv).round() as i32, "sf {sf} n {n}");
        }
    }
}

#[test]
fn dispatcher_for_every_documented_mode_satisfies_bit_budget() {
    // Public-API sanity: each mode's dispatch (when Documented)
    // satisfies sub_vector_len * count == 40 samples AND
    // slot_bits * count == excitation_vq_bits.
    for s in WIDEBAND_HIGH_BAND_SUBMODES {
        if let HbInnovationMapping::Documented { codebook, count } =
            HbInnovationMapping::for_mode(&s)
        {
            assert_eq!(
                codebook.sub_vector_len() * usize::from(count),
                HB_SUBFRAME_SAMPLES,
                "mode {} sample count",
                s.mode_id
            );
            assert_eq!(
                u32::from(codebook.slot_bits()) * u32::from(count),
                u32::from(s.excitation_vq_bits),
                "mode {} bit budget",
                s.mode_id
            );
        }
    }
}

#[test]
fn codebook_row_zero_for_each_shape_is_accessible_through_public_api() {
    // Public re-exports include the codebook accessor; this hits each
    // documented shape via the top-level free function.
    assert!(hb_innovation_sub_vector(HbInnovationCodebook::HbSv8_128, 0).is_some());
    assert!(hb_innovation_sub_vector(HbInnovationCodebook::HbSv10_32, 0).is_some());
    // Out-of-range for both.
    assert!(hb_innovation_sub_vector(HbInnovationCodebook::HbSv8_128, 128).is_none());
    assert!(hb_innovation_sub_vector(HbInnovationCodebook::HbSv10_32, 32).is_none());
}
