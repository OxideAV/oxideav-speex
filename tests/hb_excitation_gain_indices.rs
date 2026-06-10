//! Round r269 — integration tests for the wideband **high-band
//! fixed-codebook gain index primitive**.
//!
//! Builds synthetic wideband high-band bodies via the public
//! [`oxideav_speex::BitWriter`], parses them through the round-r160
//! [`oxideav_speex::WidebandHighBandBody::parse`], and verifies that
//! the new
//! [`oxideav_speex::WidebandHighBandBody::hb_excitation_gain_indices`]
//! accessor resolves the per-sub-frame `Excitation gain` indices that
//! were written, for every documented high-band mode 0..=4.
//!
//! Spec basis: Speex Codec Manual §10.4 / Table 10.1 — the
//! `Excitation gain` sub-frame row widths `0 / 5 / 4 / 4 / 4` for
//! modes 0..=4, with no frame-level gain factor (unlike the
//! narrowband Table 9.1 `OL Exc gain` row).
//!
//! No external library source consulted; the test exercises only the
//! crate's public API surface.

use oxideav_speex::{
    BitReader, BitWriter, HbExcitationGainIndex, WidebandHighBandBody, WidebandHighBandSubmode,
    HIGH_BAND_SUBFRAMES_PER_FRAME, WIDEBAND_HIGH_BAND_SUBMODES,
};

/// Build a synthetic high-band body for `submode` whose four
/// sub-frames carry the supplied raw excitation-gain indices, with
/// zero LSP + zero excitation-VQ fields (irrelevant to this test).
fn synth_body(submode: &WidebandHighBandSubmode, gains: [u32; 4]) -> Vec<u8> {
    let mut w = BitWriter::new();
    if submode.lsp_bits > 0 {
        w.write(0, u32::from(submode.lsp_bits)).unwrap();
    }
    for &g in gains.iter() {
        if submode.excitation_gain_bits > 0 {
            w.write(g, u32::from(submode.excitation_gain_bits)).unwrap();
        }
        // Zero out the excitation-VQ field in 32-bit chunks (mode 4's
        // field is 80 bits wide, above the writer's 32-bit guard).
        let mut remaining = u32::from(submode.excitation_vq_bits);
        while remaining > 0 {
            let chunk = remaining.min(32);
            w.write(0, chunk).unwrap();
            remaining -= chunk;
        }
    }
    w.pad_to_byte().unwrap();
    w.into_bytes()
}

#[test]
fn parse_then_resolve_round_trips_written_gains_for_every_documented_mode() {
    // Per-mode in-range gain patterns (Table 10.1 widths: 0/5/4/4/4).
    let patterns: [[u32; 4]; 5] = [
        [0, 0, 0, 0],    // mode 0: no field on the wire.
        [0, 13, 30, 31], // mode 1: 5-bit indices.
        [1, 7, 14, 15],  // mode 2: 4-bit indices.
        [15, 8, 4, 2],   // mode 3: 4-bit indices.
        [3, 0, 9, 12],   // mode 4: 4-bit indices.
    ];
    for (mode, gains) in patterns.iter().enumerate() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[mode];
        let bytes = synth_body(&submode, *gains);
        let mut reader = BitReader::new(&bytes);
        let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();
        let resolved = body
            .hb_excitation_gain_indices(&submode)
            .expect("documented mode resolves");
        for (sf, slot) in resolved.iter().enumerate() {
            match (submode.excitation_gain_bits, slot) {
                (0, HbExcitationGainIndex::Absent) => {}
                (5, HbExcitationGainIndex::FiveBit(idx)) => {
                    assert_eq!(u32::from(*idx), gains[sf], "mode {mode} sf {sf}");
                }
                (4, HbExcitationGainIndex::FourBit(idx)) => {
                    assert_eq!(u32::from(*idx), gains[sf], "mode {mode} sf {sf}");
                }
                (bits, other) => {
                    panic!("mode {mode} sf {sf}: budget {bits} resolved to {:?}", other)
                }
            }
            assert_eq!(slot.bit_budget(), submode.excitation_gain_bits);
        }
    }
}

#[test]
fn silence_mode_resolves_absent_with_empty_body() {
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[0];
    let bytes = synth_body(&submode, [0; 4]);
    // Mode 0's body is zero bits; the pad still emits nothing, so the
    // buffer may be empty — the parser must consume zero bits either
    // way.
    let mut reader = BitReader::new(&bytes);
    let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();
    assert_eq!(reader.consumed_bits(), 0);
    let resolved = body.hb_excitation_gain_indices(&submode).unwrap();
    for slot in &resolved {
        assert!(slot.is_absent());
        assert_eq!(slot.raw_index(), None);
        assert_eq!(slot.entries(), None);
    }
}

#[test]
fn gain_resolution_is_independent_of_lsp_and_excitation_vq_content() {
    // Fill the LSP + excitation-VQ fields with all-ones instead of
    // zeros and confirm the resolved gain indices are unchanged — the
    // gain primitive reads only the `Excitation gain` field.
    let submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
    let gains = [5u32, 10, 0, 15];
    let mut w = BitWriter::new();
    w.write(
        (1 << u32::from(submode.lsp_bits)) - 1,
        u32::from(submode.lsp_bits),
    )
    .unwrap();
    for &g in gains.iter() {
        w.write(g, u32::from(submode.excitation_gain_bits)).unwrap();
        w.write(
            (1 << u32::from(submode.excitation_vq_bits)) - 1,
            u32::from(submode.excitation_vq_bits),
        )
        .unwrap();
    }
    w.pad_to_byte().unwrap();
    let bytes = w.into_bytes();
    let mut reader = BitReader::new(&bytes);
    let body = WidebandHighBandBody::parse(&mut reader, &submode).unwrap();
    let resolved = body.hb_excitation_gain_indices(&submode).unwrap();
    for (sf, slot) in resolved.iter().enumerate() {
        assert_eq!(
            *slot,
            HbExcitationGainIndex::FourBit(gains[sf] as u8),
            "sf {sf}"
        );
    }
    assert_eq!(resolved.len(), HIGH_BAND_SUBFRAMES_PER_FRAME as usize);
}
