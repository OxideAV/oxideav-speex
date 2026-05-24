//! Integration test: parse every audio packet of a real
//! `speexenc`-encoded narrowband file end-to-end through round-3's
//! frame-body bit-reader.
//!
//! The fixture is generated via the black-box `speexenc` binary —
//! `tests/fixtures/Makefile`'s `regen` target is the audit trail.
//! `speexenc`'s **source** is not consulted; it is invoked as an
//! opaque process and its output bytes are the test input.
//!
//! Per workspace policy this crate does NOT take a dev-dependency on
//! `oxideav-ogg`. Instead, the test contains a minimal inline Ogg
//! page-walker (≈30 lines) that pulls the Speex audio packets back
//! out of their Ogg framing — just enough to feed the codec under
//! test. No Ogg encoding, no chained-stream support, no granulepos
//! handling, no metadata; only what's needed to lift packets out of a
//! single-stream `.spx` file produced by `speexenc`.
//!
//! What is asserted:
//!
//! * The first audio packet's leading 5-bit header parses as
//!   `wideband=0, mode=5` (`quality 8` per Speex Manual Table 9.2
//!   maps to narrowband sub-mode 5 — 15 kbps, 300 bits per 20 ms
//!   frame).
//! * For every audio packet, [`NarrowbandFrameBody::parse`] consumes
//!   exactly `total_bits - 5` further bits without underflow.
//! * The cumulative cursor position lands within the packet's byte
//!   span (with at most 7 bits of padding leftover, since the encoder
//!   rounds up to a whole-byte packet — per §5.5 the encoder
//!   appends a mode-15 terminator and/or trailing zeros to pad).

use oxideav_speex::{
    NarrowbandFrameBody, NarrowbandFrameHeader, Submode, NARROWBAND_FRAME_PREFIX_BITS,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/nb_440hz_q8.spx");

/// Inline minimal Ogg page-walker. Iterates packets reassembled from
/// per-page segment tables; returns the packet bodies as owned `Vec`s.
///
/// This is NOT a general-purpose Ogg implementation — it deliberately
/// ignores chained streams, the BOS/EOS flags, the granule position,
/// the bitstream serial number, the CRC, and the page sequence
/// number. It does just one thing: stitch lacing-segment runs back
/// into Speex packets.
fn lift_ogg_packets(buf: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos + 27 <= buf.len() {
        if &buf[pos..pos + 4] != b"OggS" {
            break;
        }
        let nseg = buf[pos + 26] as usize;
        if pos + 27 + nseg > buf.len() {
            break;
        }
        let segments = &buf[pos + 27..pos + 27 + nseg];
        let mut body_pos = pos + 27 + nseg;
        for &ln in segments {
            let take = ln as usize;
            if body_pos + take > buf.len() {
                return packets;
            }
            current.extend_from_slice(&buf[body_pos..body_pos + take]);
            body_pos += take;
            if take < 255 {
                packets.push(core::mem::take(&mut current));
            }
        }
        pos = body_pos;
    }
    packets
}

#[test]
fn audio_packets_round_trip_through_narrowband_body_parser() {
    let packets = lift_ogg_packets(FIXTURE);
    // Speex stream packets: [0] = Speex header, [1] = comment header,
    // [2..] = audio. The fixture is 1 s of 8 kHz mono so the encoder
    // emits 50 frames + (possibly) a small tail; assert there's at
    // least 40 audio packets to make the test robust to encoder-side
    // packing tweaks.
    assert!(
        packets.len() >= 42,
        "expected at least 42 Ogg packets (1 header + 1 comment + ≥40 audio), got {}",
        packets.len()
    );

    let audio = &packets[2..];
    assert!(!audio.is_empty(), "no audio packets in fixture");

    // First audio packet: 5-bit prefix → wideband=0, mode=5.
    let (h0, _) = NarrowbandFrameHeader::parse_bytes(&audio[0])
        .expect("first audio packet must parse a frame header");
    assert!(!h0.wideband, "fixture is pure narrowband");
    assert_eq!(h0.mode_id, 5, "speexenc quality 8 → narrowband mode 5");

    // Every audio packet: header + body must walk without underflow,
    // and the cursor must end inside the packet (mode 5 = 300 bits ≤
    // 38 bytes = 304 bits, leaving up to 4 padding bits).
    let mut parsed = 0u32;
    for (i, pkt) in audio.iter().enumerate() {
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(pkt)
            .unwrap_or_else(|e| panic!("packet {}: header parse failed: {:?}", i, e));
        let s = match h.submode {
            Submode::Celp(s) => s,
            other => panic!("packet {}: unexpected submode {:?}", i, other),
        };
        let body = NarrowbandFrameBody::parse(&mut r, &s)
            .unwrap_or_else(|e| panic!("packet {}: body parse failed: {:?}", i, e));

        // Field-width sanity for mode 5.
        assert!(body.lsp_index < (1u32 << 30));
        assert!(body.ol_exc_gain_index < (1u8 << 5));
        for (sf_idx, sf) in body.subframes.iter().enumerate() {
            assert!(
                sf.pitch_index < (1u8 << 7),
                "packet {} subframe {}: pitch_index out of range",
                i,
                sf_idx
            );
            assert!(
                sf.pitch_gain_index < (1u8 << 7),
                "packet {} subframe {}: pitch_gain_index out of range",
                i,
                sf_idx
            );
            assert!(
                sf.innovation_gain_index < (1u8 << 3),
                "packet {} subframe {}: innovation_gain_index out of range",
                i,
                sf_idx
            );
            assert!(
                sf.innovation_vq_index < (1u128 << 48),
                "packet {} subframe {}: innovation_vq_index out of range",
                i,
                sf_idx
            );
        }

        // Cursor must land within the packet (with <= 7 bits trailing
        // padding for the byte-align tail).
        let consumed = r.consumed_bits();
        let expected = u32::from(s.total_bits);
        assert_eq!(
            consumed, expected,
            "packet {}: cursor expected at {} bits, got {}",
            i, expected, consumed
        );
        let pkt_bits = (pkt.len() as u32) * 8;
        assert!(
            consumed <= pkt_bits,
            "packet {}: parse over-ran packet ({} > {} bits)",
            i,
            consumed,
            pkt_bits
        );
        assert!(
            pkt_bits - consumed < 8,
            "packet {}: more than one padding byte left ({} bits)",
            i,
            pkt_bits - consumed
        );
        parsed += 1;
    }
    assert_eq!(
        parsed as usize,
        audio.len(),
        "must successfully parse every audio packet"
    );
}

#[test]
fn first_audio_packet_uses_full_5_bit_prefix() {
    let packets = lift_ogg_packets(FIXTURE);
    let pkt0 = &packets[2];
    // First byte top 5 bits = wideband(1) | mode_id(4); the bottom 3
    // bits start the LSP VQ field for mode 5 (30 bits of LSP). They
    // can be anything, but the prefix-consumed-bit-count must equal
    // NARROWBAND_FRAME_PREFIX_BITS regardless.
    let (_, r) = NarrowbandFrameHeader::parse_bytes(pkt0).unwrap();
    assert_eq!(r.consumed_bits(), NARROWBAND_FRAME_PREFIX_BITS);
}
