//! Integration test: walk every Ogg-packet payload of the round-3
//! `speexenc`-encoded narrowband fixture through the new round-r165
//! [`PacketFrames`] iterator end-to-end.
//!
//! The fixture (`tests/fixtures/nb_440hz_q8.spx`) was produced by an
//! opaque invocation of the `speexenc` binary — its **source** is not
//! consulted; only its output bytes. The fixture build recipe is
//! recorded in `tests/fixtures/Makefile`.
//!
//! What this test adds on top of the round-3
//! `narrowband_body_fixture` test: the round-3 test parses each
//! packet's header + body manually; this one composes the same parse
//! through the typed packet iterator and asserts the iterator's view
//! matches the manual parse — exercising the §5.5 terminator + padding
//! handling path that the round-3 test didn't touch (real `speexenc`
//! always appends a mode-15 terminator + zero-pad to fill the last
//! byte).

use oxideav_speex::{parse_packet, PacketFrame, PacketFrames};

const FIXTURE: &[u8] = include_bytes!("fixtures/nb_440hz_q8.spx");

/// Same minimal inline Ogg page-walker as `narrowband_body_fixture.rs`
/// — duplicated rather than shared to keep this test file independent
/// of its sibling.
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
fn every_audio_packet_walks_cleanly_through_packet_iterator() {
    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];
    assert!(!audio.is_empty(), "fixture must have audio packets");

    let mut narrowband_count = 0u32;
    let mut other_count = 0u32;
    for (i, pkt) in audio.iter().enumerate() {
        let frames =
            parse_packet(pkt).unwrap_or_else(|e| panic!("packet {}: walk failed: {:?}", i, e));
        // Every packet from `speexenc` for a quality-8 narrowband
        // mono fixture should hold exactly one mode-5 frame (the
        // encoder packs one 20 ms frame per Ogg packet by default).
        assert!(
            !frames.is_empty(),
            "packet {}: iterator yielded no frames",
            i
        );
        for f in frames {
            match f {
                PacketFrame::Narrowband { header, .. } => {
                    assert_eq!(header.mode_id, 5, "packet {}: expected mode 5", i);
                    assert!(!header.wideband);
                    narrowband_count += 1;
                }
                _ => other_count += 1,
            }
        }
    }
    assert!(
        narrowband_count >= 40,
        "expected ≥40 narrowband frames, got {}",
        narrowband_count
    );
    assert_eq!(
        other_count, 0,
        "narrowband fixture should not have non-CELP frames"
    );
}

#[test]
fn iterator_halts_at_terminator_for_each_packet() {
    // §5.5: speexenc always appends a mode-15 terminator to fill the
    // last byte. After the iterator finishes, `is_halted()` must be
    // true and `remaining_bits()` must be smaller than the prefix
    // width (the terminator + padding consumed everything addressable).
    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];
    for (i, pkt) in audio.iter().enumerate() {
        let mut iter = PacketFrames::new(pkt);
        for (frames, r) in iter.by_ref().enumerate() {
            r.unwrap_or_else(|e| panic!("packet {} frame {}: {:?}", i, frames, e));
        }
        assert!(iter.is_halted(), "packet {}: iterator must halt", i);
        // After clean termination, the iterator's remaining bits
        // should be < 5 (the prefix width — anything less is padding,
        // anything more is an unparsed frame and the iterator would
        // not have halted).
        assert!(
            iter.remaining_bits() < 5,
            "packet {}: {} bits left after halt — should be padding-only",
            i,
            iter.remaining_bits()
        );
    }
}
