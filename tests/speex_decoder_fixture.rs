//! Integration test: drive the real `speexenc` narrowband fixture
//! through the **top-level** [`SpeexDecoder`] (packet → `Vec<DecodedFrame>`),
//! the highest-level public decode entry point.
//!
//! Unlike `narrowband_decoder_fixture.rs` (which calls the per-frame
//! `NarrowbandDecoder` directly), this test exercises the full
//! packet-walking dispatch: the [`SpeexDecoder`] consumes each Ogg audio
//! packet as a Speex packet, the [`oxideav_speex::PacketFrames`] iterator
//! inside it splits the packet into frames, and each CELP frame is
//! decoded to PCM with the shared decoder state carried across the whole
//! stream.
//!
//! The fixture is the same `tests/fixtures/nb_440hz_q8.spx` (black-box
//! `speexenc`, source NOT consulted). No `oxideav-ogg` dev-dependency;
//! the inline page-walker lifts the audio packets.

use oxideav_speex::{DecodedFrame, SpeexDecoder};

const FIXTURE: &[u8] = include_bytes!("fixtures/nb_440hz_q8.spx");

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
fn top_level_decoder_walks_fixture_to_pcm() {
    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];
    assert!(!audio.is_empty());

    let mut decoder = SpeexDecoder::new();
    let mut nb_frames = 0usize;
    let mut total_samples = 0usize;

    for pkt in audio {
        let decoded = decoder.decode_packet(pkt).expect("packet decodes");
        for frame in decoded {
            match frame {
                DecodedFrame::Narrowband(pcm) => {
                    assert_eq!(pcm.len(), 160);
                    for &s in pcm.iter() {
                        assert!(s.is_finite(), "non-finite PCM sample");
                    }
                    nb_frames += 1;
                    total_samples += pcm.len();
                }
                DecodedFrame::Wideband { .. } => {
                    panic!("narrowband fixture should not yield wideband frames")
                }
                DecodedFrame::Control => {}
            }
        }
    }

    assert!(
        nb_frames >= 40,
        "expected ≥ 40 narrowband frames, got {nb_frames}"
    );
    assert_eq!(
        total_samples,
        nb_frames * 160,
        "every narrowband frame yields 160 samples"
    );
}

#[test]
fn top_level_decoder_is_deterministic() {
    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];

    let collect = || {
        let mut decoder = SpeexDecoder::new();
        let mut all = Vec::new();
        for pkt in audio {
            for frame in decoder.decode_packet(pkt).unwrap() {
                if let DecodedFrame::Narrowband(pcm) = frame {
                    all.extend_from_slice(pcm.as_ref());
                }
            }
        }
        all
    };

    let a = collect();
    let b = collect();
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "non-deterministic PCM at sample {i}");
    }
}
