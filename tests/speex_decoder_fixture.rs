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
                DecodedFrame::Control(_) => {}
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

/// Round r347 boundedness milestone: with the LSP base vector +
/// `LSP_MARGIN` pinned, the closed-loop decode of the real `speexenc`
/// mode-5 (q8) fixture stays **non-divergent** across the whole stream —
/// the live excitation feedback no longer drives the output to runaway
/// (1e10+) magnitudes that a genuinely unstable LPC set (poles outside
/// the unit circle) would produce, because every reconstructed LSP angle
/// is now forced inside the conformant, strictly interlaced band.
///
/// The output is not yet at reference *amplitude* (the remaining
/// cosine-series fixed-point Q-format + the absolute gain calibration are
/// a documented docs gap, so the synthesis filter sits in a not-yet-
/// reference envelope and the peak runs above full-scale i16), but it is
/// **bounded** frame-to-frame rather than growing without limit: the
/// last-quarter peak does not exceed the first-quarter peak by more than
/// a small factor. That non-growth is the concrete boundedness signal the
/// base-vector + margin pin delivers; a regression that un-pinned the LSP
/// band would send the tail magnitudes orders of magnitude above the head.
#[test]
fn closed_loop_decode_is_non_divergent() {
    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];

    let mut decoder = SpeexDecoder::new();
    let mut frame_peaks: Vec<f64> = Vec::new();
    for pkt in audio {
        for frame in decoder.decode_packet(pkt).expect("packet decodes") {
            if let DecodedFrame::Narrowband(pcm) = frame {
                let mut peak = 0.0f64;
                for &s in pcm.iter() {
                    assert!(s.is_finite(), "non-finite PCM sample");
                    peak = peak.max(s.abs());
                }
                frame_peaks.push(peak);
            }
        }
    }
    assert!(frame_peaks.len() >= 40, "fixture should decode many frames");

    // Every frame's peak is finite and far below the runaway range an
    // unstable (out-of-band LSP) filter would reach within the feedback
    // loop.
    let global_peak = frame_peaks.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        global_peak.is_finite() && global_peak < 1.0e8,
        "global peak {global_peak} is in the runaway range — LSP→LPC filter unstable"
    );

    // Non-growth: the mean peak over the last quarter of the stream is not
    // dramatically larger than over the first quarter. A divergent filter
    // would show the tail many orders of magnitude above the head.
    let n = frame_peaks.len();
    let q = (n / 4).max(1);
    let head: f64 = frame_peaks[..q].iter().sum::<f64>() / q as f64;
    let tail: f64 = frame_peaks[n - q..].iter().sum::<f64>() / q as f64;
    assert!(
        tail <= head * 50.0 + 1.0,
        "tail peak mean {tail} grew {:.1}× over head {head} — filter is diverging",
        tail / head.max(1.0)
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
