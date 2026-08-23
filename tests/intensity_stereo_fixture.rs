//! **Intensity-stereo conformance gate** (campaign B).
//!
//! Drives the staged narrowband stereo oracle
//! (`docs/audio/speex/fixtures/stereo-nb-ladder-q4/`, mirrored under
//! `tests/fixtures/`) — a 10-segment amplitude-panning ladder (bal
//! 0…31, both signs, all four `e_ratio` values) — through the crate's
//! stereo reconstruction ([`oxideav_speex::StereoDecoder`] +
//! [`oxideav_speex::SpeexStreamDecoder::decode_packet_frames_stereo`]).
//!
//! The oracle ships both `expected.pcm` (interleaved L/R) and
//! `expected-mono.pcm` (the reference `--mono` decode). Because the
//! stereo law sits *above* the mono CELP decode, the interleaved match
//! can be no better than the mono match; the gate asserts the
//! interleaved SNR **tracks** the mono SNR — i.e. the L/R
//! reconstruction (gains + §4 interpolation) adds no material error on
//! top of the (reference-imperfect) mono decode. Per the staged note
//! (`intensity-stereo.md` §4.1) the sub-frame block-phase offset is a
//! `speexdec`-pipeline detail this decoder does not reproduce, so
//! byte-exactness is bounded by it while the per-sample gains are
//! reference-correct.

use oxideav_speex::{SpeexHeader, SpeexStreamDecoder, StereoDecoder};

const INPUT: &[u8] = include_bytes!("fixtures/stereo-nb-ladder-q4/input.spx");
const EXPECTED: &[u8] = include_bytes!("fixtures/stereo-nb-ladder-q4/expected.pcm");
const EXPECTED_MONO: &[u8] = include_bytes!("fixtures/stereo-nb-ladder-q4/expected-mono.pcm");

fn lift(buf: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos + 27 <= buf.len() {
        if &buf[pos..pos + 4] != b"OggS" {
            break;
        }
        let nseg = buf[pos + 26] as usize;
        let segments = &buf[pos + 27..pos + 27 + nseg];
        let mut body_pos = pos + 27 + nseg;
        for &ln in segments {
            let take = ln as usize;
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

fn f64_pcm(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn best_snr(reference: &[f64], ours: &[f64], step: usize, max_lead: usize) -> (f64, usize) {
    let mut best = -99.0;
    let mut best_lead = 0;
    let mut lead = 0;
    while lead < max_lead {
        let n = reference.len().saturating_sub(lead).min(ours.len());
        if n > 0 {
            let (mut e, mut s) = (0.0f64, 0.0f64);
            for i in 0..n {
                let d = reference[lead + i] - ours[i];
                e += d * d;
                s += reference[lead + i] * reference[lead + i];
            }
            let snr = 10.0 * (s / (e + 1e-9)).log10();
            if snr > best {
                best = snr;
                best_lead = lead;
            }
        }
        lead += step;
    }
    (best, best_lead)
}

fn decode() -> (Vec<f64>, Vec<f64>) {
    let packets = lift(INPUT);
    let header = SpeexHeader::parse(&packets[0]).unwrap();
    let mut dec = SpeexStreamDecoder::for_header(&header).unwrap();
    let mut sd = StereoDecoder::new();
    let mut interleaved = Vec::new();
    let mut mono = Vec::new();
    for pkt in &packets[2..] {
        for (m, payload) in dec.decode_packet_frames_stereo(pkt).unwrap() {
            mono.extend(m.iter().map(|&s| f64::from(s)));
            let p = payload.unwrap_or(0b0000_0011);
            interleaved.extend(sd.interleave_frame(&m, p).iter().map(|&s| f64::from(s)));
        }
    }
    (interleaved, mono)
}

/// The in-band code-9 message is present on every audio frame (the
/// payload is `Some` for all of them) — a stereo stream that a parser
/// mis-frames would not decode cleanly.
#[test]
fn every_frame_carries_a_stereo_message() {
    let packets = lift(INPUT);
    let header = SpeexHeader::parse(&packets[0]).unwrap();
    let mut dec = SpeexStreamDecoder::for_header(&header).unwrap();
    let mut frames = 0usize;
    let mut with_payload = 0usize;
    for pkt in &packets[2..] {
        for (_m, payload) in dec.decode_packet_frames_stereo(pkt).unwrap() {
            frames += 1;
            if payload.is_some() {
                with_payload += 1;
            }
        }
    }
    assert!(
        frames > 90,
        "fixture carries ~101 audio frames, got {frames}"
    );
    assert_eq!(with_payload, frames, "every audio frame carries code 9");
}

/// The stereo reconstruction adds no material error over the mono
/// decode: the interleaved SNR tracks the mono SNR.
#[test]
fn interleaved_tracks_mono_decode() {
    let reference = f64_pcm(EXPECTED);
    let ref_mono = f64_pcm(EXPECTED_MONO);
    let (interleaved, mono) = decode();

    let (mono_snr, _) = best_snr(&ref_mono, &mono, 1, 400);
    // Interleaved leads must stay on L/R parity (even step).
    let (inter_snr, _) = best_snr(&reference, &interleaved, 2, 800);
    println!("stereo-nb-q4: mono {mono_snr:.2} dB, interleaved {inter_snr:.2} dB");

    // The mono decode floor (r450: the NB fixes lifted this fixture's
    // mono decode from 13.8 to ≈20.7 dB).
    assert!(mono_snr > 18.0, "mono decode SNR {mono_snr:.2} dB < 18");
    // The stereo law tracks it within 3 dB (measured 18.4 vs 20.7 —
    // with the mono decode now this close to the reference, the §4.1
    // sub-frame block-phase approximation is the visible term; a
    // stereo-law probe campaign is the recorded follow-up) and stays
    // well above the pre-r450 interleaved level outright.
    assert!(
        inter_snr > mono_snr - 3.0,
        "interleaved {inter_snr:.2} dB lags mono {mono_snr:.2} by > 3 dB (stereo law regressed)"
    );
    assert!(inter_snr > 16.0, "interleaved {inter_snr:.2} dB < 16");
}
