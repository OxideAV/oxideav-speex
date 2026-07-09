//! **Ultra-wideband decoder robustness / fuzz gate** (round r403).
//!
//! The [`UltraWidebandDecoder`] must be *total* over arbitrary byte
//! input: for a stream whose rate class is UWB (the out-of-band context
//! the decoder assumes), any packet — random bytes, a truncation of a
//! real frame, a bit-flipped real frame — must return `Ok`/`Err` and
//! never panic, and any decoded PCM must be finite. These are the
//! properties a decoder fed hostile / corrupted network input needs;
//! the staged fixture supplies the real packets to mutate.

use oxideav_speex::{UltraWidebandDecoder, UwbDecodedFrame};

const INPUT: &[u8] = include_bytes!("fixtures/uwb-fold-geometry/input.spx");

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

/// A tiny deterministic PRNG (SplitMix64) so the fuzz corpus is fixed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn byte(&mut self) -> u8 {
        (self.next() & 0xFF) as u8
    }
}

/// Decode a packet and assert only totality + finiteness (never panic).
fn check_total(dec: &mut UltraWidebandDecoder, pkt: &[u8]) {
    if let Ok(frames) = dec.decode_packet(pkt) {
        for f in &frames {
            if let UwbDecodedFrame::Audio(a) = f {
                assert!(
                    a.uwb_pcm.iter().all(|s| s.is_finite()),
                    "decoded PCM must be finite"
                );
                assert!(
                    a.uwb_high_band.iter().all(|s| s.is_finite()),
                    "decoded high band must be finite"
                );
            }
        }
    }
}

#[test]
fn random_packets_never_panic() {
    let mut rng = Rng(0x0000_5EED_u64);
    for len in [0usize, 1, 3, 7, 16, 40, 64, 128, 300] {
        // A fresh decoder per length keeps the corpus reproducible;
        // also feed a sequence into one decoder to exercise carried
        // state across garbage frames.
        let mut fresh = UltraWidebandDecoder::new();
        let mut shared = UltraWidebandDecoder::new();
        for _ in 0..200 {
            let pkt: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
            check_total(&mut fresh, &pkt);
            check_total(&mut shared, &pkt);
        }
    }
}

#[test]
fn truncations_of_real_frames_never_panic() {
    let packets = lift_ogg_packets(INPUT);
    let audio = &packets[2..];
    assert!(!audio.is_empty());
    for pkt in audio.iter().take(20) {
        // Every prefix length of a real frame — the decoder must survive
        // a cut at any bit-boundary-rounded byte.
        for cut in 0..=pkt.len() {
            let mut dec = UltraWidebandDecoder::new();
            check_total(&mut dec, &pkt[..cut]);
        }
    }
}

#[test]
fn bit_flips_of_real_frames_never_panic() {
    let packets = lift_ogg_packets(INPUT);
    let audio = &packets[2..];
    let mut rng = Rng(0x00C0_FFEE_u64);
    for pkt in audio.iter().take(30) {
        for _ in 0..64 {
            let mut m = pkt.clone();
            if !m.is_empty() {
                let i = (rng.next() as usize) % m.len();
                let bit = (rng.next() % 8) as u8;
                m[i] ^= 1 << bit;
            }
            let mut dec = UltraWidebandDecoder::new();
            check_total(&mut dec, &m);
        }
    }
}

#[test]
fn flat_pcm_paths_are_total_too() {
    // The i16 convenience path must be equally total.
    let mut rng = Rng(0x1234_5678_u64);
    let mut dec = UltraWidebandDecoder::new();
    for _ in 0..2_000 {
        let len = (rng.next() as usize) % 200;
        let pkt: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
        if let Ok(pcm) = dec.decode_packet_pcm_i16(&pkt) {
            assert!(pcm.len() % 640 == 0, "flat UWB PCM is whole frames");
        }
    }
}
