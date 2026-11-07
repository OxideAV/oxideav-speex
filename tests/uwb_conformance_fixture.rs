//! **Ultra-wideband speech-material tracking gate** (round r410).
//!
//! A speech-like (pitch-glide) 32 kHz reference fixture at quality 4
//! (`tests/fixtures/uwb-conformance/`, black-box `--no-enh` decode —
//! see `NOTES.md`), decoded through [`UltraWidebandDecoder`] and scored
//! absolutely against the reference.
//!
//! **Status (r483): reference-exact.** The exact float decode path in
//! all three layers reproduces this fixture at 78.5 dB on the float
//! output, within ±1 LSB on every 16-bit sample (r410 measured 2.0 dB
//! here; r450 21.9 dB).

use oxideav_speex::{UltraWidebandDecoder, UwbDecodedFrame};

const INPUT: &[u8] = include_bytes!("fixtures/uwb-conformance/uwb_q4.spx");
const EXPECTED: &[u8] = include_bytes!("fixtures/uwb-conformance/uwb_q4.noenh.pcm");

/// Fixed alignment: none. r483 reproduces the reference decoder's
/// one-sub-frame synthesis lag in every layer, so this decode lines up
/// with the reference sample for sample (r410–r480: the reference led
/// by 160 samples at 32 kHz).
const REF_LEAD: usize = 0;

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

fn decode_fixture() -> Vec<f64> {
    let packets = lift_ogg_packets(INPUT);
    assert!(packets.len() > 2, "no audio packets");
    let mut dec = UltraWidebandDecoder::new();
    dec.set_enhancer(false);
    let mut out = Vec::new();
    for (i, pkt) in packets[2..].iter().enumerate() {
        for f in dec
            .decode_packet(pkt)
            .unwrap_or_else(|e| panic!("frame {i}: {e}"))
        {
            if let UwbDecodedFrame::Audio(a) = f {
                out.extend_from_slice(a.uwb_pcm.as_ref());
            }
        }
    }
    out
}

fn score(ours: &[f64], reference: &[f64], lead: usize) -> (f64, f64, f64) {
    let n = (reference.len().saturating_sub(lead)).min(ours.len());
    assert!(n > 20_000, "comparison window too short");
    let r = &reference[lead..lead + n];
    let o = &ours[..n];
    let (mut err, mut er, mut eo, mut dot) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let d = r[i] - o[i];
        err += d * d;
        er += r[i] * r[i];
        eo += o[i] * o[i];
        dot += r[i] * o[i];
    }
    (
        10.0 * (er / (err + 1e-12)).log10(),
        dot / (er.sqrt() * eo.sqrt() + 1e-12),
        eo / er,
    )
}

/// Tracking floors at the r410 measured level (2.0 dB / 0.78 / 1.66):
/// a regression fails; the eventual outer-layer fix shows up as a
/// floor raise.
#[test]
fn uwb_speech_tracking_gate() {
    let reference: Vec<f64> = EXPECTED
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect();
    let ours = decode_fixture();
    assert_eq!(ours.len(), 101 * 640, "101 × 640-sample UWB frames");

    let (snr, corr, energy) = score(&ours, &reference, REF_LEAD);
    println!("uwb-speech-q4: {snr:.2} dB corr {corr:.4} energy {energy:.3}");

    // r483: measured 78.5 dB on the float decode (85.5 dB on the
    // rounded 16-bit output, ±1 LSB everywhere); r450: 21.91 dB / 0.9968.
    assert!(snr >= 75.0, "tracking SNR {snr:.2} dB < 75 dB");
    assert!(corr >= 0.99999, "tracking correlation {corr:.6} < 0.99999");
    assert!(
        (0.999..=1.001).contains(&energy),
        "tracking energy ratio {energy:.4} outside [0.999, 1.001]"
    );
}
