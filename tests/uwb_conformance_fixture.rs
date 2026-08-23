//! **Ultra-wideband speech-material tracking gate** (round r410).
//!
//! A speech-like (pitch-glide) 32 kHz reference fixture at quality 4
//! (`tests/fixtures/uwb-conformance/`, black-box `--no-enh` decode —
//! see `NOTES.md`), decoded through [`UltraWidebandDecoder`] and scored
//! absolutely against the reference.
//!
//! **Status: known-divergent, tracking floors.** Where the staged tone
//! fixture decodes at 19.1 dB (`uwb_fold_geometry_fixture.rs`), this
//! speech fixture measures **2.0 dB / corr 0.78 / energy 1.66** (r410)
//! — and the divergence is *not* the r410 crossover-shaped inner fold
//! (which lifted the wideband speech fixture from −12.9 to 15.6 dB):
//! the embedded 0–8 kHz half itself scores only 2.3 dB at 1.6× energy
//! here, far below the 15.6 dB the same code path scores on the
//! standalone wideband speech fixture, and the 8–16 kHz folded layer
//! overshoots ≈10×. Shaping the *outer* fold with the inner law's
//! crossover normalisation (slope scaled by analogy, ceiling 1/16)
//! barely moves this fixture and *regresses* the tone fixture — so the
//! outer-layer speech behaviour is a distinct, unpinned mechanism
//! (recorded follow-up + docs ask). This gate pins today's measured
//! level so the eventual fix shows up as a floor raise, and any
//! regression fails loudly.

use oxideav_speex::{UltraWidebandDecoder, UwbDecodedFrame};

const INPUT: &[u8] = include_bytes!("fixtures/uwb-conformance/uwb_q4.spx");
const EXPECTED: &[u8] = include_bytes!("fixtures/uwb-conformance/uwb_q4.noenh.pcm");

/// Fixed alignment: the reference decode leads ours by 160 samples at
/// 32 kHz (the double-QMF look-ahead padding, same as the tone
/// fixture).
const REF_LEAD: usize = 160;

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
        .chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect();
    let ours = decode_fixture();
    assert_eq!(ours.len(), 101 * 640, "101 × 640-sample UWB frames");

    let (snr, corr, energy) = score(&ours, &reference, REF_LEAD);
    println!("uwb-speech-q4: {snr:.2} dB corr {corr:.4} energy {energy:.3}");

    // r450: measured 21.91 dB / 0.9968 (was ≈2 dB under the r403 law).
    assert!(snr >= 19.0, "tracking SNR {snr:.2} dB < 19 dB");
    assert!(corr >= 0.99, "tracking correlation {corr:.4} < 0.99");
    assert!(
        (0.8..=2.2).contains(&energy),
        "tracking energy ratio {energy:.3} outside [0.8, 2.2]"
    );

    // The known-divergence marker: if the decode ever clears 6 dB the
    // divergence is materially fixed — raise the floors and drop the
    // "known-divergent" status in the module docs + README.
    if snr > 6.0 {
        println!("NOTE: uwb speech divergence appears fixed ({snr:.2} dB) — raise these floors");
    }
}
