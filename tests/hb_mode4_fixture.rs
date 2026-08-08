//! **High-band mode-4 (quality 10) decode gate** (campaign B).
//!
//! Drives the staged wideband quality-10 oracle
//! (`docs/audio/speex/fixtures/hb-mode4-wb-q10/`, mirrored under
//! `tests/fixtures/`) — 76 frames, NB submode 7 + HB submode 4 in every
//! frame — through the crate's [`oxideav_speex::SpeexDecoder`].
//!
//! ## What this locks
//!
//! Before campaign B, HB mode 4 (80-bit two-stage innovation) surfaced a
//! docs-gap error and a q10 wideband stream was undecodable. The
//! two-stage `sv8-128` binding (`docs/audio/speex/hb-innovation-binding.md`
//! §1/§2: five 8-bit groups × two stages, MSB sign bit, stage 2 at
//! weight 0.4) now *decodes* the stream. This gate pins:
//!
//! 1. The stream decodes without error (76 wideband frames).
//! 2. The low band (0–4 kHz, NB 7) is reference-tracking (~2 dB mean
//!    band error).
//! 3. The mode-4 high band (4–8 kHz) tracks with a **documented
//!    residual** (~13 dB mean band error at the doc-faithful magnitude).
//!    The residual is the absolute per-frame HB-innovation gain/energy
//!    law, which the staged evidence (codebook *shape* + the 0.4 stage
//!    weight, via sign-difference isolation) does not pin — recorded in
//!    the crate README as a precise docs gap. A fix drops the 4–8 kHz
//!    floor; a regression fails loudly.

use oxideav_speex::SpeexDecoder;

const INPUT: &[u8] = include_bytes!("fixtures/hb-mode4-wb-q10/input.spx");
const EXPECTED: &[u8] = include_bytes!("fixtures/hb-mode4-wb-q10/expected.pcm");

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

/// Magnitude band energies (dB) of one 320-sample 16 kHz frame:
/// 0–4 kHz (bins 1..80), 4–8 kHz (bins 80..160).
fn band_db(frame: &[f64]) -> (f64, f64) {
    let n = frame.len();
    let (mut lo, mut hi) = (0.0f64, 0.0f64);
    for k in 1..160usize {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        let w = -2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
        for (i, &x) in frame.iter().enumerate() {
            re += x * (w * i as f64).cos();
            im += x * (w * i as f64).sin();
        }
        let p = re * re + im * im;
        if k < 80 {
            lo += p;
        } else {
            hi += p;
        }
    }
    let db = |e: f64| {
        if e <= 1e-9 {
            -90.0
        } else {
            10.0 * (e / n as f64).log10()
        }
    };
    (db(lo), db(hi))
}

#[test]
fn mode4_q10_decodes_with_documented_residual() {
    let reference: Vec<f64> = EXPECTED
        .chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect();
    let packets = lift(INPUT);
    let mut dec = SpeexDecoder::new();
    let mut ours: Vec<f64> = Vec::new();
    let mut frames = 0usize;
    for pkt in &packets[2..] {
        let pcm = dec
            .decode_packet_pcm_i16(pkt)
            .expect("q10 / HB mode-4 stream must decode");
        frames += 1;
        ours.extend(pcm.iter().map(|&s| f64::from(s)));
    }
    assert!(frames >= 70, "≈76 wideband frames, got {frames}");
    assert_eq!(ours.len() % 320, 0, "whole 320-sample WB frames");

    // Per-frame magnitude band error (reference leads by 138 samples,
    // the WB double-QMF look-ahead).
    let lead = 138usize;
    let (mut slo, mut shi, mut c) = (0.0f64, 0.0f64, 0usize);
    for f in 0..75usize {
        let ro = lead + f * 320;
        let oo = f * 320;
        if ro + 320 > reference.len() || oo + 320 > ours.len() {
            break;
        }
        let rb = band_db(&reference[ro..ro + 320]);
        let ob = band_db(&ours[oo..oo + 320]);
        slo += (rb.0 - ob.0).abs();
        shi += (rb.1 - ob.1).abs();
        c += 1;
    }
    let (lo, hi) = (slo / c as f64, shi / c as f64);
    println!("hb-mode4-wb-q10 band mean|err| dB: 0-4k={lo:.2} 4-8k={hi:.2}");

    // Low band (NB 7) tracks the reference.
    assert!(lo < 3.5, "0-4 kHz mean |err| {lo:.2} dB ≥ 3.5 (regression)");
    // Mode-4 high band: documented residual floor (measured ~13 dB).
    assert!(hi < 15.0, "4-8 kHz mean |err| {hi:.2} dB ≥ 15 (regression)");
    if hi < 6.0 {
        println!("NOTE: HB mode-4 gain law appears pinned — tighten this floor");
    }
}
