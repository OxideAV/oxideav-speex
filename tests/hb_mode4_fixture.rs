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

use oxideav_speex::{QmfAnalysis, SpeexDecoder};

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
    // Mode-4 high band: r440 state-derived gain base drops the band
    // error from ~13.1 dB (gc-only gain) to ~6.1 dB. The remaining
    // residual is the unpinned exact gain law (provenance/08 records
    // the direction, deliberately not a formula — a discriminating
    // fixture pair is the recorded docs ask).
    assert!(hi < 8.0, "4-8 kHz mean |err| {hi:.2} dB ≥ 8 (regression)");
    if hi < 3.0 {
        println!("NOTE: HB mode-4 gain law appears pinned — tighten this floor");
    }
}

/// Split a 16 kHz signal into its two 8 kHz sub-bands with the crate's
/// analysis bank (whole 320-sample frames; provenance/08's instrument,
/// pinned to its measured 88–95 dB isolation by
/// `tests/qmf_band_isolation.rs`).
fn qmf_split(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut qa = QmfAnalysis::new();
    let frames = x.len() / 320;
    let mut lb = vec![0.0f64; frames * 160];
    let mut hb = vec![0.0f64; frames * 160];
    for f in 0..frames {
        qa.split_slices(
            &x[f * 320..(f + 1) * 320],
            &mut lb[f * 160..(f + 1) * 160],
            &mut hb[f * 160..(f + 1) * 160],
        );
    }
    (lb, hb)
}

fn snr_corr(reference: &[f64], ours: &[f64]) -> (f64, f64) {
    let n = reference.len().min(ours.len());
    let (mut se, mut sr, mut dot, mut ee, mut rr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (r, o) = (reference[i], ours[i]);
        let d = r - o;
        se += d * d;
        sr += r * r;
        dot += r * o;
        ee += o * o;
        rr += r * r;
    }
    let snr = if se <= 0.0 {
        99.0
    } else {
        10.0 * (sr / se).log10()
    };
    let corr = dot / (ee.sqrt() * rr.sqrt()).max(1e-30);
    (snr, corr)
}

/// **QMF-route sub-band conformance** (round r440). Splits both the
/// reference and the crate decode into their true 8 kHz sub-bands with
/// the staged prototype (the provenance/08 route — the isolation is
/// 88–95 dB, so the numbers below measure the codec, not the filter)
/// and scores each sub-band absolutely.
///
/// The isolated high band is the direct external gate on the mode-4
/// innovation path: shape errors, sign errors and gain-law errors all
/// land here undiluted by the ~20 dB louder low band.
#[test]
fn mode4_q10_qmf_subband_gate() {
    let reference: Vec<f64> = EXPECTED
        .chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect();
    let packets = lift(INPUT);
    let mut dec = SpeexDecoder::new();
    let mut ours: Vec<f64> = Vec::new();
    for pkt in &packets[2..] {
        let pcm = dec.decode_packet_pcm_i16(pkt).expect("q10 stream decodes");
        ours.extend(pcm.iter().map(|&s| f64::from(s)));
    }
    // Align at full rate. `expected.pcm` is trimmed to the 24 000-sample
    // source length (fixture notes §PCM geometry), i.e. the toolchain
    // removed the whole codec look-ahead at the front; this decoder's
    // output is untrimmed, so OUR stream is *delayed* relative to the
    // reference. Sweep that delay on low-band waveform correlation and
    // pin it (measured r440: 142 full-rate samples, low-band corr
    // ≈ 0.66 whole-file / 0.94–0.96 on voiced frames).
    let n0 = (reference.len().min(ours.len()) / 320) * 320;
    let (rlb_probe, _) = qmf_split(&reference[..n0]);
    let (olb_probe, _) = qmf_split(&ours[..n0]);
    let mut best_delay = 0usize;
    let mut best_corr = f64::NEG_INFINITY;
    for delay in 0..150usize {
        let m = (olb_probe.len() - delay).min(rlb_probe.len());
        let (_, c) = snr_corr(&rlb_probe[..m], &olb_probe[delay..delay + m]);
        if c > best_corr {
            best_corr = c;
            best_delay = delay;
        }
    }
    println!("half-band best our-delay {best_delay} (lb corr {best_corr:.4})");
    assert!(
        (70..=73).contains(&best_delay),
        "decoder delay vs the trimmed reference moved (was ≈142–144 full-rate samples, got {})",
        2 * best_delay
    );

    // Trim our leading delay at full rate and split both streams.
    let ours_aligned = &ours[2 * best_delay..];
    let n = (reference.len().min(ours_aligned.len()) / 320) * 320;
    let (ref_lb, ref_hb) = qmf_split(&reference[..n]);
    let (our_lb, our_hb) = qmf_split(&ours_aligned[..n]);

    // Skip the shared analysis transient (one half-band frame).
    let (lb_snr, lb_corr) = snr_corr(&ref_lb[160..], &our_lb[160..]);
    let (hb_snr, hb_corr) = snr_corr(&ref_hb[160..], &our_hb[160..]);
    let hb_energy_ratio: f64 = our_hb[160..].iter().map(|&v| v * v).sum::<f64>()
        / ref_hb[160..].iter().map(|&v| v * v).sum::<f64>();
    println!(
        "hb-mode4-wb-q10 QMF sub-bands: low {lb_snr:.2} dB corr {lb_corr:.4} | \
         high {hb_snr:.2} dB corr {hb_corr:.4} energy ratio {hb_energy_ratio:.3}"
    );

    // Low band: whole-file waveform correlation (measured r440: 0.38 —
    // voiced frames sit at 0.94–0.96; the whole-file figure is diluted
    // by the near-crossover fricative content, whose reconstruction
    // needs the high band too).
    assert!(
        lb_corr > 0.30,
        "low sub-band corr {lb_corr:.4} ≤ 0.30 (regression from 0.38)"
    );
    assert!(lb_snr > -3.0, "low sub-band snr {lb_snr:.2} dB ≤ -3");
    // High band: the r440 state-derived gain base + global innovation
    // polarity (both fixture-pinned; the exact gain law remains a
    // recorded docs gap). Measured r440: corr +0.44, energy ratio 0.74.
    // Positive correlation is the polarity pin — a sign regression
    // flips it hard negative.
    assert!(
        hb_corr > 0.30,
        "high sub-band corr {hb_corr:.4} ≤ 0.30 (shape/sign/gain regression)"
    );
    assert!(
        hb_energy_ratio > 0.35 && hb_energy_ratio < 2.0,
        "high sub-band energy ratio {hb_energy_ratio:.3} outside [0.35, 2]"
    );
}
