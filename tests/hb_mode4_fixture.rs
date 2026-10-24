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
//! 2. The low band (0–4 kHz, NB 7) is reference-tracking (r450: the
//!    measured mode-7 stage-2 weight closes the former +1.6 dB energy
//!    bias — ≈1.7 dB mean band error, energy ratio ≈0.99).
//! 3. The mode-4 high band (4–8 kHz) decodes through the exact gain law
//!    of `docs/audio/speex/tracer-2026-09.md` §1
//!    (`S = 0.8736·bound[q]·e_L/R`, see
//!    `oxideav_speex::hb_gc_crossover_gain`): isolated sub-band SNR
//!    ≈28.6 dB at correlation ≈0.9993, energy ratio 1.002 (r480; the
//!    per-sub-frame scale itself is gated against the staged trace by
//!    `tests/hb_gain_trace_fixture.rs`).

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
        .as_chunks::<2>()
        .0
        .iter()
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

    // Low band (NB 7) tracks the reference (r450 measured 1.74 dB).
    assert!(lo < 2.5, "0-4 kHz mean |err| {lo:.2} dB ≥ 2.5 (regression)");
    // Mode-4 high band through the r450 crossover-anchored gain law
    // (r480 measured 3.03 dB on this crude per-frame DFT metric; was
    // ~6.1 dB under the r440 fitted law and ~13.1 dB correction-only —
    // the QMF sub-band gate below is the precise one).
    assert!(hi < 4.5, "4-8 kHz mean |err| {hi:.2} dB ≥ 4.5 (regression)");
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
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect();
    let packets = lift(INPUT);
    let mut dec = SpeexDecoder::new();
    let mut ours: Vec<f64> = Vec::new();
    for pkt in &packets[2..] {
        let pcm = dec.decode_packet_pcm_i16(pkt).expect("q10 stream decodes");
        ours.extend(pcm.iter().map(|&s| f64::from(s)));
    }
    // Align at full rate — over BOTH parities. `expected.pcm` is
    // trimmed to the 24 000-sample source length (fixture notes §PCM
    // geometry): the toolchain removed the whole codec look-ahead at
    // the front, so OUR stream is *delayed* relative to the reference.
    // The r440 revision of this gate swept that delay in half-band
    // steps (even full-rate delays only) and pinned "142"; the r450
    // crafted-stream probes showed the true delay is **odd** (143) —
    // and because the QMF high band is recovered through a (−1)ⁿ
    // modulation, a one-sample parity error *negates* the recovered
    // high band (and comb-filters both bands), which is why the r440
    // gate read hb corr +0.44 under the then-inverted mode-4 polarity.
    // Sweep every full-rate delay on low-band waveform correlation.
    let n0 = (reference.len().min(ours.len()) / 320) * 320;
    let (rlb_probe, _) = qmf_split(&reference[..n0]);
    let mut best_delay = 0usize;
    let mut best_corr = f64::NEG_INFINITY;
    for delay in 100..200usize {
        let oa = &ours[delay..];
        let n = ((n0 - delay).min(rlb_probe.len() * 2) / 320) * 320;
        let (olb, _) = qmf_split(&oa[..n]);
        let m = olb.len().min(rlb_probe.len());
        let (_, c) = snr_corr(&rlb_probe[160..m], &olb[160..m]);
        if c > best_corr {
            best_corr = c;
            best_delay = delay;
        }
    }
    println!("full-rate best our-delay {best_delay} (lb corr {best_corr:.4})");
    assert_eq!(
        best_delay, 143,
        "decoder delay vs the trimmed reference moved"
    );

    // Trim our leading delay at full rate and split both streams.
    let ours_aligned = &ours[best_delay..];
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

    // r480 measured: low 25.60 dB / 0.9986; high 28.64 dB / 0.9993 /
    // energy 1.002 — the exact tracer-2026-09.md §1 gain law
    // (S = g_c·e_L/R, both +0.01 / +0.1 regularisers) replaced the r450
    // reading (low 13.72 / high 21.13 dB, energy 1.119).
    assert!(lb_corr > 0.995, "low sub-band corr {lb_corr:.4} ≤ 0.995");
    assert!(lb_snr > 22.0, "low sub-band snr {lb_snr:.2} dB ≤ 22");
    assert!(
        hb_corr > 0.998,
        "high sub-band corr {hb_corr:.4} ≤ 0.998 (shape/sign/gain regression)"
    );
    assert!(hb_snr > 25.0, "high sub-band snr {hb_snr:.2} dB ≤ 25");
    assert!(
        hb_energy_ratio > 0.97 && hb_energy_ratio < 1.03,
        "high sub-band energy ratio {hb_energy_ratio:.3} outside [0.97, 1.03]"
    );
}
