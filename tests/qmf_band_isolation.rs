//! **QMF analysis band-isolation exactness gate** (round r440).
//!
//! `docs/audio/speex/provenance/08-qmf-recovered-hb-excitation.md`
//! validates the staged 64-tap prototype (`tables/qmf-filter-h0-float.csv`)
//! as a *measurement instrument*: pure tones pushed through the textbook
//! analysis pair `h0[n]` / `h1[n] = (−1)ⁿ·h0[n]` (each decimated 2:1)
//! separate the two 8 kHz sub-bands to **88–95 dB** — measured
//! high/low-band energy ratios of −95.1 / −95.6 / −87.9 dB at
//! 1 / 2 / 3 kHz and +87.9 / +95.6 / +95.1 dB at 5 / 6 / 7 kHz. That
//! isolation is what makes the staged fixtures' sub-bands recoverable
//! from decoded PCM *without an oracle* (it is 60–70 dB better than the
//! generic FIR route round 07 recorded as leakage-dominated).
//!
//! This gate pins the crate's own [`oxideav_speex::QmfAnalysis`] as the
//! same instrument: driving the six provenance tones through it must
//! reproduce the documented isolation class. It protects
//!
//! 1. the analysis bank's mirror/sign conventions (a swapped `(−1)ⁿ`
//!    parity or a one-sample polyphase error collapses the isolation),
//! 2. the staged prototype's transcription (a corrupted tap shows up as
//!    stopband leakage), and
//! 3. every downstream per-band conformance measurement made with this
//!    bank.
//!
//! Measured outcome (r440): the crate's bank lands on the provenance
//! figures at their own 0.1 dB print precision (−95.10 / −95.61 /
//! −87.94 / +87.94 / +95.61 / +95.10 dB), so the gate pins the match at
//! ±0.5 dB rather than a loose floor.

use oxideav_speex::QmfAnalysis;

const FULL_RATE: f64 = 16_000.0;
const FRAME: usize = 320;

/// Split `x` (16 kHz) with the crate's streaming analysis bank and
/// return the two half-band signals, transient-trimmed.
fn split(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut qa = QmfAnalysis::new();
    let mut lb = vec![0.0f64; x.len() / 2];
    let mut hb = vec![0.0f64; x.len() / 2];
    for f in 0..(x.len() / FRAME) {
        let (lo, hi) = (f * FRAME, (f + 1) * FRAME);
        let (blo, bhi) = (f * FRAME / 2, (f + 1) * FRAME / 2);
        qa.split_slices(&x[lo..hi], &mut lb[blo..bhi], &mut hb[blo..bhi]);
    }
    // Drop the leading FIR transient (one half-band frame is ample).
    (lb.split_off(FRAME / 2), hb.split_off(FRAME / 2))
}

fn energy(s: &[f64]) -> f64 {
    s.iter().map(|&v| v * v).sum()
}

/// High-to-low band energy ratio in dB for a pure tone at `freq_hz`.
fn tone_band_ratio_db(freq_hz: f64) -> f64 {
    let n = 16 * FRAME; // 320 ms of tone
    let x: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * freq_hz * i as f64 / FULL_RATE).sin())
        .collect();
    let (lb, hb) = split(&x);
    10.0 * (energy(&hb) / energy(&lb)).log10()
}

/// The six provenance/08 probe tones reproduce the documented isolation
/// **exactly** on the crate's analysis bank: measured here at
/// −95.10 / −95.61 / −87.94 / +87.94 / +95.61 / +95.10 dB against the
/// provenance's −95.1 / −95.6 / −87.9 / +87.9 / +95.6 / +95.1 — i.e. to
/// the provenance table's own 0.1 dB print precision. The crate's
/// [`QmfAnalysis`] *is* the instrument provenance/08 validated, tap for
/// tap and sign for sign.
#[test]
fn provenance_tones_reproduce_documented_isolation() {
    // (frequency, provenance/08-measured hb/lb energy ratio in dB)
    let probes: [(f64, f64); 6] = [
        (1_000.0, -95.1),
        (2_000.0, -95.6),
        (3_000.0, -87.9),
        (5_000.0, 87.9),
        (6_000.0, 95.6),
        (7_000.0, 95.1),
    ];
    for (freq, documented) in probes {
        let ratio = tone_band_ratio_db(freq);
        println!("tone {freq:5.0} Hz: hb/lb ratio {ratio:+7.2} dB (prov/08 {documented:+.1})");
        assert!(
            (ratio - documented).abs() < 0.5,
            "{freq} Hz: isolation {ratio:.2} dB departs from the documented {documented:.1} dB"
        );
    }
}

/// Symmetry pin: mirror tone pairs (ν, 8000 − ν) isolate equally well —
/// the mirror highpass is the exact `(−1)ⁿ` image of the lowpass, so
/// the two stopbands are the same filter seen from either band.
#[test]
fn mirror_tone_pairs_isolate_symmetrically() {
    for (lo_f, hi_f) in [(1_000.0, 7_000.0), (2_000.0, 6_000.0), (3_000.0, 5_000.0)] {
        let lo = tone_band_ratio_db(lo_f);
        let hi = tone_band_ratio_db(hi_f);
        let asym = (lo + hi).abs();
        println!("pair {lo_f:.0}/{hi_f:.0} Hz: {lo:+.2} / {hi:+.2} dB, asymmetry {asym:.2} dB");
        assert!(
            asym < 3.0,
            "mirror pair {lo_f}/{hi_f} Hz isolation asymmetry {asym:.2} dB ≥ 3"
        );
    }
}
