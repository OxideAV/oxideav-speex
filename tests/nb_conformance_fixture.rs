//! **Narrowband decode-conformance matrix** (round r410).
//!
//! Drives the staged narrowband reference fixtures
//! (`tests/fixtures/nb-conformance/`, black-box `speexenc` /
//! `speexdec --no-enh` I/O — see `NOTES.md` there) through the crate's
//! [`NarrowbandDecoder`] and scores each decode **absolutely** (no
//! fitted gain) against the reference decoder's core (`--no-enh`)
//! output:
//!
//! * a 1 s tone-mix at qualities 1/2/3/5/7/9 — Table 9.2 sub-modes
//!   8, 2, 3, 4, 5 and 6;
//! * a 2 s speech-like pitch-glide (f0 40–180 Hz, harmonic stack with
//!   syllabic AM) at qualities 1/2/3/7 — time-varying pitch periods
//!   both above and below the 40-sample sub-frame length.
//!
//! What this gate pins (arbitrated in round r410 — the measured
//! baselines are in each fixture row below):
//!
//! * the **pitch-gain VQ column ↔ lag association** (`pitch_gain`
//!   module docs): the reversed reading scores 13.5–14.4 dB on the VQ
//!   sub-modes where the direct reading scores 2.8–5.6 dB at half the
//!   reference energy;
//! * the **in-sub-frame pitch recursion** for short periods
//!   (r450: the single-substitution repeat rule for the VQ modes and
//!   the unbounded centre-tap recursion for the forced OL modes —
//!   `gain_scaled_pitch_subframe_repeat` / `_forced`);
//! * the alignment: the decode lines up with the reference sample for
//!   sample — a delay regression fails loudly.
//!
//! r483: the reference decodes carry the codec's **default output
//! high-pass** (the `--no-enh` flag disables only the enhancer), and so
//! does [`NarrowbandDecoder`] now (`NB_OUTPUT_HIGHPASS`, measured); with
//! the transposed synthesis form, the float excitation memory and the
//! float-path LSP constants the decode is within ±1 LSB of the
//! reference on ≥ 98 % of samples.

use oxideav_speex::{
    NARROWBAND_FRAME_SAMPLES, NarrowbandDecoder, NarrowbandFrameBody, NarrowbandFrameHeader,
    Submode, saturate_i16,
};

/// Fixed alignment: none — r483 reproduces the reference decoder's
/// one-sub-frame synthesis lag, so the decode lines up with the
/// reference sample for sample (r410–r480: the reference led by 40).
const REF_LEAD: usize = 0;

struct Fixture {
    name: &'static str,
    nb_mode: u8,
    spx: &'static [u8],
    reference: &'static [u8],
    /// Floor on the absolute (unfitted) SNR of the raw decode, dB.
    /// Measured r410 values are ~1.5 dB above each floor.
    min_snr_db: f64,
    /// Floor on the share of output samples within ±1 LSB.
    min_within1: f64,
    /// Floor on the normalised correlation of the raw decode.
    min_corr: f64,
    /// Bounds on the decode/reference energy ratio.
    energy: (f64, f64),
}

/// r483 floors, every fixture: the exact float path measures 76.9–
/// 80.9 dB, ≥ 98.4 % of samples within ±1 LSB (tone fixtures ≥ 98.8 %,
/// speech ≥ 98.4 %, the max deviation 5 LSB on the pitch glide's loudest
/// pulses). r410 measured 10.5–14.4 dB raw / 13.1–19.5 dB through the
/// then-fitted output high-pass.
const MIN_SNR_DB: f64 = 74.0;
const MIN_WITHIN1: f64 = 0.98;

const FIXTURES: &[Fixture] = &[
    // --- tone-mix matrix (measured r410: raw SNR / hp SNR / corr / energy) ---
    Fixture {
        // 12.7 / 17.7 / 0.973 / 0.98
        name: "tone-q1",
        nb_mode: 8,
        spx: include_bytes!("fixtures/nb-conformance/nb_q1.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q1.noenh.pcm"),
        min_snr_db: MIN_SNR_DB,
        min_within1: MIN_WITHIN1,
        min_corr: 0.99999,
        energy: (0.999, 1.001),
    },
    Fixture {
        // 13.4 / 14.3 / 0.979 / 1.08
        name: "tone-q2",
        nb_mode: 2,
        spx: include_bytes!("fixtures/nb-conformance/nb_q2.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q2.noenh.pcm"),
        min_snr_db: MIN_SNR_DB,
        min_within1: MIN_WITHIN1,
        min_corr: 0.99999,
        energy: (0.999, 1.001),
    },
    Fixture {
        // 13.8 / 18.9 / 0.979 / 0.98
        name: "tone-q3",
        nb_mode: 3,
        spx: include_bytes!("fixtures/nb-conformance/nb_q3.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q3.noenh.pcm"),
        min_snr_db: MIN_SNR_DB,
        min_within1: MIN_WITHIN1,
        min_corr: 0.99999,
        energy: (0.999, 1.001),
    },
    Fixture {
        // 14.2 / 19.1 / 0.981 / 0.97
        name: "tone-q5",
        nb_mode: 4,
        spx: include_bytes!("fixtures/nb-conformance/nb_q5.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q5.noenh.pcm"),
        min_snr_db: MIN_SNR_DB,
        min_within1: MIN_WITHIN1,
        min_corr: 0.99999,
        energy: (0.999, 1.001),
    },
    Fixture {
        // 14.4 / 19.5 / 0.982 / 0.98
        name: "tone-q7",
        nb_mode: 5,
        spx: include_bytes!("fixtures/nb-conformance/nb_q7.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q7.noenh.pcm"),
        min_snr_db: MIN_SNR_DB,
        min_within1: MIN_WITHIN1,
        min_corr: 0.99999,
        energy: (0.999, 1.001),
    },
    Fixture {
        // 14.4 / 19.5 / 0.982 / 0.98
        name: "tone-q9",
        nb_mode: 6,
        spx: include_bytes!("fixtures/nb-conformance/nb_q9.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q9.noenh.pcm"),
        min_snr_db: MIN_SNR_DB,
        min_within1: MIN_WITHIN1,
        min_corr: 0.99999,
        energy: (0.999, 1.001),
    },
    // --- speech-like pitch-glide matrix ---
    Fixture {
        // 12.7 / 17.1 / 0.973 / 0.95
        name: "speech-q1",
        nb_mode: 8,
        spx: include_bytes!("fixtures/nb-conformance/sp_q1.spx"),
        reference: include_bytes!("fixtures/nb-conformance/sp_q1.noenh.pcm"),
        min_snr_db: MIN_SNR_DB,
        min_within1: MIN_WITHIN1,
        min_corr: 0.99999,
        energy: (0.999, 1.001),
    },
    Fixture {
        // 10.5 / 13.1 / 0.955 / 1.00
        name: "speech-q2",
        nb_mode: 2,
        spx: include_bytes!("fixtures/nb-conformance/sp_q2.spx"),
        reference: include_bytes!("fixtures/nb-conformance/sp_q2.noenh.pcm"),
        min_snr_db: MIN_SNR_DB,
        min_within1: MIN_WITHIN1,
        min_corr: 0.99999,
        energy: (0.999, 1.001),
    },
    Fixture {
        // 12.1 / 16.1 / 0.969 / 0.98
        name: "speech-q3",
        nb_mode: 3,
        spx: include_bytes!("fixtures/nb-conformance/sp_q3.spx"),
        reference: include_bytes!("fixtures/nb-conformance/sp_q3.noenh.pcm"),
        min_snr_db: MIN_SNR_DB,
        min_within1: MIN_WITHIN1,
        min_corr: 0.99999,
        energy: (0.999, 1.001),
    },
    Fixture {
        // 11.4 / 15.4 / 0.963 / 0.98
        name: "speech-q7",
        nb_mode: 5,
        spx: include_bytes!("fixtures/nb-conformance/sp_q7.spx"),
        reference: include_bytes!("fixtures/nb-conformance/sp_q7.noenh.pcm"),
        min_snr_db: MIN_SNR_DB,
        min_within1: MIN_WITHIN1,
        min_corr: 0.99999,
        energy: (0.999, 1.001),
    },
];

/// Inline minimal Ogg page-walker — identical contract to the sibling
/// fixture tests.
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

fn reference_pcm(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Decode every audio frame of a fixture through the closed-loop
/// narrowband decoder, asserting the expected Table 9.2 sub-mode.
fn decode_fixture(fx: &Fixture) -> Vec<f64> {
    let packets = lift_ogg_packets(fx.spx);
    assert!(packets.len() > 2, "{}: no audio packets", fx.name);

    let mut decoder = NarrowbandDecoder::new();

    decoder.set_enhancer(false);
    let mut pcm = Vec::new();
    for (i, pkt) in packets[2..].iter().enumerate() {
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(pkt)
            .unwrap_or_else(|e| panic!("{} frame {i}: header parse: {e}", fx.name));
        let submode = match h.submode {
            Submode::Celp(s) => s,
            other => panic!("{} frame {i}: unexpected submode {other:?}", fx.name),
        };
        assert_eq!(
            submode.mode_id, fx.nb_mode,
            "{} frame {i}: Table 9.2 sub-mode",
            fx.name
        );
        let body = NarrowbandFrameBody::parse(&mut r, &submode)
            .unwrap_or_else(|e| panic!("{} frame {i}: body parse: {e}", fx.name));
        let frame = decoder
            .decode_frame(&body, &submode)
            .unwrap_or_else(|e| panic!("{} frame {i}: decode: {e}", fx.name));
        pcm.extend_from_slice(&frame);
    }
    pcm
}

/// `(absolute_snr_db, normalised_correlation, energy_ratio)` of `ours`
/// against `reference` at a fixed reference lead. Absolute: no gain is
/// fitted, so both shape and calibration are scored.
fn score(ours: &[f64], reference: &[f64], ref_lead: usize) -> (f64, f64, f64) {
    let n = (reference.len() - ref_lead).min(ours.len());
    assert!(n > 4_000, "comparison window too short: {n}");
    let r = &reference[ref_lead..ref_lead + n];
    let o = &ours[..n];
    let mut err = 0.0f64;
    let mut er = 0.0f64;
    let mut eo = 0.0f64;
    let mut dot = 0.0f64;
    for i in 0..n {
        let d = r[i] - o[i];
        err += d * d;
        er += r[i] * r[i];
        eo += o[i] * o[i];
        dot += r[i] * o[i];
    }
    let snr = 10.0 * (er / (err + 1e-12)).log10();
    let corr = dot / (er.sqrt() * eo.sqrt() + 1e-12);
    (snr, corr, eo / er)
}

/// The conformance matrix proper: every fixture must clear its
/// absolute floors, raw and high-passed (run with `--nocapture` to see
/// the measured values).
#[test]
fn narrowband_conformance_matrix() {
    let mut failures: Vec<String> = Vec::new();
    for fx in FIXTURES {
        let reference = reference_pcm(fx.reference);
        assert_eq!(
            reference.len() % NARROWBAND_FRAME_SAMPLES,
            40,
            "{}: reference geometry (whole frames + one sub-frame)",
            fx.name
        );
        let ours = decode_fixture(fx);
        assert_eq!(ours.len() % NARROWBAND_FRAME_SAMPLES, 0);
        // The encoder pads the source with one look-ahead frame, so the
        // decode covers the reference window (score truncates to the
        // overlap).
        assert!(ours.len() + REF_LEAD >= reference.len(), "{}", fx.name);

        let (snr, corr, ratio) = score(&ours, &reference, REF_LEAD);
        // Sample exactness of the rounded output.
        let n = reference.len().min(ours.len());
        let within1 = (0..n)
            .filter(|&i| (reference[i] - f64::from(saturate_i16(ours[i]))).abs() <= 1.0)
            .count() as f64
            / n as f64;

        println!(
            "{} (mode {}): {snr:.2} dB corr {corr:.6} energy {ratio:.5} | ±1 LSB {:.3} %",
            fx.name,
            fx.nb_mode,
            100.0 * within1
        );

        if snr < fx.min_snr_db {
            failures.push(format!(
                "{}: raw SNR {snr:.2} dB < {} dB",
                fx.name, fx.min_snr_db
            ));
        }
        if corr < fx.min_corr {
            failures.push(format!(
                "{}: correlation {corr:.4} < {}",
                fx.name, fx.min_corr
            ));
        }
        if ratio < fx.energy.0 || ratio > fx.energy.1 {
            failures.push(format!(
                "{}: energy ratio {ratio:.4} outside [{}, {}]",
                fx.name, fx.energy.0, fx.energy.1
            ));
        }
        if within1 < fx.min_within1 {
            failures.push(format!(
                "{}: ±1 LSB share {within1:.4} < {}",
                fx.name, fx.min_within1
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "matrix failures:\n{}",
        failures.join("\n")
    );
}

/// The fixed alignment is the best integer lag in a 40-sample window,
/// so a delay-convention regression fails loudly.
#[test]
fn reference_lead_is_the_best_alignment() {
    for fx in FIXTURES {
        let reference = reference_pcm(fx.reference);
        let ours = decode_fixture(fx);
        let (at_lead, _, _) = score(&ours, &reference, REF_LEAD);
        let mut best = (f64::MIN, 0usize);
        for lead in 0..=40 {
            let (s, _, _) = score(&ours, &reference, lead);
            if s > best.0 {
                best = (s, lead);
            }
        }
        assert!(
            best.1 == REF_LEAD && at_lead >= best.0,
            "{}: best lag {} ({:.2} dB) vs fixed lead {REF_LEAD} ({at_lead:.2} dB)",
            fx.name,
            best.1,
            best.0
        );
    }
}

/// Decoding the same fixture twice yields bit-identical PCM — the
/// closed-loop state is deterministic.
#[test]
fn conformance_decode_is_deterministic() {
    let fx = &FIXTURES[6]; // speech-q1: OL pitch + recursion active
    let a = decode_fixture(fx);
    let b = decode_fixture(fx);
    assert_eq!(a, b);
}
