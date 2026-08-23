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
//! * the alignment: the reference decode leads by exactly its
//!   40-sample look-ahead padding — a delay regression fails loudly.
//!
//! The reference decodes carry the codec's **default output
//! high-pass** (the `--no-enh` flag disables only the enhancer), so
//! the raw floors are intrinsically limited by the crate's unfiltered
//! output at the very bottom of the band; the gate also scores the
//! decode through the fitted opt-in [`oxideav_speex::OutputHighpass`]
//! and asserts the improvement.

use oxideav_speex::{
    NarrowbandDecoder, NarrowbandFrameBody, NarrowbandFrameHeader, OutputHighpass, Submode,
    NARROWBAND_FRAME_SAMPLES,
};

/// Fixed alignment: reference decode leads ours by 40 samples at 8 kHz
/// (the reference codec's look-ahead padding, `NOTES.md`).
const REF_LEAD: usize = 40;

struct Fixture {
    name: &'static str,
    nb_mode: u8,
    spx: &'static [u8],
    reference: &'static [u8],
    /// Floor on the absolute (unfitted) SNR of the raw decode, dB.
    /// Measured r410 values are ~1.5 dB above each floor.
    min_snr_db: f64,
    /// Floor on the absolute SNR of the high-passed decode, dB.
    min_hp_snr_db: f64,
    /// Floor on the normalised correlation of the raw decode.
    min_corr: f64,
    /// Bounds on the decode/reference energy ratio.
    energy: (f64, f64),
}

const FIXTURES: &[Fixture] = &[
    // --- tone-mix matrix (measured r410: raw SNR / hp SNR / corr / energy) ---
    Fixture {
        // 12.7 / 17.7 / 0.973 / 0.98
        name: "tone-q1",
        nb_mode: 8,
        spx: include_bytes!("fixtures/nb-conformance/nb_q1.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q1.noenh.pcm"),
        min_snr_db: 11.0,
        min_hp_snr_db: 16.0,
        min_corr: 0.96,
        energy: (0.85, 1.15),
    },
    Fixture {
        // 13.4 / 14.3 / 0.979 / 1.08
        name: "tone-q2",
        nb_mode: 2,
        spx: include_bytes!("fixtures/nb-conformance/nb_q2.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q2.noenh.pcm"),
        min_snr_db: 11.5,
        min_hp_snr_db: 12.5,
        min_corr: 0.97,
        energy: (0.9, 1.2),
    },
    Fixture {
        // 13.8 / 18.9 / 0.979 / 0.98
        name: "tone-q3",
        nb_mode: 3,
        spx: include_bytes!("fixtures/nb-conformance/nb_q3.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q3.noenh.pcm"),
        min_snr_db: 12.0,
        min_hp_snr_db: 17.0,
        min_corr: 0.97,
        energy: (0.85, 1.15),
    },
    Fixture {
        // 14.2 / 19.1 / 0.981 / 0.97
        name: "tone-q5",
        nb_mode: 4,
        spx: include_bytes!("fixtures/nb-conformance/nb_q5.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q5.noenh.pcm"),
        min_snr_db: 12.5,
        min_hp_snr_db: 17.5,
        min_corr: 0.97,
        energy: (0.85, 1.15),
    },
    Fixture {
        // 14.4 / 19.5 / 0.982 / 0.98
        name: "tone-q7",
        nb_mode: 5,
        spx: include_bytes!("fixtures/nb-conformance/nb_q7.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q7.noenh.pcm"),
        min_snr_db: 12.5,
        min_hp_snr_db: 17.5,
        min_corr: 0.97,
        energy: (0.85, 1.15),
    },
    Fixture {
        // 14.4 / 19.5 / 0.982 / 0.98
        name: "tone-q9",
        nb_mode: 6,
        spx: include_bytes!("fixtures/nb-conformance/nb_q9.spx"),
        reference: include_bytes!("fixtures/nb-conformance/nb_q9.noenh.pcm"),
        min_snr_db: 12.5,
        min_hp_snr_db: 17.5,
        min_corr: 0.97,
        energy: (0.85, 1.15),
    },
    // --- speech-like pitch-glide matrix ---
    Fixture {
        // 12.7 / 17.1 / 0.973 / 0.95
        name: "speech-q1",
        nb_mode: 8,
        spx: include_bytes!("fixtures/nb-conformance/sp_q1.spx"),
        reference: include_bytes!("fixtures/nb-conformance/sp_q1.noenh.pcm"),
        min_snr_db: 11.0,
        min_hp_snr_db: 15.5,
        min_corr: 0.96,
        energy: (0.85, 1.15),
    },
    Fixture {
        // 10.5 / 13.1 / 0.955 / 1.00
        name: "speech-q2",
        nb_mode: 2,
        spx: include_bytes!("fixtures/nb-conformance/sp_q2.spx"),
        reference: include_bytes!("fixtures/nb-conformance/sp_q2.noenh.pcm"),
        min_snr_db: 9.0,
        min_hp_snr_db: 11.5,
        min_corr: 0.94,
        energy: (0.85, 1.15),
    },
    Fixture {
        // 12.1 / 16.1 / 0.969 / 0.98
        name: "speech-q3",
        nb_mode: 3,
        spx: include_bytes!("fixtures/nb-conformance/sp_q3.spx"),
        reference: include_bytes!("fixtures/nb-conformance/sp_q3.noenh.pcm"),
        min_snr_db: 10.5,
        min_hp_snr_db: 14.5,
        min_corr: 0.95,
        energy: (0.85, 1.15),
    },
    Fixture {
        // 11.4 / 15.4 / 0.963 / 0.98
        name: "speech-q7",
        nb_mode: 5,
        spx: include_bytes!("fixtures/nb-conformance/sp_q7.spx"),
        reference: include_bytes!("fixtures/nb-conformance/sp_q7.noenh.pcm"),
        min_snr_db: 10.0,
        min_hp_snr_db: 14.0,
        min_corr: 0.95,
        energy: (0.85, 1.15),
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
        .chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Decode every audio frame of a fixture through the closed-loop
/// narrowband decoder, asserting the expected Table 9.2 sub-mode.
fn decode_fixture(fx: &Fixture) -> Vec<f64> {
    let packets = lift_ogg_packets(fx.spx);
    assert!(packets.len() > 2, "{}: no audio packets", fx.name);

    let mut decoder = NarrowbandDecoder::new();
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
            REF_LEAD,
            "{}: reference geometry (decode + look-ahead lead)",
            fx.name
        );
        let ours = decode_fixture(fx);
        assert_eq!(ours.len() % NARROWBAND_FRAME_SAMPLES, 0);
        // The encoder pads the source with one look-ahead frame, so the
        // decode covers the reference window (score truncates to the
        // overlap).
        assert!(ours.len() + REF_LEAD >= reference.len(), "{}", fx.name);

        let (snr, corr, ratio) = score(&ours, &reference, REF_LEAD);

        let mut hp_pcm = ours.clone();
        let mut hp = OutputHighpass::for_sample_rate(8_000);
        hp.process_slice(&mut hp_pcm);
        let (hp_snr, hp_corr, _) = score(&hp_pcm, &reference, REF_LEAD);

        println!(
            "{} (mode {}): raw {snr:.2} dB corr {corr:.4} energy {ratio:.4} | hp {hp_snr:.2} dB corr {hp_corr:.4}",
            fx.name, fx.nb_mode
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
        if hp_snr < fx.min_hp_snr_db {
            failures.push(format!(
                "{}: high-passed SNR {hp_snr:.2} dB < {} dB",
                fx.name, fx.min_hp_snr_db
            ));
        }
        // The reference has the default output high-pass active, so the
        // fitted high-pass must move every decode closer to it.
        assert!(
            hp_snr > snr,
            "{}: high-pass should improve the match ({snr:.2} → {hp_snr:.2})",
            fx.name
        );
    }
    assert!(
        failures.is_empty(),
        "matrix failures:\n{}",
        failures.join("\n")
    );
}

/// The r410 arbitration is only meaningful at the fixed look-ahead
/// alignment: verify the fixed-lead score is the best integer lag in a
/// ±40-sample window, so a delay-convention regression fails loudly.
#[test]
fn reference_lead_is_the_best_alignment() {
    for fx in FIXTURES {
        let reference = reference_pcm(fx.reference);
        let ours = decode_fixture(fx);
        let (at_lead, _, _) = score(&ours, &reference, REF_LEAD);
        let mut best = (f64::MIN, 0usize);
        for lead in 0..=(2 * REF_LEAD) {
            let (s, _, _) = score(&ours, &reference, lead);
            if s > best.0 {
                best = (s, lead);
            }
        }
        // Allow a 1-sample / 0.8 dB grace: this RAW comparison holds
        // our un-high-passed decode against the reference's default
        // high-passed output, whose ≈0.08 rad phase lead at 440 Hz can
        // make a pure-tone fixture score a hair higher one sample off
        // (the hp rows of the matrix are the phase-consistent metric).
        assert!(
            best.1.abs_diff(REF_LEAD) <= 1 && at_lead >= best.0 - 0.8,
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
