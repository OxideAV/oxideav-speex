//! **WB mode-1 folded high-band conformance gate** (round r393).
//!
//! Drives the staged reference fixture (`docs/audio/speex/fixtures/
//! wb-mode1-folded/`, mirrored under `tests/fixtures/wb-mode1-folded/`)
//! through the crate's [`WidebandDecoder`] and scores the decode
//! **absolutely** (no scale freedom) against the reference decoder's
//! `--no-enh` PCM. The fixture is a 16 kHz sine-mix encoded at WB
//! quality 1 (RFC 5574 wideband mode 1, 5750 bps): 101 frames, every
//! one narrowband mode 8 + high-band sub-mode 1 — the gain-only
//! **folded** high band this gate pins.
//!
//! What the gate locks (externally arbitrated in round r393, see
//! `src/hb_fold.rs` module docs):
//!
//! * the folded reconstruction law `e_hb[n] = K·g·(−1)ⁿ·e_lb[n]` — the
//!   alternative conventions (no `(−1)ⁿ` modulation; RMS-normalised
//!   fold source) score ≤ 0.31 high-band correlation vs ≥ 0.999 for
//!   the shipped law;
//! * the `INNOVATION_CODEBOOK_SCALE` = 1/32 Q5 row normalisation — the
//!   gate has **no scale freedom**, so a wrong absolute calibration
//!   fails the energy-ratio and SNR floors immediately;
//! * the QMF band alignment (the fixed 80-sample reference lead — the
//!   reference decoder's QMF/look-ahead padding, `notes.md`).
//!
//! Reference geometry: `expected.pcm` is 32 080 samples s16le mono
//! 16 kHz and leads the decode by 80 samples (2.5 ms QMF/look-ahead
//! padding); the gate aligns at that fixed lag rather than searching,
//! so a delay-convention regression fails loudly.

use oxideav_speex::{QmfAnalysis, WidebandDecoder};

const INPUT: &[u8] = include_bytes!("fixtures/wb-mode1-folded/input.spx");
const EXPECTED: &[u8] = include_bytes!("fixtures/wb-mode1-folded/expected.pcm");

/// Fixed alignment: the reference decode leads ours by 80 samples at
/// 16 kHz (40 per 8 kHz half-band) — the reference QMF/look-ahead
/// padding documented in the fixture notes.
const REF_LEAD_FULL: usize = 80;
const REF_LEAD_HALF: usize = 40;

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

fn reference_pcm() -> Vec<f64> {
    EXPECTED
        .chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn decode_fixture() -> Vec<f64> {
    let packets = lift_ogg_packets(INPUT);
    assert!(packets.len() > 2, "Ogg stream must carry audio packets");
    let audio = &packets[2..];
    assert_eq!(audio.len(), 101, "fixture carries 101 audio frames");

    let mut dec = WidebandDecoder::new();
    let mut out = Vec::new();
    for (i, pkt) in audio.iter().enumerate() {
        let frame = dec
            .decode_packet(pkt)
            .unwrap_or_else(|e| panic!("frame {i} failed to decode: {e}"));
        out.extend_from_slice(&frame.wideband_pcm);
    }
    out
}

/// `(absolute_snr_db, normalised_correlation)` of `ours` against
/// `reference` at a fixed lead of `reference`. Absolute: no gain is
/// fitted, so both shape and calibration are scored.
fn score(ours: &[f64], reference: &[f64], ref_lead: usize) -> (f64, f64) {
    let n = (reference.len() - ref_lead).min(ours.len());
    assert!(n > 10_000, "comparison window too short: {n}");
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
    (snr, corr)
}

/// Split a 16 kHz signal into its QMF half-bands (both signals are
/// measured through the same analysis bank, so the comparison domain is
/// identical for reference and decode).
fn split_bands(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut qa = QmfAnalysis::new();
    let mut low = Vec::new();
    let mut high = Vec::new();
    for frame in x.chunks(320) {
        if frame.len() < 320 {
            break;
        }
        let mut l = [0.0f64; 160];
        let mut h = [0.0f64; 160];
        qa.split_slices(frame, &mut l, &mut h);
        low.extend_from_slice(&l);
        high.extend_from_slice(&h);
    }
    (low, high)
}

#[test]
fn folded_high_band_decode_matches_reference() {
    let reference = reference_pcm();
    assert_eq!(reference.len(), 32_080, "reference PCM geometry");

    let ours = decode_fixture();
    assert_eq!(ours.len(), 101 * 320, "101 × 320-sample wideband frames");

    // --- Full-signal absolute conformance (measured r393: 16.7 dB /
    // corr 0.9894; floors leave margin for platform float variance). ---
    let (snr, corr) = score(&ours, &reference, REF_LEAD_FULL);
    assert!(snr >= 14.0, "full-signal absolute SNR {snr:.2} dB < 14 dB");
    assert!(corr >= 0.985, "full-signal correlation {corr:.4} < 0.985");

    // --- Absolute energy calibration (no scale freedom): a missing or
    // doubled INNOVATION_CODEBOOK_SCALE / fold constant fails here. ---
    let e_ours: f64 = ours.iter().map(|v| v * v).sum();
    let e_ref: f64 = reference.iter().map(|v| v * v).sum();
    let ratio = e_ours / e_ref;
    assert!(
        (0.85..=1.15).contains(&ratio),
        "decode/reference energy ratio {ratio:.4} outside [0.85, 1.15]"
    );

    // --- Per-band conformance. The high band is the folded-law gate
    // proper (measured r393: 38.9 dB absolute / corr 0.99994). ---
    let (ref_low, ref_high) = split_bands(&reference);
    let (our_low, our_high) = split_bands(&ours);

    let (hsnr, hcorr) = score(&our_high, &ref_high, REF_LEAD_HALF);
    assert!(
        hsnr >= 30.0,
        "high-band (folded) absolute SNR {hsnr:.2} dB < 30 dB"
    );
    assert!(hcorr >= 0.999, "high-band correlation {hcorr:.4} < 0.999");

    let (lsnr, lcorr) = score(&our_low, &ref_low, REF_LEAD_HALF);
    assert!(lsnr >= 12.0, "low-band absolute SNR {lsnr:.2} dB < 12 dB");
    assert!(lcorr >= 0.975, "low-band correlation {lcorr:.4} < 0.975");

    // The folded high band carries real (non-silent) energy at the
    // right level: ratio pinned tightly because the fold gain law has
    // no other external anchor.
    let he_ours: f64 = our_high.iter().map(|v| v * v).sum();
    let he_ref: f64 = ref_high.iter().map(|v| v * v).sum();
    let hratio = he_ours / he_ref;
    assert!(
        (0.9..=1.1).contains(&hratio),
        "high-band energy ratio {hratio:.4} outside [0.9, 1.1]"
    );
}

/// The reference decode was produced with the codec's **default output
/// high-pass** active (`--no-enh` disables only the perceptual
/// enhancer; the manual's codec-control table documents the high-pass
/// as default-on). Applying the crate's fitted [`OutputHighpass`]
/// (r393, opt-in) to the raw decode must therefore move it *closer* to
/// the reference — this pins both the filter fit and the pipeline
/// reading (measured: 16.7 dB raw → 18.3 dB high-passed).
#[test]
fn output_highpass_improves_reference_match() {
    use oxideav_speex::OutputHighpass;

    let reference = reference_pcm();
    let mut ours = decode_fixture();
    let (raw_snr, _) = score(&ours, &reference, REF_LEAD_FULL);

    let mut hp = OutputHighpass::for_sample_rate(16_000);
    hp.process_slice(&mut ours);
    let (hp_snr, hp_corr) = score(&ours, &reference, REF_LEAD_FULL);

    assert!(
        hp_snr > raw_snr + 0.5,
        "high-pass should improve the match: raw {raw_snr:.2} dB vs {hp_snr:.2} dB"
    );
    assert!(hp_snr >= 16.0, "high-passed SNR {hp_snr:.2} dB < 16 dB");
    assert!(hp_corr >= 0.985, "high-passed correlation {hp_corr:.4}");
}

/// The arbitration is meaningful only if the fixture really is the
/// all-mode-1 stream the docs describe: re-verify the framing facts the
/// fixture notes pin (101 frames, NB mode 8, HB sub-mode 1).
#[test]
fn fixture_framing_matches_staged_notes() {
    use oxideav_speex::{
        BitReader, NarrowbandFrameBody, NarrowbandFrameHeader, Submode, WidebandHighBandBody,
        WidebandHighBandFrameHeader, WidebandSubmode,
    };

    let packets = lift_ogg_packets(INPUT);
    let audio = &packets[2..];
    assert_eq!(audio.len(), 101);

    for (i, pkt) in audio.iter().enumerate() {
        let mut r = BitReader::new(pkt);
        let h = NarrowbandFrameHeader::parse(&mut r).unwrap();
        let nb_sub = match h.submode {
            Submode::Celp(s) => s,
            other => panic!("frame {i}: unexpected NB submode {other:?}"),
        };
        assert_eq!(nb_sub.mode_id, 8, "frame {i}: NB mode");
        let _body = NarrowbandFrameBody::parse(&mut r, &nb_sub).unwrap();
        let hb = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        assert!(hb.wideband, "frame {i}: high-band flag");
        let hb_sub = match hb.submode {
            WidebandSubmode::Documented(s) => s,
            other => panic!("frame {i}: unexpected HB submode {other:?}"),
        };
        assert_eq!(hb_sub.mode_id, 1, "frame {i}: HB sub-mode");
        let _hb_body = WidebandHighBandBody::parse(&mut r, &hb_sub).unwrap();
    }
}
