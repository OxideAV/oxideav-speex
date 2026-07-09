//! **Ultra-wideband 3-layer fold-geometry conformance gate** (round
//! r403).
//!
//! Drives the staged reference fixture (`docs/audio/speex/fixtures/
//! uwb-fold-geometry/`, mirrored under `tests/fixtures/
//! uwb-fold-geometry/`) through the crate's [`UltraWidebandDecoder`] and
//! scores the decode **absolutely** (no scale freedom) against the
//! reference decoder's `--no-enh` 32 kHz PCM. The fixture is a 32 kHz
//! sine mix encoded at UWB quality 1 (RFC 5574 ultra-wideband mode 1,
//! 7550 bps): 101 frames, every one narrowband mode 8 + first high-band
//! sub-mode 1 + **second** high-band sub-mode 1 — the stacked
//! (NB→WB→UWB) folded high bands.
//!
//! What the gate locks:
//!
//! * the **embedded wideband layers** (NB + first high band), validated
//!   through the outer-QMF low half (measured r403: 21.6 dB / corr
//!   0.997) — the first external validation of the UWB path's first two
//!   layers end-to-end;
//! * the **second-layer fold source geometry** pinned in r403 — the
//!   first-high-band excitation, linear-interpolated to 16 kHz, re-folded
//!   into 8–16 kHz (measured high-band correlation 0.93, energy ratio
//!   0.95); the earlier QMF-recombined generalisation scored ≈0 here;
//! * the full 32 kHz reconstruction (measured 19.1 dB / corr 0.994).
//!
//! Reference geometry: `expected.pcm` is 64 160 samples s16le mono
//! 32 kHz and leads the decode by 160 samples (double the wideband
//! fixture's 80-sample QMF/look-ahead padding — the extra sub-band
//! stage); the gate aligns at that fixed lag.

use oxideav_speex::{QmfAnalysis, UltraWidebandDecoder, UwbDecodedFrame};

const INPUT: &[u8] = include_bytes!("fixtures/uwb-fold-geometry/input.spx");
const EXPECTED: &[u8] = include_bytes!("fixtures/uwb-fold-geometry/expected.pcm");

/// Fixed alignment: the reference decode leads ours by 160 samples at
/// 32 kHz (80 per 16 kHz half-band) — twice the wideband fixture's lead.
const REF_LEAD_FULL: usize = 160;
const REF_LEAD_HALF: usize = 80;

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

    let mut dec = UltraWidebandDecoder::new();
    let mut out = Vec::new();
    for (i, pkt) in audio.iter().enumerate() {
        for f in dec
            .decode_packet(pkt)
            .unwrap_or_else(|e| panic!("frame {i} failed to decode: {e}"))
        {
            if let UwbDecodedFrame::Audio(a) = f {
                out.extend_from_slice(a.uwb_pcm.as_ref());
            }
        }
    }
    out
}

/// `(absolute_snr_db, normalised_correlation)` of `ours` against
/// `reference` at a fixed lead of `reference`. Absolute: no gain is
/// fitted, so both shape and calibration are scored.
fn score(ours: &[f64], reference: &[f64], ref_lead: usize) -> (f64, f64) {
    let n = (reference.len() - ref_lead).min(ours.len());
    assert!(n > 20_000, "comparison window too short: {n}");
    let r = &reference[ref_lead..ref_lead + n];
    let o = &ours[..n];
    let (mut err, mut er, mut eo, mut dot) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
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

/// Split a 32 kHz signal into its outer-QMF half-bands: low = 0–16 kHz
/// (the embedded wideband layer), high = 8–16 kHz effective (the second
/// folded layer). Both signals are measured through the same analysis
/// bank so the comparison domain is identical for reference and decode.
fn split_outer(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut qa = QmfAnalysis::new();
    let (mut low, mut high) = (Vec::new(), Vec::new());
    for frame in x.chunks(640) {
        if frame.len() < 640 {
            break;
        }
        let mut l = [0.0f64; 320];
        let mut h = [0.0f64; 320];
        qa.split_slices(frame, &mut l, &mut h);
        low.extend_from_slice(&l);
        high.extend_from_slice(&h);
    }
    (low, high)
}

#[test]
fn uwb_three_layer_decode_matches_reference() {
    let reference = reference_pcm();
    assert_eq!(reference.len(), 64_160, "reference PCM geometry");

    let ours = decode_fixture();
    assert_eq!(ours.len(), 101 * 640, "101 × 640-sample UWB frames");

    // --- Full-signal absolute conformance (measured r403: 19.1 dB /
    // corr 0.994; floors leave margin for platform float variance). ---
    let (snr, corr) = score(&ours, &reference, REF_LEAD_FULL);
    assert!(snr >= 16.0, "full-signal absolute SNR {snr:.2} dB < 16 dB");
    assert!(corr >= 0.985, "full-signal correlation {corr:.4} < 0.985");

    // --- Absolute energy calibration (no scale freedom): the 25×
    // overshoot of the pre-r403 fold source fails here immediately. ---
    let e_ours: f64 = ours.iter().map(|v| v * v).sum();
    let e_ref: f64 = reference.iter().map(|v| v * v).sum();
    let ratio = e_ours / e_ref;
    assert!(
        (0.85..=1.15).contains(&ratio),
        "decode/reference energy ratio {ratio:.4} outside [0.85, 1.15]"
    );

    // --- Per-band conformance. ---
    let (ref_low, ref_high) = split_outer(&reference);
    let (our_low, our_high) = split_outer(&ours);

    // Low 16 kHz half = the embedded wideband layers (NB + first HB) —
    // externally validated end-to-end for the first time (r403).
    let (lsnr, lcorr) = score(&our_low, &ref_low, REF_LEAD_HALF);
    assert!(
        lsnr >= 18.0,
        "embedded wideband (low 16 kHz) SNR {lsnr:.2} dB < 18 dB"
    );
    assert!(
        lcorr >= 0.99,
        "embedded wideband correlation {lcorr:.4} < 0.99"
    );

    // Second folded layer (8–16 kHz) — the r403 fold-source gate proper.
    let (hsnr, hcorr) = score(&our_high, &ref_high, REF_LEAD_HALF);
    assert!(hsnr >= 6.0, "second-layer (folded) SNR {hsnr:.2} dB < 6 dB");
    assert!(hcorr >= 0.88, "second-layer correlation {hcorr:.4} < 0.88");
    let he_ours: f64 = our_high.iter().map(|v| v * v).sum();
    let he_ref: f64 = ref_high.iter().map(|v| v * v).sum();
    let hratio = he_ours / he_ref;
    assert!(
        (0.75..=1.25).contains(&hratio),
        "second-layer energy ratio {hratio:.4} outside [0.75, 1.25]"
    );
}

/// The reference decode carries the codec's **default output high-pass**
/// (`--no-enh` disables only the perceptual enhancer). Applying the
/// crate's fitted [`OutputHighpass`] at the 32 kHz output rate must move
/// the decode *closer* to the reference — pinning both the filter fit at
/// the ultra-wideband rate and the pipeline reading (measured: 19.1 dB
/// raw → 21.3 dB high-passed, mirroring the wideband fixture's
/// +1.6 dB). See `docs/audio/speex/decoder-output-empirical.md`.
#[test]
fn output_highpass_improves_reference_match() {
    use oxideav_speex::OutputHighpass;

    let reference = reference_pcm();
    let mut ours = decode_fixture();
    let (raw_snr, _) = score(&ours, &reference, REF_LEAD_FULL);

    let mut hp = OutputHighpass::for_sample_rate(32_000);
    hp.process_slice(&mut ours);
    let (hp_snr, hp_corr) = score(&ours, &reference, REF_LEAD_FULL);

    assert!(
        hp_snr > raw_snr + 1.0,
        "high-pass should improve the match: raw {raw_snr:.2} dB vs {hp_snr:.2} dB"
    );
    assert!(hp_snr >= 20.0, "high-passed SNR {hp_snr:.2} dB < 20 dB");
    assert!(hp_corr >= 0.99, "high-passed correlation {hp_corr:.4}");
}

/// The header-driven top-level path ([`SpeexStreamDecoder`]) decodes the
/// same fixture identically to the direct [`UltraWidebandDecoder`] walk.
#[test]
fn stream_decoder_path_matches_direct_uwb_decode() {
    use oxideav_speex::{SpeexHeader, SpeexStreamDecoder};

    let packets = lift_ogg_packets(INPUT);
    let header = SpeexHeader::parse(&packets[0]).expect("header packet parses");
    assert!(header.is_ultrawideband(), "fixture is an UWB stream");

    let mut stream = SpeexStreamDecoder::for_header(&header).expect("uwb dispatch");
    assert_eq!(stream.output_rate_hz(), 32_000);
    assert_eq!(stream.frame_samples(), 640);

    let mut via_stream: Vec<i16> = Vec::new();
    for pkt in &packets[2..] {
        via_stream.extend(stream.decode_packet_pcm_i16(pkt).expect("stream decode"));
    }

    let mut direct = UltraWidebandDecoder::new();
    let mut via_direct: Vec<i16> = Vec::new();
    for pkt in &packets[2..] {
        via_direct.extend(direct.decode_packet_pcm_i16(pkt).expect("direct decode"));
    }

    assert_eq!(via_stream.len(), 101 * 640);
    assert_eq!(
        via_stream, via_direct,
        "header-driven and direct ultra-wideband decodes must be bit-identical"
    );
}

/// Two independent decoder instances produce byte-identical PCM — the
/// decode is deterministic.
#[test]
fn uwb_decode_is_deterministic() {
    let a = decode_fixture();
    let b = decode_fixture();
    assert_eq!(a, b, "two decodes of the fixture must be identical");
}

/// The arbitration is meaningful only if the fixture really is the
/// all-mode-1 three-layer stream the docs describe: re-verify the
/// framing facts the fixture notes pin (101 frames, NB mode 8, both
/// high-band layers sub-mode 1).
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
        // Layer 0: narrowband, mode 8.
        let h = NarrowbandFrameHeader::parse(&mut r).unwrap();
        let nb_sub = match h.submode {
            Submode::Celp(s) => s,
            other => panic!("frame {i}: unexpected NB submode {other:?}"),
        };
        assert_eq!(nb_sub.mode_id, 8, "frame {i}: NB mode");
        let _nb_body = NarrowbandFrameBody::parse(&mut r, &nb_sub).unwrap();

        // Layer 1: first high band, sub-mode 1.
        let hb1 = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        assert!(hb1.wideband, "frame {i}: first high-band flag");
        let hb1_sub = match hb1.submode {
            WidebandSubmode::Documented(s) => s,
            other => panic!("frame {i}: unexpected HB1 submode {other:?}"),
        };
        assert_eq!(hb1_sub.mode_id, 1, "frame {i}: first HB sub-mode");
        let _hb1_body = WidebandHighBandBody::parse(&mut r, &hb1_sub).unwrap();

        // Layer 2: second high band, sub-mode 1 (the UWB layer).
        let hb2 = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        assert!(hb2.wideband, "frame {i}: second high-band flag");
        let hb2_sub = match hb2.submode {
            WidebandSubmode::Documented(s) => s,
            other => panic!("frame {i}: unexpected HB2 submode {other:?}"),
        };
        assert_eq!(hb2_sub.mode_id, 1, "frame {i}: second HB sub-mode");
        let _hb2_body = WidebandHighBandBody::parse(&mut r, &hb2_sub).unwrap();
    }
}
