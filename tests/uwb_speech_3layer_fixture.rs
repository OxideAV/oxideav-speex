//! **Ultra-wideband 3-layer speech conformance gate** (campaign A).
//!
//! Drives the staged speech oracle (`docs/audio/speex/fixtures/
//! uwb-speech-3layer/`, mirrored under `tests/fixtures/
//! uwb-speech-3layer/`) — the deterministic formant-speech UWB stream
//! (quality 1, all frames NB-8 / HB-1 / UWB-1) that supersedes the
//! near-stationary tone fixtures as the outer-fold oracle
//! (`docs/audio/speex/hb-folded-gain.md` §7.4, `notes.md`) — through
//! the crate's [`UltraWidebandDecoder`].
//!
//! ## What this gate locks
//!
//! 1. **Bit-exact framing against the reference's own per-frame trace.**
//!    The staged `frame-trace.txt` records, for every one of the 126
//!    audio frames, the reference decoder's parse: NB sub-mode, both
//!    high-band layers' sub-mode, their 12-bit LSP MSVQ stage indices
//!    `(i1, i2)`, and all four 5-bit folded-gain indices per layer.
//!    [`framing_matches_reference_trace`] re-parses the stream and
//!    asserts **every** field matches — proving the crate's whole UWB
//!    parse + index-extraction path (three-layer §5.5 walk, Table
//!    10.1 field widths, the `HbLspStages` 6+6 split, gain-index
//!    resolution) is reference-exact across all 126 frames. This
//!    isolates any remaining PCM divergence to the *reconstruction*
//!    (fold / synthesis) stage, not the framing.
//!
//! 2. **Per-band decode tracking.** [`decode_tracks_reference`] scores
//!    the 32 kHz decode against `expected.pcm` per 640-sample frame in
//!    three bands (0–4 / 4–8 / 8–16 kHz). The narrowband band is
//!    reference-accurate (≈1 dB); the two high-band fold layers carry a
//!    documented residual (campaign A) pinned here as a tracking floor
//!    so a fix shows up as a floor raise and a regression fails loudly.
//!
//! ## Recorded divergence (campaign A differential-debug)
//!
//! The framing being bit-exact (test 1) localises the residual to the
//! folded high-band **reconstruction**. Measured per-band mean |error|
//! over the 111 non-silent frames: 0–4 kHz ≈ 1.1 dB, 4–8 kHz ≈ 4.8 dB,
//! 8–16 kHz ≈ 7.1 dB. The high-band error concentrates on the
//! high-formant frames (weak low band, strong high bands, e.g. the /i/
//! vowel run) where the mode-1 folded law's saturation behaviour and
//! the outer layer's fold source/scale are only pinned by the
//! near-stationary tone fixtures. `docs/audio/speex/hb-folded-gain.md`
//! §7.3/§7.4 describe a kneeless `C·|Â(π)|` law with the WB
//! *synthesized* high-band signal as the outer source; on the
//! frames whose fold source (the WB high band) is itself accurate, that
//! law drives the 8–16 kHz band to <1 dB — but it conflicts with the
//! `uwb-fold-geometry` tone oracle (whose outer band carries
//! independent tones mode-1 folded cannot reproduce), so the exact
//! reconciling reference fact is a docs gap (see the crate README).

use oxideav_speex::{
    BitReader, HbLspStages, NarrowbandFrameBody, NarrowbandFrameHeader, Submode,
    UltraWidebandDecoder, UwbDecodedFrame, WidebandHighBandBody, WidebandHighBandFrameHeader,
    WidebandSubmode,
};

const INPUT: &[u8] = include_bytes!("fixtures/uwb-speech-3layer/input.spx");
const EXPECTED: &[u8] = include_bytes!("fixtures/uwb-speech-3layer/expected.pcm");
const TRACE: &str = include_str!("fixtures/uwb-speech-3layer/frame-trace.txt");

/// The reference decode leads ours by 160 samples at 32 kHz (the
/// double-QMF look-ahead padding, same as the tone fixture).
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

/// One row of the staged `frame-trace.txt` (the reference decoder's
/// per-frame parse, framing from Manual Tables 9.1/10.1 only).
struct TraceRow {
    frame: usize,
    hb_lsp: (u8, u8),
    hb_gains: [u8; 4],
    uwb_lsp: (u8, u8),
    uwb_gains: [u8; 4],
}

fn parse_trace() -> Vec<TraceRow> {
    let mut rows = Vec::new();
    for line in TRACE.lines() {
        let l = line.trim();
        if l.starts_with('#') || l.is_empty() {
            continue;
        }
        // Tokenise, treating '(', ')' and ',' as separators so the
        // `(i1,i2)` LSP columns split cleanly.
        let cleaned: String = l
            .chars()
            .map(|c| if matches!(c, '(' | ')' | ',') { ' ' } else { c })
            .collect();
        let t: Vec<&str> = cleaned.split_whitespace().collect();
        // frame nb hb i1 i2 g0 g1 g2 g3 uwb i1 i2 u0 u1 u2 u3 e04 e48 e816
        if t.len() < 16 {
            continue;
        }
        let u = |i: usize| t[i].parse::<u8>().unwrap();
        rows.push(TraceRow {
            frame: t[0].parse().unwrap(),
            hb_lsp: (u(3), u(4)),
            hb_gains: [u(5), u(6), u(7), u(8)],
            uwb_lsp: (u(10), u(11)),
            uwb_gains: [u(12), u(13), u(14), u(15)],
        });
    }
    rows
}

/// Direct DFT band energies (dB) of one 640-sample 32 kHz frame:
/// 0–4 kHz (bins 1..80), 4–8 kHz (80..160), 8–16 kHz (160..320), with a
/// −90 dB silence floor. The same filter measures reference and decode,
/// so the comparison domain is identical.
fn band_energies(frame: &[f64]) -> (f64, f64, f64) {
    let n = frame.len();
    let mut power = vec![0.0f64; n / 2 + 1];
    for (k, pw) in power.iter_mut().enumerate() {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        let w = -2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
        for (i, &x) in frame.iter().enumerate() {
            let ang = w * (i as f64);
            re += x * ang.cos();
            im += x * ang.sin();
        }
        *pw = re * re + im * im;
    }
    let sum = |lo: usize, hi: usize| power[lo..hi].iter().sum::<f64>() / (n as f64);
    let db = |e: f64| if e <= 1e-9 { -90.0 } else { 10.0 * e.log10() };
    (db(sum(1, 80)), db(sum(80, 160)), db(sum(160, 320)))
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
    let mut dec = UltraWidebandDecoder::new();
    let mut out = Vec::new();
    for pkt in &packets[2..] {
        for f in dec.decode_packet(pkt).unwrap() {
            if let UwbDecodedFrame::Audio(a) = f {
                out.extend_from_slice(a.uwb_pcm.as_ref());
            }
        }
    }
    out
}

/// **Bit-exact framing against the reference's own per-frame trace.**
/// Every NB / HB / UWB sub-mode, both high-band layers' LSP MSVQ stage
/// indices, and all eight folded-gain indices must match the staged
/// `frame-trace.txt` for all 126 audio frames.
#[test]
fn framing_matches_reference_trace() {
    let packets = lift_ogg_packets(INPUT);
    let audio = &packets[2..];
    assert_eq!(audio.len(), 126, "fixture carries 126 audio frames");
    let trace = parse_trace();
    assert_eq!(trace.len(), 126, "trace carries 126 rows");

    for (i, pkt) in audio.iter().enumerate() {
        let mut r = BitReader::new(pkt);
        let nbh = NarrowbandFrameHeader::parse(&mut r).unwrap();
        let nb_sub = match nbh.submode {
            Submode::Celp(s) => s,
            other => panic!("frame {i}: unexpected NB submode {other:?}"),
        };
        assert_eq!(nb_sub.mode_id, 8, "frame {i}: NB mode");
        let _nb = NarrowbandFrameBody::parse(&mut r, &nb_sub).unwrap();

        // Layer 1 — wideband high band (4–8 kHz).
        let hb1h = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        let hb1_sub = match hb1h.submode {
            WidebandSubmode::Documented(s) => s,
            other => panic!("frame {i}: unexpected HB1 submode {other:?}"),
        };
        assert_eq!(hb1_sub.mode_id, 1, "frame {i}: HB1 mode");
        let hb1 = WidebandHighBandBody::parse(&mut r, &hb1_sub).unwrap();

        // Layer 2 — ultra-wideband high band (8–16 kHz).
        let hb2h = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        let hb2_sub = match hb2h.submode {
            WidebandSubmode::Documented(s) => s,
            other => panic!("frame {i}: unexpected HB2 submode {other:?}"),
        };
        assert_eq!(hb2_sub.mode_id, 1, "frame {i}: HB2 mode");
        let hb2 = WidebandHighBandBody::parse(&mut r, &hb2_sub).unwrap();

        let s1 = HbLspStages::from_packed(hb1.lsp_index, &hb1_sub).unwrap();
        let s2 = HbLspStages::from_packed(hb2.lsp_index, &hb2_sub).unwrap();
        let g1: Vec<u8> = hb1
            .subframes
            .iter()
            .map(|s| s.excitation_gain_index)
            .collect();
        let g2: Vec<u8> = hb2
            .subframes
            .iter()
            .map(|s| s.excitation_gain_index)
            .collect();

        let tr = &trace[i];
        assert_eq!(tr.frame, i, "trace row ordering");
        assert_eq!(
            (s1.stage1, s1.stage2),
            tr.hb_lsp,
            "frame {i}: WB-HB LSP MSVQ (i1,i2)"
        );
        assert_eq!(g1[..], tr.hb_gains, "frame {i}: WB-HB folded-gain indices");
        assert_eq!(
            (s2.stage1, s2.stage2),
            tr.uwb_lsp,
            "frame {i}: UWB-HB LSP MSVQ (i1,i2)"
        );
        assert_eq!(
            g2[..],
            tr.uwb_gains,
            "frame {i}: UWB-HB folded-gain indices"
        );
    }
}

/// **Per-band decode tracking** against `expected.pcm`. The narrowband
/// band is reference-accurate; the high-band fold layers carry the
/// documented campaign-A residual, pinned as tracking floors.
#[test]
fn decode_tracks_reference() {
    let reference = reference_pcm();
    assert_eq!(reference.len(), 80_000, "reference PCM geometry");
    let ours = decode_fixture();
    assert_eq!(ours.len(), 126 * 640, "126 × 640-sample UWB frames");

    // Full-signal absolute SNR (no fitted gain). Measured 16.33 dB.
    let n = (reference.len() - REF_LEAD).min(ours.len());
    let (mut err, mut er) = (0.0f64, 0.0f64);
    for i in 0..n {
        let d = reference[REF_LEAD + i] - ours[i];
        err += d * d;
        er += reference[REF_LEAD + i] * reference[REF_LEAD + i];
    }
    let snr = 10.0 * (er / (err + 1e-12)).log10();
    println!("uwb-speech full: {snr:.2} dB");
    assert!(snr >= 15.5, "full-signal SNR {snr:.2} dB < 15.5 dB");

    // Per-frame per-band mean |error| over the non-silent frames.
    let mut sum = [0.0f64; 3];
    let mut cnt = 0usize;
    for f in 0..126usize {
        let ro = REF_LEAD + f * 640;
        let oo = f * 640;
        if ro + 640 > reference.len() || oo + 640 > ours.len() {
            break;
        }
        let rb = band_energies(&reference[ro..ro + 640]);
        let ob = band_energies(&ours[oo..oo + 640]);
        if rb.2 > -20.0 {
            sum[0] += (rb.0 - ob.0).abs();
            sum[1] += (rb.1 - ob.1).abs();
            sum[2] += (rb.2 - ob.2).abs();
            cnt += 1;
        }
    }
    let m = [
        sum[0] / cnt as f64,
        sum[1] / cnt as f64,
        sum[2] / cnt as f64,
    ];
    println!(
        "uwb-speech band mean|err| dB (n={cnt}): 0-4k={:.2} 4-8k={:.2} 8-16k={:.2}",
        m[0], m[1], m[2]
    );

    // Narrowband band is reference-accurate (r450 measured 0.85 dB).
    assert!(m[0] < 1.3, "0-4 kHz mean |err| {:.2} dB ≥ 1.3", m[0]);
    // High-band fold layers under the r450 crossover-anchored,
    // innovation-only-source fold law (measured 0.92 / 3.17 dB; was
    // 4.77 / 7.06 dB at campaign A). The 8–16 kHz tail is the outer
    // (second-layer) fold, still on the r403 excitation-source law.
    assert!(
        m[1] < 1.3,
        "4-8 kHz mean |err| {:.2} dB ≥ 1.3 (regression, was 0.84)",
        m[1]
    );
    assert!(
        m[2] < 1.7,
        "8-16 kHz mean |err| {:.2} dB ≥ 1.7 (regression, was 1.12)",
        m[2]
    );

    // If the high-band error ever drops below 3 dB the fold divergence
    // is materially fixed — raise these floors and update the README.
    if m[2] < 3.0 {
        println!("NOTE: uwb outer-fold divergence appears fixed — raise floors");
    }
}
