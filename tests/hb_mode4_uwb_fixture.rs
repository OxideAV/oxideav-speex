//! **Ultra-wideband quality-10 (stacked mode-4) conformance gate**
//! (round r446, `docs/audio/speex/fixtures/hb-mode4-uwb-q10/`).
//!
//! The 32 kHz quality-10 default is the *stacked* case no other staged
//! fixture covers: the inner (4–8 kHz) high-band layer runs in
//! **submode 4** (the 80-bit two-stage innovation, 352-bit layer) while
//! the outer (8–16 kHz) layer runs in the folded **submode 1** — so the
//! outer fold's source is itself an innovation-coded high band rather
//! than the folded one every previously staged UWB fixture used
//! (fixture `notes.md`). A decoder with the submode-4 codebook right
//! but the fold source wrong (or vice versa) fails here in a different
//! band from the other fixtures.
//!
//! ## What this gate locks
//!
//! 1. **Bit-exact framing against the reference's own per-frame
//!    trace** — for all 76 frames: NB submode 7, inner high-band
//!    submode 4 with its 12-bit LSP MSVQ pair, per-sub-frame 4-bit gain
//!    correction and all ten `(sign, 7-bit index)` pairs (the packed
//!    80-bit field compared verbatim), and outer submode 1 with its LSP
//!    pair and four 5-bit folded-gain indices.
//! 2. **The stream decodes** through [`UltraWidebandDecoder`] (before
//!    campaign B a q10 stream was undecodable; this is the first gate
//!    that decodes mode 4 *inside* the embedded UWB recursion, exercised
//!    with the r440 state-derived gain base + polarity).
//! 3. **Per-band tracking** against `expected.pcm`, floors pinned at the
//!    r446 measured values so a reconstruction fix shows up as a floor
//!    raise and a regression fails loudly.

use oxideav_speex::{
    BitReader, HbLspStages, NarrowbandFrameBody, NarrowbandFrameHeader, Submode,
    UltraWidebandDecoder, UwbDecodedFrame, WidebandHighBandBody, WidebandHighBandFrameHeader,
    WidebandSubmode,
};

const INPUT: &[u8] = include_bytes!("fixtures/hb-mode4-uwb-q10/input.spx");
const EXPECTED: &[u8] = include_bytes!("fixtures/hb-mode4-uwb-q10/expected.pcm");
const TRACE: &str = include_str!("fixtures/hb-mode4-uwb-q10/frame-trace.txt");

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

/// One trace row: the inner mode-4 layer (LSP pair + per-sub-frame gain
/// correction + packed 80-bit excitation-VQ field) and the outer mode-1
/// layer (LSP pair + four folded-gain indices).
struct TraceRow {
    frame: usize,
    hb1_lsp: (u8, u8),
    hb1_gc: [u8; 4],
    hb1_vq: [u128; 4],
    hb2_lsp: (u8, u8),
    hb2_gains: [u8; 4],
}

fn parse_lsp(section: &str) -> (u8, u8) {
    let at = section.find("lsp=(").expect("lsp pair") + 5;
    let close = section[at..].find(')').expect("lsp close") + at;
    let mut it = section[at..close].split(',');
    (
        it.next().unwrap().trim().parse().unwrap(),
        it.next().unwrap().trim().parse().unwrap(),
    )
}

/// Parse one `[gc= G ±idx ×10]` sub-frame block into the 4-bit gain
/// correction and the packed 80-bit excitation-VQ field (ten 8-bit
/// `[sign][7-bit index]` groups MSB-first — stage 1 then stage 2, the
/// bitstream order the trace prints).
fn parse_mode4_block(block: &str) -> (u8, u128) {
    let block = block.strip_prefix("gc=").expect("block starts with gc=");
    let mut tokens: Vec<&str> = Vec::new();
    let mut cur = block;
    while !cur.is_empty() {
        cur = cur.trim_start();
        if cur.is_empty() {
            break;
        }
        if let Some(stripped) = cur.strip_prefix(['+', '-']) {
            tokens.push(&cur[..1]);
            cur = stripped;
        } else {
            let end = cur.find(|c: char| !c.is_ascii_digit()).unwrap_or(cur.len());
            tokens.push(&cur[..end]);
            cur = &cur[end..];
        }
    }
    let gc: u8 = tokens[0].parse().expect("gc index");
    assert_eq!(tokens.len(), 21, "ten sign/index pairs per sub-frame");
    let mut vq = 0u128;
    for t in (1..21).step_by(2) {
        let sign = u128::from(tokens[t] == "-");
        let idx: u128 = tokens[t + 1].parse().expect("codebook index");
        vq = (vq << 8) | (sign << 7) | (idx & 0x7F);
    }
    (gc, vq)
}

fn parse_trace() -> Vec<TraceRow> {
    let mut rows = Vec::new();
    for line in TRACE.lines() {
        let l = line.trim();
        if !l.starts_with("frame") {
            continue;
        }
        let mut sections = l.split(" | ");
        let head = sections.next().unwrap();
        let hb1 = sections.next().expect("hb1 section");
        let hb2 = sections.next().expect("hb2 section");
        let frame: usize = head
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .expect("frame index");
        assert!(head.contains("nb=7"), "frame {frame}: NB submode 7");
        assert!(hb1.contains("m=4"), "frame {frame}: inner submode 4");
        assert!(hb2.contains("m=1"), "frame {frame}: outer submode 1");

        let mut gc = [0u8; 4];
        let mut vq = [0u128; 4];
        let mut rest = hb1;
        for s in 0..4 {
            let open = rest.find('[').expect("sub-frame block");
            let close = rest[open..].find(']').expect("block close") + open;
            let (g, v) = parse_mode4_block(&rest[open + 1..close]);
            gc[s] = g;
            vq[s] = v;
            rest = &rest[close + 1..];
        }

        let gtail = hb2.find("g=").expect("outer gains") + 2;
        let gains: Vec<u8> = hb2[gtail..]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        assert_eq!(gains.len(), 4, "four outer folded gains");

        rows.push(TraceRow {
            frame,
            hb1_lsp: parse_lsp(hb1),
            hb1_gc: gc,
            hb1_vq: vq,
            hb2_lsp: parse_lsp(hb2),
            hb2_gains: [gains[0], gains[1], gains[2], gains[3]],
        });
    }
    rows
}

/// Direct DFT band energies (dB) of one 640-sample 32 kHz frame:
/// 0–4 kHz (bins 1..80), 4–8 kHz (80..160), 8–16 kHz (160..320).
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
        for f in dec.decode_packet(pkt).expect("q10 UWB stream decodes") {
            if let UwbDecodedFrame::Audio(a) = f {
                out.extend_from_slice(a.uwb_pcm.as_ref());
            }
        }
    }
    out
}

/// **Bit-exact framing against the reference's per-frame trace** — the
/// first framing validation of submode 4 inside the embedded UWB
/// recursion (352-bit inner layer + 36-bit outer layer).
#[test]
fn framing_matches_reference_trace() {
    let packets = lift_ogg_packets(INPUT);
    let audio = &packets[2..];
    assert_eq!(audio.len(), 76, "fixture carries 76 audio frames");
    let trace = parse_trace();
    assert_eq!(trace.len(), 76, "trace carries 76 rows");

    for (i, pkt) in audio.iter().enumerate() {
        let mut r = BitReader::new(pkt);
        let nbh = NarrowbandFrameHeader::parse(&mut r).unwrap();
        let nb_sub = match nbh.submode {
            Submode::Celp(s) => s,
            other => panic!("frame {i}: unexpected NB submode {other:?}"),
        };
        assert_eq!(nb_sub.mode_id, 7, "frame {i}: NB mode");
        let _nb = NarrowbandFrameBody::parse(&mut r, &nb_sub).unwrap();

        // Layer 1 — wideband high band (4–8 kHz), submode 4.
        let hb1h = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        let hb1_sub = match hb1h.submode {
            WidebandSubmode::Documented(s) => s,
            other => panic!("frame {i}: unexpected HB1 submode {other:?}"),
        };
        assert_eq!(hb1_sub.mode_id, 4, "frame {i}: HB1 mode");
        let hb1 = WidebandHighBandBody::parse(&mut r, &hb1_sub).unwrap();

        // Layer 2 — ultra-wideband high band (8–16 kHz), submode 1.
        let hb2h = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        let hb2_sub = match hb2h.submode {
            WidebandSubmode::Documented(s) => s,
            other => panic!("frame {i}: unexpected HB2 submode {other:?}"),
        };
        assert_eq!(hb2_sub.mode_id, 1, "frame {i}: HB2 mode");
        let hb2 = WidebandHighBandBody::parse(&mut r, &hb2_sub).unwrap();

        let tr = &trace[i];
        assert_eq!(tr.frame, i, "trace row ordering");
        let s1 = HbLspStages::from_packed(hb1.lsp_index, &hb1_sub).unwrap();
        assert_eq!((s1.stage1, s1.stage2), tr.hb1_lsp, "frame {i}: HB1 LSP");
        for s in 0..4 {
            assert_eq!(
                hb1.subframes[s].excitation_gain_index, tr.hb1_gc[s],
                "frame {i} sub {s}: gain correction"
            );
            assert_eq!(
                hb1.subframes[s].excitation_vq_index, tr.hb1_vq[s],
                "frame {i} sub {s}: packed 80-bit mode-4 excitation field"
            );
        }
        let s2 = HbLspStages::from_packed(hb2.lsp_index, &hb2_sub).unwrap();
        assert_eq!((s2.stage1, s2.stage2), tr.hb2_lsp, "frame {i}: HB2 LSP");
        let g2: Vec<u8> = hb2
            .subframes
            .iter()
            .map(|s| s.excitation_gain_index)
            .collect();
        assert_eq!(g2[..], tr.hb2_gains, "frame {i}: outer folded gains");
    }
}

/// **The stacked q10 stream decodes and tracks the reference.**
#[test]
fn decode_tracks_reference() {
    let reference = reference_pcm();
    assert_eq!(reference.len(), 48_000, "reference PCM geometry");
    let ours = decode_fixture();
    assert_eq!(ours.len(), 76 * 640, "76 × 640-sample UWB frames");

    // `expected.pcm` is source-length-trimmed (fixture notes §PCM
    // geometry), so our untrimmed decode is *delayed* against it. Sweep
    // the delay on full-signal correlation and pin it.
    let mut best_delay = 0usize;
    let mut best_corr = f64::NEG_INFINITY;
    for delay in 0..400usize {
        let n = (ours.len() - delay).min(reference.len());
        let (mut dot, mut ee, mut rr) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            dot += reference[i] * ours[delay + i];
            ee += ours[delay + i] * ours[delay + i];
            rr += reference[i] * reference[i];
        }
        let c = dot / (ee.sqrt() * rr.sqrt()).max(1e-30);
        if c > best_corr {
            best_corr = c;
            best_delay = delay;
        }
    }
    println!("our-delay {best_delay} (corr {best_corr:.4})");
    // Measured r446: 351 samples at 32 kHz — the triple-QMF decode
    // chain's look-ahead against the source-length-trimmed reference
    // (the 16 kHz fixture pins ≈142; one more QMF stage doubles it
    // plus its own filter delay). A moved delay means the filterbank
    // timeline changed.
    assert!(
        (348..=354).contains(&best_delay),
        "decoder delay vs the trimmed reference moved (got {best_delay}, was ≈351)"
    );
    assert!(
        best_corr > 0.98,
        "full-signal alignment correlation {best_corr:.4} ≤ 0.98 (was 0.9967, r450)"
    );

    let ours = &ours[best_delay..];
    let n = (reference.len().min(ours.len()) / 640) * 640;

    // Full-signal absolute SNR (no fitted gain).
    let (mut err, mut er) = (0.0f64, 0.0f64);
    for i in 0..n {
        let d = reference[i] - ours[i];
        err += d * d;
        er += reference[i] * reference[i];
    }
    let snr = 10.0 * (er / (err + 1e-12)).log10();

    // Per-frame per-band mean |error| over the non-silent frames.
    let mut sum = [0.0f64; 3];
    let mut cnt = 0usize;
    for f in 0..n / 640 {
        let o = f * 640;
        let rb = band_energies(&reference[o..o + 640]);
        let ob = band_energies(&ours[o..o + 640]);
        if rb.0 > -20.0 {
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
        "hb-mode4-uwb-q10 full {snr:.2} dB | band mean|err| dB (n={cnt}): \
         0-4k={:.2} 4-8k={:.2} 8-16k={:.2}",
        m[0], m[1], m[2]
    );

    // r446 measured: full 0.03 dB; bands 1.29 / 5.55 / 7.98 dB.
    //
    // - 0–4 kHz: the embedded NB-7 low band is reference-tracking.
    // - 4–8 kHz: the inner **mode-4** layer through the r440
    //   state-derived gain base + polarity — 5.55 dB here, replicating
    //   the wideband fixture's ≈6.1 dB on a *second* stream whose
    //   high-band LSP pair varies frame to frame (the WB oracle holds
    //   it constant), so the level-tracking law is not a one-fixture
    //   artifact. The remaining residual is the unpinned exact gain
    //   law (provenance/08 — recorded docs gap).
    // - 8–16 kHz: the outer folded layer over an innovation-coded
    //   source — 7.98 dB, the same documented campaign-A outer-fold
    //   residual as `uwb-speech-3layer` (≈7.1 dB); the exact outer
    //   source normalisation is the recorded `hb-folded-gain.md` §7.4
    //   docs gap.
    //
    // r450 (crossover-anchored laws through all three layers):
    // measured full 20.16 dB, bands 0.10 / 0.49 / 0.70 dB.
    assert!(snr > 17.0, "full-signal SNR {snr:.2} dB < 17 (was 20.16)");
    assert!(
        m[0] < 0.5,
        "0-4 kHz mean |err| {:.2} dB ≥ 0.5 (was 0.10)",
        m[0]
    );
    assert!(
        m[1] < 1.2,
        "4-8 kHz mean |err| {:.2} dB ≥ 1.2 (was 0.49)",
        m[1]
    );
    assert!(
        m[2] < 1.5,
        "8-16 kHz mean |err| {:.2} dB ≥ 1.5 (was 0.70)",
        m[2]
    );
    if m[1] < 3.0 {
        println!("NOTE: inner mode-4 gain law appears pinned — tighten this floor");
    }
}
