//! **Mode-4 gain-base discrimination gates** (round r446,
//! `tests/fixtures/hb-mode4-gain-probes/` — see its `NOTES.md`).
//!
//! Provenance/08 names "a fixture pair differing only in the low-band
//! content at fixed high-band bits, or vice versa" as the measurement
//! that would close #329 residual 1; this crate generated that pair
//! black-box (r410 conformance-fixture precedent) and the QMF-recovered
//! measurement over it settles the *drivers*:
//!
//! 1. [`lbvar_pins_base_not_lowband`] — with the high-band content
//!    held fixed, a **31.6 dB** low-band sweep moves the recovered
//!    per-sub-frame gain by only ≈ 2 dB: the reference's gain base is
//!    **not** the same frame's low-band level (a causal `lb²` base
//!    would swing ≈ 63 dB). Provenance/08's low-band R² was natural
//!    speech co-variation.
//! 2. [`hbvar_pins_backward_adaptive_base`] — with the low band held
//!    fixed, an 18 dB high-band sweep is tracked by the recovered gain
//!    while the transmitted 4-bit correction stays parked at the grid
//!    bottom: the base is **backward-adaptive decoder state** (recent
//!    high-band excitation memory), not the transmitted field.
//! 3. [`crate_decode_tracks_probes`] — under the r450
//!    **crossover-anchored gain law**
//!    (`g = gc_recon·|A_hb(π)|·rms(e_lb)/|A_lb(π)|`, measured by
//!    crafted-bitstream probing — `oxideav_speex::hb_gc_crossover_gain`)
//!    the crate decodes both off-manifold probe streams to
//!    **≤ 0.4 dB** per-segment band error in *both* bands — the r440
//!    fitted law's 6…27 dB known divergence this gate used to pin is
//!    closed, and the ceilings are now conformance floors. (The r446
//!    "backward-adaptive memory" reading of finding 2 is superseded:
//!    the tracking was the transmitted high-band *envelope* moving
//!    `|A_hb(π)|` plus the correction steps, not decoder gain memory —
//!    crafted streams that hold every field constant show no memory at
//!    all.)

use oxideav_speex::{
    decode_hb_subframe_mode4_f32, BitReader, NarrowbandFrameBody, NarrowbandFrameHeader,
    QmfAnalysis, SpeexDecoder, Submode, WidebandHighBandBody, WidebandHighBandFrameHeader,
    WidebandSubmode,
};

const LBVAR_SPX: &[u8] = include_bytes!("fixtures/hb-mode4-gain-probes/lbvar.spx");
const LBVAR_PCM: &[u8] = include_bytes!("fixtures/hb-mode4-gain-probes/lbvar.noenh.pcm");
const HBVAR_SPX: &[u8] = include_bytes!("fixtures/hb-mode4-gain-probes/hbvar.spx");
const HBVAR_PCM: &[u8] = include_bytes!("fixtures/hb-mode4-gain-probes/hbvar.noenh.pcm");

const HB_ORDER: usize = 8;
const SUBFRAME: usize = 40;
const FRAME_HB: usize = 160;
const SEG_FRAMES: usize = 16;

fn lift(buf: &[u8]) -> Vec<Vec<u8>> {
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
        let segments: Vec<u8> = buf[pos + 27..pos + 27 + nseg].to_vec();
        let mut body_pos = pos + 27 + nseg;
        for &ln in &segments {
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

/// Per sub-frame: the 4-bit gain-correction index and the packed
/// 80-bit mode-4 excitation field, parsed off the wire by the crate.
fn parse_stream(spx: &[u8]) -> Vec<[(u8, u128); 4]> {
    let packets = lift(spx);
    let mut frames = Vec::new();
    for pkt in &packets[2..] {
        let mut r = BitReader::new(pkt);
        let nbh = NarrowbandFrameHeader::parse(&mut r).unwrap();
        let nb_sub = match nbh.submode {
            Submode::Celp(s) => s,
            other => panic!("unexpected NB submode {other:?}"),
        };
        assert_eq!(nb_sub.mode_id, 7, "q10 stream: NB submode 7");
        let _ = NarrowbandFrameBody::parse(&mut r, &nb_sub).unwrap();
        let hbh = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        let hb_sub = match hbh.submode {
            WidebandSubmode::Documented(s) => s,
            other => panic!("unexpected HB submode {other:?}"),
        };
        assert_eq!(hb_sub.mode_id, 4, "q10 stream: HB submode 4");
        let hb = WidebandHighBandBody::parse(&mut r, &hb_sub).unwrap();
        let mut subs = [(0u8, 0u128); 4];
        for (s, slot) in subs.iter_mut().enumerate() {
            *slot = (
                hb.subframes[s].excitation_gain_index,
                hb.subframes[s].excitation_vq_index,
            );
        }
        frames.push(subs);
    }
    frames
}

/// Two-band split with a zero-input flush frame (whole-signal
/// convolution equivalence).
fn qmf_split(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut qa = QmfAnalysis::new();
    let frames = x.len() / 320;
    let mut lb = vec![0.0f64; (frames + 1) * FRAME_HB];
    let mut hb = vec![0.0f64; (frames + 1) * FRAME_HB];
    let zeros = [0.0f64; 320];
    for f in 0..=frames {
        let input = if f < frames {
            &x[f * 320..(f + 1) * 320]
        } else {
            &zeros[..]
        };
        qa.split_slices(
            input,
            &mut lb[f * FRAME_HB..(f + 1) * FRAME_HB],
            &mut hb[f * FRAME_HB..(f + 1) * FRAME_HB],
        );
    }
    (lb, hb)
}

fn winr(sig: &[f64], start: i64, len: usize) -> Vec<f64> {
    (start..start + len as i64)
        .map(|i| {
            if i >= 0 && (i as usize) < sig.len() {
                sig[i as usize]
            } else {
                0.0
            }
        })
        .collect()
}

/// Global order-8 LPC + monic inverse filter (legitimate per stream:
/// the high-band spectral *shape* is constant by construction).
fn inverse_filter_global_lpc(x: &[f64]) -> Vec<f64> {
    let mut r = [0.0f64; HB_ORDER + 1];
    for (m, slot) in r.iter_mut().enumerate() {
        *slot = x[m..].iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
    }
    r[0] *= 1.0001;
    let mut a = [0.0f64; HB_ORDER];
    let mut err = r[0];
    for i in 0..HB_ORDER {
        let mut acc = r[i + 1];
        for j in 0..i {
            acc -= a[j] * r[i - j];
        }
        let k = acc / err;
        let mut na = a;
        na[i] = k;
        for j in 0..i {
            na[j] = a[j] - k * a[i - 1 - j];
        }
        a = na;
        err *= 1.0 - k * k;
    }
    let mut e = vec![0.0f64; x.len()];
    for n in 0..x.len() {
        let mut v = x[n];
        for (k, &ak) in a.iter().enumerate() {
            if n > k {
                v -= ak * x[n - k - 1];
            }
        }
        e[n] = v;
    }
    e
}

struct Row {
    frame: usize,
    gc_index: u8,
    proj: f64,
    rho: f64,
    lb_frame_rms: f64,
}

/// The provenance/08 measurement over one probe stream at the pinned
/// −40 recovered-band offset.
fn measure(spx: &[u8], pcm_bytes: &[u8]) -> Vec<Row> {
    let pcm: Vec<f64> = pcm_bytes
        .chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (lb, hb) = qmf_split(&pcm);
    let exc = inverse_filter_global_lpc(&hb);
    let offset = -40i64;
    let mut rows = Vec::new();
    for (f, subs) in parse_stream(spx).iter().enumerate() {
        let fstart = f as i64 * FRAME_HB as i64 + offset;
        let lseg = winr(&lb, fstart, FRAME_HB);
        let lb_frame_rms = (lseg.iter().map(|&v| v * v).sum::<f64>() / FRAME_HB as f64).sqrt();
        for (s, &(gc_index, vq)) in subs.iter().enumerate() {
            let shape = decode_hb_subframe_mode4_f32(vq);
            let v: Vec<f64> = shape.iter().map(|&x| f64::from(x)).collect();
            let vv: f64 = v.iter().map(|&b| b * b).sum();
            if vv <= 0.0 {
                continue; // all-zero innovation (index 43 across groups)
            }
            let start = fstart + s as i64 * SUBFRAME as i64;
            let e = winr(&exc, start, SUBFRAME);
            let ee: f64 = e.iter().map(|&a| a * a).sum();
            if ee <= 0.0 {
                continue;
            }
            let ev: f64 = e.iter().zip(&v).map(|(&a, &b)| a * b).sum();
            rows.push(Row {
                frame: f,
                gc_index,
                proj: ev / vv,
                rho: ev / (ee.sqrt() * vv.sqrt()),
                lb_frame_rms,
            });
        }
    }
    rows
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Per-segment medians of `log10 |proj|`, `log10 lb_rms`, and gc index
/// over rows whose recovered excitation matches the innovation at
/// least at `rho_min` (0.6 = the doc regime; the quietest `hbvar`
/// segment sits mostly below it under the global whitening fit, so
/// that gate lowers the cut — medians stay robust).
fn segment_medians(rows: &[Row], rho_min: f64) -> Vec<(f64, f64, f64, usize)> {
    let mut out = Vec::new();
    for k in 0..5usize {
        let sel: Vec<&Row> = rows
            .iter()
            .filter(|r| {
                r.frame >= k * SEG_FRAMES && r.frame < (k + 1) * SEG_FRAMES && r.rho.abs() > rho_min
            })
            .collect();
        assert!(
            sel.len() >= 30,
            "segment {k}: only {} usable rows",
            sel.len()
        );
        let mut p: Vec<f64> = sel.iter().map(|r| r.proj.abs().log10()).collect();
        let mut l: Vec<f64> = sel.iter().map(|r| r.lb_frame_rms.log10()).collect();
        let mut g: Vec<f64> = sel.iter().map(|r| f64::from(r.gc_index)).collect();
        out.push((median(&mut p), median(&mut l), median(&mut g), sel.len()));
    }
    out
}

/// **The gain base is not the same frame's low-band level.**
#[test]
fn lbvar_pins_base_not_lowband() {
    let rows = measure(LBVAR_SPX, LBVAR_PCM);
    let usable: Vec<f64> = rows
        .iter()
        .filter(|r| r.rho.abs() > 0.0)
        .map(|r| r.rho.abs())
        .collect();
    let mean_rho = usable.iter().sum::<f64>() / usable.len() as f64;
    let segs = segment_medians(&rows, 0.6);
    let proj: Vec<f64> = segs.iter().map(|s| s.0).collect();
    let lb: Vec<f64> = segs.iter().map(|s| s.1).collect();
    println!("lbvar mean|rho| {mean_rho:.4}; per-seg median log10 proj {proj:?} lb {lb:?}");
    // Measured r446: mean |ρ| 0.9198; proj medians span 0.104 log10
    // (≈2 dB) while the recovered low band spans 1.53 log10 (≈31 dB).
    assert!(mean_rho > 0.88, "recovered-excitation match degraded");
    let proj_span = proj.iter().cloned().fold(f64::MIN, f64::max)
        - proj.iter().cloned().fold(f64::MAX, f64::min);
    let lb_span =
        lb.iter().cloned().fold(f64::MIN, f64::max) - lb.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        lb_span > 1.4,
        "low-band sweep collapsed ({lb_span:.2} log10)"
    );
    assert!(
        proj_span < 0.25,
        "recovered gain now tracks the low band ({proj_span:.2} log10 over {lb_span:.2}) — \
         the base-driver conclusion changed, re-derive"
    );
}

/// **The gain base is backward-adaptive decoder state.**
#[test]
fn hbvar_pins_backward_adaptive_base() {
    let rows = measure(HBVAR_SPX, HBVAR_PCM);
    let usable: Vec<f64> = rows
        .iter()
        .filter(|r| r.rho.abs() > 0.0)
        .map(|r| r.rho.abs())
        .collect();
    let mean_rho = usable.iter().sum::<f64>() / usable.len() as f64;
    let segs = segment_medians(&rows, 0.3);
    let proj: Vec<f64> = segs.iter().map(|s| s.0).collect();
    let gc: Vec<f64> = segs.iter().map(|s| s.2).collect();
    println!("hbvar mean|rho| {mean_rho:.4}; per-seg median log10 proj {proj:?} gc {gc:?}");
    assert!(mean_rho > 0.70, "recovered-excitation match degraded");
    // Measured r446: recovered gain rises ≈0.9 log10 (18 dB) across
    // the high-band sweep while the median transmitted correction
    // stays at the grid bottom (0–2 of 15) in every segment.
    let rise = proj[4] - proj[0];
    assert!(
        rise > 0.6,
        "recovered gain no longer tracks the high-band level (rise {rise:.2} log10)"
    );
    for (k, &g) in gc.iter().enumerate() {
        assert!(
            g <= 3.0,
            "segment {k}: median gc {g} — the correction now carries the \
             adaptation, the backward-adaptive conclusion changed"
        );
    }
}

/// Band energies (dB) of one 320-sample 16 kHz frame:
/// 0–4 kHz (bins 1..80), 4–8 kHz (80..160).
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

fn decode_ours(spx: &[u8]) -> Vec<f64> {
    let packets = lift(spx);
    let mut dec = SpeexDecoder::new();
    let mut out = Vec::new();
    for pkt in &packets[2..] {
        let pcm = dec.decode_packet_pcm_i16(pkt).expect("probe decodes");
        out.extend(pcm.iter().map(|&s| f64::from(s)));
    }
    out
}

/// Per-segment (mean low-band err, mean high-band err) of our decode
/// against the reference, at the swept whole-file alignment.
fn segment_band_errors(spx: &[u8], pcm_bytes: &[u8]) -> (usize, Vec<(f64, f64)>) {
    let reference: Vec<f64> = pcm_bytes
        .chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect();
    let ours = decode_ours(spx);
    let mut best = (0usize, f64::MIN);
    for d in 0..300usize {
        let n = (ours.len() - d).min(reference.len());
        let (mut dot, mut ee, mut rr) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            dot += reference[i] * ours[d + i];
            ee += ours[d + i] * ours[d + i];
            rr += reference[i] * reference[i];
        }
        let c = dot / (ee.sqrt() * rr.sqrt()).max(1e-30);
        if c > best.1 {
            best = (d, c);
        }
    }
    let o = &ours[best.0..];
    let n = (reference.len().min(o.len()) / 320) * 320;
    let mut segs = [(0.0f64, 0.0f64, 0usize); 5];
    for f in 0..n / 320 {
        let rb = band_db(&reference[f * 320..(f + 1) * 320]);
        let ob = band_db(&o[f * 320..(f + 1) * 320]);
        let k = (f / SEG_FRAMES).min(4);
        segs[k].0 += (rb.0 - ob.0).abs();
        segs[k].1 += (rb.1 - ob.1).abs();
        segs[k].2 += 1;
    }
    (
        best.0,
        segs.iter()
            .map(|&(a, b, c)| (a / c as f64, b / c as f64))
            .collect(),
    )
}

/// **Known divergence of the fitted `(gc·lb_rms)²` law off the natural
/// speech manifold** — ceilings pinned; a landed gain law collapses
/// them (and the two measurement gates above say what it must be).
#[test]
fn crate_decode_tracks_probes() {
    let (d1, lbv) = segment_band_errors(LBVAR_SPX, LBVAR_PCM);
    let (d2, hbv) = segment_band_errors(HBVAR_SPX, HBVAR_PCM);
    println!("lbvar delay {d1}: per-seg (lo,hi) dB {lbv:?}");
    println!("hbvar delay {d2}: per-seg (lo,hi) dB {hbv:?}");
    assert!(
        (140..=150).contains(&d1) && (140..=150).contains(&d2),
        "decoder delay moved"
    );

    // The low band decodes essentially exactly on both probes
    // (measured ≤ 0.08 dB per segment).
    for (k, &(lo, _)) in lbv.iter().enumerate() {
        assert!(lo < 0.5, "lbvar seg {k}: low band {lo:.2} dB ≥ 0.5");
    }
    for (k, &(lo, _)) in hbv.iter().enumerate() {
        assert!(lo < 0.5, "hbvar seg {k}: low band {lo:.2} dB ≥ 0.5");
    }
    // r450 crossover-anchored law: both probe streams decode to
    // ≤ 0.4 dB per-segment high-band error (measured lbvar
    // 0.07/0.02/0.04/0.03/0.06, hbvar 0.37/0.23/0.09/0.05/0.03 —
    // the r446 fitted-law divergence of 6…27 dB is closed). Floor at
    // 1 dB with cross-platform margin.
    for k in 0..5 {
        assert!(
            lbv[k].1 < 1.0,
            "lbvar seg {k}: 4–8 kHz {:.2} dB ≥ 1.0 (gain-law regression)",
            lbv[k].1
        );
        assert!(
            hbv[k].1 < 1.0,
            "hbvar seg {k}: 4–8 kHz {:.2} dB ≥ 1.0 (gain-law regression)",
            hbv[k].1
        );
    }
}
