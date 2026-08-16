//! **QMF-recovered mode-4 excitation, second fixture** (round r446) —
//! the replication `provenance/08-qmf-recovered-hb-excitation.md`
//! itself asks for ("re-emitting that trace … would double the evidence
//! base"; its only blocker was its own trace reader, and the staged
//! `hb-mode4-uwb-q10/frame-trace.txt` parses fine under this crate's).
//!
//! The staged 32 kHz quality-10 oracle carries its mode-4 innovation in
//! the **inner** high-band layer, so recovering the isolated 4–8 kHz
//! sub-band takes a **two-stage** QMF analysis (32 kHz → 2×16 kHz →
//! the low half again → 2×8 kHz), both stages the crate's own
//! [`QmfAnalysis`]. Two properties make this fixture the discriminator
//! the wideband one is not:
//!
//! * its transmitted high-band **LSP pair varies frame to frame**
//!   (the WB oracle holds `(14, 50)` throughout), so envelope-dependent
//!   terms in the gain law become visible — the per-frame order-8 LPC
//!   fit replaces provenance/08's global fit accordingly;
//! * the recovered band sits under a *stacked* outer folded layer, so
//!   the isolation of the staged prototype is exercised against real
//!   8–16 kHz content, not silence.
//!
//! ## What this gate locks
//!
//! 1. **The §1 binding replicates on the second fixture, oracle-free**:
//!    mean |ρ| ≈ 0.93 at the same unique −40 alignment (the same
//!    one-sub-frame look-ahead as the WB fixture, surviving a second
//!    analysis stage), sign positive throughout.
//! 2. **The r440 fixed-2 gain reading stays serviceable** on rows the
//!    WB law was never fitted to (R² ≈ 0.75).
//! 3. **New docs-gap evidence, pinned as a fact**: the fixed-2
//!    residual is more than half explained by *which LSP envelope the
//!    frame transmits* — a term provenance/08 could not see (its
//!    fixture holds the envelope constant) and deliberately did not
//!    fit. A per-sub-frame **state** term (the previous sub-frame's
//!    recovered excitation RMS) likewise beats the same-frame low-band
//!    level on this fixture. Neither is adopted into the decode path:
//!    the free-fit exponents are not stable across the two fixtures,
//!    and the workspace standard (provenance/07/08) is to record the
//!    direction, not to fit a loose formula. The refined ask lives in
//!    the crate README.

use oxideav_speex::QmfAnalysis;

const EXPECTED: &[u8] = include_bytes!("fixtures/hb-mode4-uwb-q10/expected.pcm");
const TRACE: &str = include_str!("fixtures/hb-mode4-uwb-q10/frame-trace.txt");

const HB_ORDER: usize = 8;
const SUBFRAME: usize = 40;
const FRAME_HB: usize = 160;

/// One inner-layer trace sub-frame: gain correction + ten (sign, index)
/// pairs, plus the frame's transmitted 12-bit LSP MSVQ pair.
#[derive(Clone, Copy)]
struct TraceSubframe {
    gc_index: u8,
    pairs: [(bool, u8); 10],
    lsp: (u8, u8),
}

impl TraceSubframe {
    fn is_all_zero(&self) -> bool {
        self.pairs.iter().all(|&(_, idx)| idx == 43)
    }
}

fn parse_trace() -> Vec<[TraceSubframe; 4]> {
    let mut frames = Vec::new();
    for line in TRACE.lines() {
        let l = line.trim();
        if !l.starts_with("frame") {
            continue;
        }
        let hb1 = l.split(" | ").nth(1).expect("hb1 section");
        let at = hb1.find("lsp=(").expect("lsp") + 5;
        let close = hb1[at..].find(')').unwrap() + at;
        let mut it = hb1[at..close].split(',');
        let lsp: (u8, u8) = (
            it.next().unwrap().trim().parse().unwrap(),
            it.next().unwrap().trim().parse().unwrap(),
        );
        let mut subframes = Vec::new();
        let mut rest = hb1;
        while let Some(open) = rest.find('[') {
            let close = rest[open..].find(']').expect("block close") + open;
            let block = rest[open + 1..close].strip_prefix("gc=").expect("gc block");
            rest = &rest[close + 1..];
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
            let gc_index: u8 = tokens[0].parse().unwrap();
            let mut pairs = [(false, 0u8); 10];
            for (k, pair) in pairs.iter_mut().enumerate() {
                *pair = (tokens[1 + 2 * k] == "-", tokens[2 + 2 * k].parse().unwrap());
            }
            subframes.push(TraceSubframe {
                gc_index,
                pairs,
                lsp,
            });
        }
        assert_eq!(subframes.len(), 4, "four inner sub-frames per frame");
        frames.push([subframes[0], subframes[1], subframes[2], subframes[3]]);
    }
    frames
}

/// Rebuild the mode-4 innovation shape from the trace pairs via the
/// staged binding (stage 2 at weight 0.4, sign-applied `sv8-128` rows)
/// — through the crate's own decoder path.
fn innovation(sub: &TraceSubframe) -> [f32; SUBFRAME] {
    let mut vq = 0u128;
    for &(sign, idx) in &sub.pairs {
        vq = (vq << 8) | u128::from((u8::from(sign) << 7) | (idx & 0x7F));
    }
    oxideav_speex::decode_hb_subframe_mode4_f32(vq)
}

/// One two-band split with a zero-input flush frame (whole-signal
/// convolution equivalence — the landing-1 convention).
fn split(x: &[f64], frame_full: usize) -> (Vec<f64>, Vec<f64>) {
    let mut qa = QmfAnalysis::new();
    let half = frame_full / 2;
    let frames = x.len() / frame_full;
    let mut lo = vec![0.0f64; (frames + 1) * half];
    let mut hi = vec![0.0f64; (frames + 1) * half];
    let zeros = vec![0.0f64; frame_full];
    for f in 0..=frames {
        let input = if f < frames {
            &x[f * frame_full..(f + 1) * frame_full]
        } else {
            &zeros[..]
        };
        qa.split_slices(
            input,
            &mut lo[f * half..(f + 1) * half],
            &mut hi[f * half..(f + 1) * half],
        );
    }
    (lo, hi)
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

/// Order-8 autocorrelation LPC over one 160-sample recovered frame
/// (the per-frame fit — this fixture's transmitted envelope varies, so
/// provenance/08's global fit is not legitimate here), then the monic
/// inverse filter over the frame with 8 samples of preceding context.
fn frame_excitation(hb: &[f64], start: i64) -> Option<[f64; FRAME_HB]> {
    let seg = winr(hb, start, FRAME_HB);
    let mut r = [0.0f64; HB_ORDER + 1];
    for (m, slot) in r.iter_mut().enumerate() {
        *slot = seg[m..].iter().zip(seg.iter()).map(|(&a, &b)| a * b).sum();
    }
    if r[0] <= 0.0 {
        return None;
    }
    r[0] *= 1.0001;
    let mut a = [0.0f64; HB_ORDER];
    let mut err = r[0];
    for i in 0..HB_ORDER {
        if err <= 0.0 {
            return None;
        }
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
    let ctx = winr(hb, start - HB_ORDER as i64, HB_ORDER);
    let mut exc = [0.0f64; FRAME_HB];
    for n in 0..FRAME_HB {
        let mut v = seg[n];
        for (k, &ak) in a.iter().enumerate() {
            let idx = n as i64 - k as i64 - 1;
            v -= ak
                * if idx >= 0 {
                    seg[idx as usize]
                } else {
                    ctx[(idx + HB_ORDER as i64) as usize]
                };
        }
        exc[n] = v;
    }
    Some(exc)
}

struct Row {
    frame: usize,
    sub: usize,
    lsp: (u8, u8),
    gc_index: u8,
    /// `None` for the all-zero-innovation sub-frames (index 43 across
    /// all ten groups) — their excitation RMS still feeds the
    /// state-term chain.
    proj_rho: Option<(f64, f64)>,
    hb_exc_rms: f64,
    lb_frame_rms: f64,
}

/// The offset-invariant inputs: the two recovered 8 kHz sub-bands and
/// the parsed trace (hoisted out of the alignment sweep).
struct Recovered {
    lb: Vec<f64>,
    hb: Vec<f64>,
    frames: Vec<[TraceSubframe; 4]>,
}

fn recover() -> Recovered {
    let pcm: Vec<f64> = EXPECTED
        .chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect();
    // Two-stage split: 32 kHz → (0–8 kHz @16 kHz, 8–16 kHz), then the
    // low half again → (0–4 kHz @8 kHz, 4–8 kHz @8 kHz).
    let (lo16, _hi16) = split(&pcm, 640);
    let (lb, hb) = split(&lo16, 320);
    Recovered {
        lb,
        hb,
        frames: parse_trace(),
    }
}

fn measure(rec: &Recovered, offset: i64) -> Vec<Row> {
    let (lb, hb) = (&rec.lb, &rec.hb);
    let mut rows = Vec::new();
    for (f, subs) in rec.frames.iter().enumerate() {
        let start = f as i64 * FRAME_HB as i64 + offset;
        let Some(exc) = frame_excitation(hb, start) else {
            continue;
        };
        let lseg = winr(lb, start, FRAME_HB);
        let lb_frame_rms = (lseg.iter().map(|&v| v * v).sum::<f64>() / FRAME_HB as f64).sqrt();
        for (s, sub) in subs.iter().enumerate() {
            let e = &exc[s * SUBFRAME..(s + 1) * SUBFRAME];
            let ee: f64 = e.iter().map(|&v| v * v).sum();
            if ee <= 0.0 {
                continue;
            }
            let hb_exc_rms = (ee / SUBFRAME as f64).sqrt();
            let proj_rho = if sub.is_all_zero() {
                None
            } else {
                let v: Vec<f64> = innovation(sub).iter().map(|&x| f64::from(x)).collect();
                let ev: f64 = e.iter().zip(&v).map(|(&a, &b)| a * b).sum();
                let vv: f64 = v.iter().map(|&b| b * b).sum();
                Some((ev / vv, ev / (ee.sqrt() * vv.sqrt())))
            };
            rows.push(Row {
                frame: f,
                sub: s,
                lsp: sub.lsp,
                gc_index: sub.gc_index,
                proj_rho,
                hb_exc_rms,
                lb_frame_rms,
            });
        }
    }
    rows
}

fn mean_abs_rho(rows: &[Row]) -> (f64, usize) {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for r in rows {
        if let Some((_, rho)) = r.proj_rho {
            sum += rho.abs();
            n += 1;
        }
    }
    (sum / n.max(1) as f64, n)
}

/// **Binding replication on the second fixture** — through two QMF
/// stages and a per-frame LPC fit, with the transmitted envelope
/// varying frame to frame.
#[test]
fn uwb_recovered_excitation_confirms_binding() {
    let rec = recover();
    let mut best = (0i64, 0.0f64);
    let mut second = 0.0f64;
    for offset in -260i64..=59 {
        let rows = measure(&rec, offset);
        let (m, n) = mean_abs_rho(&rows);
        if n < 200 {
            continue;
        }
        if m > best.1 {
            second = best.1;
            best = (offset, m);
        } else if m > second {
            second = m;
        }
    }
    println!(
        "uwb alignment sweep: peak mean|rho| {:.4} at {}, runner-up {:.4}",
        best.1, best.0, second
    );
    assert_eq!(
        best.0, -40,
        "correlation peak moved off the −40 sub-frame look-ahead"
    );
    assert!(
        best.1 > 3.0 * second,
        "peak {:.4} lacks a factor-3 margin over {:.4}",
        best.1,
        second
    );

    let rows = measure(&rec, -40);
    let (mean, n) = mean_abs_rho(&rows);
    let positive = rows
        .iter()
        .filter(|r| matches!(r.proj_rho, Some((_, rho)) if rho > 0.0))
        .count() as f64
        / n as f64;
    println!(
        "uwb rows {n}: mean|rho| {mean:.4} positive {:.1}%",
        100.0 * positive
    );
    // Measured r446: 298 rows, mean |ρ| 0.9316 (the WB fixture: 0.88),
    // positive on ~97 % — the binding holds with the envelope varying.
    assert!(n >= 290, "measurable sub-frames {n} < 290");
    assert!(mean > 0.90, "mean |rho| {mean:.4} ≤ 0.90");
    assert!(positive > 0.95, "positive share {positive:.3} ≤ 0.95");
}

/// Least squares of `y ≈ Σ cᵢ·xᵢ + intercept`; returns (R², rms·20).
#[allow(clippy::needless_range_loop)] // dense normal-equation indexing
fn fit(y: &[f64], xs: &[&[f64]]) -> (f64, f64, Vec<f64>) {
    let n = y.len();
    let k = xs.len() + 1;
    // Normal equations over [x…, 1].
    let col = |i: usize, r: usize| -> f64 {
        if i < xs.len() {
            xs[i][r]
        } else {
            1.0
        }
    };
    let mut m = vec![vec![0.0f64; k + 1]; k];
    for r in 0..n {
        for i in 0..k {
            for j in 0..k {
                m[i][j] += col(i, r) * col(j, r);
            }
            m[i][k] += col(i, r) * y[r];
        }
    }
    // Gaussian elimination.
    for i in 0..k {
        let piv = (i..k)
            .max_by(|&a, &b| m[a][i].abs().total_cmp(&m[b][i].abs()))
            .unwrap();
        m.swap(i, piv);
        let d = m[i][i];
        for j in i..=k {
            m[i][j] /= d;
        }
        for r in 0..k {
            if r != i {
                let f = m[r][i];
                for j in i..=k {
                    m[r][j] -= f * m[i][j];
                }
            }
        }
    }
    let coef: Vec<f64> = (0..k).map(|i| m[i][k]).collect();
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let (mut ss_res, mut ss_tot) = (0.0f64, 0.0f64);
    for r in 0..n {
        let pred: f64 = (0..k).map(|i| coef[i] * col(i, r)).sum();
        ss_res += (y[r] - pred) * (y[r] - pred);
        ss_tot += (y[r] - mean_y) * (y[r] - mean_y);
    }
    (
        1.0 - ss_res / ss_tot,
        20.0 * (ss_res / n as f64).sqrt(),
        coef,
    )
}

/// **Gain-direction on the LSP-varying fixture** — the r440 fixed-2
/// reading stays serviceable, and the *new* facts are pinned: the
/// fixed-2 residual is majority-explained by the transmitted LSP
/// envelope class, and a decoder-state term (previous sub-frame's
/// excitation RMS) outperforms the same-frame low-band level. No law
/// is adopted (unstable exponents across fixtures — README docs ask).
#[test]
fn uwb_gain_direction_and_envelope_dependence() {
    let rows = measure(&recover(), -40);
    // The doc's regression regime: |ρ| > 0.6, positive projection.
    let by_key: std::collections::HashMap<(usize, usize), f64> = rows
        .iter()
        .map(|r| ((r.frame, r.sub), r.hb_exc_rms))
        .collect();
    let gcb = |idx: u8| {
        f64::from(oxideav_speex::reconstruct_hb_exc_gain(
            oxideav_speex::HbExcitationGainIndex::FourBit(idx),
        ))
    };
    let mut y = Vec::new();
    let mut x_gc = Vec::new();
    let mut x_lb = Vec::new();
    let mut x_prev = Vec::new();
    let mut lsp_class = Vec::new();
    for r in &rows {
        let Some((proj, rho)) = r.proj_rho else {
            continue;
        };
        if rho <= 0.6 || proj <= 0.0 {
            continue;
        }
        let prev_key = if r.sub > 0 {
            (r.frame, r.sub - 1)
        } else if r.frame > 0 {
            (r.frame - 1, 3)
        } else {
            continue;
        };
        let Some(&prev_rms) = by_key.get(&prev_key) else {
            continue;
        };
        y.push(proj.log10());
        x_gc.push(gcb(r.gc_index).log10());
        x_lb.push(r.lb_frame_rms.log10());
        x_prev.push(prev_rms.log10());
        lsp_class.push(r.lsp);
    }
    let n = y.len();
    println!("regression rows: {n}");
    assert!(n >= 280, "regression rows {n} < 280");

    // (1) r440 fixed-2 reading — residual against the fixed prediction.
    let fixed2: Vec<f64> = (0..n)
        .map(|i| y[i] - 2.0 * x_gc[i] - 2.0 * x_lb[i])
        .collect();
    let mean_f2 = fixed2.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - mean_y) * (v - mean_y)).sum();
    let ss_f2: f64 = fixed2.iter().map(|v| (v - mean_f2) * (v - mean_f2)).sum();
    let r2_fixed2 = 1.0 - ss_f2 / ss_tot;
    println!(
        "fixed-2 (gc·lb)²: R² {r2_fixed2:.3} rms {:.2} dB",
        20.0 * (ss_f2 / n as f64).sqrt()
    );
    // Measured r446: 0.746 (the WB fixture's own rows: 0.79).
    assert!(
        r2_fixed2 > 0.70,
        "fixed-2 law no longer tracks the second fixture (R² {r2_fixed2:.3})"
    );

    // (2) Envelope dependence: the share of the fixed-2 residual
    // explained by the transmitted LSP class (between-class variance).
    let mut classes: std::collections::HashMap<(u8, u8), (f64, usize)> =
        std::collections::HashMap::new();
    for i in 0..n {
        let e = classes.entry(lsp_class[i]).or_insert((0.0, 0));
        e.0 += fixed2[i];
        e.1 += 1;
    }
    let mut ss_within = 0.0f64;
    for i in 0..n {
        let (sum, cnt) = classes[&lsp_class[i]];
        let d = fixed2[i] - sum / cnt as f64;
        ss_within += d * d;
    }
    let between_share = 1.0 - ss_within / ss_f2;
    println!(
        "LSP-class share of the fixed-2 residual: {:.1}% across {} classes",
        100.0 * between_share,
        classes.len()
    );
    // Measured r446: ≈56 % over 17 classes — the envelope term the
    // constant-LSP WB fixture could not show. This is the pinned new
    // docs-gap evidence.
    assert!(classes.len() >= 10, "LSP variety collapsed");
    assert!(
        between_share > 0.35,
        "envelope dependence vanished ({:.1}% — was ≈56%)",
        100.0 * between_share
    );

    // (3) State term: gc + previous sub-frame's excitation RMS beats
    // gc + same-frame low-band level (free fits, both two-term).
    let xg: &[f64] = &x_gc;
    let (r2_lb, rms_lb, c_lb) = fit(&y, &[xg, &x_lb]);
    let (r2_prev, rms_prev, c_prev) = fit(&y, &[xg, &x_prev]);
    println!(
        "free gc+lb: R² {r2_lb:.3} rms {rms_lb:.2} dB exps [{:.2}, {:.2}] | \
         free gc+prev-exc: R² {r2_prev:.3} rms {rms_prev:.2} dB exps [{:.2}, {:.2}]",
        c_lb[0], c_lb[1], c_prev[0], c_prev[1]
    );
    // Measured r446: 0.815 vs 0.900.
    assert!(
        r2_prev > r2_lb + 0.04,
        "state term no longer dominates (prev {r2_prev:.3} vs lb {r2_lb:.3})"
    );
}
