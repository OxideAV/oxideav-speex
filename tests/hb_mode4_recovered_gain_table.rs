//! **QMF-recovered high-band excitation — crate-machinery replication**
//! (round r446, `docs/audio/speex/provenance/08-qmf-recovered-hb-excitation.md`).
//!
//! Docs round 8 recovered the isolated 4–8 kHz sub-band of the staged
//! `hb-mode4-wb-q10` oracle **from staged bytes alone** — QMF-analysing
//! `expected.pcm` with the staged 64-tap prototype, inverse-filtering the
//! recovered high band through a global order-8 LPC fit (legitimate: the
//! fixture transmits the same high-band LSP pair `(14, 50)` in all 76
//! frames), and projecting each sub-frame's excitation estimate onto the
//! innovation rebuilt from `frame-trace.txt` per the
//! `hb-innovation-binding.md` §1 binding. Its measured results are staged
//! as `tables/hb-mode4-recovered-gain.csv` (299 rows).
//!
//! This suite re-runs that whole measurement **through the crate's own
//! machinery** — [`QmfAnalysis`] as the analysis instrument,
//! [`decode_hb_subframe_mode4_f32`] as the innovation rebuild (so the
//! 80-bit group packing, the leading sign bit and the 0.4 stage-2 weight
//! are all exercised), and [`reconstruct_hb_exc_gain`] for the
//! gain-correction column — and pins:
//!
//! 1. The provenance/08 headline numbers reproduce: mean |ρ| ≈ 0.86,
//!    ≥ ~90 % of sub-frames above 0.8, sign positive on ≥ ~96 % — the
//!    submode-4 binding confirmed oracle-free by crate code.
//! 2. The alignment is a *unique* correlation peak (a one-sub-frame
//!    look-ahead in the recovered-band timeline), factor-3 margin over
//!    every other delay — the provenance/08 alignment-sweep control.
//! 3. Per-row agreement with the staged CSV (docs checkout only,
//!    skip-if-absent): ρ, `gc_recon`, `lb_frame_rms`, `hb_exc_rms`.
//! 4. The staged table's gain *direction* — `log g` on the doc's
//!    fixed-2 exponents of `gc_recon` and `lb_frame_rms` reaches the
//!    doc's R² ≈ 0.79 / rms ≈ 8.9 dB. (r450: the crate's decode law is
//!    now the crafted-probe crossover-anchored linear law — see
//!    `HB_GC_CROSSOVER_SCALE` — and this regression stands as a
//!    replication of the *staged measurement*, whose loose fit the
//!    probes explain as natural-speech co-variation.)

use oxideav_speex::{
    decode_hb_subframe_mode4_f32, reconstruct_hb_exc_gain, HbExcitationGainIndex, QmfAnalysis,
};

/// Provenance/08's fixed exponents for its gain-direction regression
/// (the doc's reading, replicated here as a measurement; not the
/// crate's decode law as of r450).
const DOC_EXP_GC: f64 = 2.0;
const DOC_EXP_LB: f64 = 2.0;

const EXPECTED: &[u8] = include_bytes!("fixtures/hb-mode4-wb-q10/expected.pcm");
const TRACE: &str = include_str!("fixtures/hb-mode4-wb-q10/frame-trace.txt");

const HB_ORDER: usize = 8;
const SUBFRAME: usize = 40;
const FRAME_HB: usize = 160;

/// One trace sub-frame: the 4-bit gain-correction index and the ten
/// `(sign, index)` pairs (stage 1 then stage 2) in bitstream order.
#[derive(Clone, Copy)]
struct TraceSubframe {
    gc_index: u8,
    pairs: [(bool, u8); 10],
}

impl TraceSubframe {
    /// Pack the ten 8-bit `[sign][7-bit index]` groups MSB-first into
    /// the 80-bit on-wire excitation-VQ field — the exact layout
    /// [`decode_hb_subframe_mode4_f32`] consumes.
    fn vq_index(&self) -> u128 {
        let mut vq = 0u128;
        for &(sign, idx) in &self.pairs {
            vq = (vq << 8) | u128::from((u8::from(sign) << 7) | (idx & 0x7F));
        }
        vq
    }

    /// The innovation is wholly zero iff every group addresses the
    /// unique all-zero `sv8-128` row 43 (provenance/08: the
    /// "no excitation here" symbol).
    fn is_all_zero(&self) -> bool {
        self.pairs.iter().all(|&(_, idx)| idx == 43)
    }
}

/// Parse the staged `frame-trace.txt`: one line per frame,
/// `frame N nb=7 | hb1 m=4 lsp=(i1,i2) [gc= G ±idx ×10] ×4`.
fn parse_trace(trace: &str) -> Vec<[TraceSubframe; 4]> {
    let mut frames = Vec::new();
    for line in trace.lines() {
        let line = line.trim();
        if !line.starts_with("frame") {
            continue;
        }
        let mut subframes = Vec::new();
        let mut rest = line;
        while let Some(open) = rest.find('[') {
            let close = rest[open..]
                .find(']')
                .expect("unterminated sub-frame block")
                + open;
            let block = &rest[open + 1..close];
            rest = &rest[close + 1..];
            let block = block
                .strip_prefix("gc=")
                .expect("sub-frame block starts with gc=");
            // Tokens: the gain correction, then sign/index pairs. Signs
            // are printed as standalone `+`/`-` prefixes that may or may
            // not be space-separated from the index.
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
            let gc_index: u8 = tokens[0].parse().expect("gc index");
            let mut pairs = [(false, 0u8); 10];
            let mut t = 1usize;
            for pair in pairs.iter_mut() {
                let sign = tokens[t] == "-";
                let idx: u8 = tokens[t + 1].parse().expect("codebook index");
                *pair = (sign, idx);
                t += 2;
            }
            assert_eq!(t, tokens.len(), "trailing tokens in sub-frame block");
            subframes.push(TraceSubframe { gc_index, pairs });
        }
        assert_eq!(subframes.len(), 4, "four sub-frames per frame");
        frames.push([subframes[0], subframes[1], subframes[2], subframes[3]]);
    }
    frames
}

/// QMF-split a 16 kHz signal into its two 8 kHz sub-bands with the
/// crate's analysis bank (whole 320-sample frames) — the provenance/08
/// instrument, isolation-pinned by `tests/qmf_band_isolation.rs`. One
/// zero-input frame is flushed through at the end so the FIR tail of
/// the final frame is emitted — the staged table's whole-signal
/// convolution keeps that tail, and its last frame's rows read it.
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

/// Zero-padded window read — the staged table's convention: samples
/// before the signal start (the frame-0 look-ahead region) and past the
/// convolution tail are zero, and windowed RMS values normalise by the
/// full window length.
fn win(sig: &[f64], start: i64, len: usize) -> Vec<f64> {
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

/// Global order-8 autocorrelation LPC over the whole recovered high band
/// (the provenance/08 fit — legitimate because the fixture's high-band
/// LSP pair is constant), followed by the monic inverse filter
/// `e[n] = x[n] − Σ aₖ·x[n−k]` in the crate's `A(z) = 1 − Σ aₖ z⁻ᵏ`
/// convention.
fn inverse_filter_global_lpc(x: &[f64]) -> Vec<f64> {
    let mut r = [0.0f64; HB_ORDER + 1];
    for (m, slot) in r.iter_mut().enumerate() {
        *slot = x[m..].iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
    }
    r[0] *= 1.0001; // white-noise floor, the manual's worked value
                    // Levinson-Durbin, order 8 (prediction coefficients).
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
    subframe: usize,
    gc_recon: f64,
    proj_gain: f64,
    rho: f64,
    hb_exc_rms: f64,
    lb_frame_rms: f64,
}

/// Run the full provenance/08 measurement through crate machinery at a
/// given sub-frame window offset (in recovered-band samples).
fn measure(offset: i64) -> Vec<Row> {
    let pcm: Vec<f64> = EXPECTED
        .chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (lb, hb) = qmf_split(&pcm);
    let exc = inverse_filter_global_lpc(&hb);
    let frames = parse_trace(TRACE);

    let mut rows = Vec::new();
    for (f, subs) in frames.iter().enumerate() {
        // The staged table's per-frame low-band window sits at the same
        // aligned offset as the excitation windows: [f·160 + offset,
        // +160), zero-padded, normalised by the full window length.
        let lseg = win(&lb, f as i64 * FRAME_HB as i64 + offset, FRAME_HB);
        let lb_frame_rms = (lseg.iter().map(|&v| v * v).sum::<f64>() / FRAME_HB as f64).sqrt();
        for (s, sub) in subs.iter().enumerate() {
            if sub.is_all_zero() {
                continue;
            }
            let start = f as i64 * FRAME_HB as i64 + s as i64 * SUBFRAME as i64 + offset;
            // The staged table keeps a sub-frame only when its whole
            // 40-sample window fits inside the convolution's real
            // extent — `(pcm_len + QMF_ORDER − 1) / 2` recovered-band
            // samples (its final frame contributes one row).
            let real_len = ((pcm.len() + 63) / 2) as i64;
            if start < 0 || start + SUBFRAME as i64 > real_len {
                continue;
            }
            let e = win(&exc, start, SUBFRAME);
            let e = &e[..];
            let v: Vec<f64> = decode_hb_subframe_mode4_f32(sub.vq_index())
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            let ev: f64 = e.iter().zip(&v).map(|(&a, &b)| a * b).sum();
            let ee: f64 = e.iter().map(|&a| a * a).sum();
            let vv: f64 = v.iter().map(|&b| b * b).sum();
            rows.push(Row {
                frame: f,
                subframe: s,
                gc_recon: f64::from(reconstruct_hb_exc_gain(HbExcitationGainIndex::FourBit(
                    sub.gc_index,
                ))),
                proj_gain: ev / vv,
                rho: ev / (ee.sqrt() * vv.sqrt()).max(1e-30),
                hb_exc_rms: (ee / SUBFRAME as f64).sqrt(),
                lb_frame_rms,
            });
        }
    }
    rows
}

fn mean_abs_rho(rows: &[Row]) -> f64 {
    rows.iter().map(|r| r.rho.abs()).sum::<f64>() / rows.len() as f64
}

/// **Provenance/08 headline replication, oracle-free and standalone**
/// (mirrored fixture bytes only). The innovation rebuilt by the crate's
/// own mode-4 decode correlates with the QMF-recovered excitation at the
/// staged doc's levels, with the sign positive — the §1 binding + the §4
/// polarity confirmed by crate code end-to-end.
#[test]
fn recovered_excitation_confirms_mode4_binding() {
    // Alignment sweep first (the provenance/08 control): the peak must
    // be unique with a decisive margin.
    let mut best = (0i64, 0.0f64);
    let mut second = 0.0f64;
    for offset in -260i64..=59 {
        let rows = measure(offset);
        if rows.len() < 200 {
            continue;
        }
        let m = mean_abs_rho(&rows);
        if m > best.1 {
            second = best.1;
            best = (offset, m);
        } else if m > second {
            second = m;
        }
    }
    println!(
        "alignment sweep: peak mean|rho| {:.4} at offset {}, runner-up {:.4}",
        best.1, best.0, second
    );
    assert_eq!(
        best.0, -40,
        "correlation peak moved off the staged −40 sub-frame look-ahead"
    );
    assert!(
        best.1 > 3.0 * second,
        "alignment peak {:.4} lacks the factor-3 margin over runner-up {:.4}",
        best.1,
        second
    );

    let rows = measure(-40);
    let n = rows.len();
    let mean = mean_abs_rho(&rows);
    let mut sorted: Vec<f64> = rows.iter().map(|r| r.rho.abs()).collect();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[n / 2];
    let above08 = rows.iter().filter(|r| r.rho.abs() > 0.8).count() as f64 / n as f64;
    let positive = rows.iter().filter(|r| r.rho > 0.0).count() as f64 / n as f64;
    println!(
        "rows {n}: mean|rho| {mean:.4} median {median:.4} \
         >0.8 {:.1}% positive {:.1}%",
        100.0 * above08,
        100.0 * positive
    );
    // Staged doc: 299 rows, mean 0.8617, median 0.8963, 90.0 % > 0.8,
    // 96.7 % positive. Floors leave room for the LPC-fit and QMF-edge
    // differences between the crate pipeline and the doc's.
    assert_eq!(n, 299, "row count diverged from the staged table");
    assert!(mean > 0.84, "mean |rho| {mean:.4} ≤ 0.84");
    assert!(median > 0.87, "median |rho| {median:.4} ≤ 0.87");
    assert!(above08 > 0.87, "share above 0.8 {above08:.3} ≤ 0.87");
    assert!(positive > 0.95, "positive-sign share {positive:.3} ≤ 0.95");
}

/// **Staged-table per-row cross-check** (docs checkout only): the crate
/// pipeline reproduces `tables/hb-mode4-recovered-gain.csv` row for row.
#[test]
fn staged_recovered_gain_table_reproduced() {
    let path = std::env::var("OXIDEAV_DOCS_SPEEX_TABLES")
        .unwrap_or_else(|_| "../../docs/audio/speex/tables".to_string());
    let csv_path = format!("{path}/hb-mode4-recovered-gain.csv");
    let Ok(csv) = std::fs::read_to_string(&csv_path) else {
        eprintln!("SKIP: staged table not present at {csv_path}");
        return;
    };

    let rows = measure(-40);
    let mut staged = 0usize;
    let (mut drho, mut dlb, mut dhb, mut dproj) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for line in csv.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 9 {
            continue;
        }
        let frame: usize = f[0].parse().unwrap();
        let subframe: usize = f[1].parse().unwrap();
        let want_gc: f64 = f[3].parse().unwrap();
        let want_proj: f64 = f[4].parse().unwrap();
        let want_rho: f64 = f[5].parse().unwrap();
        let want_hb_rms: f64 = f[6].parse().unwrap();
        let want_lb_rms: f64 = f[7].parse().unwrap();
        let got = rows
            .iter()
            .find(|r| r.frame == frame && r.subframe == subframe)
            .unwrap_or_else(|| panic!("staged row ({frame},{subframe}) missing from crate rows"));
        staged += 1;
        // gc_recon is table arithmetic — exact to print precision.
        assert!(
            (got.gc_recon - want_gc).abs() < 5e-4,
            "({frame},{subframe}) gc_recon {} vs staged {want_gc}",
            got.gc_recon
        );
        drho.push((got.rho - want_rho).abs());
        dlb.push(((got.lb_frame_rms / want_lb_rms).ln()).abs());
        dhb.push(((got.hb_exc_rms / want_hb_rms).ln()).abs());
        dproj.push(((got.proj_gain / want_proj).abs().ln()).abs());
        // Projection sign: pinned in the doc's own regression regime
        // (|ρ| > 0.6); below it the projection is noise-dominated and
        // its sign is not meaningful.
        if want_rho.abs() > 0.6 {
            assert_eq!(
                got.proj_gain.is_sign_positive(),
                want_proj.is_sign_positive(),
                "({frame},{subframe}) projection sign flipped"
            );
        }
    }
    let stats = |d: &mut Vec<f64>| -> (f64, f64, f64) {
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        d.sort_by(f64::total_cmp);
        (mean, d[(d.len() * 9) / 10], d[d.len() - 1])
    };
    let (rho_mean, rho_p90, rho_max) = stats(&mut drho);
    let (lb_mean, _, lb_max) = stats(&mut dlb);
    let (hb_mean, hb_p90, hb_max) = stats(&mut dhb);
    let (proj_mean, proj_p90, proj_max) = stats(&mut dproj);
    println!(
        "{staged} staged rows: |Δrho| mean {rho_mean:.4} p90 {rho_p90:.4} max {rho_max:.4} | \
         |Δln lb_rms| mean {lb_mean:.6} max {lb_max:.6} | \
         |Δln hb_rms| mean {hb_mean:.4} p90 {hb_p90:.4} max {hb_max:.4} | \
         |Δln proj| mean {proj_mean:.4} p90 {proj_p90:.4} max {proj_max:.4}"
    );
    assert_eq!(staged, 299, "staged table row count");
    // The low-band level column is filter-free instrument output — the
    // crate's analysis bank reproduces it essentially exactly (the
    // measured max is < 1e-5 log). The excitation-derived columns (ρ,
    // RMS, projection) sit behind the order-8 LPC fit, whose fine
    // details the doc does not pin; they agree tightly in the mass of
    // the distribution, with a small tail on sparse low-gain (gc = 0)
    // sub-frames where a near-zero excitation makes both pipelines
    // noise-dominated (those rows sit below the doc's own |ρ| > 0.6
    // regression cut).
    assert!(
        lb_max < 1e-3,
        "lb_frame_rms is instrument-exact ({lb_max:.2e})"
    );
    assert!(rho_mean < 0.04, "mean |Δrho| {rho_mean:.4} ≥ 0.04");
    assert!(rho_p90 < 0.08, "p90 |Δrho| {rho_p90:.4} ≥ 0.08");
    assert!(hb_mean < 0.09, "mean |Δln hb_rms| {hb_mean:.4} ≥ 0.09");
    assert!(hb_p90 < 0.15, "p90 |Δln hb_rms| {hb_p90:.4} ≥ 0.15");
    assert!(proj_mean < 0.05, "mean |Δln proj| {proj_mean:.4} ≥ 0.05");
    assert!(proj_p90 < 0.10, "p90 |Δln proj| {proj_p90:.4} ≥ 0.10");
}

/// **Gain-law direction gate** on the staged table (docs checkout only):
/// the doc's fixed-exponent reading — `g ∝ (gc_recon · lb_frame_rms)²`,
/// [`DOC_EXP_GC`] = [`DOC_EXP_LB`] = 2 — reaches the
/// doc's regression quality (R² ≈ 0.79, rms ≈ 8.9 dB with both exponents
/// fixed at 2) on the staged rows, and the transmitted correction alone
/// explains nothing (R² ≈ 0.005). The absolute intercept stays free —
/// provenance/08 deliberately asserts no closed-form law.
#[test]
fn gain_law_direction_matches_staged_regression() {
    let path = std::env::var("OXIDEAV_DOCS_SPEEX_TABLES")
        .unwrap_or_else(|_| "../../docs/audio/speex/tables".to_string());
    let csv_path = format!("{path}/hb-mode4-recovered-gain.csv");
    let Ok(csv) = std::fs::read_to_string(&csv_path) else {
        eprintln!("SKIP: staged table not present at {csv_path}");
        return;
    };
    // (log10 gc_recon, log10 lb_frame_rms, log10 |proj_gain|), rows with
    // |rho| > 0.6 (the doc's exclusion, 279 of 299).
    let mut pts: Vec<(f64, f64, f64)> = Vec::new();
    for line in csv.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 9 {
            continue;
        }
        let gc: f64 = f[3].parse().unwrap();
        let proj: f64 = f[4].parse().unwrap();
        let rho: f64 = f[5].parse().unwrap();
        let lb: f64 = f[7].parse().unwrap();
        if rho.abs() > 0.6 && proj.abs() > 0.0 && gc > 0.0 && lb > 0.0 {
            pts.push((gc.log10(), lb.log10(), proj.abs().log10()));
        }
    }
    println!("{} regression rows (doc: 279)", pts.len());
    assert!((270..=290).contains(&pts.len()), "row filter diverged");

    // dB-domain helper: rms residual of y − pred over the best (free)
    // intercept, plus R² against the y variance.
    let fit = |pred: &dyn Fn(&(f64, f64, f64)) -> f64| -> (f64, f64) {
        let n = pts.len() as f64;
        let mean_res = pts.iter().map(|p| p.2 - pred(p)).sum::<f64>() / n;
        let ss_res: f64 = pts.iter().map(|p| (p.2 - pred(p) - mean_res).powi(2)).sum();
        let mean_y = pts.iter().map(|p| p.2).sum::<f64>() / n;
        let ss_tot: f64 = pts.iter().map(|p| (p.2 - mean_y).powi(2)).sum();
        let rms_db = 20.0 * (ss_res / n).sqrt();
        (1.0 - ss_res / ss_tot, rms_db)
    };

    let e_gc = DOC_EXP_GC;
    let e_lb = DOC_EXP_LB;
    let (r2_crate, rms_crate) = fit(&|p| e_gc * p.0 + e_lb * p.1);
    let (r2_gc_only, _) = {
        // gc-only: best single-variable linear fit on log gc.
        let n = pts.len() as f64;
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
        let my = pts.iter().map(|p| p.2).sum::<f64>() / n;
        let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.2 - my)).sum();
        let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
        let b = sxy / sxx;
        (fit(&move |p: &(f64, f64, f64)| b * p.0).0, b)
    };
    let (r2_lin, rms_lin) = fit(&|p| p.0 + p.1);
    println!(
        "fixed-2 law: R² {r2_crate:.3} rms {rms_crate:.2} dB | gc-only best-slope R² {r2_gc_only:.3} \
         | both-at-1 R² {r2_lin:.3} rms {rms_lin:.2} dB"
    );
    // Doc: fixed-2 costs almost nothing vs the free fit (8.89 dB, R²≈0.79);
    // gc-only explains R² = 0.005; both-at-1 costs a lot (12.86 dB).
    assert!(
        r2_crate > 0.75,
        "fixed-2 law R² {r2_crate:.3} ≤ 0.75 (doc ≈ 0.79)"
    );
    assert!(
        rms_crate < 9.5,
        "fixed-2 law rms {rms_crate:.2} dB ≥ 9.5 (doc ≈ 8.89)"
    );
    assert!(
        r2_gc_only < 0.10,
        "gc-only R² {r2_gc_only:.3} ≥ 0.10 (doc: 0.005 — the correction is not the gain)"
    );
    assert!(
        rms_lin > rms_crate + 2.0,
        "both-at-1 rms {rms_lin:.2} dB not clearly worse than fixed-2 {rms_crate:.2} dB"
    );
}
