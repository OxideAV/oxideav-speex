//! **Folded-band scale-law validation** against the staged envelope
//! sweep (campaign B).
//!
//! `docs/audio/speex/hb-folded-gain.md` §7.3/§7.5 pin the folded
//! high-band scale as `s = C·|Â(π)|` (linear, **kneeless**), and the
//! staged `fixtures/fold-envelope-sweep/sweep-measurements.txt`
//! (mirrored here) records, for each forced high-band LSP stage-1 index
//! `i1`, the reference decoder's measured per-band scale ratio
//! (`band_mean = log10(scale ratio)`) relative to the `i1 = 0` setting.
//!
//! Under the law, `band_mean(i1) = log10(|Â_i1(π)| / |Â_0(π)|)`. This
//! gate reconstructs the crate's own `|Â(π)|` for those same forced
//! settings (LSP stage-1 index `i1`, stage-2 index 20) and asserts it
//! reproduces the staged band-means — a decode-free confirmation that
//! the crate's crossover response *is* the reference's normalising
//! response in the shallow-to-mid envelope range.
//!
//! ## Finding (campaign B)
//!
//! The crate's `|Â(π)|` matches the staged sweep to ~0.5 dB for
//! `i1 ∈ {0, 8, 20}` and diverges (up to ~3 dB) for the near-degenerate
//! deep envelopes `i1 ∈ {33, 49, 63}` (`|Â(π)| ≲ 0.05`) — exactly the
//! regime `hb-folded-gain.md` §7.3 flags as unpinned ("suspected
//! reference-side LSP margin enforcement"). This validates the kneeless
//! scale law where it is pinned and localises the residual to the
//! flagged deep-envelope regime.

use oxideav_speex::{
    hb_crossover_response, hb_crossover_response_bwexp, hb_crossover_response_from_lsp,
    hb_lsp_with_base_q10, hb_subframe_lpc_set_with_base, reconstruct_hb_lsp_q10, HbLspStages,
    HbSubFrameLsp, HB_FOLD_ENVELOPE_COMPRESSION_GAMMA,
};

const MEASUREMENTS: &str = include_str!("fixtures/fold-envelope-sweep/sweep-measurements.txt");

/// `|Â(π)|` for the forced setting (stage-1 index `i1`, stage-2 index 20)
/// at the last (fully interpolated on a first frame) sub-frame.
fn a_pi(i1: u8) -> f64 {
    let stages = HbLspStages {
        stage1: i1,
        stage2: 20,
    };
    let curr = reconstruct_hb_lsp_q10(stages).expect("valid stage indices");
    let sub = HbSubFrameLsp::first_frame(&curr);
    let sets = hb_subframe_lpc_set_with_base(&sub);
    hb_crossover_response(&sets[3])
}

/// Parse the `wb-inner` rows of the staged sweep: `layer i1 band_mean …`.
fn staged_wb_inner() -> Vec<(u8, f64)> {
    let mut rows = Vec::new();
    for line in MEASUREMENTS.lines() {
        let l = line.trim();
        if l.starts_with('#') || l.is_empty() {
            continue;
        }
        let t: Vec<&str> = l.split_whitespace().collect();
        if t.len() < 3 || t[0] != "wb-inner" {
            continue;
        }
        if let (Ok(i1), Ok(bm)) = (t[1].parse::<u8>(), t[2].parse::<f64>()) {
            rows.push((i1, bm));
        }
    }
    rows
}

/// The forced setting's reconstructed high-band LSP set in radians
/// (base-added Q10, the pre-margin reconstruction domain).
fn lsp_radians(i1: u8) -> [f64; 8] {
    let stages = HbLspStages {
        stage1: i1,
        stage2: 20,
    };
    let delta = reconstruct_hb_lsp_q10(stages).expect("valid stage indices");
    let q10 = hb_lsp_with_base_q10(&delta);
    let mut out = [0.0f64; 8];
    for (o, &v) in out.iter_mut().zip(q10.iter()) {
        *o = f64::from(v) / 1024.0;
    }
    out
}

/// **§7.6 parity pin** (r440): the closed-form odd-parity LSP product
/// `Π 4·cos²(ωᵢ/2)` equals the crate's direct polynomial evaluation of
/// `|Â(π)|` through LSP→LPC, for every swept envelope — the discrete
/// parity choice `hb-folded-gain.md` §7.6 validates (the even class is
/// wrong by 3–10 dB) realised on the crate's own reconstruction.
#[test]
fn odd_parity_lsp_product_equals_direct_evaluation() {
    for i1 in [0u8, 8, 20, 33, 49, 63] {
        let direct = a_pi(i1);
        let product = hb_crossover_response_from_lsp(&lsp_radians(i1));
        let dlog = (direct / product).log10().abs();
        println!("i1={i1:2}: direct {direct:.5} odd-product {product:.5} |Δlog10| {dlog:.4}");
        assert!(
            dlog < 0.02,
            "i1={i1}: odd-parity product departs from direct |Â(π)| by {dlog:.4} log10"
        );
    }
}

/// **Staged-table cross-check** (r440): the crate's `|Â(π)|` chain
/// reproduces the `dlog10_Api_odd_parity` and
/// `dlog10_Api_odd_bwexp_0p944` columns of the newly staged
/// `tables/hb-fold-envelope-vs-transmitted.csv` — the docs
/// collaborator's independent computation of the same quantities from
/// the same staged codebooks. Skips (loudly) when the docs tree is not
/// present (standalone CI checkout).
#[test]
fn crossover_response_reproduces_staged_csv_columns() {
    let path = std::env::var("OXIDEAV_DOCS_SPEEX_TABLES")
        .unwrap_or_else(|_| "../../docs/audio/speex/tables".to_string());
    let csv_path = format!("{path}/hb-fold-envelope-vs-transmitted.csv");
    let Ok(csv) = std::fs::read_to_string(&csv_path) else {
        eprintln!("SKIP: staged table not present at {csv_path}");
        return;
    };
    let a0 = a_pi(0);
    let lpc0 = lpc_for(0);
    let g0 = hb_crossover_response_bwexp(&lpc0, HB_FOLD_ENVELOPE_COMPRESSION_GAMMA);
    let mut checked = 0usize;
    for line in csv.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 4 {
            continue;
        }
        let i1: u8 = f[0].parse().unwrap();
        let want_odd: f64 = f[1].parse().unwrap();
        let want_bwexp: f64 = f[3].parse().unwrap();
        let got_odd = (a_pi(i1) / a0).log10();
        let got_bwexp =
            (hb_crossover_response_bwexp(&lpc_for(i1), HB_FOLD_ENVELOPE_COMPRESSION_GAMMA) / g0)
                .log10();
        println!(
            "i1={i1:2}: odd {got_odd:+.4} (staged {want_odd:+.4}) | γ-expanded {got_bwexp:+.4} (staged {want_bwexp:+.4})"
        );
        assert!(
            (got_odd - want_odd).abs() < 0.02,
            "i1={i1}: odd-parity column mismatch {got_odd:.4} vs {want_odd:.4}"
        );
        assert!(
            (got_bwexp - want_bwexp).abs() < 0.02,
            "i1={i1}: γ-expanded column mismatch {got_bwexp:.4} vs {want_bwexp:.4}"
        );
        checked += 1;
    }
    assert!(checked >= 6, "staged CSV should carry the six settings");
}

/// The forced setting's last-sub-frame order-8 LPC set.
fn lpc_for(i1: u8) -> [f64; 8] {
    let stages = HbLspStages {
        stage1: i1,
        stage2: 20,
    };
    let curr = reconstruct_hb_lsp_q10(stages).expect("valid stage indices");
    let sub = HbSubFrameLsp::first_frame(&curr);
    let sets = hb_subframe_lpc_set_with_base(&sub);
    sets[3]
}

#[test]
fn crossover_response_reproduces_staged_band_means() {
    let rows = staged_wb_inner();
    assert!(rows.len() >= 6, "sweep carries the wb-inner rows");
    let a0 = a_pi(0);
    let mut max_shallow = 0.0f64;
    for (i1, band_mean) in rows {
        let pred = (a_pi(i1) / a0).log10();
        let diff = (pred - band_mean).abs();
        println!("i1={i1:2} pred={pred:+.4} staged={band_mean:+.4} diff={diff:.4}");
        // Shallow-to-mid envelopes: the crate's |Â(π)| is the reference's
        // normalising response to ~0.5 dB (0.05 log10).
        if i1 <= 20 {
            max_shallow = max_shallow.max(diff);
        }
    }
    assert!(
        max_shallow < 0.05,
        "shallow/mid |Â(π)| mismatch {max_shallow:.4} log10 > 0.05 (≈0.5 dB)"
    );
}
