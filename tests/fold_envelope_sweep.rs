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
    hb_crossover_response, hb_subframe_lpc_set_with_base, reconstruct_hb_lsp_q10, HbLspStages,
    HbSubFrameLsp,
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
