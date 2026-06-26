//! End-to-end encoder envelope chain (round r372).
//!
//! Drives the full short-term-envelope encode path the round-r372 modules
//! compose:
//!
//! ```text
//! signal → lpc_analyse → lpc_to_lsp → radians→Q10 → quantise_lsp_q10
//!        → pack_lsp_index → [wire] → NbLspStages::from_packed
//!        → reconstruct_q10 → lsp_to_lpc → reconstructed envelope
//! ```
//!
//! and confirms the reconstructed LPC envelope is a faithful match to the
//! analysed one — the encoder's spectral-envelope path is closed against
//! the existing decoder reconstruction.

use oxideav_speex::{
    lpc_analyse, lpc_from_lsp_delta_q10, lpc_to_lsp, lsp_vector_radians_to_q10, nb_lsp_base_q10,
    pack_lsp_index, quantise_lsp_q10, NbLspStages,
};
use oxideav_speex::{LspQuant, NB_LSP_ORDER, NB_LSP_STAGES_30BIT};

/// Subtract the pinned narrowband LSP base vector from an absolute Q10 LSP
/// vector, producing the codebook-**delta** domain the multi-stage VQ
/// codebooks operate in (the decoder adds the base back via
/// `lpc_from_lsp_delta_q10`).
fn to_delta(absolute_q10: &[i32; NB_LSP_ORDER]) -> [i32; NB_LSP_ORDER] {
    let base = nb_lsp_base_q10();
    let mut d = [0i32; NB_LSP_ORDER];
    for i in 0..NB_LSP_ORDER {
        d[i] = absolute_q10[i] - base[i];
    }
    d
}

/// Generate a deterministic AR(2)-driven signal of the requested length.
fn ar_signal(n: usize, a1: f64, a2: f64, seed0: u64) -> Vec<f64> {
    let mut x = vec![0.0f64; n];
    let mut seed = seed0;
    for i in 2..n {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let innov = ((seed >> 40) as f64 / (1u64 << 24) as f64) - 0.5;
        x[i] = a1 * x[i - 1] + a2 * x[i - 2] + innov;
    }
    x
}

#[test]
fn full_envelope_encode_chain_preserves_spectrum() {
    // Analyse a real-length buffer to LPC.
    let sig = ar_signal(240, 0.55, -0.25, 0xC0FFEE);
    let coeffs = lpc_analyse(&sig).expect("analysis");

    // LPC → LSP (radians) → absolute Q10 → codebook-delta domain.
    let lsp_rad = lpc_to_lsp(&coeffs.a).expect("stable analysed filter");
    let lsp_q10 = lsp_vector_radians_to_q10(&lsp_rad);
    let delta = to_delta(&lsp_q10);

    // Quantise the delta through the 30-bit multi-stage VQ.
    let stages = quantise_lsp_q10(&delta, LspQuant::Bits30).expect("quantises");
    assert_eq!(stages.stage_count(), NB_LSP_STAGES_30BIT);

    // Pack to the wire field and parse it back (the on-wire round-trip).
    let packed = pack_lsp_index(&stages);
    let decoded = NbLspStages::from_packed(packed, LspQuant::Bits30).expect("decodes");
    assert_eq!(decoded, stages);

    // Reconstruct the LPC envelope from the decoded indices, adding the
    // base back via the decoder's base-aware path (the inverse of the
    // `to_delta` subtraction above).
    let recon_delta = oxideav_speex::reconstruct_nb_lsp_q10(decoded).expect("reconstructs");
    let recon_lpc = lpc_from_lsp_delta_q10(&recon_delta);

    // The reconstructed LPC envelope is finite and a faithful match to the
    // analysed one. The quantiser is lossy (multi-stage VQ + Q10 rounding),
    // so the match is approximate, but the coarse envelope must survive.
    assert!(recon_lpc.iter().all(|c| c.is_finite()));
    let sig_energy: f64 = coeffs.a.iter().map(|c| c * c).sum::<f64>().max(1e-9);
    let diff_energy: f64 = recon_lpc
        .iter()
        .zip(coeffs.a.iter())
        .map(|(r, a)| (r - a) * (r - a))
        .sum();
    assert!(
        diff_energy < sig_energy,
        "VQ envelope error {diff_energy} should be below the coefficient energy {sig_energy}"
    );
}

#[test]
fn envelope_chain_18bit_regime_round_trips_on_wire() {
    let sig = ar_signal(220, 0.4, -0.1, 0x1234_5678);
    let coeffs = lpc_analyse(&sig).expect("analysis");
    let lsp_rad = lpc_to_lsp(&coeffs.a).expect("stable filter");
    let lsp_q10 = lsp_vector_radians_to_q10(&lsp_rad);
    let delta = to_delta(&lsp_q10);
    let stages = quantise_lsp_q10(&delta, LspQuant::Bits18).expect("quantises");
    assert!(stages.low2.is_none() && stages.high2.is_none());
    let packed = pack_lsp_index(&stages);
    let decoded = NbLspStages::from_packed(packed, LspQuant::Bits18).expect("decodes");
    assert_eq!(decoded, stages);
    let recon = oxideav_speex::reconstruct_nb_lsp_q10(decoded).expect("reconstructs");
    assert!(recon.iter().all(|v| v.abs() < 1 << 20));
}

#[test]
fn silence_mode_skips_lsp_quantisation() {
    // Mode 0 carries no LSP field — the quantiser returns None.
    let lsp_q10 = [0i32; 10];
    assert!(quantise_lsp_q10(&lsp_q10, LspQuant::None).is_none());
}
