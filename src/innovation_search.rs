//! Innovation (fixed-codebook) sub-vector search (encoder, round r382
//! scope).
//!
//! After the adaptive-codebook (pitch) contribution has been chosen
//! ([`crate::abs_search`]), the remaining excitation is coded by the
//! **fixed / innovation codebook**. Speex does not use an algebraic
//! codebook: each 40-sample sub-frame is split into sub-vectors of
//! 5 / 8 / 10 / 20 samples, and each sub-vector is quantised
//! independently against a bit-rate-dependent codebook, the results
//! concatenated (MAN §9.2 p.31; companion §2.3). The fixed-codebook gain
//! `g = g_frame · g_subf` is chosen **before** the codebook search (open
//! loop, companion §2.3), so with the gain fixed the search reduces to a
//! per-sub-vector nearest-neighbour selection.
//!
//! ## What this module does
//!
//! Given the residual excitation `r[n]` that the innovation should
//! approximate (the sub-frame's target excitation with the pitch
//! contribution already removed, in the same integer-scaled domain as
//! the staged codebook rows) and the fixed gain `g`, for each sub-vector
//! position it picks the codebook row `c*` minimising the squared error
//!
//! ```text
//! Σ_k ( r[base + k] − g · c[k] )²
//! ```
//!
//! and concatenates the chosen indices into the packed `innovation_vq`
//! field exactly as [`crate::innovation::decode_subframe`] parses it
//! (sub-vector 0 in the most-significant index bits). The result
//! reconstructs, through that decoder path, to `g · c*` — the
//! magnitude-correct innovation the sub-frame transmits.
//!
//! ## Scope / fidelity
//!
//! This is a **functional** (round-trippable) sub-vector VQ, not the
//! reference encoder's perceptually-filtered joint search: it selects in
//! the residual-excitation domain with the open-loop gain, matching the
//! companion's "gain chosen before the search" structure. A full
//! analysis-by-synthesis refinement (filtering each candidate through
//! the weighted synthesis and updating the target between sub-vectors)
//! is a later refinement; the reference's exact per-sub-vector filtered
//! metric is not pinned by the staged manual (recorded as a fidelity
//! gap, not a correctness one — the chosen indices always decode to a
//! valid innovation).

use crate::innovation::{sub_vector, InnovationCodebook, SUBFRAME_SAMPLES};

/// Result of an innovation sub-vector search over one 40-sample sub-frame.
#[derive(Debug, Clone, PartialEq)]
pub struct InnovationChoice {
    /// The per-sub-vector chosen codebook indices (length = `count`),
    /// sub-vector 0 first.
    pub indices: Vec<u32>,
    /// The packed on-wire `innovation_vq` field (sub-vector 0 in the
    /// most-significant `index_bits` bits), matching the layout
    /// [`crate::innovation::decode_subframe`] parses.
    pub packed: u128,
    /// The summed squared error of the chosen quantisation against the
    /// residual (in the residual-excitation domain).
    pub error: f64,
}

/// Search the innovation codebook for the best per-sub-vector
/// quantisation of `residual` at fixed gain `gain`.
///
/// `codebook` and `count` come from the sub-mode's
/// [`crate::innovation::InnovationMapping`] (`count` sub-vectors of
/// `codebook.sub_vector_len()` samples cover the 40-sample sub-frame).
/// Returns the chosen indices, the packed field, and the residual-domain
/// error.
pub fn search_innovation(
    residual: &[f64; SUBFRAME_SAMPLES],
    gain: f64,
    codebook: InnovationCodebook,
    count: u8,
) -> InnovationChoice {
    let sv_len = codebook.sub_vector_len();
    let n_entries = codebook.entries();
    let bits = u32::from(codebook.index_bits());
    debug_assert_eq!(
        sv_len * usize::from(count),
        SUBFRAME_SAMPLES,
        "innovation sub-vectors must cover 40 samples"
    );

    let mut indices = Vec::with_capacity(usize::from(count));
    let mut packed: u128 = 0;
    let mut total_error = 0.0_f64;

    for sv in 0..count {
        let base = usize::from(sv) * sv_len;
        let target = &residual[base..base + sv_len];

        let mut best_idx = 0u32;
        let mut best_err = f64::INFINITY;
        for idx in 0..n_entries {
            let Some(row) = sub_vector(codebook, idx) else {
                continue;
            };
            let mut err = 0.0_f64;
            for (k, &t) in target.iter().enumerate() {
                let d = t - gain * f64::from(row[k]);
                err += d * d;
            }
            if err < best_err {
                best_err = err;
                best_idx = idx;
            }
        }

        indices.push(best_idx);
        total_error += best_err;
        // Sub-vector 0 occupies the most-significant index bits: shift
        // the already-accumulated bits up and append this index.
        packed = (packed << bits) | u128::from(best_idx);
    }

    InnovationChoice {
        indices,
        packed,
        error: total_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::innovation::decode_subframe;
    use crate::submode::NARROWBAND_SUBMODES;

    #[test]
    fn recovers_exact_codebook_concatenation() {
        // Build a residual that is exactly gain·(concatenated codebook
        // rows) for chosen indices; the search must recover those indices.
        let codebook = InnovationCodebook::Sv10_32; // 10 samples, 32 entries
        let count = 4u8;
        let gain = 1.0_f64;
        let chosen = [3u32, 17, 0, 31];
        let mut residual = [0.0_f64; SUBFRAME_SAMPLES];
        for (sv, &idx) in chosen.iter().enumerate() {
            let row = sub_vector(codebook, idx).unwrap();
            let base = sv * 10;
            for (k, &v) in row.iter().enumerate() {
                residual[base + k] = gain * f64::from(v);
            }
        }
        let out = search_innovation(&residual, gain, codebook, count);
        assert_eq!(out.indices, chosen.to_vec());
        assert!(out.error < 1e-6, "exact match should have ~zero error");
    }

    #[test]
    fn packed_field_decodes_back_through_decoder() {
        // The packed field the search emits must round-trip through the
        // decoder's decode_subframe to the same gain·rows.
        let codebook = InnovationCodebook::Sv10_16; // mode 2: 10 samples, 16 entries
        let count = 4u8;
        let gain = 1.0_f64;
        let chosen = [5u32, 10, 2, 15];
        let mut residual = [0.0_f64; SUBFRAME_SAMPLES];
        for (sv, &idx) in chosen.iter().enumerate() {
            let row = sub_vector(codebook, idx).unwrap();
            let base = sv * 10;
            for (k, &v) in row.iter().enumerate() {
                residual[base + k] = gain * f64::from(v);
            }
        }
        let out = search_innovation(&residual, gain, codebook, count);
        assert_eq!(out.indices, chosen.to_vec());

        // Mode 2 uses this codebook; decode the packed field.
        let submode = &NARROWBAND_SUBMODES[2];
        let decoded = decode_subframe(submode, out.packed).unwrap();
        for (sv, &idx) in chosen.iter().enumerate() {
            let row = sub_vector(codebook, idx).unwrap();
            let base = sv * 10;
            for k in 0..10 {
                assert_eq!(decoded[base + k], row[k], "sv {sv} sample {k}");
            }
        }
    }

    #[test]
    fn zero_residual_and_gain_pick_lowest_error_row() {
        // With a zero residual and a nonzero gain, the search picks the
        // row closest to zero (smallest energy). Just assert it returns a
        // valid choice and finite error.
        let codebook = InnovationCodebook::Sv5_64;
        let residual = [0.0_f64; SUBFRAME_SAMPLES];
        let out = search_innovation(&residual, 2.0, codebook, 8);
        assert_eq!(out.indices.len(), 8);
        assert!(out.error.is_finite());
    }

    #[test]
    fn search_never_worse_than_arbitrary_index() {
        let codebook = InnovationCodebook::Sv8_128;
        let count = 5u8;
        let gain = 0.7_f64;
        let mut residual = [0.0_f64; SUBFRAME_SAMPLES];
        for (n, slot) in residual.iter_mut().enumerate() {
            *slot = ((n as f64) * 0.3).sin() * 100.0;
        }
        let out = search_innovation(&residual, gain, codebook, count);
        // Compute the error of an all-zero-index quantisation.
        let mut alt_err = 0.0_f64;
        for sv in 0..usize::from(count) {
            let row = sub_vector(codebook, 0).unwrap();
            let base = sv * 8;
            for k in 0..8 {
                let d = residual[base + k] - gain * f64::from(row[k]);
                alt_err += d * d;
            }
        }
        assert!(
            out.error <= alt_err + 1e-9,
            "search worse than index-0 baseline"
        );
    }

    #[test]
    fn twenty_sample_mode8_covers_subframe() {
        // Mode 8: Sv20_32, count 2 (2 × 20 = 40).
        let codebook = InnovationCodebook::Sv20_32;
        let out = search_innovation(&[0.0_f64; SUBFRAME_SAMPLES], 1.0, codebook, 2);
        assert_eq!(out.indices.len(), 2);
        // Packed field is 2 × 5 = 10 bits wide.
        assert!(out.packed < (1u128 << 10));
    }
}
