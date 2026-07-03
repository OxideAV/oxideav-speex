//! High-band innovation (fixed-codebook) sub-vector search — the
//! encoder counterpart of [`crate::hb_innovation::decode_hb_subframe`]
//! (round r385 scope).
//!
//! Per *The Speex Codec Manual* §10.3, *"the encoding of the high-band
//! excitation is done in a way similar to that of the narrowband
//! innovation"*: each 40-sample high-band sub-frame is split into
//! sub-vectors quantised independently against the mode's codebook and
//! concatenated. There is **no pitch prediction in the high band**
//! (§10.2), so the search target is the whole high-band excitation
//! residual, and with the excitation gain chosen open-loop the search
//! reduces to a per-sub-vector nearest-neighbour selection — the same
//! structure as the narrowband [`crate::innovation_search`], plus the
//! `HbSv8_128` codebook's **1-bit polarity sign**: each candidate row is
//! scored in both polarities and the winning `(index, sign)` pair packs
//! as `index << 1 | sign` within its 8-bit slot (MSB-first), exactly the
//! layout the decoder splits.
//!
//! ## Scope / fidelity
//!
//! Functional (round-trippable) sub-vector VQ in the residual domain at
//! a fixed open-loop gain, matching the narrowband search's posture.
//! The reference encoder's perceptually-filtered metric is not pinned by
//! the staged manual (the same recorded fidelity gap as the narrowband
//! search — a fidelity refinement, not a correctness one: the chosen
//! slots always decode to a valid high-band innovation).

use crate::hb_innovation::{hb_innovation_sub_vector, HbInnovationCodebook, HB_SUBFRAME_SAMPLES};

/// Result of a high-band innovation search over one 40-sample sub-frame.
#[derive(Debug, Clone, PartialEq)]
pub struct HbInnovationChoice {
    /// The per-sub-vector chosen codebook row indices (length = `count`),
    /// sub-vector 0 first. Sign bits are **not** folded in here — see
    /// [`Self::signs`].
    pub indices: Vec<u32>,
    /// The per-sub-vector polarity signs (`true` = negated row), aligned
    /// with [`Self::indices`]. Always all-`false` for a codebook without
    /// a sign bit ([`HbInnovationCodebook::HbSv10_32`]).
    pub signs: Vec<bool>,
    /// The packed on-wire `excitation_vq_index` field (sub-vector 0 in
    /// the most-significant slot bits; for a signed codebook each slot is
    /// `index << 1 | sign`), matching the layout
    /// [`crate::hb_innovation::decode_hb_subframe`] parses.
    pub packed: u128,
    /// The summed squared error of the chosen quantisation against the
    /// residual (in the residual-excitation domain).
    pub error: f64,
}

/// Search the high-band innovation codebook for the best per-sub-vector
/// quantisation of `residual` at fixed gain `gain`.
///
/// `codebook` and `count` come from the sub-mode's
/// [`crate::hb_innovation::HbInnovationMapping`] (`count` sub-vectors of
/// `codebook.sub_vector_len()` samples cover the 40-sample sub-frame).
/// For [`HbInnovationCodebook::HbSv8_128`] each candidate row is scored
/// in both polarities (the sign bit doubles the effective codebook).
pub fn search_hb_innovation(
    residual: &[f64; HB_SUBFRAME_SAMPLES],
    gain: f64,
    codebook: HbInnovationCodebook,
    count: u8,
) -> HbInnovationChoice {
    let sv_len = codebook.sub_vector_len();
    let n_entries = codebook.entries();
    let slot_bits = u32::from(codebook.slot_bits());
    let has_sign = codebook.has_sign_bit();
    debug_assert_eq!(
        sv_len * usize::from(count),
        HB_SUBFRAME_SAMPLES,
        "hb innovation sub-vectors must cover 40 samples"
    );

    let mut indices = Vec::with_capacity(usize::from(count));
    let mut signs = Vec::with_capacity(usize::from(count));
    let mut packed: u128 = 0;
    let mut total_error = 0.0_f64;

    for sv in 0..count {
        let base = usize::from(sv) * sv_len;
        let target = &residual[base..base + sv_len];

        let mut best_idx = 0u32;
        let mut best_sign = false;
        let mut best_err = f64::INFINITY;
        for idx in 0..n_entries {
            let Some(row) = hb_innovation_sub_vector(codebook, idx) else {
                continue;
            };
            // Score the positive polarity, and — for a signed codebook —
            // the negated row as a second candidate.
            let mut err_pos = 0.0_f64;
            let mut err_neg = 0.0_f64;
            for (k, &t) in target.iter().enumerate() {
                let c = gain * f64::from(row[k]);
                let dp = t - c;
                err_pos += dp * dp;
                if has_sign {
                    let dn = t + c;
                    err_neg += dn * dn;
                }
            }
            if err_pos < best_err {
                best_err = err_pos;
                best_idx = idx;
                best_sign = false;
            }
            if has_sign && err_neg < best_err {
                best_err = err_neg;
                best_idx = idx;
                best_sign = true;
            }
        }

        indices.push(best_idx);
        signs.push(best_sign);
        total_error += best_err;
        // Sub-vector 0 occupies the most-significant slot bits; a signed
        // codebook's slot is `index << 1 | sign` (MSB-first within the
        // slot), matching the decoder's split.
        let slot = if has_sign {
            (u128::from(best_idx) << 1) | u128::from(best_sign)
        } else {
            u128::from(best_idx)
        };
        packed = (packed << slot_bits) | slot;
    }

    HbInnovationChoice {
        indices,
        signs,
        packed,
        error: total_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hb_innovation::decode_hb_subframe;
    use crate::wideband::WidebandHighBandSubmode;

    #[test]
    fn recovers_exact_codebook_concatenation_sv10() {
        // Residual == gain·(concatenated rows) → the search recovers the
        // planted indices with ~zero error (unsigned codebook).
        let codebook = HbInnovationCodebook::HbSv10_32;
        let count = 4u8;
        let gain = 1.5_f64;
        let chosen = [3u32, 17, 0, 31];
        let mut residual = [0.0_f64; HB_SUBFRAME_SAMPLES];
        for (sv, &idx) in chosen.iter().enumerate() {
            let row = hb_innovation_sub_vector(codebook, idx).unwrap();
            for (k, &v) in row.iter().enumerate() {
                residual[sv * 10 + k] = gain * f64::from(v);
            }
        }
        let out = search_hb_innovation(&residual, gain, codebook, count);
        assert_eq!(out.indices, chosen.to_vec());
        assert!(out.signs.iter().all(|&s| !s), "unsigned codebook");
        assert!(out.error < 1e-6, "exact match should have ~zero error");
    }

    #[test]
    fn recovers_planted_signs_sv8() {
        // Signed codebook: plant negated rows in some sub-vectors and
        // confirm the search recovers both the indices and the signs.
        let codebook = HbInnovationCodebook::HbSv8_128;
        let count = 5u8;
        let gain = 0.8_f64;
        let chosen = [10u32, 20, 30, 40, 50];
        let planted_signs = [false, true, false, true, true];
        let mut residual = [0.0_f64; HB_SUBFRAME_SAMPLES];
        for (sv, (&idx, &neg)) in chosen.iter().zip(planted_signs.iter()).enumerate() {
            let row = hb_innovation_sub_vector(codebook, idx).unwrap();
            for (k, &v) in row.iter().enumerate() {
                let s = if neg { -1.0 } else { 1.0 };
                residual[sv * 8 + k] = s * gain * f64::from(v);
            }
        }
        let out = search_hb_innovation(&residual, gain, codebook, count);
        // A planted row could tie with another row's opposite polarity
        // only if the codebook contains an exact negated duplicate; check
        // the reconstruction rather than raw indices to stay robust.
        assert!(out.error < 1e-6, "planted rows should match exactly");
        let submode = WidebandHighBandSubmode::for_id(3).unwrap();
        let decoded = decode_hb_subframe(&submode, out.packed).unwrap();
        for n in 0..HB_SUBFRAME_SAMPLES {
            let want = residual[n] / gain;
            assert!(
                (f64::from(decoded[n]) - want).abs() < 1e-9,
                "sample {n}: decoded {} want {want}",
                decoded[n]
            );
        }
    }

    #[test]
    fn packed_field_decodes_back_through_decoder_mode_2() {
        // Mode 2 = 4 × HbSv10_32: the packed field must round-trip
        // through the decoder to the chosen rows.
        let codebook = HbInnovationCodebook::HbSv10_32;
        let chosen = [5u32, 10, 2, 15];
        let mut residual = [0.0_f64; HB_SUBFRAME_SAMPLES];
        for (sv, &idx) in chosen.iter().enumerate() {
            let row = hb_innovation_sub_vector(codebook, idx).unwrap();
            for (k, &v) in row.iter().enumerate() {
                residual[sv * 10 + k] = f64::from(v);
            }
        }
        let out = search_hb_innovation(&residual, 1.0, codebook, 4);
        assert_eq!(out.indices, chosen.to_vec());

        let submode = WidebandHighBandSubmode::for_id(2).unwrap();
        let decoded = decode_hb_subframe(&submode, out.packed).unwrap();
        for (sv, &idx) in chosen.iter().enumerate() {
            let row = hb_innovation_sub_vector(codebook, idx).unwrap();
            for k in 0..10 {
                assert_eq!(decoded[sv * 10 + k], row[k], "sv {sv} sample {k}");
            }
        }
    }

    #[test]
    fn sign_bit_never_worse_than_positive_only() {
        // For any residual, allowing the sign candidate can only reduce
        // the error versus a positive-only search.
        let codebook = HbInnovationCodebook::HbSv8_128;
        let mut residual = [0.0_f64; HB_SUBFRAME_SAMPLES];
        for (n, slot) in residual.iter_mut().enumerate() {
            *slot = -((n as f64) * 0.37).sin() * 90.0;
        }
        let gain = 1.2_f64;
        let out = search_hb_innovation(&residual, gain, codebook, 5);

        // Positive-only baseline error.
        let mut pos_err = 0.0_f64;
        for sv in 0..5usize {
            let target = &residual[sv * 8..sv * 8 + 8];
            let mut best = f64::INFINITY;
            for idx in 0..codebook.entries() {
                let row = hb_innovation_sub_vector(codebook, idx).unwrap();
                let mut err = 0.0_f64;
                for (k, &t) in target.iter().enumerate() {
                    let d = t - gain * f64::from(row[k]);
                    err += d * d;
                }
                best = best.min(err);
            }
            pos_err += best;
        }
        assert!(
            out.error <= pos_err + 1e-9,
            "signed search {} worse than positive-only {}",
            out.error,
            pos_err
        );
    }

    #[test]
    fn packed_field_fits_mode_bit_budget() {
        // Mode 2: 4 × 5 = 20 bits; mode 3: 5 × 8 = 40 bits.
        let residual = [1.0_f64; HB_SUBFRAME_SAMPLES];
        let m2 = search_hb_innovation(&residual, 1.0, HbInnovationCodebook::HbSv10_32, 4);
        assert!(m2.packed < (1u128 << 20));
        let m3 = search_hb_innovation(&residual, 1.0, HbInnovationCodebook::HbSv8_128, 5);
        assert!(m3.packed < (1u128 << 40));
    }
}
