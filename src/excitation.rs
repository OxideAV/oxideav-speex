//! Narrowband **raw excitation composition** primitive (round r244 scope).
//!
//! Composes the r241 adaptive-codebook contribution `ea[n]` (raw integer
//! `[i32; 40]`) with the r220 innovation sub-vector `c[n]` (raw integer
//! `[i16; 40]`) into a single per-sub-frame raw-integer excitation vector
//! `e_raw[n]` matching Speex Codec Manual §8.4 / CELP companion §2.3
//! structural composition:
//!
//! > `e[n] = p[n] + c[n]`
//!
//! where `p[n] = ea[n]` is the adaptive-codebook contribution and `c[n]`
//! is the fixed-codebook (innovation) contribution. This module supplies
//! the per-sample widening sum as a typed primitive matching the r234 /
//! r241 raw-integer convention, leaving the downstream Q-format pin for
//! a follow-up round.
//!
//! ## Scope (positive form)
//!
//! Given:
//!
//! * an adaptive-codebook contribution `ea[n]` as `[i32; 40]` (the r241
//!   [`crate::adaptive_contribution_subframe`] output);
//! * an innovation sub-vector `c[n]` as `[i16; 40]` (the r220
//!   [`crate::decode_innovation_subframe`] output);
//!
//! this module produces a typed `[i32; 40]` raw-integer per-sub-frame
//! excitation vector through a per-sample widening sum
//! `e_raw[n] = ea[n] + i32::from(c[n])`.
//!
//! ## Why a Q-format-agnostic raw-integer sum
//!
//! Both inputs are raw integer values in raw codebook units:
//!
//! * `ea[n]` from r241 is a raw integer dot product of post-bias gain
//!   integers with `i16` historical samples (see
//!   [`crate::adaptive_contribution`] module docs for the recorded
//!   pitch-gain Q-format gap).
//! * `c[n]` from r220 is the raw `i16` sub-vector entry concatenated off
//!   the staged narrowband innovation codebooks (see
//!   [`crate::innovation`] module docs for the fixed-codebook gain
//!   composition status — the gain pin lives in CELP companion §9 as a
//!   documented open-loop scalar quantiser, distinct from the static
//!   lookup tables that drive the present primitive).
//!
//! Adding two raw integer signals into a raw integer sum is well-defined
//! without committing to a Q-format choice: whatever Q-format the
//! downstream `e[n]` consumer eventually picks, the same single
//! arithmetic shift applies to all 40 elements of the vector. This
//! mirrors the r241 design (see [`crate::adaptive_contribution`] module
//! docs for the same argument applied to the three-tap dot product).
//!
//! ## Headroom argument
//!
//! Per [`crate::adaptive_contribution`] the per-sample `ea[n]` term is
//! bounded analytically by
//! `3 × |g|_max × |e|_max ≤ 3 × 159 × i16::MAX ≈ 1.6 × 10⁷`, well below
//! `i32::MAX ≈ 2.1 × 10⁹`. The added `c[n]` widens an `i16` into the
//! same `i32` accumulator with `|c| ≤ i16::MAX ≈ 3.3 × 10⁴`. The summed
//! magnitude is bounded by `1.6 × 10⁷ + 3.3 × 10⁴ ≈ 1.6 × 10⁷`, still
//! orders of magnitude below `i32::MAX`. Using checked arithmetic at the
//! per-sample level is therefore unnecessary; the widening `i32` cast on
//! `c[n]` plus the integer add suffice.
//!
//! ## Stream-start behaviour
//!
//! With the all-zero default [`crate::ExcitationBuffer`] from r234, the
//! r241 `ea[n]` term is identically zero across the whole sub-frame for
//! any tap triple (see r241 module docs for the proof). The composed
//! `e_raw[n]` at stream start therefore equals `c[n]` exactly — the
//! envelope follows the first-frame innovation sub-vector with the
//! continuous "no spurious transient" property inherited from r241.
//!
//! ## Feedback-loop note
//!
//! The §8.4 / §2.3 composition closes a feedback loop: once `e_raw[n]`
//! is determined at the eventual Q-format pin, the resulting `i16`
//! samples feed the [`crate::ExcitationBuffer`] for the next sub-frame's
//! r241 adaptive-codebook step. This module's `[i32; 40]` output is the
//! pre-Q-format-pin intermediate; the buffer-push step lives downstream
//! of the Q-format choice (a single right-shift over the vector reduces
//! it to the `i16` storage type the buffer expects). The present
//! primitive surfaces the algebra of the composition; the saturating
//! `i32 → i16` step lives one layer further down the pipeline.
//!
//! ## Spec basis (positive-only paths-read statement)
//!
//! All structural anchors come from in-tree material:
//!
//! * `docs/audio/speex/speex-manual.pdf` §8.4 ("Innovation") — names the
//!   excitation composition `e[n] = p[n] + c[n]` where `p[n]` is the
//!   long-term-predictor contribution and `c[n]` is the fixed-codebook
//!   contribution.
//! * `docs/audio/speex/speex-celp-companion.md` §2.3 ("Innovation (fixed
//!   codebook)") — paraphrases the same composition: *"Final excitation
//!   `e[n] = p[n] + c[n]`, c[n] from the fixed codebook"*.
//! * `docs/audio/speex/speex-celp-companion.md` §2.2 ("Pitch (adaptive
//!   codebook)") — confirms the adaptive-codebook contribution is itself
//!   `p[n] = β · e[n − T]` in the single-tap formulation and Eq. 9.1 in
//!   the three-tap formulation, both of which r241 surfaces as raw `i32`.
//!
//! ## Module structure
//!
//! Mirrors r241's [`crate::adaptive_contribution`] layout: a whole-vector
//! batch helper [`raw_excitation_subframe`] returning `[i32; 40]` plus a
//! per-sample helper [`raw_excitation_sample`] returning a single `i32`,
//! both pure and free of side effects.
//!
//! ## What this module DOES
//!
//! * [`raw_excitation_subframe`] — per-sample widening sum of an `ea`
//!   batch (`[i32; 40]`) with a `c` batch (`[i16; 40]`) into a single
//!   raw-integer `[i32; 40]`.
//! * [`raw_excitation_sample`] — single-sample helper returning the sum
//!   of one `ea[n]` element with one widened `c[n]` element.
//! * [`RAW_EXCITATION_SAMPLES`] — restated `40` constant matching
//!   [`crate::innovation::SUBFRAME_SAMPLES`] and re-exported at the
//!   crate root.
//!
//! ## What this module DOES NOT do
//!
//! * Q-format scaling. The output `e_raw[n]` is in raw integer units;
//!   the downstream consumer applies a single arithmetic shift once the
//!   pitch-gain + fixed-codebook-gain Q-formats are pinned (see
//!   [`crate::adaptive_contribution`] module docs for the deferred gap).
//! * Fixed-codebook gain scaling. The §9.2 fixed-codebook gain
//!   `g_frame × g_subf` is a scalar quantiser product the CELP companion
//!   §9 marks as a documented-but-non-table open-loop computation; its
//!   integration into the composition is a follow-up round.
//! * Saturating `i32 → i16` reduction. The buffer-push step downstream
//!   of the Q-format choice handles the storage-width reduction once
//!   the Q-format choice is pinned.
//! * Excitation-buffer mutation. The composition is a pure batch
//!   primitive; pushing the resulting samples into the
//!   [`crate::ExcitationBuffer`] lives with the caller after the
//!   Q-format reduction step.
//! * High-band path. Per §10.2 the wideband high-band has the simpler
//!   `e[n] = c[n]` composition (the high-band adaptive-codebook path is
//!   absent); the high band consumes its own fixed-codebook output
//!   directly via [`crate::decode_hb_subframe`].
//! * Encoder-side analysis-by-synthesis loop. The composition surfaced
//!   here is the decoder-side per-sub-frame reconstruction step.

use crate::innovation::SUBFRAME_SAMPLES;

/// Number of raw-excitation samples per CELP sub-frame.
///
/// Restates [`crate::innovation::SUBFRAME_SAMPLES`] = `40` (per Speex
/// Manual §9.1: *"subdivided in 4 sub-frames of 5 ms each (40
/// samples)"*) at the composition layer so the public API surface
/// names the dimension where the consumer reads it.
pub const RAW_EXCITATION_SAMPLES: usize = SUBFRAME_SAMPLES;

/// Compose one CELP sub-frame's raw-integer excitation
/// `e_raw[n] = ea[n] + i32::from(c[n])`.
///
/// Inputs:
///
/// * `ea` — the r241 adaptive-codebook contribution `[i32; 40]` from
///   [`crate::adaptive_contribution_subframe`].
/// * `c` — the r220 fixed-codebook sub-vector `[i16; 40]` from
///   [`crate::decode_innovation_subframe`].
///
/// Returns the per-sample widening sum as a `[i32; 40]` raw-integer
/// vector. Both inputs are already raw integer values; the output is
/// therefore also a raw integer value (Q-format-agnostic, matching the
/// r234 / r241 convention).
///
/// The output element `e_raw[n]` is bounded analytically by
/// `|ea[n]| + |c[n]| ≤ 1.6 × 10⁷ + 3.3 × 10⁴ ≈ 1.6 × 10⁷`, well below
/// `i32::MAX`; the `i32 + i32` accumulator therefore cannot overflow
/// for any conformant input.
#[inline]
pub fn raw_excitation_subframe(
    ea: &[i32; RAW_EXCITATION_SAMPLES],
    c: &[i16; RAW_EXCITATION_SAMPLES],
) -> [i32; RAW_EXCITATION_SAMPLES] {
    let mut out = [0i32; RAW_EXCITATION_SAMPLES];
    for (n, slot) in out.iter_mut().enumerate() {
        *slot = ea[n] + i32::from(c[n]);
    }
    out
}

/// Compose one sample of the raw excitation at sub-frame position `n`.
///
/// Convenience for callers that walk samples one at a time (e.g. an
/// encoder's analysis-by-synthesis loop). The result matches
/// `raw_excitation_subframe(ea, c)[n]`.
///
/// `n` is the output position within the sub-frame
/// (`0..RAW_EXCITATION_SAMPLES`).
#[inline]
pub fn raw_excitation_sample(n: usize, ea_n: i32, c_n: i16) -> i32 {
    debug_assert!(
        n < RAW_EXCITATION_SAMPLES,
        "sub-frame sample position out of range"
    );
    // `n` participates only as a debug guard; the composition itself is
    // index-free once the two input scalars are in hand.
    let _ = n;
    ea_n + i32::from(c_n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_codebook::ExcitationBuffer;
    use crate::adaptive_contribution::adaptive_contribution_subframe;
    use crate::innovation::decode_subframe;
    use crate::pitch_gain::PitchGainTaps;
    use crate::submode::NARROWBAND_SUBMODES;

    /// Both inputs zero → output zero.
    #[test]
    fn both_zero_yields_zero() {
        let ea = [0i32; RAW_EXCITATION_SAMPLES];
        let c = [0i16; RAW_EXCITATION_SAMPLES];
        let out = raw_excitation_subframe(&ea, &c);
        assert!(out.iter().all(|&v| v == 0));
    }

    /// Zero `ea` → output equals widened `c`.
    #[test]
    fn zero_ea_yields_widened_c() {
        let ea = [0i32; RAW_EXCITATION_SAMPLES];
        let mut c = [0i16; RAW_EXCITATION_SAMPLES];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (i as i16) * 7 - 100;
        }
        let out = raw_excitation_subframe(&ea, &c);
        for (n, &v) in out.iter().enumerate() {
            assert_eq!(v, i32::from(c[n]), "n={n}");
        }
    }

    /// Zero `c` → output equals `ea` exactly (no widening loss).
    #[test]
    fn zero_c_yields_ea_unchanged() {
        let mut ea = [0i32; RAW_EXCITATION_SAMPLES];
        for (i, slot) in ea.iter_mut().enumerate() {
            *slot = (i as i32) * 1_000 - 20_000;
        }
        let c = [0i16; RAW_EXCITATION_SAMPLES];
        let out = raw_excitation_subframe(&ea, &c);
        for (n, &v) in out.iter().enumerate() {
            assert_eq!(v, ea[n], "n={n}");
        }
    }

    /// Linearity in `ea`: `raw_excitation(ea1 + ea2, c) == raw(ea1, c) + raw(0, ea2)`.
    /// (The "+ raw(0, ea2)" path is exercised separately; this checks the
    /// distributive sum split.)
    #[test]
    fn linearity_in_ea_term() {
        let mut ea1 = [0i32; RAW_EXCITATION_SAMPLES];
        let mut ea2 = [0i32; RAW_EXCITATION_SAMPLES];
        let mut c = [0i16; RAW_EXCITATION_SAMPLES];
        for i in 0..RAW_EXCITATION_SAMPLES {
            ea1[i] = (i as i32) * 13 - 100;
            ea2[i] = (i as i32) * -17 + 200;
            c[i] = (i as i16) * 3 - 40;
        }
        let mut ea_sum = [0i32; RAW_EXCITATION_SAMPLES];
        for i in 0..RAW_EXCITATION_SAMPLES {
            ea_sum[i] = ea1[i] + ea2[i];
        }
        let out_combined = raw_excitation_subframe(&ea_sum, &c);
        let out_a = raw_excitation_subframe(&ea1, &c);
        let zero_c = [0i16; RAW_EXCITATION_SAMPLES];
        let out_b = raw_excitation_subframe(&ea2, &zero_c);
        for n in 0..RAW_EXCITATION_SAMPLES {
            assert_eq!(out_combined[n], out_a[n] + out_b[n], "n={n}");
        }
    }

    /// Pointwise pin: every output element equals the documented formula.
    #[test]
    fn pointwise_pin_matches_formula() {
        let mut ea = [0i32; RAW_EXCITATION_SAMPLES];
        let mut c = [0i16; RAW_EXCITATION_SAMPLES];
        for i in 0..RAW_EXCITATION_SAMPLES {
            ea[i] = (i as i32) * 1_234 - 5_678;
            c[i] = (i as i16) * 31 - 17;
        }
        let out = raw_excitation_subframe(&ea, &c);
        for n in 0..RAW_EXCITATION_SAMPLES {
            assert_eq!(out[n], ea[n] + i32::from(c[n]), "n={n}");
        }
    }

    /// Per-sample helper agrees with the batch path elementwise.
    #[test]
    fn per_sample_helper_matches_batch() {
        let mut ea = [0i32; RAW_EXCITATION_SAMPLES];
        let mut c = [0i16; RAW_EXCITATION_SAMPLES];
        for i in 0..RAW_EXCITATION_SAMPLES {
            ea[i] = (i as i32) * -73 + 4_000;
            c[i] = (i as i16) * 19 - 250;
        }
        let batch = raw_excitation_subframe(&ea, &c);
        for n in 0..RAW_EXCITATION_SAMPLES {
            let single = raw_excitation_sample(n, ea[n], c[n]);
            assert_eq!(single, batch[n], "n={n}");
        }
    }

    /// Stream-start composition with the r241 path: empty buffer yields
    /// `ea = 0`, so `e_raw == widened(c)` exactly.
    #[test]
    fn stream_start_envelope_follows_innovation_only() {
        let buffer = ExcitationBuffer::new();
        // Pick mode 6 — a Documented dispatch (8 × Sv5_256) — and feed a
        // small non-trivial index so c[n] is non-zero across the sub-frame.
        let submode = NARROWBAND_SUBMODES[6];
        // 8 sub-vectors × 8 bits each — distinct indices land distinct
        // codebook rows. Pick a small set: [3, 7, 1, 12, 5, 9, 2, 6].
        let indices: [u32; 8] = [3, 7, 1, 12, 5, 9, 2, 6];
        let mut packed: u128 = 0;
        for &i in &indices {
            packed = (packed << 8) | u128::from(i);
        }
        let c = decode_subframe(&submode, packed).expect("mode 6 documented");
        let taps = PitchGainTaps { taps: [60, 70, 80] };
        // Pick any in-spec pitch period.
        let ea = adaptive_contribution_subframe(50, taps, &buffer).expect("in-range pitch");
        // Stream-start sanity: ea is all-zero.
        assert!(
            ea.iter().all(|&v| v == 0),
            "stream-start adaptive contribution should be all-zero"
        );
        let out = raw_excitation_subframe(&ea, &c);
        // With ea = 0, e_raw[n] should equal widened c[n] verbatim.
        for n in 0..RAW_EXCITATION_SAMPLES {
            assert_eq!(out[n], i32::from(c[n]), "n={n}");
        }
    }

    /// Silence sub-mode (mode 0) → both terms zero → composed envelope
    /// identically zero across the sub-frame.
    #[test]
    fn silence_subframe_composes_to_zero_envelope() {
        let buffer = ExcitationBuffer::new();
        let silence = NARROWBAND_SUBMODES[0];
        let c = decode_subframe(&silence, 0).expect("silence yields zero c");
        let ea =
            adaptive_contribution_subframe(50, PitchGainTaps::SILENCE, &buffer).expect("in-range");
        let out = raw_excitation_subframe(&ea, &c);
        assert!(
            out.iter().all(|&v| v == 0),
            "silence composition should yield all-zero envelope"
        );
    }

    /// Headroom: even at the analytic worst case `(|ea| + |c|)` stays
    /// well below `i32::MAX`, so the per-sample sum never overflows.
    #[test]
    fn headroom_argument_holds_at_analytic_bounds() {
        // The r241 analytic bound on |ea[n]| is 3 × 159 × i16::MAX ≈ 1.6e7.
        let ea_bound: i32 = 3 * 159 * i32::from(i16::MAX);
        let c_bound: i32 = i32::from(i16::MAX);
        // Sum stays comfortably within i32::MAX.
        let envelope = ea_bound.checked_add(c_bound).expect("sum fits in i32");
        assert!(envelope < i32::MAX);
        // Symmetric on the negative side.
        let neg_ea_bound: i32 = 3 * 159 * i32::from(i16::MIN);
        let neg_c_bound: i32 = i32::from(i16::MIN);
        let neg_envelope = neg_ea_bound
            .checked_add(neg_c_bound)
            .expect("negative sum fits in i32");
        assert!(neg_envelope > i32::MIN);
    }

    /// Worked example: known `ea` + known `c` → handsumed composed values.
    #[test]
    fn worked_example_handsumed_values() {
        let mut ea = [0i32; RAW_EXCITATION_SAMPLES];
        let mut c = [0i16; RAW_EXCITATION_SAMPLES];
        ea[0] = 1_000;
        ea[1] = -500;
        ea[39] = 12_345;
        c[0] = 7;
        c[1] = -3;
        c[39] = -45;
        let out = raw_excitation_subframe(&ea, &c);
        assert_eq!(out[0], 1_007);
        assert_eq!(out[1], -503);
        assert_eq!(out[39], 12_300);
        // Untouched indices remain zero + 0 = 0.
        for (n, &v) in out.iter().enumerate().take(39).skip(2) {
            assert_eq!(v, 0, "untouched n={n}");
        }
    }

    /// Composition is commutative wrt swapping signs across the two
    /// terms (sanity-check for the integer-addition algebra).
    #[test]
    fn negation_commutes_through_composition() {
        let mut ea = [0i32; RAW_EXCITATION_SAMPLES];
        let mut c = [0i16; RAW_EXCITATION_SAMPLES];
        for i in 0..RAW_EXCITATION_SAMPLES {
            ea[i] = (i as i32) * 11 - 50;
            c[i] = (i as i16) * 5 - 20;
        }
        let out = raw_excitation_subframe(&ea, &c);
        // Negate both inputs, expect negated output.
        let mut ea_neg = [0i32; RAW_EXCITATION_SAMPLES];
        let mut c_neg = [0i16; RAW_EXCITATION_SAMPLES];
        for i in 0..RAW_EXCITATION_SAMPLES {
            ea_neg[i] = -ea[i];
            c_neg[i] = -c[i];
        }
        let out_neg = raw_excitation_subframe(&ea_neg, &c_neg);
        for n in 0..RAW_EXCITATION_SAMPLES {
            assert_eq!(out_neg[n], -out[n], "n={n}");
        }
    }

    /// Pre-existing buffer state + non-zero innovation → composed output
    /// is `ea + widened(c)` elementwise, exercising the r241 / r220
    /// composition end-to-end with a non-stream-start buffer.
    #[test]
    fn buffer_state_combined_with_innovation_composes_correctly() {
        let mut buffer = ExcitationBuffer::new();
        // Push a non-trivial history pattern so ea is non-zero.
        for i in 0..150 {
            let s = ((i * 13 + 7) & 0xff) as i16 - 128;
            buffer.push(s);
        }
        // Mode 8 — Documented dispatch (2 × Sv20_32, 5 bits each = 10 bits total).
        let submode = NARROWBAND_SUBMODES[8];
        let packed: u128 = (3u128 << 5) | 7u128; // two 5-bit indices.
        let c = decode_subframe(&submode, packed).expect("mode 8 documented");
        let taps = PitchGainTaps { taps: [30, 50, 20] };
        let period = 60u16;
        let ea = adaptive_contribution_subframe(period, taps, &buffer).expect("in-range");
        let out = raw_excitation_subframe(&ea, &c);
        // Re-derive elementwise from the two inputs and compare.
        for n in 0..RAW_EXCITATION_SAMPLES {
            let expected = ea[n] + i32::from(c[n]);
            assert_eq!(out[n], expected, "n={n}");
        }
    }
}
