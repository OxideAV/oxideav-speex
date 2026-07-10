//! Wideband high-band **folded excitation** (HB mode 1) — the
//! reconstruction law for the gain-only high-band sub-mode, arbitrated
//! externally against the staged reference decode fixture (round r393).
//!
//! ## The law
//!
//! High-band sub-mode 1 (Table 10.1, manual §10.4) transmits a 5-bit
//! excitation gain per sub-frame and **no innovation vector**. Per
//! manual §10.2 / §10.3 and the staged clean-room note
//! `docs/audio/speex/hb-folded-gain.md`, the decoder reconstructs the
//! high-band excitation by **folding the already-decoded low-band
//! (narrowband) excitation** into the high band and scaling it by the
//! reconstructed gain:
//!
//! ```text
//! e_hb[n] = K · g · (−1)ⁿ · e_lb[n]        n = 0 .. 39 (per sub-frame)
//! g       = fold_quant_bound[qi]           (5-bit index qi, staged table)
//! K       = HB_FOLD_RECONSTRUCTION_MULT
//! ```
//!
//! where `e_lb[n]` is the same sub-frame's composed low-band excitation
//! `e[n] = p[n] + c[n]` ([`crate::NarrowbandDecoder::last_frame_excitation`])
//! and the `(−1)ⁿ` factor is the sample-level spectral fold: modulating
//! by the Nyquist carrier mirrors the excitation spectrum, so the
//! low-band fine structure lands frequency-reversed in the high band —
//! exactly the QMF axis reversal manual §10.2 describes. The parity
//! convention (`+1` at even `n`) is pinned **jointly with this crate's
//! QMF synthesis modulation convention** ([`crate::QmfSynthesis`],
//! `g1[n] = −2·(−1)ⁿ·h0[n]`): the fixture arbitration below validated
//! the composed chain fold → synthesis → QMF against the reference
//! decoder's PCM, so the two sign choices are pinned as a pair.
//!
//! ## External arbitration (staged fixture, round r393)
//!
//! `docs/audio/speex/fixtures/wb-mode1-folded/` supplies a real
//! WB-mode-1 stream (101 frames, every high-band frame sub-mode 1) plus
//! the reference decoder's `--no-enh` PCM. Replaying the stream through
//! this crate with candidate fold conventions and regressing the
//! reference high-band half-band (QMF analysis split, best-lag aligned)
//! onto each candidate's output:
//!
//! | candidate                        | HB correlation | HB shape SNR |
//! |----------------------------------|---------------:|-------------:|
//! | `g · e_lb[n]` (no modulation)    | 0.31           | 0.5 dB       |
//! | `g · (−1)ⁿ · e_lb[n]`            | **0.9999**     | **36.3 dB**  |
//! | `g · e_lb[n] / rms(e_lb)`        | 0.03           | 0.0 dB       |
//!
//! The `(−1)ⁿ` fold is decisive. The residual scalar `K` was measured
//! by per-sub-frame least squares over all 398 non-trivial sub-frames:
//! energy-weighted mean `0.3549` (std ≈ 1 %, no dependence on the gain
//! index — so it is a constant of the law, not a table effect), with a
//! systematic-error window of ≈ ±1 % inherited from the not-yet-bit-exact
//! low-band envelope. This crate adopts the reading `K = 1/(2·√2)`
//! (= 0.35355, inside the measured window): one √2 per QMF half-band
//! energy normalisation of the two-filterbank chain the folded
//! excitation traverses. The exact reference constant within that ±1 %
//! window remains open until the low band is bit-exact (recorded
//! follow-up, README).
//!
//! ## Crossover shaping (round r410)
//!
//! The flat `K` above is what the near-stationary tone fixture pins —
//! but the r410 speech fixture (`tests/fixtures/wb-conformance/wb_q4`)
//! showed it overshooting the reference by up to **130× per frame** in
//! envelope troughs. Synthetic oracle streams sweeping the high-band
//! LSP envelope against the reference decoder (see
//! `tests/fixtures/wb-conformance/NOTES.md`) pin the missing factor:
//! the fold scale is **proportional to the high-band envelope's
//! magnitude response at the 4 kHz QMF crossover**, saturating at the
//! flat constant:
//!
//! ```text
//! e_hb[n] = min( C · |A_hb(π)| , K ) · g · (−1)ⁿ · e_lb[n]
//! ```
//!
//! ([`HB_FOLD_CROSSOVER_SLOPE`] / [`HB_FOLD_CROSSOVER_CEILING`],
//! applied via [`folded_hb_scale`] / the per-sub-frame
//! [`folded_hb_excitation_subframe_shaped`]). Both real-stream anchors
//! (`wb-mode1-folded`, `uwb-fold-geometry`) sit in the saturated
//! region and are unchanged; the speech fixture moves from −12.9 dB /
//! 20× energy to 15.6 dB / 1.01× energy full-signal. A residual
//! per-frame factor of up to ≈6× on some speech frames remains
//! unattributed (recorded follow-up + docs ask).
//!
//! ## Scope
//!
//! * The **wideband** decoder applies the shaped law with the embedded
//!   narrowband frame's excitation as the fold source (same 8 kHz
//!   half-band geometry, sub-frame for sub-frame) — pinned by the
//!   fixtures.
//! * The **ultra-wideband** second layer keeps the flat outer law
//!   (`UWB_FOLD_RECONSTRUCTION_MULT`): its staged fixture's layer-2
//!   envelope sits in the saturated region, so the shaping is
//!   unobservable there; its fold *source* at the 16 kHz half-band
//!   geometry — see [`crate::uwb_decoder`] — is the r403 pin.

use crate::codebooks::HB_LPC_ORDER;
use crate::hb_innovation::HB_SUBFRAME_SAMPLES;

/// The constant scalar `K` of the folded reconstruction
/// `e_hb[n] = K · g · (−1)ⁿ · e_lb[n]`.
///
/// Adopted reading `1/(2·√2)` of the fixture-measured window
/// `0.3516 … 0.3549` (module docs); the staged `fold_quant_bound` table
/// carries no separate reconstruction multiplier
/// (`docs/audio/speex/provenance/02-speex-gain-quant.md`), so this
/// constant is the whole non-table part of the law.
pub const HB_FOLD_RECONSTRUCTION_MULT: f64 = 0.353_553_390_593_273_8;

/// Slope constant `C` of the **crossover-shaped** folded reconstruction
/// (round r410):
///
/// ```text
///   e_hb[n] = min( C · |A_hb(π)| , K ) · g · (−1)ⁿ · e_lb[n]
/// ```
///
/// Black-box oracle probing (synthetic single-fold streams sweeping the
/// 12-bit high-band LSP envelope against the reference decoder — see
/// `tests/fixtures/wb-conformance/NOTES.md`) shows the reference's
/// mode-1 high-band level is **attenuated in proportion to the
/// high-band synthesis filter's magnitude response at the 4 kHz QMF
/// crossover** (`ω = π` of the frequency-reversed half-band domain):
/// across 16 probed envelopes with `|A_hb(π)| ≤ 1.4` the
/// reference-to-fold ratio tracks `C·|A_hb(π)|` within ±13 % (measured
/// slope `C ≈ 0.171…0.189`, adopted `0.17`), while it tracks *no*
/// power of the filter's gross gain (impulse-response energy spans the
/// same sweep with a 66× unexplained spread). This is the same
/// crossover-response compensation the staged provenance names for the
/// sibling 4-bit gain-correction path ("corrects for the LPC-response
/// difference at the 4 kHz QMF cutoff", `provenance/02` /
/// `hb-folded-gain.md` §3).
pub const HB_FOLD_CROSSOVER_SLOPE: f64 = 0.17;

/// Saturation ceiling of the crossover-shaped law — the r393 flat
/// constant [`HB_FOLD_RECONSTRUCTION_MULT`].
///
/// Both real-stream anchors sit in the saturated region: the
/// `wb-mode1-folded` tone fixture (per-sub-frame `|A_hb(π)|`
/// 2.4…3.6) and the `uwb-fold-geometry` fixture's embedded wideband
/// layer (`|A_hb(π)| = 3.56`) both match the **flat** `K = 0.35355`
/// law sample-accurately (high band 38.9 dB / corr 0.9999, and
/// 21.7 dB embedded-layer respectively), which a linear-through-zero
/// crossover law cannot reproduce (it would need `K/C ≈ |A| ≥ 2` to
/// saturate). The oracle sweep only reaches `|A_hb(π)| ≤ 1.4`, where
/// the law is linear; the two fixtures pin the ceiling. The exact
/// shape of the transition (hard `min` vs a smooth knee) between
/// `|A| ≈ 1.4` and `|A| ≈ 2.4` is unobserved — recorded follow-up.
pub const HB_FOLD_CROSSOVER_CEILING: f64 = HB_FOLD_RECONSTRUCTION_MULT;

/// Magnitude response `|A_hb(e^{jπ})|` of the high-band short-term
/// predictor at the QMF crossover, for an order-8 LPC set in the
/// crate's synthesis convention (`x[n] = e[n] + Σ a[i]·x[n−1−i]`, i.e.
/// `A(z) = 1 − Σ a[i]·z^{−(i+1)}`).
///
/// At `z = e^{jπ} = −1`: `A(−1) = 1 − Σ a[i]·(−1)^{i+1}`.
#[inline]
pub fn hb_crossover_response(lpc: &[f64; HB_LPC_ORDER]) -> f64 {
    let mut acc = 1.0f64;
    for (i, &a) in lpc.iter().enumerate() {
        // (−1)^{i+1}: −1 for even i, +1 for odd i.
        if i % 2 == 0 {
            acc += a;
        } else {
            acc -= a;
        }
    }
    acc.abs()
}

/// The effective per-sub-frame fold scale of the crossover-shaped law:
/// `k = min(C·|A_hb(π)|, K) · g` ([`HB_FOLD_CROSSOVER_SLOPE`] /
/// [`HB_FOLD_CROSSOVER_CEILING`]), where `lpc` is the sub-frame's
/// interpolated order-8 high-band LPC set — the same set that drives
/// the synthesis filter the folded excitation feeds.
#[inline]
pub fn folded_hb_scale(gain: f32, lpc: &[f64; HB_LPC_ORDER]) -> f64 {
    (HB_FOLD_CROSSOVER_SLOPE * hb_crossover_response(lpc)).min(HB_FOLD_CROSSOVER_CEILING)
        * f64::from(gain)
}

/// Reconstruct one high-band mode-1 sub-frame's excitation with the
/// r410 **crossover-shaped** folded law:
///
/// ```text
///   e_hb[n] = min( C · |A_hb(π)| , K ) · g · (−1)ⁿ · e_lb[n]
/// ```
///
/// This supersedes the flat r393 law
/// ([`folded_hb_excitation_subframe`], kept for API compatibility) —
/// the flat constant matched the (near-stationary-envelope) tone
/// fixture but overshoots real speech by up to 130× per frame in
/// envelope troughs; the crossover shaping is what the reference
/// decoder's black-box envelope sweep pins (module docs of
/// [`hb_crossover_response`]).
#[inline]
pub fn folded_hb_excitation_subframe_shaped(
    exc_lb: &[f32; HB_SUBFRAME_SAMPLES],
    gain: f32,
    lpc: &[f64; HB_LPC_ORDER],
) -> [f64; HB_SUBFRAME_SAMPLES] {
    let k = folded_hb_scale(gain, lpc);
    let mut out = [0.0f64; HB_SUBFRAME_SAMPLES];
    for (n, (slot, &e)) in out.iter_mut().zip(exc_lb.iter()).enumerate() {
        let folded = k * f64::from(e);
        *slot = if n % 2 == 0 { folded } else { -folded };
    }
    out
}

/// Reconstruct one high-band mode-1 sub-frame's excitation by folding
/// the matching low-band excitation sub-frame.
///
/// `exc_lb` is the composed low-band excitation `e[n] = p[n] + c[n]` of
/// the **same** sub-frame
/// ([`crate::NarrowbandDecoder::last_frame_excitation`], 40-sample
/// slice); `gain` is the reconstructed 5-bit folded gain
/// ([`crate::reconstruct_hb_exc_gain`] on the
/// [`crate::HbExcitationGainIndex::FiveBit`] index). Returns the
/// high-band excitation `e_hb[n] = K·g·(−1)ⁿ·exc_lb[n]` ready for the
/// order-8 high-band synthesis filter.
///
/// A zero low-band excitation (stream start, silence) folds to zero
/// regardless of the gain — the graceful shape the earlier
/// (pre-fixture) rounds pinned for the gain-only mode.
#[inline]
pub fn folded_hb_excitation_subframe(
    exc_lb: &[f32; HB_SUBFRAME_SAMPLES],
    gain: f32,
) -> [f64; HB_SUBFRAME_SAMPLES] {
    let k = HB_FOLD_RECONSTRUCTION_MULT * f64::from(gain);
    let mut out = [0.0f64; HB_SUBFRAME_SAMPLES];
    for (n, (slot, &e)) in out.iter_mut().zip(exc_lb.iter()).enumerate() {
        let folded = k * f64::from(e);
        *slot = if n % 2 == 0 { folded } else { -folded };
    }
    out
}

/// Slice variant of the folded law for the ultra-wideband second layer
/// (80-sample sub-frames at the 16 kHz half-band): writes
/// `out[n] = K·g·(−1)ⁿ·exc_src[n]` for slices of any (equal, even)
/// length.
///
/// The **law** is the one the wideband fixture pins
/// ([`folded_hb_excitation_subframe`]); the ultra-wideband fold
/// *source* fed to this variant is the crate's recursion-consistent
/// generalisation (see [`crate::uwb_decoder`] — the reference's exact
/// source geometry at the 16 kHz half-band is a recorded gap, the
/// staged fixture being wideband-only).
#[inline]
pub fn folded_hb_excitation_slice(exc_src: &[f64], gain: f32, out: &mut [f64]) {
    debug_assert_eq!(exc_src.len(), out.len());
    let k = HB_FOLD_RECONSTRUCTION_MULT * f64::from(gain);
    for (n, (slot, &e)) in out.iter_mut().zip(exc_src.iter()).enumerate() {
        let folded = k * e;
        *slot = if n % 2 == 0 { folded } else { -folded };
    }
}

/// The constant scalar of the **ultra-wideband second-layer** folded
/// reconstruction — the outer (8–16 kHz) sub-band stage's analogue of
/// [`HB_FOLD_RECONSTRUCTION_MULT`].
///
/// The outer fold's source is the reconstructed **first-high-band
/// excitation** ([`crate::WidebandDecoder::last_hb_excitation`], the
/// 4–8 kHz layer), linear-interpolated to the 16 kHz half-band geometry
/// and re-folded into 8–16 kHz. Because that source has already been
/// scaled by the inner [`HB_FOLD_RECONSTRUCTION_MULT`], the outer stage
/// needs a **smaller** multiplier: the value `1/16` (=
/// `HB_FOLD_RECONSTRUCTION_MULT²/2`) sits inside the window measured by
/// the staged 3-layer fixture (`docs/audio/speex/fixtures/
/// uwb-fold-geometry/`): per-sub-frame least squares gives `≈ 0.060`,
/// the energy-ratio match `≈ 0.064`, and `1/16 = 0.0625` lands between
/// them. Like the inner constant this is **fixture-calibrated** (adopted
/// reading, not a staged table value) and carries the same ±1 % residue
/// until the low band is bit-exact — see [`crate::uwb_decoder`].
pub const UWB_FOLD_RECONSTRUCTION_MULT: f64 = 0.0625;

/// Linear-interpolate a 160-sample 8 kHz high-band excitation up to the
/// 320-sample 16 kHz second-layer geometry.
///
/// The ultra-wideband second high-band layer synthesises a 320-sample
/// 16 kHz half-band; its fold source (the embedded wideband layer's
/// 160-sample first-high-band excitation) is brought to that rate by
/// 2× linear interpolation (`out[2i] = src[i]`,
/// `out[2i+1] = ½(src[i]+src[i+1])`). The fixture arbitration measured
/// linear interpolation at high-band correlation **0.93** vs **0.85**
/// for nearest-sample repetition — the smoother upsample is the pinned
/// choice.
#[inline]
pub fn upsample_hb_excitation_linear(src: &[f64], out: &mut [f64]) {
    debug_assert_eq!(out.len(), 2 * src.len());
    let last = src.len().saturating_sub(1);
    for i in 0..src.len() {
        let a = src[i];
        let b = src[(i + 1).min(last)];
        out[2 * i] = a;
        out[2 * i + 1] = 0.5 * (a + b);
    }
}

/// Reconstruct one ultra-wideband second-layer sub-frame's excitation by
/// re-folding the (already upsampled) first-high-band excitation.
///
/// Identical in form to [`folded_hb_excitation_slice`] — the same
/// `(−1)ⁿ` spectral fold — but scaled by
/// [`UWB_FOLD_RECONSTRUCTION_MULT`] rather than the inner
/// [`HB_FOLD_RECONSTRUCTION_MULT`], because the outer stage's source has
/// already been through the inner fold (module + constant docs).
#[inline]
pub fn folded_uwb_excitation_slice(exc_src: &[f64], gain: f32, out: &mut [f64]) {
    debug_assert_eq!(exc_src.len(), out.len());
    let k = UWB_FOLD_RECONSTRUCTION_MULT * f64::from(gain);
    for (n, (slot, &e)) in out.iter_mut().zip(exc_src.iter()).enumerate() {
        let folded = k * e;
        *slot = if n % 2 == 0 { folded } else { -folded };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> [f32; HB_SUBFRAME_SAMPLES] {
        let mut e = [0.0f32; HB_SUBFRAME_SAMPLES];
        for (i, s) in e.iter_mut().enumerate() {
            *s = (i as f32) * 3.5 - 40.0;
        }
        e
    }

    /// Pointwise pin of the law: K·g·(−1)ⁿ·e[n], +1 at even n.
    #[test]
    fn pointwise_pin_matches_law() {
        let e = ramp();
        let g = 2.25f32;
        let out = folded_hb_excitation_subframe(&e, g);
        for (n, &v) in out.iter().enumerate() {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            let want = HB_FOLD_RECONSTRUCTION_MULT * f64::from(g) * sign * f64::from(e[n]);
            assert!((v - want).abs() < 1e-12, "n={n}: {v} vs {want}");
        }
    }

    /// The fold is linear in the gain.
    #[test]
    fn linear_in_gain() {
        let e = ramp();
        let single = folded_hb_excitation_subframe(&e, 0.7);
        let double = folded_hb_excitation_subframe(&e, 1.4);
        for n in 0..HB_SUBFRAME_SAMPLES {
            assert!((double[n] - 2.0 * single[n]).abs() < 1e-12, "n={n}");
        }
    }

    /// A zero excitation folds to zero for every gain.
    #[test]
    fn zero_excitation_folds_to_zero() {
        let zero = [0.0f32; HB_SUBFRAME_SAMPLES];
        for g in [0.0f32, 0.30498, 14.69497] {
            let out = folded_hb_excitation_subframe(&zero, g);
            assert!(out.iter().all(|&v| v == 0.0), "g={g}");
        }
    }

    /// The (−1)ⁿ modulation mirrors the spectrum: a constant (DC)
    /// excitation folds to the Nyquist alternation, i.e. the fold moves
    /// energy from one band edge to the other.
    #[test]
    fn dc_folds_to_nyquist_alternation() {
        let dc = [8.0f32; HB_SUBFRAME_SAMPLES];
        let out = folded_hb_excitation_subframe(&dc, 1.0);
        let expect = HB_FOLD_RECONSTRUCTION_MULT * 8.0;
        for (n, &v) in out.iter().enumerate() {
            let want = if n % 2 == 0 { expect } else { -expect };
            assert!((v - want).abs() < 1e-12, "n={n}");
        }
        // Sums to ~0: no DC survives the fold.
        assert!(out.iter().sum::<f64>().abs() < 1e-9);
    }

    /// The slice variant matches the fixed sub-frame entry elementwise
    /// on the 40-sample geometry.
    #[test]
    fn slice_variant_matches_subframe_entry() {
        let e = ramp();
        let g = 1.75f32;
        let fixed = folded_hb_excitation_subframe(&e, g);
        let src: Vec<f64> = e.iter().map(|&v| f64::from(v)).collect();
        let mut out = vec![0.0f64; HB_SUBFRAME_SAMPLES];
        folded_hb_excitation_slice(&src, g, &mut out);
        for n in 0..HB_SUBFRAME_SAMPLES {
            assert!((out[n] - fixed[n]).abs() < 1e-9, "n={n}");
        }
    }

    /// The adopted constant sits inside the fixture-measured window.
    #[test]
    fn reconstruction_mult_is_in_measured_window() {
        let k = std::hint::black_box(HB_FOLD_RECONSTRUCTION_MULT);
        assert!((k - 1.0 / (2.0 * 2.0f64.sqrt())).abs() < 1e-15);
        assert!(k > 0.3516 - 1e-4);
        assert!(k < 0.3549 + 1e-4);
    }

    /// Linear 2× upsample: even samples copy, odd samples average with
    /// the next; the last sample holds (no phantom step past the end).
    #[test]
    fn linear_upsample_interpolates_and_holds_tail() {
        let src = [2.0f64, 4.0, 10.0];
        let mut out = [0.0f64; 6];
        upsample_hb_excitation_linear(&src, &mut out);
        assert_eq!(out, [2.0, 3.0, 4.0, 7.0, 10.0, 10.0]);
    }

    /// The outer UWB fold multiplier sits inside the 3-layer fixture's
    /// measured window (LS ≈ 0.060, energy-match ≈ 0.064) and is the
    /// clean reading `1/16`.
    #[test]
    fn uwb_reconstruction_mult_is_in_measured_window() {
        let k = std::hint::black_box(UWB_FOLD_RECONSTRUCTION_MULT);
        assert!((k - 1.0 / 16.0).abs() < 1e-15);
        assert!((0.058..=0.066).contains(&k));
        // Smaller than the inner constant: the source is already folded.
        assert!(k < HB_FOLD_RECONSTRUCTION_MULT);
    }

    /// The UWB fold has the same `(−1)ⁿ` shape as the inner fold, scaled
    /// by the outer multiplier rather than the inner one.
    #[test]
    fn uwb_fold_uses_outer_multiplier_and_nyquist_sign() {
        let src: Vec<f64> = (0..8).map(|i| i as f64 - 3.5).collect();
        let g = 2.0f32;
        let mut out = vec![0.0f64; 8];
        folded_uwb_excitation_slice(&src, g, &mut out);
        for (n, (&o, &s)) in out.iter().zip(src.iter()).enumerate() {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            let want = UWB_FOLD_RECONSTRUCTION_MULT * f64::from(g) * sign * s;
            assert!((o - want).abs() < 1e-12, "n={n}: {o} vs {want}");
        }
    }

    // ---------- r410 crossover-shaped law ----------

    /// `|A(−1)|` of the flat (all-zero) LPC set is 1 — the shaped scale
    /// then sits in the linear region at `C · g`.
    #[test]
    fn crossover_response_of_flat_envelope_is_unity() {
        let flat = [0.0f64; 8];
        assert_eq!(hb_crossover_response(&flat), 1.0);
        let k = folded_hb_scale(1.0, &flat);
        assert!((k - HB_FOLD_CROSSOVER_SLOPE).abs() < 1e-12);
    }

    /// Alternating-sign evaluation: `A(−1) = 1 − Σ a[i]·(−1)^{i+1}` —
    /// pinned against a directly evaluated example.
    #[test]
    fn crossover_response_matches_direct_evaluation() {
        let lpc = [0.5f64, -0.25, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0];
        // A(−1) = 1 − (0.5·(−1) + (−0.25)·(+1) + 0.125·(−1))
        //       = 1 − (−0.5 − 0.25 − 0.125) = 1.875.
        assert!((hb_crossover_response(&lpc) - 1.875).abs() < 1e-12);
    }

    /// The shaped scale saturates at the flat r393 constant: an
    /// envelope with a large crossover response cannot push the fold
    /// above `K = HB_FOLD_RECONSTRUCTION_MULT`.
    #[test]
    fn shaped_scale_saturates_at_flat_constant() {
        // Alternating taps drive |A(−1)| = |1 − 8·0.9| = 6.2 — far
        // above the knee at K/C ≈ 2.08.
        let peaky = [-0.9f64, 0.9, -0.9, 0.9, -0.9, 0.9, -0.9, 0.9];
        let k = folded_hb_scale(1.0, &peaky);
        assert!((k - HB_FOLD_RECONSTRUCTION_MULT).abs() < 1e-12);
        // Gain scales linearly through the saturation.
        assert!((folded_hb_scale(2.5, &peaky) - 2.5 * HB_FOLD_RECONSTRUCTION_MULT).abs() < 1e-9);
    }

    /// In the linear region the scale is `C · |A(−1)| · g` — the oracle
    /// sweep's measured proportionality.
    #[test]
    fn shaped_scale_is_linear_below_the_knee() {
        // |A(−1)| = 1 − 0.5 = 0.5 (single tap +0.5 at i = 0 → term −0.5
        // ... A(−1) = 1 − 0.5·(−1) = 1.5). Use a value below the knee:
        let lpc = [-0.4f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let resp = hb_crossover_response(&lpc); // 1 + (−0.4)·... = 0.6
        assert!((resp - 0.6).abs() < 1e-12);
        let k = folded_hb_scale(3.0, &lpc);
        assert!((k - HB_FOLD_CROSSOVER_SLOPE * 0.6 * 3.0).abs() < 1e-12);
        assert!(k < HB_FOLD_RECONSTRUCTION_MULT);
    }

    /// The shaped sub-frame reconstruction keeps the `(−1)ⁿ` fold and
    /// applies the shaped scale sample-for-sample.
    #[test]
    fn shaped_subframe_matches_scale_and_fold() {
        let mut exc = [0.0f32; HB_SUBFRAME_SAMPLES];
        for (i, e) in exc.iter_mut().enumerate() {
            *e = (i as f32) - 20.0;
        }
        let lpc = [-0.4f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let g = 1.5f32;
        let out = folded_hb_excitation_subframe_shaped(&exc, g, &lpc);
        let k = folded_hb_scale(g, &lpc);
        for (n, (&o, &e)) in out.iter().zip(exc.iter()).enumerate() {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!((o - k * sign * f64::from(e)).abs() < 1e-9, "n={n}");
        }
    }
}
