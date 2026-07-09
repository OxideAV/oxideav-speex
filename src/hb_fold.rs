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
//! ## Scope
//!
//! * The **wideband** decoder applies this law with the embedded
//!   narrowband frame's excitation as the fold source (same 8 kHz
//!   half-band geometry, sub-frame for sub-frame) — pinned by the
//!   fixture.
//! * The **ultra-wideband** second layer reuses the same law; its fold
//!   *source* at the 16 kHz half-band geometry is not covered by the
//!   (wideband) fixture — see [`crate::uwb_decoder`] for the
//!   generalisation this crate applies there and the recorded gap.

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
}
