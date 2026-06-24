//! Two-band **QMF synthesis filterbank** — the final wideband
//! recombination of the low-band (0–4 kHz) and high-band (4–8 kHz)
//! 8 kHz half-band signals into a single 16 kHz wideband PCM stream
//! (round r365 scope).
//!
//! ## What the staged material pins
//!
//! *The Speex Codec Manual* §10 states the structure: *"the Speex
//! approach uses a quadrature mirror filter (QMF) to split the band in
//! two. The 16 kHz signal is thus divided into two 8 kHz signals, one
//! representing the low band (0-4 kHz), the other the high band
//! (4-8 kHz)."* §10.2 adds that *"the QMF folds the 4-8 kHz band into
//! 4-0 kHz (reversing the frequency axis)"*. The 64-tap QMF **prototype
//! lowpass filter `h0`** is staged as pure data
//! ([`crate::qmf_h0_float`] / [`crate::qmf_h0_q15`]).
//!
//! ## Clean-room basis for the synthesis structure
//!
//! Given the prototype `h0` and the manual's two-band split structure,
//! the *reconstruction* is the **classical two-band quadrature-mirror
//! filterbank** (Croisier–Esteban–Galand, 1976) — a standard multirate
//! DSP construction covered in any signal-processing textbook, exactly
//! the same clean-room category the Speex LSP→LPC trace doc
//! (`docs/audio/speex/gain-quantiser-and-lsp-lpc-trace.md` §5) grants
//! for the LSP polynomial reconstruction: *"a textbook DSP procedure …
//! no codec-specific tuning required."* No Speex source was consulted to
//! derive it.
//!
//! For a real prototype lowpass `h0[n]` of even length `L`, the
//! two-band QMF relations are:
//!
//! ```text
//!   analysis lowpass     h0[n]
//!   analysis highpass    h1[n] = (-1)^n · h0[n]
//!   synthesis lowpass    g0[n] = 2 · h0[n]
//!   synthesis highpass   g1[n] = -2 · h1[n] = -2 · (-1)^n · h0[n]
//! ```
//!
//! The synthesis stage upsamples each 8 kHz half-band by 2 (zero-stuff),
//! filters the low band by `g0` and the high band by `g1`, and sums:
//!
//! ```text
//!   y[m] = Σ_k g0[k]·u_lb[m-k]  +  Σ_k g1[k]·u_hb[m-k]
//! ```
//!
//! where `u_lb` / `u_hb` are the 2× zero-stuffed half-bands. With the
//! CEG mirror relation `H1(z) = H0(-z)` and `G0 = 2H0`, `G1 = -2H1`, the
//! aliasing terms introduced by the 2× decimation/interpolation cancel,
//! reconstructing the full 16 kHz band. The factor-2 synthesis gain
//! compensates the upsampler's energy split.
//!
//! ### Polyphase implementation
//!
//! Zero-stuffing then convolving by a 64-tap filter does redundant
//! work (half the products multiply a stuffed zero). The standard
//! **polyphase** form computes the two output phases of each 16 kHz
//! sample pair directly from the 8 kHz half-band samples, halving the
//! filter length per phase. This module implements the polyphase form;
//! [`QmfSynthesis::reconstruct_frame`] over a constant-zero high band
//! is unit-tested to equal the equivalent direct upsample-filter-sum, so
//! the two forms are pinned identical.
//!
//! ## Frequency folding of the high band
//!
//! Per §10.2 the high-band 8 kHz signal carries the 4–8 kHz content with
//! the frequency axis **reversed** (the QMF analysis folds 4–8 kHz down
//! to 0–4 kHz). The synthesis highpass `g1` maps it back up to 4–8 kHz;
//! the `(-1)^n` modulation in `g1` performs exactly that spectral
//! reflection, so no separate axis-flip of the half-band samples is
//! required — it is intrinsic to the mirror-filter synthesis.
//!
//! ## State
//!
//! The synthesis filters are FIR, so the only state carried across
//! frames is the **tail of the two half-band input histories** (the
//! `L-1` most recent samples of each band needed to start the next
//! frame's convolution). [`QmfSynthesis`] holds those two ring tails so
//! a stream of wideband frames reconstructs continuously with no
//! inter-frame discontinuity.

use crate::codebooks::{qmf_h0_float, QMF_FILTER_LEN};

/// Number of 8 kHz half-band samples per wideband frame (one band).
pub const QMF_HALF_BAND_FRAME: usize = crate::wb_synthesis::HB_FRAME_SAMPLES;

/// Number of 16 kHz wideband output samples per frame — twice the
/// half-band frame length (the 2× synthesis interpolation).
pub const QMF_WIDEBAND_FRAME: usize = 2 * QMF_HALF_BAND_FRAME;

/// Two-band QMF synthesis filterbank with cross-frame history.
///
/// Recombines the per-frame low-band (`x_lb`) + high-band (`x_hb`)
/// 8 kHz half-band signals into a 16 kHz wideband frame. The FIR
/// histories of both bands persist across [`reconstruct_frame`] calls so
/// a continuous wideband stream reconstructs without boundary artifacts.
///
/// [`reconstruct_frame`]: QmfSynthesis::reconstruct_frame
#[derive(Debug, Clone)]
pub struct QmfSynthesis {
    /// Previous low-band samples (most-recent-last), length `QMF_FILTER_LEN`.
    lb_hist: [f64; QMF_FILTER_LEN],
    /// Previous high-band samples (most-recent-last), length `QMF_FILTER_LEN`.
    hb_hist: [f64; QMF_FILTER_LEN],
}

impl Default for QmfSynthesis {
    fn default() -> Self {
        Self::new()
    }
}

impl QmfSynthesis {
    /// A fresh synthesis filterbank with zeroed histories (stream start).
    pub fn new() -> Self {
        Self {
            lb_hist: [0.0; QMF_FILTER_LEN],
            hb_hist: [0.0; QMF_FILTER_LEN],
        }
    }

    /// Reconstruct one 16 kHz wideband frame from the two 8 kHz
    /// half-band frames.
    ///
    /// `low_band` is the narrowband-synthesised 0–4 kHz signal; `high_band`
    /// is the §10 high-band-synthesised (frequency-folded) 4–8 kHz signal.
    /// Both are one wideband frame long ([`QMF_HALF_BAND_FRAME`]). Returns
    /// the [`QMF_WIDEBAND_FRAME`]-sample 16 kHz output; the band histories
    /// are advanced so the next call continues seamlessly.
    ///
    /// ## Polyphase reconstruction
    ///
    /// For each input pair the two output phases are
    ///
    /// ```text
    ///   y[2i]   = Σ_j g0_even[j]·lb[i-j] + g1_even[j]·hb[i-j]
    ///   y[2i+1] = Σ_j g0_odd[j] ·lb[i-j] + g1_odd[j] ·hb[i-j]
    /// ```
    ///
    /// where `g0_even`/`g0_odd` are the even/odd polyphase components of
    /// `g0 = 2·h0` and `g1_even`/`g1_odd` of `g1 = -2·(-1)^n·h0`. The
    /// `(-1)^n` sign collapses into the polyphase split: on the even
    /// phase `(-1)^{2k} = +1`, on the odd phase `(-1)^{2k+1} = -1`, so
    /// the high-band even/odd polyphase taps are `+2·h0_even` /
    /// `-2·h0_odd` (up to the shared `-2` synthesis gain folded below).
    pub fn reconstruct_frame(
        &mut self,
        low_band: &[f64; QMF_HALF_BAND_FRAME],
        high_band: &[f64; QMF_HALF_BAND_FRAME],
    ) -> [f64; QMF_WIDEBAND_FRAME] {
        let h0 = qmf_h0_float();
        // Polyphase tap count: even/odd indices of the 64-tap prototype.
        // L = QMF_FILTER_LEN is even, so each phase has L/2 taps.
        let half = QMF_FILTER_LEN / 2;

        let mut out = [0.0f64; QMF_WIDEBAND_FRAME];
        for i in 0..QMF_HALF_BAND_FRAME {
            let mut y_even = 0.0f64;
            let mut y_odd = 0.0f64;
            // Convolve the two polyphase phases against the band samples
            // ending at input index `i` (drawing on history for j>i).
            for j in 0..half {
                // Synthesis filter g0 = 2·h0, g1 = -2·(-1)^n·h0.
                // Even output phase uses prototype even taps h0[2j];
                // odd output phase uses prototype odd taps h0[2j+1].
                let h_even = h0[2 * j];
                let h_odd = h0[2 * j + 1];
                let lb = self.band_sample(low_band, &self.lb_hist, i, j);
                let hb = self.band_sample(high_band, &self.hb_hist, i, j);

                // g0 = 2·h0 (lowpass, no sign modulation).
                // g1 even phase: -2·(+1)·h0_even ; odd phase: -2·(-1)·h0_odd.
                y_even += 2.0 * h_even * lb - 2.0 * h_even * hb;
                y_odd += 2.0 * h_odd * lb + 2.0 * h_odd * hb;
            }
            out[2 * i] = y_even;
            out[2 * i + 1] = y_odd;
        }

        // Advance the histories: append this frame's tail.
        self.push_history(low_band, high_band);
        out
    }

    /// Fetch band sample `s[i - j]`, falling back to the persisted
    /// history tail for negative absolute indices. `hist` holds the
    /// previous frame samples most-recent-last (index `L-1` is the
    /// sample immediately preceding `s[0]`).
    #[inline]
    fn band_sample(
        &self,
        frame: &[f64; QMF_HALF_BAND_FRAME],
        hist: &[f64; QMF_FILTER_LEN],
        i: usize,
        j: usize,
    ) -> f64 {
        if j <= i {
            frame[i - j]
        } else {
            // Need sample at absolute position (i - j) < 0.
            // hist[L-1] is position -1, hist[L-2] is -2, …
            let back = j - i; // ≥ 1
            if back <= QMF_FILTER_LEN {
                hist[QMF_FILTER_LEN - back]
            } else {
                0.0
            }
        }
    }

    /// Roll this frame's last `QMF_FILTER_LEN` samples of each band into
    /// the history tails (most-recent-last) for the next frame.
    fn push_history(
        &mut self,
        low_band: &[f64; QMF_HALF_BAND_FRAME],
        high_band: &[f64; QMF_HALF_BAND_FRAME],
    ) {
        Self::roll(&mut self.lb_hist, low_band);
        Self::roll(&mut self.hb_hist, high_band);
    }

    fn roll(hist: &mut [f64; QMF_FILTER_LEN], frame: &[f64; QMF_HALF_BAND_FRAME]) {
        // The frame is longer than the history window, so the new history
        // is simply the frame's last QMF_FILTER_LEN samples.
        let start = QMF_HALF_BAND_FRAME - QMF_FILTER_LEN;
        hist.copy_from_slice(&frame[start..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zeros() -> [f64; QMF_HALF_BAND_FRAME] {
        [0.0; QMF_HALF_BAND_FRAME]
    }

    /// Direct (non-polyphase) reference: zero-stuff each band by 2, FIR
    /// filter by g0 / g1, and sum. Used to pin the polyphase output.
    fn reference_reconstruct(
        lb: &[f64; QMF_HALF_BAND_FRAME],
        hb: &[f64; QMF_HALF_BAND_FRAME],
    ) -> [f64; QMF_WIDEBAND_FRAME] {
        let h0 = qmf_h0_float();
        // Upsample by 2 (zero-stuff) into 16 kHz grids.
        let mut u_lb = [0.0f64; QMF_WIDEBAND_FRAME];
        let mut u_hb = [0.0f64; QMF_WIDEBAND_FRAME];
        for i in 0..QMF_HALF_BAND_FRAME {
            u_lb[2 * i] = lb[i];
            u_hb[2 * i] = hb[i];
        }
        let mut out = [0.0f64; QMF_WIDEBAND_FRAME];
        for m in 0..QMF_WIDEBAND_FRAME {
            let mut acc = 0.0f64;
            for (k, &h) in h0.iter().enumerate() {
                if k > m {
                    break;
                }
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                let g0 = 2.0 * h; // synthesis lowpass
                let g1 = -2.0 * sign * h; // synthesis highpass
                acc += g0 * u_lb[m - k] + g1 * u_hb[m - k];
            }
            out[m] = acc;
        }
        out
    }

    #[test]
    fn output_length_is_twice_half_band() {
        assert_eq!(QMF_WIDEBAND_FRAME, 2 * QMF_HALF_BAND_FRAME);
        assert_eq!(QMF_WIDEBAND_FRAME, 320);
    }

    #[test]
    fn silence_reconstructs_to_silence() {
        let mut q = QmfSynthesis::new();
        let out = q.reconstruct_frame(&zeros(), &zeros());
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Polyphase output equals the direct upsample-filter-sum reference
    /// on the first frame (zero history), pinning the two forms identical.
    #[test]
    fn polyphase_matches_direct_reference_first_frame() {
        let mut lb = zeros();
        let mut hb = zeros();
        // Deterministic pseudo-signal.
        for i in 0..QMF_HALF_BAND_FRAME {
            lb[i] = ((i * 7 % 13) as f64 - 6.0) * 0.1;
            hb[i] = ((i * 5 % 11) as f64 - 5.0) * 0.05;
        }
        let mut q = QmfSynthesis::new();
        let got = q.reconstruct_frame(&lb, &hb);
        let want = reference_reconstruct(&lb, &hb);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-12, "polyphase {g} vs direct {w}");
        }
    }

    /// A pure low-band input (high band silent) reconstructs a finite,
    /// non-silent 16 kHz signal — the low band passes through.
    #[test]
    fn low_band_only_passes_through() {
        let mut lb = zeros();
        for (i, s) in lb.iter_mut().enumerate() {
            *s = (i as f64 * 0.1).sin();
        }
        let mut q = QmfSynthesis::new();
        let out = q.reconstruct_frame(&lb, &zeros());
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|&v| v != 0.0));
    }

    /// History carries across frames: a second frame's leading samples
    /// see the first frame's tail (the FIR memory is live), so a
    /// two-frame run differs from re-running frame 2 from a fresh bank.
    #[test]
    fn history_carries_across_frames() {
        let mut lb = zeros();
        let mut hb = zeros();
        for i in 0..QMF_HALF_BAND_FRAME {
            lb[i] = (i as f64 * 0.07).cos();
            hb[i] = (i as f64 * 0.03).sin() * 0.3;
        }
        let mut q = QmfSynthesis::new();
        let _f0 = q.reconstruct_frame(&lb, &hb);
        let f1_cont = q.reconstruct_frame(&lb, &hb);

        let mut fresh = QmfSynthesis::new();
        let f1_fresh = fresh.reconstruct_frame(&lb, &hb);

        // The continued frame's early samples must differ from the
        // fresh-bank frame (non-zero history feeds the leading taps).
        assert!(
            f1_cont
                .iter()
                .zip(f1_fresh.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "history should influence the continued frame"
        );
    }

    /// DC / unity-passband gain pin: a constant low-band input
    /// reconstructs (after the FIR transient) to the **same constant** at
    /// 16 kHz. This holds because the staged prototype is normalised with
    /// `Σ h0 ≈ 1` and `Σ h0_even = Σ h0_odd ≈ 0.5`, so the factor-2
    /// synthesis gain gives `2·0.5·c = c` on both output phases. Protects
    /// the synthesis-gain normalisation against accidental change.
    #[test]
    fn constant_low_band_reconstructs_unity_passband() {
        let c = 0.37f64;
        let lb = [c; QMF_HALF_BAND_FRAME];
        let mut q = QmfSynthesis::new();
        // Run two frames so the second is past the 64-tap FIR transient.
        let _ = q.reconstruct_frame(&lb, &zeros());
        let out = q.reconstruct_frame(&lb, &zeros());
        // Steady-state samples (well past the transient) ≈ c.
        for &v in &out[QMF_FILTER_LEN..] {
            assert!((v - c).abs() < 1e-3, "steady-state {v} should ≈ {c}");
        }
    }

    /// The high-band synthesis highpass injects 4–8 kHz energy: a
    /// non-silent high band alone produces a finite, non-silent output
    /// distinct from the low-band-only case.
    #[test]
    fn high_band_only_is_finite_and_nonsilent() {
        let mut hb = zeros();
        for (i, s) in hb.iter_mut().enumerate() {
            *s = (i as f64 * 0.2).sin() * 0.5;
        }
        let mut q = QmfSynthesis::new();
        let out = q.reconstruct_frame(&zeros(), &hb);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|&v| v != 0.0));
    }
}
