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
    /// This is the fixed-length wideband convenience over
    /// [`Self::reconstruct_slices`], which accepts any half-band frame
    /// length ≥ [`QMF_FILTER_LEN`] (the ultra-wideband outer filterbank
    /// runs the same structure at 320-sample half-band frames).
    pub fn reconstruct_frame(
        &mut self,
        low_band: &[f64; QMF_HALF_BAND_FRAME],
        high_band: &[f64; QMF_HALF_BAND_FRAME],
    ) -> [f64; QMF_WIDEBAND_FRAME] {
        let mut out = [0.0f64; QMF_WIDEBAND_FRAME];
        self.reconstruct_slices(low_band, high_band, &mut out);
        out
    }

    /// Reconstruct one full-rate frame from two half-band frames of
    /// **any** common length — the slice-generic core of the two-band
    /// synthesis bank.
    ///
    /// The two-band mirror recombination is independent of the frame
    /// length (the carried state is only the `QMF_FILTER_LEN`-sample tail
    /// of each band), so the same bank serves the wideband inner
    /// filterbank (160-sample half-bands → 320-sample 16 kHz frames) and
    /// the ultra-wideband **outer** filterbank (320-sample 16 kHz
    /// half-bands → 640-sample 32 kHz frames) — §2.2's *"Embedded
    /// wideband structure (scalable sampling rate)"* repeats the same QMF
    /// split one level up.
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
    ///
    /// # Panics
    ///
    /// If `low_band.len() != high_band.len()`, if
    /// `out.len() != 2 * low_band.len()`, or if the half-band frame is
    /// shorter than [`QMF_FILTER_LEN`] (the history roll requires a full
    /// filter window per frame).
    pub fn reconstruct_slices(&mut self, low_band: &[f64], high_band: &[f64], out: &mut [f64]) {
        let n = low_band.len();
        assert_eq!(n, high_band.len(), "half-band frames must match");
        assert_eq!(out.len(), 2 * n, "output must be twice the half-band");
        assert!(n >= QMF_FILTER_LEN, "half-band frame shorter than filter");

        let h0 = qmf_h0_float();
        // Polyphase tap count: even/odd indices of the 64-tap prototype.
        // L = QMF_FILTER_LEN is even, so each phase has L/2 taps.
        let half = QMF_FILTER_LEN / 2;

        for i in 0..n {
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
                let lb = Self::band_sample(low_band, &self.lb_hist, i, j);
                let hb = Self::band_sample(high_band, &self.hb_hist, i, j);

                // g0 = 2·h0 (lowpass, no sign modulation).
                // g1 even phase: -2·(+1)·h0_even ; odd phase: -2·(-1)·h0_odd.
                y_even += 2.0 * h_even * lb - 2.0 * h_even * hb;
                y_odd += 2.0 * h_odd * lb + 2.0 * h_odd * hb;
            }
            out[2 * i] = y_even;
            out[2 * i + 1] = y_odd;
        }

        // Advance the histories: append this frame's tail.
        Self::roll(&mut self.lb_hist, low_band);
        Self::roll(&mut self.hb_hist, high_band);
    }

    /// Fetch band sample `s[i - j]`, falling back to the persisted
    /// history tail for negative absolute indices. `hist` holds the
    /// previous frame samples most-recent-last (index `L-1` is the
    /// sample immediately preceding `s[0]`).
    #[inline]
    fn band_sample(frame: &[f64], hist: &[f64; QMF_FILTER_LEN], i: usize, j: usize) -> f64 {
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

    fn roll(hist: &mut [f64; QMF_FILTER_LEN], frame: &[f64]) {
        // The frame is at least as long as the history window (asserted
        // by the callers), so the new history is simply the frame's last
        // QMF_FILTER_LEN samples.
        let start = frame.len() - QMF_FILTER_LEN;
        hist.copy_from_slice(&frame[start..]);
    }
}

/// Two-band QMF **analysis** filterbank with cross-frame history — the
/// encode-direction inverse of [`QmfSynthesis`] (round r385 scope).
///
/// Splits one 320-sample 16 kHz wideband frame into the two 8 kHz
/// half-band signals the sub-band CELP encoder consumes: the low band
/// (0–4 kHz) and the high band (4–8 kHz folded to 0–4 kHz, §10.2's
/// frequency-axis reversal — intrinsic to the `(-1)ⁿ` mirror
/// modulation, exactly as in the synthesis direction).
///
/// ## Clean-room basis
///
/// Same category as [`QmfSynthesis`]: *The Speex Codec Manual* §10 pins
/// the structure (*"the Speex approach uses a quadrature mirror filter
/// (QMF) to split the band in two"*) and the 64-tap prototype `h0` is
/// staged as pure data; the analysis relations are the classical
/// two-band Croisier–Esteban–Galand construction (textbook multirate
/// DSP):
///
/// ```text
///   lb[i] = Σ_k h0[k] · x[2i − k]           (lowpass, decimate by 2)
///   hb[i] = Σ_k (−1)^k h0[k] · x[2i − k]    (mirror highpass, decimate by 2)
/// ```
///
/// The r365 perfect-reconstruction test pins this analysis against the
/// synthesis bank: analysis → synthesis recovers the input up to the
/// filterbank group delay.
///
/// ## State
///
/// The analysis filters are FIR; the carried state is the tail of the
/// full-rate input history (the `QMF_FILTER_LEN` most recent 16 kHz
/// samples) so a continuous stream splits without frame-boundary
/// discontinuities.
#[derive(Debug, Clone)]
pub struct QmfAnalysis {
    /// Previous full-rate input samples (most-recent-last).
    input_hist: [f64; QMF_FILTER_LEN],
}

impl Default for QmfAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl QmfAnalysis {
    /// A fresh analysis filterbank with zeroed history (stream start).
    pub fn new() -> Self {
        Self {
            input_hist: [0.0; QMF_FILTER_LEN],
        }
    }

    /// Split one 320-sample 16 kHz wideband frame into its two
    /// 160-sample 8 kHz half-band frames `(low_band, high_band)`.
    ///
    /// The input history advances so the next call continues seamlessly.
    /// This is the fixed-length wideband convenience over
    /// [`Self::split_slices`], which accepts any even full-rate frame
    /// length ≥ `2 ×` [`QMF_FILTER_LEN`] (the ultra-wideband outer
    /// filterbank runs the same split on 640-sample 32 kHz frames).
    pub fn split_frame(
        &mut self,
        input: &[f64; QMF_WIDEBAND_FRAME],
    ) -> ([f64; QMF_HALF_BAND_FRAME], [f64; QMF_HALF_BAND_FRAME]) {
        let mut lb = [0.0f64; QMF_HALF_BAND_FRAME];
        let mut hb = [0.0f64; QMF_HALF_BAND_FRAME];
        self.split_slices(input, &mut lb, &mut hb);
        (lb, hb)
    }

    /// Split one full-rate frame of **any** even length into its two
    /// half-band frames — the slice-generic core of the two-band
    /// analysis bank.
    ///
    /// The mirror split is independent of the frame length (the carried
    /// state is only the `QMF_FILTER_LEN`-sample tail of the full-rate
    /// input), so the same bank serves the wideband inner filterbank
    /// (320-sample 16 kHz frames → 160-sample half-bands) and the
    /// ultra-wideband **outer** filterbank (640-sample 32 kHz frames →
    /// 320-sample 16 kHz half-bands) — the §2.2 embedded structure
    /// repeats the same QMF split one level up.
    ///
    /// # Panics
    ///
    /// If `low_band.len() != high_band.len()`, if
    /// `input.len() != 2 * low_band.len()`, or if `input` is shorter
    /// than [`QMF_FILTER_LEN`] (the history roll requires a full filter
    /// window per frame).
    pub fn split_slices(&mut self, input: &[f64], low_band: &mut [f64], high_band: &mut [f64]) {
        let n = low_band.len();
        assert_eq!(n, high_band.len(), "half-band frames must match");
        assert_eq!(input.len(), 2 * n, "input must be twice the half-band");
        assert!(
            input.len() >= QMF_FILTER_LEN,
            "full-rate frame shorter than filter"
        );

        let h0 = qmf_h0_float();
        for i in 0..n {
            let m = 2 * i; // full-rate output position kept by the decimator
            let mut acc_l = 0.0f64;
            let mut acc_h = 0.0f64;
            for (k, &h) in h0.iter().enumerate() {
                let x = self.input_sample(input, m, k);
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                acc_l += h * x;
                acc_h += sign * h * x;
            }
            low_band[i] = acc_l;
            high_band[i] = acc_h;
        }

        // Advance the history: the frame is at least as long as the
        // window (asserted above), so the new history is the frame's
        // last QMF_FILTER_LEN samples.
        let start = input.len() - QMF_FILTER_LEN;
        self.input_hist.copy_from_slice(&input[start..]);
    }

    /// Fetch full-rate input sample `x[m − k]`, drawing on the persisted
    /// history for negative absolute positions (`hist[L−1]` is position
    /// −1, `hist[L−2]` is −2, …).
    #[inline]
    fn input_sample(&self, frame: &[f64], m: usize, k: usize) -> f64 {
        if k <= m {
            frame[m - k]
        } else {
            let back = k - m; // ≥ 1
            if back <= QMF_FILTER_LEN {
                self.input_hist[QMF_FILTER_LEN - back]
            } else {
                0.0
            }
        }
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

    /// Two-band QMF **analysis** reference (test-only; the decoder needs
    /// only synthesis). Splits a 16 kHz signal `x` into the two 8 kHz
    /// half-bands: `lb = downsample2(h0 * x)`, `hb = downsample2(h1 * x)`
    /// with `h1[n] = (-1)^n h0[n]`. `x` is indexed with zero extension for
    /// negative indices (start-of-stream).
    fn analysis(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let h0 = qmf_h0_float();
        let n = x.len();
        let mut lb = Vec::with_capacity(n / 2);
        let mut hb = Vec::with_capacity(n / 2);
        // Keep every other full-rate output sample (decimate by 2).
        let mut m = 0;
        while m < n {
            let mut acc_l = 0.0f64;
            let mut acc_h = 0.0f64;
            for (k, &h) in h0.iter().enumerate() {
                if k > m {
                    break;
                }
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                acc_l += h * x[m - k];
                acc_h += sign * h * x[m - k];
            }
            lb.push(acc_l);
            hb.push(acc_h);
            m += 2;
        }
        (lb, hb)
    }

    /// **Perfect-reconstruction** pin: a 16 kHz full-band signal pushed
    /// through QMF analysis (split → 2 half-bands) and back through the
    /// synthesis filterbank recovers the **original signal up to the
    /// filterbank group delay** in the steady-state region. This is the
    /// decisive sample-correctness check for the two-band QMF: the
    /// alias-cancellation + amplitude-distortion-free property of the
    /// CEG mirror relation must hold for the staged prototype.
    #[test]
    fn analysis_synthesis_perfect_reconstruction() {
        // Build a multi-frame 16 kHz test signal with both low- and
        // high-frequency content so both bands are exercised.
        let total = 4 * QMF_WIDEBAND_FRAME;
        let mut x = vec![0.0f64; total];
        for (i, s) in x.iter_mut().enumerate() {
            let t = i as f64;
            *s = (t * 0.05).sin() + 0.5 * (t * 0.9).sin() + 0.3 * (t * 1.7).cos();
        }
        // Analysis → two half-bands (length total/2 each), through the
        // public streaming filterbank (pinned identical to the direct
        // whole-signal reference in
        // `streaming_analysis_matches_direct_reference`).
        let mut qa = QmfAnalysis::new();
        let mut lb = Vec::with_capacity(total / 2);
        let mut hb = Vec::with_capacity(total / 2);
        for f in 0..(total / QMF_WIDEBAND_FRAME) {
            let mut frame = [0.0f64; QMF_WIDEBAND_FRAME];
            frame.copy_from_slice(&x[f * QMF_WIDEBAND_FRAME..(f + 1) * QMF_WIDEBAND_FRAME]);
            let (lbf, hbf) = qa.split_frame(&frame);
            lb.extend_from_slice(&lbf);
            hb.extend_from_slice(&hbf);
        }

        // Synthesis frame-by-frame through the stateful bank.
        let mut q = QmfSynthesis::new();
        let mut y = vec![0.0f64; total];
        let frames = (lb.len()) / QMF_HALF_BAND_FRAME;
        for f in 0..frames {
            let mut lbf = [0.0f64; QMF_HALF_BAND_FRAME];
            let mut hbf = [0.0f64; QMF_HALF_BAND_FRAME];
            lbf.copy_from_slice(&lb[f * QMF_HALF_BAND_FRAME..(f + 1) * QMF_HALF_BAND_FRAME]);
            hbf.copy_from_slice(&hb[f * QMF_HALF_BAND_FRAME..(f + 1) * QMF_HALF_BAND_FRAME]);
            let out = q.reconstruct_frame(&lbf, &hbf);
            y[f * QMF_WIDEBAND_FRAME..(f + 1) * QMF_WIDEBAND_FRAME].copy_from_slice(&out);
        }

        // The analysis-synthesis cascade delays by ≈ QMF_FILTER_LEN-1
        // samples. Search the small delay window for the offset that best
        // aligns y to x, then assert the steady-state error is tiny.
        let mut best_delay = 0usize;
        let mut best_err = f64::INFINITY;
        for d in 0..QMF_FILTER_LEN {
            let mut err = 0.0f64;
            let lo = QMF_FILTER_LEN;
            let hi = total - QMF_FILTER_LEN;
            for i in lo..hi {
                if i + d < total {
                    let e = y[i + d] - x[i];
                    err += e * e;
                }
            }
            if err < best_err {
                best_err = err;
                best_delay = d;
            }
        }
        // Energy of the reference in the same window for a relative bound.
        let lo = QMF_FILTER_LEN;
        let hi = total - QMF_FILTER_LEN;
        let ref_energy: f64 = x[lo..hi].iter().map(|&s| s * s).sum();
        let rel = best_err / ref_energy;
        assert!(
            rel < 1e-3,
            "QMF perfect reconstruction failed: rel err {rel} at delay {best_delay}"
        );
    }

    /// The public streaming [`QmfAnalysis`] equals the direct
    /// whole-signal analysis reference sample-for-sample across multiple
    /// frames, pinning the streaming history handling exact.
    #[test]
    fn streaming_analysis_matches_direct_reference() {
        let total = 3 * QMF_WIDEBAND_FRAME;
        let mut x = vec![0.0f64; total];
        for (i, s) in x.iter_mut().enumerate() {
            *s = ((i * 11 % 29) as f64 - 14.0) * 0.07 + (i as f64 * 0.31).sin();
        }
        let (want_lb, want_hb) = analysis(&x);

        let mut qa = QmfAnalysis::new();
        let mut got_lb = Vec::new();
        let mut got_hb = Vec::new();
        for f in 0..(total / QMF_WIDEBAND_FRAME) {
            let mut frame = [0.0f64; QMF_WIDEBAND_FRAME];
            frame.copy_from_slice(&x[f * QMF_WIDEBAND_FRAME..(f + 1) * QMF_WIDEBAND_FRAME]);
            let (lbf, hbf) = qa.split_frame(&frame);
            got_lb.extend_from_slice(&lbf);
            got_hb.extend_from_slice(&hbf);
        }

        assert_eq!(got_lb.len(), want_lb.len());
        for (i, (g, w)) in got_lb.iter().zip(want_lb.iter()).enumerate() {
            assert!((g - w).abs() < 1e-12, "lb sample {i}: {g} vs {w}");
        }
        for (i, (g, w)) in got_hb.iter().zip(want_hb.iter()).enumerate() {
            assert!((g - w).abs() < 1e-12, "hb sample {i}: {g} vs {w}");
        }
    }

    /// A silent input splits to two silent half-bands and leaves the
    /// history zero.
    #[test]
    fn analysis_of_silence_is_silent() {
        let mut qa = QmfAnalysis::new();
        let (lb, hb) = qa.split_frame(&[0.0; QMF_WIDEBAND_FRAME]);
        assert!(lb.iter().all(|&v| v == 0.0));
        assert!(hb.iter().all(|&v| v == 0.0));
    }

    /// A pure low-frequency 16 kHz input lands (essentially) all its
    /// energy in the low band; a near-Nyquist input lands it in the high
    /// band — the band-split direction pin.
    #[test]
    fn analysis_separates_low_and_high_frequencies() {
        let mut frame_lo = [0.0f64; QMF_WIDEBAND_FRAME];
        let mut frame_hi = [0.0f64; QMF_WIDEBAND_FRAME];
        for i in 0..QMF_WIDEBAND_FRAME {
            // ~250 Hz at 16 kHz — deep inside the low band.
            frame_lo[i] = (2.0 * std::f64::consts::PI * i as f64 * 250.0 / 16_000.0).sin();
            // ~7 kHz at 16 kHz — deep inside the high band.
            frame_hi[i] = (2.0 * std::f64::consts::PI * i as f64 * 7_000.0 / 16_000.0).sin();
        }
        let energy = |band: &[f64]| band.iter().map(|&v| v * v).sum::<f64>();

        // Run two frames so the second is past the FIR transient.
        let mut qa = QmfAnalysis::new();
        let _ = qa.split_frame(&frame_lo);
        let (lb, hb) = qa.split_frame(&frame_lo);
        assert!(
            energy(&lb) > 50.0 * energy(&hb),
            "low tone: lb energy {} should dominate hb {}",
            energy(&lb),
            energy(&hb)
        );

        let mut qa = QmfAnalysis::new();
        let _ = qa.split_frame(&frame_hi);
        let (lb, hb) = qa.split_frame(&frame_hi);
        assert!(
            energy(&hb) > 50.0 * energy(&lb),
            "high tone: hb energy {} should dominate lb {}",
            energy(&hb),
            energy(&lb)
        );
    }

    /// Analysis history carries across frames: the continued split of a
    /// second frame differs from a fresh-bank split of the same frame.
    #[test]
    fn analysis_history_carries_across_frames() {
        let mut frame = [0.0f64; QMF_WIDEBAND_FRAME];
        for (i, s) in frame.iter_mut().enumerate() {
            *s = (i as f64 * 0.13).cos();
        }
        let mut cont = QmfAnalysis::new();
        let _ = cont.split_frame(&frame);
        let (lb_cont, _) = cont.split_frame(&frame);

        let mut fresh = QmfAnalysis::new();
        let (lb_fresh, _) = fresh.split_frame(&frame);

        assert!(
            lb_cont
                .iter()
                .zip(lb_fresh.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "history should influence the continued frame's split"
        );
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

    /// The slice-generic entries at the wideband geometry are exactly the
    /// fixed-length array paths — same filter, same history handling.
    #[test]
    fn slice_paths_match_array_paths_at_wideband_geometry() {
        let mut x = [0.0f64; QMF_WIDEBAND_FRAME];
        for (i, s) in x.iter_mut().enumerate() {
            *s = (i as f64 * 0.17).sin() + 0.4 * (i as f64 * 1.3).cos();
        }
        // Analysis: array vs slice over two frames (history exercised).
        let mut qa_arr = QmfAnalysis::new();
        let mut qa_sl = QmfAnalysis::new();
        for _ in 0..2 {
            let (lb_a, hb_a) = qa_arr.split_frame(&x);
            let mut lb_s = [0.0f64; QMF_HALF_BAND_FRAME];
            let mut hb_s = [0.0f64; QMF_HALF_BAND_FRAME];
            qa_sl.split_slices(&x, &mut lb_s, &mut hb_s);
            assert_eq!(lb_a, lb_s);
            assert_eq!(hb_a, hb_s);

            // Synthesis: array vs slice on the same half-bands.
            let mut qs_arr = QmfSynthesis::new();
            let mut qs_sl = QmfSynthesis::new();
            let y_a = qs_arr.reconstruct_frame(&lb_a, &hb_a);
            let mut y_s = [0.0f64; QMF_WIDEBAND_FRAME];
            qs_sl.reconstruct_slices(&lb_s, &hb_s, &mut y_s);
            assert_eq!(y_a, y_s);
        }
    }

    /// Perfect reconstruction at the **ultra-wideband outer geometry**:
    /// a 32 kHz-rate signal in 640-sample frames split to 320-sample
    /// half-bands and recombined recovers the input up to the filterbank
    /// group delay — the same CEG property, one level up the §2.2
    /// embedded ladder.
    #[test]
    fn slice_analysis_synthesis_perfect_reconstruction_at_uwb_geometry() {
        const FULL: usize = 640;
        const HALF: usize = FULL / 2;
        let total = 4 * FULL;
        let mut x = vec![0.0f64; total];
        for (i, s) in x.iter_mut().enumerate() {
            let t = i as f64;
            *s = (t * 0.03).sin() + 0.5 * (t * 0.7).sin() + 0.3 * (t * 1.9).cos();
        }
        let mut qa = QmfAnalysis::new();
        let mut lb = vec![0.0f64; total / 2];
        let mut hb = vec![0.0f64; total / 2];
        for f in 0..(total / FULL) {
            let (lo, hi) = (f * FULL, (f + 1) * FULL);
            let (blo, bhi) = (f * HALF, (f + 1) * HALF);
            qa.split_slices(&x[lo..hi], &mut lb[blo..bhi], &mut hb[blo..bhi]);
        }
        let mut qs = QmfSynthesis::new();
        let mut y = vec![0.0f64; total];
        for f in 0..(total / FULL) {
            let (lo, hi) = (f * FULL, (f + 1) * FULL);
            let (blo, bhi) = (f * HALF, (f + 1) * HALF);
            qs.reconstruct_slices(&lb[blo..bhi], &hb[blo..bhi], &mut y[lo..hi]);
        }

        // Find the aligning delay, then bound the steady-state error.
        let mut best_err = f64::INFINITY;
        for d in 0..QMF_FILTER_LEN {
            let mut err = 0.0f64;
            for i in QMF_FILTER_LEN..(total - QMF_FILTER_LEN) {
                if i + d < total {
                    let e = y[i + d] - x[i];
                    err += e * e;
                }
            }
            best_err = best_err.min(err);
        }
        let ref_energy: f64 = x[QMF_FILTER_LEN..total - QMF_FILTER_LEN]
            .iter()
            .map(|&s| s * s)
            .sum();
        assert!(
            best_err / ref_energy < 1e-3,
            "outer-geometry PR failed: rel err {}",
            best_err / ref_energy
        );
    }

    /// The slice entries reject mismatched geometry loudly.
    #[test]
    #[should_panic(expected = "output must be twice the half-band")]
    fn reconstruct_slices_rejects_bad_output_length() {
        let mut q = QmfSynthesis::new();
        let lb = [0.0f64; 128];
        let hb = [0.0f64; 128];
        let mut out = [0.0f64; 200]; // not 256
        q.reconstruct_slices(&lb, &hb, &mut out);
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
