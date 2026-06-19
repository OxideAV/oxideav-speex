//! High-band LPC synthesis filter (round r340 scope).
//!
//! The wideband decoder reconstructs the high-band (4–8 kHz, folded to
//! the 0–4 kHz half-band by the QMF) signal by running the high-band
//! excitation `e_hb[n]` through the order-8 all-pole synthesis filter
//! `1/A_hb(z)`. *The Speex Codec Manual* §10.1 states the high-band
//! linear prediction is *"very similar to what is done for narrowband"*
//! — the only difference is the LSP quantiser (12-bit MSVQ) and the LPC
//! order (8 for the high band vs 10 for narrowband). The synthesis
//! recurrence is therefore identical to the narrowband
//! [`crate::SynthesisFilter`] (§9.4 *"the synthesis filter
//! S(z) = 1/A(z)"*), only at order 8:
//!
//! ```text
//! x_hb[n] = e_hb[n] + Σ_{i=1}^{8} a_hb[i]·x_hb[n−i]
//! ```
//!
//! where `a_hb[i]` are the [`crate::lpc_from_hb_lsp_q10`]-produced
//! order-8 LPC coefficients of `A_hb(z) = 1 − a_hb[0]·z⁻¹ − … −
//! a_hb[7]·z⁻⁸` (the `−a` sign fold is already baked into the
//! coefficient vector, so the recurrence is a plain *add*).
//!
//! ## High-band excitation (no pitch term)
//!
//! Unlike the narrowband filter input (the §8.4 sum `e[n] = p[n] +
//! c[n]`), the high-band excitation is **just** the gain-scaled
//! innovation `e_hb[n] = c_hb[n]` — there is no high-band adaptive
//! codebook (§10.2 *"there's no pitch prediction for the high-band"*).
//! [`crate::gain_scaled_hb_innovation_subframe`] produces that input as
//! `[f32; 40]`; this filter consumes it directly.
//!
//! ## Filter state across sub-frames and frames
//!
//! Identical to the narrowband filter: the high-band synthesis is an
//! IIR recurrence whose `8`-sample memory carries straight through
//! sub-frame and frame boundaries. The high-band frame is four
//! 40-sample sub-frames (companion §1 / §5.1; the QMF gives each
//! half-band the same 8 kHz / 20 ms / 4-sub-frame cadence as
//! narrowband), each filtered with its own interpolated high-band LPC
//! set, and the history persists.
//!
//! ## Numeric domain
//!
//! High-band LPC coefficients are produced in `f64`
//! ([`crate::lpc_from_hb_lsp_q10`]). This module filters in `f64` to
//! avoid imposing a fixed-point Q-format the in-repo manual does not
//! specify for the synthesis recurrence (the same Q-format-agnostic
//! posture [`crate::synthesis`] adopts; the reference decoder's exact
//! `Q`-domains are part of the recorded LSP→LPC Q-format gap — see
//! [`crate::lsp_to_lpc`]). The `f64` output is the high-band 8 kHz
//! half-band signal that the QMF synthesis filterbank recombines with
//! the narrowband (low-band) signal into the 16 kHz wideband PCM.

use crate::codebooks::HB_LPC_ORDER;

/// All-pole order-8 high-band LPC synthesis filter `1/A_hb(z)` with
/// persistent history.
///
/// Construct with [`HbSynthesisFilter::new`] (zero history — the
/// stream-start convention, matching the all-zero high-band excitation
/// history the decoder begins from), feed the gain-scaled high-band
/// excitation through [`HbSynthesisFilter::process`], and the filter
/// carries its `8`-sample memory across calls. Structurally identical to
/// the narrowband [`crate::SynthesisFilter`], at the high-band order 8.
#[derive(Debug, Clone)]
pub struct HbSynthesisFilter {
    /// Last [`HB_LPC_ORDER`] synthesised samples, most-recent last.
    /// `history[HB_LPC_ORDER-1]` is `x[n−1]`, `history[0]` is
    /// `x[n−HB_LPC_ORDER]`.
    history: [f64; HB_LPC_ORDER],
}

impl Default for HbSynthesisFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl HbSynthesisFilter {
    /// A fresh filter with zero history (stream start).
    pub fn new() -> Self {
        Self {
            history: [0.0; HB_LPC_ORDER],
        }
    }

    /// Filter one high-band excitation block `e_hb[n]` through
    /// `1/A_hb(z)` using the order-8 LPC coefficients `a_hb` (the
    /// [`crate::lpc_from_hb_lsp_q10`] convention: `x[n] = e[n] + Σ
    /// a[i]·x[n−1−i]`), writing the synthesised samples into `out` and
    /// advancing the filter history.
    ///
    /// `excitation` and `out` must be the same length. The history
    /// carries across calls so successive sub-frames compose into a
    /// continuous IIR output.
    pub fn process(&mut self, lpc: &[f64; HB_LPC_ORDER], excitation: &[f64], out: &mut [f64]) {
        debug_assert_eq!(excitation.len(), out.len());
        for (slot, &e) in out.iter_mut().zip(excitation.iter()) {
            // x[n] = e[n] + Σ_{i=0}^{7} a[i]·x[n−1−i].
            let mut acc = e;
            for (i, &coeff) in lpc.iter().enumerate() {
                acc += coeff * self.history[HB_LPC_ORDER - 1 - i];
            }
            self.history.rotate_left(1);
            self.history[HB_LPC_ORDER - 1] = acc;
            *slot = acc;
        }
    }

    /// Convenience: allocate + filter one 40-sample high-band sub-frame,
    /// returning the synthesised high-band signal as `[f64; 40]`. The
    /// input is the gain-scaled high-band excitation
    /// (`[crate::gain_scaled_hb_innovation_subframe`]).
    pub fn process_subframe(
        &mut self,
        lpc: &[f64; HB_LPC_ORDER],
        excitation: &[f64; crate::hb_innovation::HB_SUBFRAME_SAMPLES],
    ) -> [f64; crate::hb_innovation::HB_SUBFRAME_SAMPLES] {
        let mut out = [0.0f64; crate::hb_innovation::HB_SUBFRAME_SAMPLES];
        self.process(lpc, excitation, &mut out);
        out
    }

    /// Read-only view of the current filter history (most-recent last).
    pub fn history(&self) -> &[f64; HB_LPC_ORDER] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_lpc_passes_excitation_through() {
        // A_hb(z) = 1 ⇒ 1/A_hb(z) = 1, output equals excitation.
        let mut f = HbSynthesisFilter::new();
        let lpc = [0.0_f64; HB_LPC_ORDER];
        let exc = [1.0_f64, -2.0, 3.0, -4.0, 5.0];
        let mut out = [0.0_f64; 5];
        f.process(&lpc, &exc, &mut out);
        assert_eq!(out, exc);
    }

    #[test]
    fn single_pole_geometric_response() {
        // A_hb(z) = 1 − a·z⁻¹ ⇒ one-pole filter; unit impulse gives the
        // geometric series 1, a, a², a³, … .
        let mut f = HbSynthesisFilter::new();
        let a = 0.5_f64;
        let mut lpc = [0.0_f64; HB_LPC_ORDER];
        lpc[0] = a;
        let exc = [1.0_f64, 0.0, 0.0, 0.0, 0.0];
        let mut out = [0.0_f64; 5];
        f.process(&lpc, &exc, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - a).abs() < 1e-12);
        assert!((out[2] - a * a).abs() < 1e-12);
        assert!((out[3] - a * a * a).abs() < 1e-12);
        assert!((out[4] - a.powi(4)).abs() < 1e-12);
    }

    #[test]
    fn history_carries_across_calls() {
        // Two successive process() calls equal one call over the
        // concatenated block — the IIR memory persists across sub-frame
        // boundaries.
        let a = 0.5_f64;
        let mut lpc = [0.0_f64; HB_LPC_ORDER];
        lpc[0] = a;
        let exc = [1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0];

        let mut whole = HbSynthesisFilter::new();
        let mut out_whole = [0.0_f64; 6];
        whole.process(&lpc, &exc, &mut out_whole);

        let mut split = HbSynthesisFilter::new();
        let mut out_a = [0.0_f64; 3];
        let mut out_b = [0.0_f64; 3];
        split.process(&lpc, &exc[0..3], &mut out_a);
        split.process(&lpc, &exc[3..6], &mut out_b);

        for i in 0..3 {
            assert!((out_whole[i] - out_a[i]).abs() < 1e-12);
            assert!((out_whole[3 + i] - out_b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn order_8_recurrence_uses_eighth_tap() {
        // An order-8 filter with only a[7] non-zero must reach back 8
        // samples. Feed 9 impulses; the 9th output should pick up the
        // a[7] feedback from output 0.
        let mut lpc = [0.0_f64; HB_LPC_ORDER];
        lpc[HB_LPC_ORDER - 1] = 0.5; // a[7], reaches x[n-8]
        let mut exc = [0.0_f64; 9];
        exc[0] = 1.0;
        let mut f = HbSynthesisFilter::new();
        let mut out = [0.0_f64; 9];
        f.process(&lpc, &exc, &mut out);
        // out[0] = 1. out[1..8] = 0 (no nearer taps).
        assert!((out[0] - 1.0).abs() < 1e-12);
        for v in &out[1..8] {
            assert!(v.abs() < 1e-12);
        }
        // out[8] = a[7]·x[0] = 0.5.
        assert!((out[8] - 0.5).abs() < 1e-12, "out[8] = {}", out[8]);
    }

    #[test]
    fn process_subframe_matches_process() {
        let mut lpc = [0.0_f64; HB_LPC_ORDER];
        lpc[0] = 0.3;
        lpc[1] = -0.1;
        let mut exc = [0.0_f64; crate::hb_innovation::HB_SUBFRAME_SAMPLES];
        for (i, e) in exc.iter_mut().enumerate() {
            *e = (i as f64) * 0.5 - 5.0;
        }
        let mut fa = HbSynthesisFilter::new();
        let out_sf = fa.process_subframe(&lpc, &exc);

        let mut fb = HbSynthesisFilter::new();
        let mut out_p = [0.0_f64; crate::hb_innovation::HB_SUBFRAME_SAMPLES];
        fb.process(&lpc, &exc, &mut out_p);
        assert_eq!(out_sf, out_p);
    }

    #[test]
    fn default_is_zero_history() {
        let f = HbSynthesisFilter::default();
        assert_eq!(f.history(), &[0.0_f64; HB_LPC_ORDER]);
    }

    #[test]
    fn silent_excitation_stays_silent_from_zero_history() {
        // Zero excitation into a fresh (zero-history) filter stays zero
        // for any LPC — there is nothing to ring.
        let mut lpc = [0.0_f64; HB_LPC_ORDER];
        lpc[0] = 0.9;
        lpc[1] = -0.4;
        let exc = [0.0_f64; crate::hb_innovation::HB_SUBFRAME_SAMPLES];
        let mut f = HbSynthesisFilter::new();
        let out = f.process_subframe(&lpc, &exc);
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
