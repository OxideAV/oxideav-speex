//! Wideband **high-band** sub-frame LSP interpolation (round r350 scope).
//!
//! The high-band analogue of the narrowband [`crate::lsp_interp`] path.
//! Where the narrowband decoder interpolates its ten-coefficient LSP
//! vector across the four sub-frames of a 20 ms frame (manual §9.1), the
//! high band does the **same** linear interpolation over its eight LSP
//! coefficients.
//!
//! ## Spec basis: §10.1 makes high-band LP "very similar to narrowband"
//!
//! *The Speex Codec Manual* §10.1 ("Linear Prediction") states:
//!
//! > "The linear prediction part used for the high-band is **very
//! > similar to what is done for narrowband. The only difference is
//! > that we use only 12 bits to encode the high-band LSP's** using a
//! > multi-stage vector quantizer (MSVQ)."
//!
//! The manual names exactly **one** difference between the high-band and
//! narrowband linear-prediction paths — the 12-bit LSP MSVQ quantiser
//! (vs. the narrowband 18-/30-bit VQ). Everything else, including the
//! §9.1 per-sub-frame LSP interpolation
//!
//! > "The LSP's are considered to be associated to the 4th sub-frames
//! > and the LSP's associated to the first 3 sub-frames are linearly
//! > interpolated using the current and previous LSP coefficients."
//!
//! is therefore identical for the high band. Earlier rounds noted the
//! high-band sub-frame interpolation as a possible docs gap (the
//! [`crate::hb_lsp`] module's "What this module does NOT do" tail); the
//! §10.1 "only difference is the 12 bits" sentence resolves it — the
//! interpolation rule is the §9.1 narrowband rule, applied verbatim at
//! the order-8 high-band LSP order.
//!
//! ## Interpolation weights (identical to the narrowband §9.1 rule)
//!
//! Sub-frame 4 carries the current LSPs unchanged; sub-frames 1..3 are
//! the linear interpolation between the previous frame's LSPs and the
//! current frame's LSPs. For sub-frame `s` (0-indexed in `0..4`),
//! `k = s + 1`:
//!
//! ```text
//! out = ((4 − k)·prev + k·curr) / 4
//! ```
//!
//! ## Output Q-format
//!
//! As in [`crate::lsp_interp`], the undivided weighted sum
//! `(4−k)·prev + k·curr` is kept in **Q12** (= the Q10 input format plus
//! two bits from the un-divided ×4 weight sum) so the arithmetic is
//! exact integer with no rounding-direction question. The high-band LSP
//! input is Q[`crate::hb_lsp::HB_LSP_OUTPUT_Q`] = Q10 (matching the
//! narrowband choice), so the output is Q12 exactly as on the
//! narrowband side. The downstream high-band LSP→LPC converter rescales
//! Q12 → radians with a single shift.
//!
//! ## First-frame initialisation
//!
//! At stream start there is no previous high-band LSP set;
//! [`HbSubFrameLsp::first_frame`] takes `prev = curr`, making every
//! sub-frame equal to the current LSPs (no spurious transient), the same
//! convention the narrowband [`crate::lsp_interp::NbSubFrameLsp`] adopts.
//!
//! ## What this module does NOT do
//!
//! * No LSP → LPC conversion (that is [`crate::lsp_to_lpc`]'s
//!   high-band path).
//! * No base-vector addition — the interpolation runs on the
//!   **codebook-delta** LSP vectors exactly as the narrowband
//!   [`crate::lsp_interp`] does; the pinned high-band base vector is a
//!   pure translation that survives the interpolation and is added by
//!   the bounded high-band LSP→LPC path that consumes this primitive
//!   (the high-band counterpart of
//!   [`crate::lsp_to_lpc::subframe_lpc_set_with_base`]).

use crate::codebooks::HB_LPC_ORDER;
use crate::lsp_interp::NB_LSP_SUBFRAMES_PER_FRAME;

/// Number of LSP sub-frames produced per wideband high-band frame
/// (equal to the narrowband [`NB_LSP_SUBFRAMES_PER_FRAME`] = 4; the
/// high band has the same 4-sub-frame structure as the narrowband
/// frame, §10).
pub const HB_LSP_SUBFRAMES_PER_FRAME: usize = NB_LSP_SUBFRAMES_PER_FRAME;

/// Q-shift of the per-sub-frame interpolated high-band LSP output
/// relative to the underlying LSP frequency unit (output unit =
/// `1 / 2^12`). The input Q-format
/// ([`crate::hb_lsp::HB_LSP_OUTPUT_Q`] = 10) plus two extra bits from
/// the un-divided weight-multiplication. Matches the narrowband
/// [`crate::lsp_interp::NB_LSP_INTERP_OUTPUT_Q`].
pub const HB_LSP_INTERP_OUTPUT_Q: u32 = crate::hb_lsp::HB_LSP_OUTPUT_Q + 2;

/// Per-sub-frame interpolated high-band LSP coefficients for a single
/// wideband frame.
///
/// Layout: `subframes[s][i]` is the `i`-th of the eight high-band LSP
/// coefficients for sub-frame `s` (0-indexed in
/// `0..HB_LSP_SUBFRAMES_PER_FRAME`). Values are in
/// Q[`HB_LSP_INTERP_OUTPUT_Q`] = Q12 fixed-point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HbSubFrameLsp {
    /// Per-sub-frame eight-coefficient high-band LSP vector in Q12.
    pub subframes: [[i32; HB_LPC_ORDER]; HB_LSP_SUBFRAMES_PER_FRAME],
}

impl HbSubFrameLsp {
    /// Build the per-sub-frame high-band LSP vectors for a steady-state
    /// frame given the previous and current frame's reconstructed
    /// high-band LSPs in the input Q[`crate::hb_lsp::HB_LSP_OUTPUT_Q`] =
    /// Q10 format.
    ///
    /// Output values are in Q[`HB_LSP_INTERP_OUTPUT_Q`] = Q12, using the
    /// §9.1 weights `{(3,1),(2,2),(1,3),(0,4)}` (§10.1: identical to
    /// narrowband).
    pub fn new(prev_q10: &[i32; HB_LPC_ORDER], curr_q10: &[i32; HB_LPC_ORDER]) -> Self {
        let mut subframes = [[0i32; HB_LPC_ORDER]; HB_LSP_SUBFRAMES_PER_FRAME];
        for (s, slot) in subframes.iter_mut().enumerate() {
            let k_idx = (s + 1) as i32;
            let w_prev = 4 - k_idx; // {3, 2, 1, 0}
            let w_curr = k_idx; // {1, 2, 3, 4}
            for i in 0..HB_LPC_ORDER {
                slot[i] = prev_q10[i] * w_prev + curr_q10[i] * w_curr;
            }
        }
        Self { subframes }
    }

    /// Build the per-sub-frame high-band LSP vectors for the **first
    /// frame** of a stream (no previous high-band LSP available —
    /// `prev_q10` is taken to be equal to `curr_q10`, so every
    /// sub-frame equals `curr_q10` re-expressed in Q12). Mirrors the
    /// narrowband [`crate::lsp_interp::NbSubFrameLsp::first_frame`].
    pub fn first_frame(curr_q10: &[i32; HB_LPC_ORDER]) -> Self {
        Self::new(curr_q10, curr_q10)
    }

    /// Convenience accessor for sub-frame `s` (0-indexed).
    pub fn subframe(&self, s: usize) -> Option<&[i32; HB_LPC_ORDER]> {
        self.subframes.get(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hb_lsp::HB_LSP_OUTPUT_Q;

    fn lsp(v: [i32; HB_LPC_ORDER]) -> [i32; HB_LPC_ORDER] {
        v
    }

    #[test]
    fn order_is_eight_and_four_subframes() {
        assert_eq!(HB_LPC_ORDER, 8);
        assert_eq!(HB_LSP_SUBFRAMES_PER_FRAME, 4);
    }

    #[test]
    fn output_q_format_is_input_plus_two_bits() {
        assert_eq!(HB_LSP_INTERP_OUTPUT_Q, HB_LSP_OUTPUT_Q + 2);
        assert_eq!(HB_LSP_INTERP_OUTPUT_Q, 12);
        // Both bands share the same interpolation Q-format.
        assert_eq!(
            HB_LSP_INTERP_OUTPUT_Q,
            crate::lsp_interp::NB_LSP_INTERP_OUTPUT_Q
        );
    }

    #[test]
    fn subframe_4_equals_4x_current_in_q12() {
        // §9.1 (via §10.1): last sub-frame (s=3, k=4) carries the
        // current LSPs unchanged → 4·curr in Q12.
        let prev = lsp([100, -200, 300, -400, 500, -600, 700, -800]);
        let curr = lsp([1, 2, 3, 4, 5, 6, 7, 8]);
        let out = HbSubFrameLsp::new(&prev, &curr);
        for (i, &c) in curr.iter().enumerate() {
            assert_eq!(out.subframes[3][i], 4 * c);
        }
    }

    #[test]
    fn subframe_weights_match_narrowband_rule() {
        let prev = lsp([100; HB_LPC_ORDER]);
        let curr = lsp([200; HB_LPC_ORDER]);
        let out = HbSubFrameLsp::new(&prev, &curr);
        for i in 0..HB_LPC_ORDER {
            assert_eq!(out.subframes[0][i], 3 * 100 + 200); // (3,1)
            assert_eq!(out.subframes[1][i], 2 * 100 + 2 * 200); // (2,2)
            assert_eq!(out.subframes[2][i], 100 + 3 * 200); // (1,3)
            assert_eq!(out.subframes[3][i], 4 * 200); // (0,4)
        }
    }

    #[test]
    fn first_frame_produces_constant_envelope() {
        let curr = lsp([7, -13, 19, -23, 29, -31, 37, -41]);
        let out = HbSubFrameLsp::first_frame(&curr);
        for sf in &out.subframes {
            for (i, &c) in curr.iter().enumerate() {
                assert_eq!(sf[i], 4 * c);
            }
        }
    }

    #[test]
    fn interpolation_is_per_coefficient_independent() {
        let prev_a = lsp([10, 20, 30, 40, 50, 60, 70, 80]);
        let prev_b = lsp([10, 0, 30, 40, 50, 60, 70, 80]); // perturb idx 1
        let curr = lsp([1; HB_LPC_ORDER]);
        let out_a = HbSubFrameLsp::new(&prev_a, &curr);
        let out_b = HbSubFrameLsp::new(&prev_b, &curr);
        for s in 0..HB_LSP_SUBFRAMES_PER_FRAME {
            for i in 0..HB_LPC_ORDER {
                if i == 1 {
                    continue;
                }
                assert_eq!(
                    out_a.subframes[s][i], out_b.subframes[s][i],
                    "coefficient {i} contaminated by perturbation of coefficient 1"
                );
            }
        }
    }

    #[test]
    fn interpolation_envelope_is_monotone_when_inputs_are_monotone() {
        let prev = lsp([10; HB_LPC_ORDER]);
        let curr = lsp([20; HB_LPC_ORDER]);
        let out = HbSubFrameLsp::new(&prev, &curr);
        for i in 0..HB_LPC_ORDER {
            assert!(out.subframes[0][i] < out.subframes[1][i]);
            assert!(out.subframes[1][i] < out.subframes[2][i]);
            assert!(out.subframes[2][i] < out.subframes[3][i]);
        }
    }

    #[test]
    fn interpolation_handles_negative_coefficients() {
        let prev = lsp([-100; HB_LPC_ORDER]);
        let curr = lsp([100; HB_LPC_ORDER]);
        let out = HbSubFrameLsp::new(&prev, &curr);
        for i in 0..HB_LPC_ORDER {
            assert_eq!(out.subframes[0][i], -200);
            assert_eq!(out.subframes[1][i], 0);
            assert_eq!(out.subframes[2][i], 200);
            assert_eq!(out.subframes[3][i], 400);
        }
    }

    #[test]
    fn subframe_accessor_returns_none_for_out_of_range_index() {
        let zero = lsp([0; HB_LPC_ORDER]);
        let out = HbSubFrameLsp::first_frame(&zero);
        assert!(out.subframe(HB_LSP_SUBFRAMES_PER_FRAME).is_none());
        assert!(out.subframe(0).is_some());
        assert!(out.subframe(HB_LSP_SUBFRAMES_PER_FRAME - 1).is_some());
    }

    #[test]
    fn output_is_subframes_per_frame_long() {
        let zero = lsp([0; HB_LPC_ORDER]);
        let out = HbSubFrameLsp::first_frame(&zero);
        assert_eq!(out.subframes.len(), HB_LSP_SUBFRAMES_PER_FRAME);
        for sf in out.subframes.iter() {
            assert_eq!(sf.len(), HB_LPC_ORDER);
        }
    }
}
