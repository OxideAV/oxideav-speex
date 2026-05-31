//! Narrowband sub-frame LSP interpolation (round r200 scope).
//!
//! The Speex narrowband frame is 160 samples = 4 sub-frames × 40 samples
//! (`docs/audio/speex/speex-celp-companion.md` §1 + the
//! `NarrowbandSubmode::SUBFRAMES_PER_FRAME = 4` constant in
//! [`crate::submode`]). Per the Speex manual §9.1:
//!
//! > "The LSP's are considered to be associated to the 4th sub-frames
//! > and the LSP's associated to the first 3 sub-frames are linearly
//! > interpolated using the current and previous LSP coefficients."
//!
//! This module performs that linear interpolation. Given the previous
//! frame's reconstructed LSPs and the current frame's reconstructed LSPs
//! (both in the Q[[`crate::lsp::NB_LSP_OUTPUT_Q`]] = Q10 fixed-point
//! format produced by [`crate::lsp::reconstruct_q10`]), it returns one
//! ten-coefficient LSP vector per sub-frame.
//!
//! ## Interpolation weights
//!
//! Sub-frame 4 carries the current LSPs unchanged (§9.1: "associated to
//! the 4th sub-frame"). Sub-frames 1..3 are produced by **linear
//! interpolation** between the previous frame's LSPs and the current
//! frame's LSPs. Linear interpolation across four equally-spaced
//! sub-frames yields the unique weight set:
//!
//! ```text
//! sub-frame 1 (s=0): out = (3·prev + 1·curr) / 4
//! sub-frame 2 (s=1): out = (2·prev + 2·curr) / 4
//! sub-frame 3 (s=2): out = (1·prev + 3·curr) / 4
//! sub-frame 4 (s=3): out = (0·prev + 4·curr) / 4 = curr
//! ```
//!
//! Equivalently: for sub-frame `s` (0-indexed in `0..4`), `k = s + 1`,
//! `out = ((4 − k)·prev + k·curr) / 4`.
//!
//! ## Output Q-format
//!
//! To keep the interpolation **exact integer arithmetic** with no
//! rounding direction question, this module emits the per-sub-frame
//! result in Q12 (= Q10 + 2). The weighted sum `(4−k)·prev + k·curr`
//! has the form `Q10 × integer`, which by definition is a value
//! expressed in `1/(2^10) = 1/1024` units multiplied by an integer
//! factor in `1..=4`. The undivided sum therefore lives in
//! `1/1024` units scaled by 4 — i.e. naturally in `4/1024 = 1/256` if
//! we kept the multiplier outside, but here we treat the multiplier
//! as part of the value to obtain a clean Q-format. Multiplying a
//! Q10 value by 4 lands it in Q12 (one extra factor-of-4 worth of
//! sub-binary-point bits). The four weights `{4, 3, 2, 1}` are all
//! between 0 and 4 inclusive, so the sum is at most `4·max(|prev|) +
//! 4·max(|curr|)` which trivially fits in `i32` for any plausible
//! LSP magnitude.
//!
//! Choosing Q12 over Q10-with-rounded-division (i.e. dividing by 4
//! and re-emitting Q10) means:
//!
//! * Bit-exact arithmetic at every stage — no rounding choice for the
//!   spec to be silent about.
//! * The downstream LSP→LPC conversion can rescale to whatever
//!   Q-format it wants with a single arithmetic shift.
//! * Sub-frame 4's output (`k=4`) is `4·curr` in Q12, which is
//!   numerically identical to `curr` re-expressed in Q12 — i.e. the
//!   "carries the current LSPs unchanged" property of §9.1 is
//!   preserved without an arithmetic-shift special case.
//!
//! ## First-frame initialisation
//!
//! For the **first frame** of a stream, there is no "previous frame"
//! LSP to interpolate from. The Speex manual is silent on the
//! initialisation convention; the conservative behaviour adopted here
//! is to expose a separate constructor [`NbSubFrameLsp::first_frame`]
//! that initialises `prev_q10` to be **equal** to the current LSPs.
//! This makes every sub-frame's interpolated output equal to the
//! current LSPs (since `((4−k)·X + k·X)/4 = X` for any `k`), which is
//! the only choice that produces a continuous interpolation envelope
//! at stream start (`prev = curr` ⇒ no spurious LSP transient on the
//! first frame). A separate constructor surfaces the
//! initialisation explicitly so callers see when it is in use and so
//! the test suite can lock the contract.
//!
//! Any future docs gap fill that overrides this convention only
//! needs to change [`NbSubFrameLsp::first_frame`]; the steady-state
//! [`NbSubFrameLsp::new`] is independent of how `prev_q10` was
//! obtained.
//!
//! ## What this module does NOT do
//!
//! * No LSP → LPC conversion. The Speex manual §9.1 says the
//!   interpolated LSPs are "converted back to the LPC filter Â(z)"
//!   but does not specify the conversion algorithm. The
//!   `docs/audio/speex/speex-celp-companion.md` is similarly silent
//!   (it covers the table data; the conversion is algorithmic, not a
//!   lookup). Implementing the conversion here would require fishing
//!   a specific algorithm from outside the allow-list. Reported as a
//!   docs gap in the round r200 final report.
//! * No high-band (wideband / ultra-wideband) LSP interpolation. The
//!   high-band uses a 12-bit MSVQ over 8 LSPs (`HB_LPC_ORDER = 8`,
//!   `HB_LSP_STAGE_ENTRIES = 64`); a sibling [`crate::wideband`]
//!   module would carry that path. Out of scope for r200.
//! * No encoder-side analysis. Producing the current frame's LSPs
//!   from a windowed signal is a separate analysis path.

use crate::codebooks::NB_LSP_ORDER;
use crate::submode::NarrowbandSubmode;

/// Number of LSP sub-frames produced per Speex narrowband frame
/// (equal to [`NarrowbandSubmode::SUBFRAMES_PER_FRAME`]).
pub const NB_LSP_SUBFRAMES_PER_FRAME: usize = NarrowbandSubmode::SUBFRAMES_PER_FRAME as usize;

/// Q-shift of the per-sub-frame interpolated LSP output relative to the
/// underlying LSP frequency unit (output unit = `1 / 2^12 ≈ 2.44e-4`).
///
/// This is the input Q-format ([`crate::lsp::NB_LSP_OUTPUT_Q`] = 10)
/// plus two extra bits from the un-divided weight-multiplication. See
/// module docs for the Q-format rationale.
pub const NB_LSP_INTERP_OUTPUT_Q: u32 = crate::lsp::NB_LSP_OUTPUT_Q + 2;

/// Per-sub-frame interpolated narrowband LSP coefficients for a single
/// frame.
///
/// Layout: `subframes[s][i]` is the `i`-th LSP coefficient for
/// sub-frame `s` (0-indexed in `0..SUBFRAMES_PER_FRAME`). Values are in
/// Q[`NB_LSP_INTERP_OUTPUT_Q`] = Q12 fixed-point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NbSubFrameLsp {
    /// Per-sub-frame ten-coefficient LSP vector in Q12 fixed-point.
    pub subframes: [[i32; NB_LSP_ORDER]; NB_LSP_SUBFRAMES_PER_FRAME],
}

impl NbSubFrameLsp {
    /// Build the per-sub-frame LSP vectors for a steady-state frame
    /// given the previous and current frame's reconstructed LSPs in
    /// the input Q[`crate::lsp::NB_LSP_OUTPUT_Q`] = Q10 format.
    ///
    /// Output values are in Q[`NB_LSP_INTERP_OUTPUT_Q`] = Q12 (see
    /// module docs).
    pub fn new(prev_q10: &[i32; NB_LSP_ORDER], curr_q10: &[i32; NB_LSP_ORDER]) -> Self {
        let mut subframes = [[0i32; NB_LSP_ORDER]; NB_LSP_SUBFRAMES_PER_FRAME];
        for (s, slot) in subframes.iter_mut().enumerate() {
            // k_idx = sub-frame ordinal in 1..=4 (manual §9.1 sub-frame
            // indexing).
            let k_idx = (s + 1) as i32;
            let w_prev = 4 - k_idx; // {3, 2, 1, 0}
            let w_curr = k_idx; // {1, 2, 3, 4}
            for i in 0..NB_LSP_ORDER {
                slot[i] = prev_q10[i] * w_prev + curr_q10[i] * w_curr;
            }
        }
        Self { subframes }
    }

    /// Build the per-sub-frame LSP vectors for the **first frame** of a
    /// stream (no previous LSP available — `prev_q10` is taken to be
    /// equal to `curr_q10`, producing four sub-frames each equal to
    /// `curr_q10` re-expressed in Q12). See module docs for why this
    /// initialisation was chosen.
    pub fn first_frame(curr_q10: &[i32; NB_LSP_ORDER]) -> Self {
        Self::new(curr_q10, curr_q10)
    }

    /// Convenience accessor for sub-frame `s` (0-indexed).
    pub fn subframe(&self, s: usize) -> Option<&[i32; NB_LSP_ORDER]> {
        self.subframes.get(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp::NB_LSP_OUTPUT_Q;

    /// Helper: build a Q10 LSP vector from `i32`-typed seed values.
    fn lsp(v: [i32; NB_LSP_ORDER]) -> [i32; NB_LSP_ORDER] {
        v
    }

    #[test]
    fn subframe_4_equals_4x_current_in_q12() {
        // §9.1: "LSPs associated to the 4th sub-frame" — the last
        // sub-frame's output (s=3, k=4) must be exactly 4·curr in Q12,
        // which equals `curr` re-expressed in Q12 (curr is Q10).
        let prev = lsp([100, -200, 300, -400, 500, -600, 700, -800, 900, -1000]);
        let curr = lsp([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let out = NbSubFrameLsp::new(&prev, &curr);
        for (i, &c) in curr.iter().enumerate() {
            assert_eq!(out.subframes[3][i], 4 * c);
        }
    }

    #[test]
    fn subframe_1_uses_three_quarters_prev_one_quarter_current() {
        let prev = lsp([100; NB_LSP_ORDER]);
        let curr = lsp([200; NB_LSP_ORDER]);
        let out = NbSubFrameLsp::new(&prev, &curr);
        // sub-frame 1 (s=0, k=1): out = 3·prev + 1·curr = 300 + 200 = 500.
        let expected = 3 * 100 + 200;
        for &v in &out.subframes[0] {
            assert_eq!(v, expected);
        }
    }

    #[test]
    fn subframe_2_uses_equal_weights() {
        let prev = lsp([100; NB_LSP_ORDER]);
        let curr = lsp([200; NB_LSP_ORDER]);
        let out = NbSubFrameLsp::new(&prev, &curr);
        // sub-frame 2 (s=1, k=2): out = 2·prev + 2·curr = 200 + 400 = 600.
        for i in 0..NB_LSP_ORDER {
            assert_eq!(out.subframes[1][i], 2 * 100 + 2 * 200);
        }
    }

    #[test]
    fn subframe_3_uses_one_quarter_prev_three_quarters_current() {
        let prev = lsp([100; NB_LSP_ORDER]);
        let curr = lsp([200; NB_LSP_ORDER]);
        let out = NbSubFrameLsp::new(&prev, &curr);
        // sub-frame 3 (s=2, k=3): out = 1·prev + 3·curr = 100 + 600 = 700.
        let expected = 100 + 3 * 200;
        for &v in &out.subframes[2] {
            assert_eq!(v, expected);
        }
    }

    #[test]
    fn weights_sum_to_four_per_subframe() {
        // For every sub-frame the prev + curr weight sum is exactly 4,
        // which is what makes the unscaled output a clean Q12 value
        // (input Q10 × 4 = Q12).
        let unit = lsp([1; NB_LSP_ORDER]);
        let out = NbSubFrameLsp::new(&unit, &unit);
        for s in 0..NB_LSP_SUBFRAMES_PER_FRAME {
            for i in 0..NB_LSP_ORDER {
                assert_eq!(out.subframes[s][i], 4, "sub-frame {s} coeff {i}");
            }
        }
    }

    #[test]
    fn output_q_format_is_input_plus_two_bits() {
        assert_eq!(NB_LSP_INTERP_OUTPUT_Q, NB_LSP_OUTPUT_Q + 2);
        assert_eq!(NB_LSP_INTERP_OUTPUT_Q, 12);
    }

    #[test]
    fn interpolation_is_per_coefficient_independent() {
        // Coefficient i depends only on prev[i] and curr[i]; swapping
        // coefficient j (j != i) must not affect coefficient i.
        let prev_a = lsp([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        let prev_b = lsp([10, 0, 30, 40, 50, 60, 70, 80, 90, 100]); // perturb idx 1
        let curr = lsp([1; NB_LSP_ORDER]);
        let out_a = NbSubFrameLsp::new(&prev_a, &curr);
        let out_b = NbSubFrameLsp::new(&prev_b, &curr);
        for s in 0..NB_LSP_SUBFRAMES_PER_FRAME {
            for i in 0..NB_LSP_ORDER {
                if i == 1 {
                    // Only coefficient 1 should differ.
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
        // For a coefficient where curr > prev, the per-sub-frame value
        // must increase strictly from sub-frame 1 to sub-frame 4.
        let prev = lsp([10; NB_LSP_ORDER]);
        let curr = lsp([20; NB_LSP_ORDER]);
        let out = NbSubFrameLsp::new(&prev, &curr);
        for i in 0..NB_LSP_ORDER {
            // 3·10 + 1·20 = 50; 2·10 + 2·20 = 60; 1·10 + 3·20 = 70; 4·20 = 80.
            assert!(out.subframes[0][i] < out.subframes[1][i]);
            assert!(out.subframes[1][i] < out.subframes[2][i]);
            assert!(out.subframes[2][i] < out.subframes[3][i]);
        }
    }

    #[test]
    fn first_frame_produces_constant_envelope() {
        // First-frame init: every sub-frame's output equals
        // `4 * curr[i]` (i.e. `curr` in Q12) — no transient.
        let curr = lsp([7, -13, 19, -23, 29, -31, 37, -41, 43, -47]);
        let out = NbSubFrameLsp::first_frame(&curr);
        for sf in &out.subframes {
            for (i, &c) in curr.iter().enumerate() {
                assert_eq!(sf[i], 4 * c);
            }
        }
    }

    #[test]
    fn subframe_accessor_returns_none_for_out_of_range_index() {
        let zero = lsp([0; NB_LSP_ORDER]);
        let out = NbSubFrameLsp::first_frame(&zero);
        assert!(out.subframe(NB_LSP_SUBFRAMES_PER_FRAME).is_none());
        assert!(out.subframe(NB_LSP_SUBFRAMES_PER_FRAME + 5).is_none());
        assert!(out.subframe(0).is_some());
        assert!(out.subframe(NB_LSP_SUBFRAMES_PER_FRAME - 1).is_some());
    }

    #[test]
    fn interpolation_handles_negative_coefficients() {
        // LSP codebook entries are signed bytes; the reconstructed
        // values can be negative. Interpolation must preserve sign
        // correctly.
        let prev = lsp([-100; NB_LSP_ORDER]);
        let curr = lsp([100; NB_LSP_ORDER]);
        let out = NbSubFrameLsp::new(&prev, &curr);
        // sub-frame 1: 3·(-100) + 1·100 = -200
        // sub-frame 2: 2·(-100) + 2·100 = 0
        // sub-frame 3: 1·(-100) + 3·100 = 200
        // sub-frame 4: 0·(-100) + 4·100 = 400
        for i in 0..NB_LSP_ORDER {
            assert_eq!(out.subframes[0][i], -200);
            assert_eq!(out.subframes[1][i], 0);
            assert_eq!(out.subframes[2][i], 200);
            assert_eq!(out.subframes[3][i], 400);
        }
    }

    #[test]
    fn output_is_subframes_per_frame_long() {
        let zero = lsp([0; NB_LSP_ORDER]);
        let out = NbSubFrameLsp::first_frame(&zero);
        assert_eq!(out.subframes.len(), NB_LSP_SUBFRAMES_PER_FRAME);
        assert_eq!(out.subframes.len(), 4);
        for sf in out.subframes.iter() {
            assert_eq!(sf.len(), NB_LSP_ORDER);
        }
    }
}
