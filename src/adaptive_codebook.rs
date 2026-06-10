//! Narrowband adaptive-codebook (long-term predictor) index resolution
//! and excitation history buffer (round r234 scope).
//!
//! This module supplies the typed index-arithmetic helpers and the
//! excitation history buffer the §9.2 adaptive-codebook step needs to
//! turn a per-sub-frame `(pitch_period, [g0, g1, g2])` pair into the
//! three historical-buffer reads per output sample. The actual gain
//! multiplication + Q-format choice stay deferred (see "What this module
//! does not do" below).
//!
//! ## Spec basis
//!
//! Speex Codec Manual §9.2 "Sub-Frame Analysis-by-Synthesis" gives the
//! three-tap adaptive-codebook contribution as:
//!
//! > Eq. 9.1: `ea[n] = g0·e[n − T − 1] + g1·e[n − T] + g2·e[n − T + 1]`
//!
//! and immediately follows with the **excitation-repeat rule** for short
//! pitch periods:
//!
//! > "It is worth noting that when the pitch is smaller than the
//! > sub-frame size, we repeat the excitation at a period T. For example,
//! > when `n − T + 1 ≥ 0`, we use `n − 2T + 1` instead."
//!
//! Companion §2.2 paraphrases the rule as "*When `T < sub-frame size`,
//! the excitation is repeated at period `T` (for `n − T + 1 ≥ 0`, use
//! `n − 2T + 1`)*".
//!
//! Both statements describe the same procedure: whenever a per-tap
//! lookback would point at a sample position **inside** the current
//! sub-frame (which has not been emitted yet at the time the tap is
//! applied), step the index back by `T` until it lies strictly in the
//! historical region.
//!
//! ## Pitch-period range
//!
//! [`crate::narrowband_body::PITCH_PERIOD_MIN`] = 17 and
//! [`crate::narrowband_body::PITCH_PERIOD_MAX`] = 144 (§9.2: *"the pitch
//! period is encoded with 7 bits in the [17, 144] range"*). The tap
//! that reaches the furthest back into history is `g0`'s
//! `e[n − T − 1]`; at `n = 0`, `T = 144` this is `e[−145]`. The
//! history buffer therefore needs to retain at least
//! [`EXCITATION_HISTORY_LEN`] = 145 past samples to satisfy any
//! conformant pitch-period choice without truncation.
//!
//! ## Output Q-format
//!
//! This module is **Q-format-agnostic**. It surfaces:
//!
//! * Typed signed lookback offsets `k ∈ [−T_max − 1, −1]` as `i32`
//!   (always strictly negative — the repeat rule guarantees no current-
//!   sub-frame position survives, and `i32` widens the post-substitution
//!   range so callers can index a buffer without overflow concerns).
//! * A typed [`ExcitationBuffer`] holding the last
//!   [`EXCITATION_HISTORY_LEN`] samples of the emitted excitation as
//!   `i16` — matching the narrowband sub-frame innovation sample type
//!   (`c[n]` from [`crate::innovation::decode_subframe`]) and the staged
//!   codebook entry width.
//!
//! The downstream `ea[n] = g0·e[n-T-1] + g1·e[n-T] + g2·e[n-T+1]`
//! multiplication is **not** performed here: the gains from
//! [`crate::pitch_gain::reconstruct`] are surfaced as post-bias signed
//! integers in raw codebook units (Q-format gap per [`crate::pitch_gain`]
//! module docs), so the multiplication output Q-format would inherit
//! that ambiguity. Keeping the buffer + index resolution at the
//! integer-offset layer means the eventual long-term-predictor step
//! can pin its Q-format choice in a single follow-up round without
//! re-deriving the index arithmetic.
//!
//! ## What this module DOES
//!
//! * [`EXCITATION_HISTORY_LEN`] — historical-buffer length sufficient
//!   for any conformant pitch period at `n = 0`.
//! * [`ExcitationBuffer`] — typed `i16` rolling history with `push_*`
//!   helpers and indexed `lookup` returning the historical sample at a
//!   negative offset (or `None` if the offset is out of range).
//! * [`resolve_lookback`] — pure function applying the §9.2 repeat
//!   rule. Given a signed candidate lookback `k` and the pitch period
//!   `T`, returns the substituted lookback that lies strictly in the
//!   historical region (`k' < 0`). Iterates the documented
//!   `k ← k − T` substitution while `k ≥ 0`.
//! * [`subframe_lookback_indices`] — for a single 40-sample CELP
//!   sub-frame and a pitch period `T`, returns a
//!   `[[i32; 3]; 40]` matrix where row `n` is the three substituted
//!   lookback offsets `[n − T − 1, n − T, n − T + 1]` after repeat-rule
//!   resolution. Each value is strictly negative (`< 0`).
//!
//! ## What this module does NOT do
//!
//! * No gain multiplication. The §9.2 / Eq. 9.1 sum
//!   `ea[n] = g0·e[n-T-1] + g1·e[n-T] + g2·e[n-T+1]` needs a documented
//!   Q-format for the post-bias `g0, g1, g2` triple (see
//!   [`crate::pitch_gain`] module docs for the recorded gap). The
//!   multiplication output value is therefore deferred.
//! * No `e[n] = p[n] + c[n]` composition. The final excitation sum
//!   per §8.4 needs the gain-scaled adaptive contribution `p[n] = ea[n]`
//!   plus the gain-scaled fixed-codebook contribution `c[n]`. The
//!   fixed-codebook gain itself is the documented scalar quantiser
//!   product `g_frame × g_subf` (§9.2 + CELP companion §2.3); that
//!   quantiser is an open-loop scalar computation, not a static lookup,
//!   and is marked as a documented-but-non-table item in the staged
//!   `docs/audio/speex/speex-celp-companion.md` §9.
//! * No high-band path. The wideband sub-band-CELP high band carries
//!   **no adaptive codebook** (§10.2: *"there's no pitch prediction for
//!   the high-band"*), so the high band never consults this module.
//! * No buffer reset on packet loss / DTX (mode 0). Callers manage the
//!   buffer's lifecycle.

use crate::innovation::SUBFRAME_SAMPLES;
use crate::narrowband_body::PITCH_PERIOD_MAX;

/// Length of the historical excitation buffer required for the
/// narrowband adaptive-codebook path.
///
/// The tap reaching furthest back is `g0`'s `e[n − T − 1]`; at
/// `n = 0` and `T = PITCH_PERIOD_MAX` (= 144) this is `e[−145]`.
/// The buffer therefore needs **145** past samples to satisfy any
/// in-spec pitch period at the start of a sub-frame.
pub const EXCITATION_HISTORY_LEN: usize = PITCH_PERIOD_MAX as usize + 1;

/// Number of taps in the long-term predictor convolution
/// (`g0`, `g1`, `g2` of §9.2 Eq. 9.1; same as
/// [`crate::pitch_gain::PITCH_GAIN_TAPS`]).
pub const ADAPTIVE_CODEBOOK_TAPS: usize = 3;

/// Per-output-sample tap-relative offsets `(−T − 1, −T, −T + 1)` from
/// §9.2 Eq. 9.1, expressed as integer offsets from the pitch period
/// `T`.
///
/// Tap index 0 corresponds to `g0` (`e[n − T − 1]`), 1 to `g1`
/// (`e[n − T]`), 2 to `g2` (`e[n − T + 1]`).
pub const TAP_PITCH_OFFSETS: [i32; ADAPTIVE_CODEBOOK_TAPS] = [-1, 0, 1];

/// Apply the §9.2 excitation-repeat rule to a candidate lookback `k`.
///
/// Manual §9.2: *"when the pitch is smaller than the sub-frame size,
/// we repeat the excitation at a period T. For example, when
/// `n − T + 1 ≥ 0`, we use `n − 2T + 1` instead."*
///
/// Iterates `k ← k − T` while `k ≥ 0`; returns the first `k` that lies
/// strictly in the historical region (`k < 0`). For any `T` ≥ 1 the
/// loop terminates in at most `⌈k / T⌉ + 1` iterations.
///
/// `t` is the pitch period in samples (the de-biased period from
/// [`crate::NarrowbandSubFrameIndices::pitch_period`]); the in-spec
/// range is `[PITCH_PERIOD_MIN, PITCH_PERIOD_MAX]` = `[17, 144]` but
/// this function accepts any `t ≥ 1`.
///
/// # Examples
///
/// ```
/// use oxideav_speex::resolve_lookback;
///
/// // Large T → no substitution needed.
/// assert_eq!(resolve_lookback(/* k = */ -50, /* t = */ 60), -50);
///
/// // The manual's worked example: n - T + 1 ≥ 0 → use n - 2T + 1.
/// // For n = 0, T = 17, the g2 tap is 0 - 17 + 1 = -16 (< 0; no
/// // substitution). For n = 22, T = 17, g2 tap is 22 - 17 + 1 = 6
/// // (≥ 0) → substitute 6 - 17 = -11.
/// assert_eq!(resolve_lookback(/* k = */ 6, /* t = */ 17), -11);
/// ```
#[inline]
pub fn resolve_lookback(mut k: i32, t: u16) -> i32 {
    debug_assert!(t >= 1, "pitch period must be at least 1");
    let step = i32::from(t);
    while k >= 0 {
        k -= step;
    }
    k
}

/// Resolve the three per-tap lookback offsets for one output sample
/// position `n` within a 40-sample CELP sub-frame, applying the §9.2
/// repeat rule.
///
/// Returns `[k0, k1, k2]` where `kj` is the substituted lookback for
/// tap `j` (corresponding to gain `gj` in §9.2 Eq. 9.1). Each `kj` is
/// strictly negative (`< 0`).
///
/// `n` is the output position within the sub-frame (`0..SUBFRAME_SAMPLES`).
/// `t` is the de-biased pitch period (see
/// [`crate::NarrowbandSubFrameIndices::pitch_period`]).
#[inline]
pub fn sample_lookback_indices(n: u32, t: u16) -> [i32; ADAPTIVE_CODEBOOK_TAPS] {
    debug_assert!(
        (n as usize) < SUBFRAME_SAMPLES,
        "sub-frame sample position out of range"
    );
    let n = n as i32;
    let t_i = i32::from(t);
    [
        resolve_lookback(n - t_i + TAP_PITCH_OFFSETS[0], t),
        resolve_lookback(n - t_i + TAP_PITCH_OFFSETS[1], t),
        resolve_lookback(n - t_i + TAP_PITCH_OFFSETS[2], t),
    ]
}

/// Resolve the per-sample, per-tap lookback offsets for one entire
/// 40-sample CELP sub-frame.
///
/// Returns `[[k0_0, k1_0, k2_0], …, [k0_39, k1_39, k2_39]]` — row `n`
/// is the result of [`sample_lookback_indices`] for that `n`.
///
/// This is the pure index resolution; combining it with the
/// [`ExcitationBuffer::lookup`] historical reads and the three β tap
/// gains (per [`crate::pitch_gain`]) would produce the §9.2 / Eq. 9.1
/// `ea[n]` sum once the gain Q-format is pinned.
pub fn subframe_lookback_indices(t: u16) -> [[i32; ADAPTIVE_CODEBOOK_TAPS]; SUBFRAME_SAMPLES] {
    let mut out = [[0i32; ADAPTIVE_CODEBOOK_TAPS]; SUBFRAME_SAMPLES];
    for (n, slot) in out.iter_mut().enumerate() {
        *slot = sample_lookback_indices(n as u32, t);
    }
    out
}

/// Errors from [`ExcitationBuffer::lookup`] and friends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExcitationError {
    /// The lookback offset was non-negative (the historical region is
    /// `k < 0`). Either the caller failed to apply
    /// [`resolve_lookback`] or supplied an invalid sample position.
    NonHistoricalOffset { offset: i32 },
    /// The lookback offset reached further back than the
    /// [`EXCITATION_HISTORY_LEN`] = 145 sample window — only reachable
    /// with a pitch period greater than [`PITCH_PERIOD_MAX`] = 144,
    /// which is outside the §9.2 documented range.
    OffsetOutOfHistory { offset: i32 },
}

impl core::fmt::Display for ExcitationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ExcitationError::NonHistoricalOffset { offset } => write!(
                f,
                "excitation buffer: non-historical offset {offset} (expected k < 0)"
            ),
            ExcitationError::OffsetOutOfHistory { offset } => write!(
                f,
                "excitation buffer: offset {offset} reaches further back than the {EXCITATION_HISTORY_LEN}-sample history window"
            ),
        }
    }
}

impl std::error::Error for ExcitationError {}

/// Rolling buffer of past emitted excitation samples.
///
/// Stores the most recent [`EXCITATION_HISTORY_LEN`] = 145 samples of
/// the per-sample excitation signal `e[·]` that the §9.2 adaptive
/// codebook needs to reference. The buffer surfaces samples through a
/// negative-offset addressing convention matching the manual's `e[n − k]`
/// notation: `lookup(-1)` returns the most recently appended sample,
/// `lookup(-2)` the one before, and so on through
/// `lookup(-EXCITATION_HISTORY_LEN as i32)`.
///
/// The buffer initialises with all-zero history, matching the
/// stream-start "no prior excitation" condition (no LSP transient is
/// introduced for the first frame since the all-zero history makes
/// `ea[0..40] = 0` for any choice of gains).
#[derive(Debug, Clone)]
pub struct ExcitationBuffer {
    /// Ring storage; `cursor` points at the next-write index, so the
    /// most recently appended sample is at `(cursor + N - 1) % N`.
    storage: [i16; EXCITATION_HISTORY_LEN],
    /// Next-write position into `storage`.
    cursor: usize,
}

impl Default for ExcitationBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl ExcitationBuffer {
    /// Construct an all-zero history buffer (the stream-start
    /// initialisation).
    pub const fn new() -> Self {
        Self {
            storage: [0i16; EXCITATION_HISTORY_LEN],
            cursor: 0,
        }
    }

    /// Append a single sample to the buffer, evicting the oldest entry.
    pub fn push(&mut self, sample: i16) {
        self.storage[self.cursor] = sample;
        self.cursor += 1;
        if self.cursor == EXCITATION_HISTORY_LEN {
            self.cursor = 0;
        }
    }

    /// Append a slice of samples in order (the new "most recent" sample
    /// is the last element of `samples`).
    pub fn extend_from_slice(&mut self, samples: &[i16]) {
        for &s in samples {
            self.push(s);
        }
    }

    /// Look up a historical sample at a negative offset `k`.
    ///
    /// `lookup(-1)` returns the most recently pushed sample,
    /// `lookup(-2)` the one before it, and so on. Returns
    /// `Err(NonHistoricalOffset)` for `k ≥ 0` (the historical region is
    /// strictly `k < 0`); returns `Err(OffsetOutOfHistory)` for
    /// `k < -EXCITATION_HISTORY_LEN as i32`.
    pub fn lookup(&self, k: i32) -> Result<i16, ExcitationError> {
        if k >= 0 {
            return Err(ExcitationError::NonHistoricalOffset { offset: k });
        }
        let back = (-k) as usize;
        if back > EXCITATION_HISTORY_LEN {
            return Err(ExcitationError::OffsetOutOfHistory { offset: k });
        }
        // back ∈ [1, EXCITATION_HISTORY_LEN]. Index back from `cursor`.
        let idx = (self.cursor + EXCITATION_HISTORY_LEN - back) % EXCITATION_HISTORY_LEN;
        Ok(self.storage[idx])
    }

    /// Length of the addressable historical region.
    pub const fn capacity(&self) -> usize {
        EXCITATION_HISTORY_LEN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrowband_body::{PITCH_PERIOD_MAX, PITCH_PERIOD_MIN};

    // ---------- resolve_lookback ----------

    #[test]
    fn resolve_lookback_no_substitution_for_already_historical() {
        // k < 0 already → unchanged.
        for &k in &[-1, -2, -10, -50, -100, -145] {
            for &t in &[17u16, 40, 80, 144] {
                assert_eq!(resolve_lookback(k, t), k, "k={k}, t={t} already historical");
            }
        }
    }

    #[test]
    fn resolve_lookback_manual_worked_example_n_minus_2t_plus_1() {
        // Manual §9.2: "when n - T + 1 ≥ 0, we use n - 2T + 1 instead."
        // For n = 22, T = 17: candidate k = 22 - 17 + 1 = 6 ≥ 0
        //   → resolved = 22 - 2·17 + 1 = -11.
        assert_eq!(resolve_lookback(6, 17), -11);

        // For n = 17, T = 17: g1 candidate = 0 ≥ 0 → resolved = -17.
        assert_eq!(resolve_lookback(0, 17), -17);

        // For n = 0, T = 17: g2 candidate = 0 - 17 + 1 = -16 (< 0)
        // → no substitution. Cross-check via resolve_lookback directly.
        assert_eq!(resolve_lookback(-16, 17), -16);
    }

    #[test]
    fn resolve_lookback_iterates_for_very_short_pitch() {
        // For T = 17 the candidate k = 22 from above resolves to -11
        // after one step; a candidate k = 33 (= 22 + 11) needs two:
        // 33 → 16 → -1.
        assert_eq!(resolve_lookback(33, 17), -1);

        // k = 50, T = 17 → 50 → 33 → 16 → -1.
        assert_eq!(resolve_lookback(50, 17), -1);
    }

    #[test]
    fn resolve_lookback_always_strictly_negative_for_in_spec_range() {
        // Sweep every valid (n, t) pair and confirm each of the three
        // tap-resolved lookbacks is < 0.
        for n in 0..SUBFRAME_SAMPLES as u32 {
            for t in PITCH_PERIOD_MIN..=PITCH_PERIOD_MAX {
                let taps = sample_lookback_indices(n, t);
                for (i, k) in taps.iter().enumerate() {
                    assert!(
                        *k < 0,
                        "tap {i} for n={n}, t={t} resolved to {k} (expected < 0)"
                    );
                }
            }
        }
    }

    // ---------- sample_lookback_indices ----------

    #[test]
    fn sample_lookback_n_zero_matches_raw_offsets_for_large_t() {
        // For n = 0 and T large enough that no substitution is needed,
        // the three taps must be exactly (-T-1, -T, -T+1).
        for t in [PITCH_PERIOD_MIN, 40, 80, PITCH_PERIOD_MAX] {
            let taps = sample_lookback_indices(0, t);
            assert_eq!(taps[0], -(i32::from(t) + 1), "g0 tap @ n=0, t={t}");
            assert_eq!(taps[1], -i32::from(t), "g1 tap @ n=0, t={t}");
            assert_eq!(taps[2], -(i32::from(t) - 1), "g2 tap @ n=0, t={t}");
        }
    }

    #[test]
    fn sample_lookback_indices_g2_substitution_at_n_eq_t_minus_1() {
        // At n = T - 1, the g2 candidate is n - T + 1 = 0 (≥ 0)
        // → substituted to -T. Pin the boundary.
        for t in [PITCH_PERIOD_MIN, 30, 144] {
            let n = u32::from(t) - 1;
            if (n as usize) < SUBFRAME_SAMPLES {
                let taps = sample_lookback_indices(n, t);
                assert_eq!(taps[2], -i32::from(t), "g2 substitution @ n={n}, t={t}");
            }
        }
    }

    #[test]
    fn sample_lookback_indices_n_inside_sub_frame_for_short_pitch() {
        // T = 17, n = 16 (last position before the very-short-pitch
        // wrap): g0 = 16 - 17 - 1 = -2; g1 = 16 - 17 = -1;
        // g2 = 16 - 17 + 1 = 0 → substitute -17.
        let taps = sample_lookback_indices(16, 17);
        assert_eq!(taps[0], -2);
        assert_eq!(taps[1], -1);
        assert_eq!(taps[2], -17);

        // T = 17, n = 17 (first position whose g1 wraps):
        // g0 = -1; g1 = 0 → -17; g2 = 1 → -16.
        let taps = sample_lookback_indices(17, 17);
        assert_eq!(taps[0], -1);
        assert_eq!(taps[1], -17);
        assert_eq!(taps[2], -16);
    }

    // ---------- subframe_lookback_indices ----------

    #[test]
    fn subframe_lookback_indices_row_n_matches_sample_lookback() {
        for t in [PITCH_PERIOD_MIN, 33, 80, PITCH_PERIOD_MAX] {
            let mat = subframe_lookback_indices(t);
            assert_eq!(mat.len(), SUBFRAME_SAMPLES);
            for (n, row_actual) in mat.iter().enumerate() {
                let row = sample_lookback_indices(n as u32, t);
                assert_eq!(*row_actual, row, "row {n} mismatch for t={t}");
            }
        }
    }

    #[test]
    fn subframe_lookback_indices_for_long_pitch_is_pure_offsets() {
        // For T ≥ SUBFRAME_SAMPLES + 1 = 41 no g2 tap ever wraps:
        // n ∈ [0, 40), n - T + 1 ≤ 39 - 41 + 1 = -1 < 0 → no substitution.
        // The matrix is then exactly the raw offsets `(n - T - 1, n - T, n - T + 1)`.
        for t in [41u16, 80, PITCH_PERIOD_MAX] {
            let mat = subframe_lookback_indices(t);
            for (n, row) in mat.iter().enumerate() {
                let n_i = n as i32;
                let t_i = i32::from(t);
                assert_eq!(row[0], n_i - t_i - 1, "g0 @ n={n}, t={t}");
                assert_eq!(row[1], n_i - t_i, "g1 @ n={n}, t={t}");
                assert_eq!(row[2], n_i - t_i + 1, "g2 @ n={n}, t={t}");
            }
        }
    }

    #[test]
    fn subframe_lookback_indices_every_value_strictly_negative() {
        for t in PITCH_PERIOD_MIN..=PITCH_PERIOD_MAX {
            let mat = subframe_lookback_indices(t);
            for (n, row) in mat.iter().enumerate() {
                for (i, k) in row.iter().enumerate() {
                    assert!(
                        *k < 0,
                        "(n={n}, tap={i}, t={t}) resolved to {k} (expected < 0)"
                    );
                }
            }
        }
    }

    #[test]
    fn subframe_lookback_indices_within_history_window() {
        // Every resolved lookback must be reachable through the
        // EXCITATION_HISTORY_LEN-sample window — i.e. -k ≤ HISTORY_LEN.
        let max_back = EXCITATION_HISTORY_LEN as i32;
        for t in PITCH_PERIOD_MIN..=PITCH_PERIOD_MAX {
            let mat = subframe_lookback_indices(t);
            for (n, row) in mat.iter().enumerate() {
                for (i, k) in row.iter().enumerate() {
                    assert!(
                        -*k <= max_back,
                        "(n={n}, tap={i}, t={t}) lookback {k} exceeds {EXCITATION_HISTORY_LEN}-sample window"
                    );
                }
            }
        }
    }

    // ---------- ExcitationBuffer ----------

    #[test]
    fn excitation_buffer_starts_all_zero() {
        let buf = ExcitationBuffer::new();
        for k in 1..=EXCITATION_HISTORY_LEN {
            assert_eq!(buf.lookup(-(k as i32)).unwrap(), 0);
        }
    }

    #[test]
    fn excitation_buffer_push_then_lookup_minus_one() {
        let mut buf = ExcitationBuffer::new();
        buf.push(42);
        assert_eq!(buf.lookup(-1).unwrap(), 42);
        // Older entries are still zero.
        assert_eq!(buf.lookup(-2).unwrap(), 0);
    }

    #[test]
    fn excitation_buffer_extend_from_slice_orders_correctly() {
        // Last element of the slice is the most recent.
        let mut buf = ExcitationBuffer::new();
        buf.extend_from_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(buf.lookup(-1).unwrap(), 5);
        assert_eq!(buf.lookup(-2).unwrap(), 4);
        assert_eq!(buf.lookup(-3).unwrap(), 3);
        assert_eq!(buf.lookup(-4).unwrap(), 2);
        assert_eq!(buf.lookup(-5).unwrap(), 1);
        assert_eq!(buf.lookup(-6).unwrap(), 0);
    }

    #[test]
    fn excitation_buffer_rejects_non_historical_offset() {
        let buf = ExcitationBuffer::new();
        for k in 0..=5 {
            assert_eq!(
                buf.lookup(k),
                Err(ExcitationError::NonHistoricalOffset { offset: k })
            );
        }
    }

    #[test]
    fn excitation_buffer_rejects_offset_out_of_history() {
        let buf = ExcitationBuffer::new();
        // The boundary: -EXCITATION_HISTORY_LEN is the deepest valid
        // lookback; one further is out of range.
        let deepest = -(EXCITATION_HISTORY_LEN as i32);
        assert!(buf.lookup(deepest).is_ok());
        assert_eq!(
            buf.lookup(deepest - 1),
            Err(ExcitationError::OffsetOutOfHistory {
                offset: deepest - 1
            })
        );
    }

    #[test]
    fn excitation_buffer_rolls_through_full_window() {
        // Push EXCITATION_HISTORY_LEN distinct samples then verify
        // every historical position reflects the push order.
        let mut buf = ExcitationBuffer::new();
        let n = EXCITATION_HISTORY_LEN;
        for i in 0..n {
            buf.push((i as i16).wrapping_add(1));
        }
        // The most recently pushed is i = n - 1 → value n.
        for k in 1..=n {
            let expected = ((n - k) as i16).wrapping_add(1);
            assert_eq!(buf.lookup(-(k as i32)).unwrap(), expected, "k={k}");
        }
    }

    #[test]
    fn excitation_buffer_wrap_around_preserves_recent_window() {
        // Push 2 × HISTORY_LEN samples; the first HISTORY_LEN are
        // evicted and the buffer should contain only the second half.
        let mut buf = ExcitationBuffer::new();
        let n = EXCITATION_HISTORY_LEN;
        for i in 0..(2 * n) {
            buf.push(i as i16);
        }
        // Most recent: index 2n - 1 → value 2n - 1.
        // Lookup(-1) = 2n - 1, lookup(-2) = 2n - 2, …, lookup(-n) = n.
        for k in 1..=n {
            let expected = (2 * n - k) as i16;
            assert_eq!(buf.lookup(-(k as i32)).unwrap(), expected, "k={k}");
        }
    }

    #[test]
    fn excitation_buffer_default_matches_new() {
        let a = ExcitationBuffer::new();
        let b = ExcitationBuffer::default();
        // Default's all-zero contract is identical to new().
        for k in 1..=EXCITATION_HISTORY_LEN {
            let kk = -(k as i32);
            assert_eq!(a.lookup(kk).unwrap(), b.lookup(kk).unwrap());
        }
    }

    #[test]
    fn excitation_buffer_capacity_is_history_len() {
        let buf = ExcitationBuffer::new();
        assert_eq!(buf.capacity(), EXCITATION_HISTORY_LEN);
    }

    // ---------- Public constants ----------

    #[test]
    fn excitation_history_len_matches_pitch_max_plus_one() {
        // The deepest tap reach is e[n - T - 1] at n = 0, T = MAX
        // → e[-(MAX + 1)]. History must be at least MAX + 1.
        assert_eq!(
            EXCITATION_HISTORY_LEN,
            PITCH_PERIOD_MAX as usize + 1,
            "history length must satisfy deepest tap reach"
        );
    }

    #[test]
    fn tap_pitch_offsets_match_eq_9_1() {
        // §9.2 Eq. 9.1: ea[n] = g0·e[n-T-1] + g1·e[n-T] + g2·e[n-T+1]
        // Tap 0 → -1, tap 1 → 0, tap 2 → +1 relative to -T.
        assert_eq!(TAP_PITCH_OFFSETS, [-1, 0, 1]);
        assert_eq!(TAP_PITCH_OFFSETS.len(), ADAPTIVE_CODEBOOK_TAPS);
        assert_eq!(ADAPTIVE_CODEBOOK_TAPS, 3);
    }
}
