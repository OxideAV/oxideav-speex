//! **Forced (open-loop) pitch-gain reconstruction** (round r356 scope) —
//! the frame-level 4-bit *forced* pitch-gain field used by narrowband
//! modes 1, 2, and 8, where the 3-tap pitch-gain VQ
//! ([`crate::pitch_gain`]) is *not* transmitted.
//!
//! ## Where this sits in the decode path
//!
//! Narrowband Table 9.1 carries two mutually-exclusive pitch-gain
//! representations (companion §3 / §2.2):
//!
//! * **Per-sub-frame 3-tap VQ** (`Pitch gain` row, 5 or 7 bits) — modes
//!   2..=7 carry a vector-quantised `(g0, g1, g2)` triple per sub-frame,
//!   reconstructed by [`crate::reconstruct_pitch_gain`].
//! * **Frame-level forced gain** (`OL pitch gain` row, 4 bits) — modes
//!   1, 2, and 8 carry a *single* frame-level scalar pitch gain that
//!   applies to all four sub-frames, paired with the frame-level
//!   open-loop pitch period (`OL pitch` row). This is the
//!   "forced-pitch-gain" path: the encoder forces one pitch coefficient
//!   for the whole frame rather than searching a per-sub-frame codebook.
//!
//! Before this module, the frame-level `OL pitch gain` field was parsed
//! ([`crate::NarrowbandFrameBody::ol_pitch_gain_index`]) but discarded —
//! the decoder fell back to [`crate::PitchGainTaps::SILENCE`] for modes
//! 1, 2, and 8, so their adaptive-codebook (pitch) contribution `p[n]`
//! was always zero and only the innovation `c[n]` drove synthesis. This
//! module reconstructs the forced gain so those three modes get a real
//! long-term-predictor contribution.
//!
//! ## Reconstruction law (numeric facts)
//!
//! The clean-room extraction staged at
//! `docs/audio/speex/provenance/02-speex-gain-quant.md` ("Scalar
//! constants" table) pins the forced-pitch-gain quantiser as a 4-bit
//! linear scalar quantiser:
//!
//! > OL pitch-gain quantiser (forced-pitch-gain field, 4-bit):
//! > encode `15 · ol_pitch_coef` → clamp `[0, 15]`; decode
//! > `pitch_coef = 0.066667 · quant`.
//!
//! i.e. the 4-bit index `quant ∈ 0..=15` reconstructs to a scalar pitch
//! coefficient `pitch_coef = 0.066667 · quant` spanning `0.0 .. 1.0`
//! (`0.066667 · 15 = 1.0`). The coefficient is a **single** long-term
//! predictor tap applied at the open-loop lag `T`:
//!
//! ```text
//!   p[n] = pitch_coef · e[n − T]
//! ```
//!
//! There is no `e[n−T−1]` / `e[n−T+1]` spread for the forced path — only
//! the centre tap is non-zero (unlike the 3-tap VQ).
//!
//! ## Q6 tap representation
//!
//! The downstream §9.2 convolution
//! ([`crate::gain_scaled_pitch_subframe`]) consumes the post-bias β taps
//! in **Q6** (`GAIN_SCALING = 64`, `GAIN_SHIFT = 6`; same provenance
//! table) and divides the integer dot product by `64`. To reuse that one
//! well-tested convolution, the forced coefficient is expressed as a Q6
//! **centre tap**:
//!
//! ```text
//!   g1 = round(pitch_coef · 64) = round(0.066667 · quant · 64)
//!   g0 = g2 = 0
//! ```
//!
//! so that `gain_scaled_pitch_subframe` reconstructs
//! `p[n] = (g1 · e[n−T]) / 64 ≈ pitch_coef · e[n−T]`. The Q6 grid step
//! (`1/64 ≈ 0.0156`) is finer than the quantiser step
//! (`0.066667 · 64 ≈ 4.27` Q6 units per index), so every distinct `quant`
//! maps to a distinct centre tap with no collision.
//!
//! ## What this module DOES
//!
//! * [`forced_pitch_coef`] — reconstruct the scalar float pitch
//!   coefficient `0.066667 · quant` from the 4-bit index.
//! * [`forced_pitch_gain_taps`] — build the single-centre-tap
//!   [`PitchGainTaps`] (Q6) the §9.2 convolution consumes.
//!
//! ## What this module DOES NOT do
//!
//! * No encoder-side forced-gain search (the `15 · ol_pitch_coef`
//!   forward direction + the `0.9` damping the encoder applies when
//!   forming `ol_pitch_coef` are encoder-only — recorded as scalar facts
//!   in `provenance/02` but not exercised by the decoder).
//! * No per-sub-frame 3-tap VQ (that is [`crate::pitch_gain`]).
//! * No high-band path (§10.2: the high band has no pitch predictor).

use crate::pitch_gain::{PitchGainTaps, PITCH_GAIN_TAPS};

/// Bit width of the frame-level forced (open-loop) pitch-gain field
/// (Table 9.1 `OL pitch gain` row, modes 1 / 2 / 8).
pub const FORCED_PITCH_GAIN_BITS: u8 = 4;

/// Number of distinct forced-pitch-gain indices (`2^4 = 16`,
/// `quant ∈ 0..=15`).
pub const FORCED_PITCH_GAIN_LEVELS: u8 = 1 << FORCED_PITCH_GAIN_BITS;

/// Forced-pitch-gain decode step (`provenance/02`: decode
/// `pitch_coef = 0.066667 · quant`). The reciprocal of the encode
/// multiplier `15` (`0.066667 · 15 = 1.0`), so index `15` reconstructs
/// to a unit pitch coefficient and index `0` to silence.
pub const FORCED_PITCH_GAIN_STEP: f32 = 0.066667;

/// Q6 pitch-gain scaling (`GAIN_SCALING = 64`, `GAIN_SHIFT = 6`) — the
/// factor the §9.2 convolution divides by, restated here so the forced
/// coefficient can be expressed as an equivalent Q6 centre tap.
const FORCED_Q6_SCALING: f32 = 64.0;

/// Reconstruct the scalar float forced pitch coefficient from a 4-bit
/// forced-pitch-gain index `quant`.
///
/// Returns `pitch_coef = min(0.066667 · quant, 0.99)` (`provenance/02`
/// law + the r450 crafted-stream cap measurement: a `T = 50` mode-8
/// probe grid recovers the reference's effective centre tap as the
/// exact float `0.066667·quant` at every index — 0.2666 at 4 … 0.9333
/// at 14, resid ≤ 0.04 % — and **0.9900** at index 15, i.e. the unit
/// coefficient is clamped just under unity for loop stability). The
/// index is masked to its 4-bit field width, so any stray high bits a
/// hand-built body might carry are ignored (the wire field is only 4
/// bits wide). Index `0` → `0.0` (no pitch).
#[inline]
pub fn forced_pitch_coef(quant: u8) -> f32 {
    let q = quant & (FORCED_PITCH_GAIN_LEVELS - 1);
    (FORCED_PITCH_GAIN_STEP * f32::from(q)).min(0.99)
}

/// Build the single-centre-tap [`PitchGainTaps`] (Q6) for the forced
/// (open-loop) pitch-gain index `quant`.
///
/// The forced path applies one pitch coefficient at the open-loop lag
/// `T` (`p[n] = pitch_coef · e[n−T]`), so only the centre tap `g1` (the
/// `e[n−T]` multiplier) is non-zero; the `e[n−T−1]` / `e[n−T+1]` taps are
/// zero. The centre tap is the Q6 round of the float coefficient
/// (`round(0.066667 · quant · 64)`), so feeding the result through
/// [`crate::gain_scaled_pitch_subframe`] (which divides the Q6 dot
/// product by `64`) reconstructs the float-domain contribution
/// `p[n] ≈ pitch_coef · e[n−T]` in the same normalised signal domain as
/// the gain-scaled innovation `c[n]`.
///
/// Index `0` yields [`PitchGainTaps::SILENCE`] exactly (`0.066667 · 0 =
/// 0`), matching the no-pitch convention.
pub fn forced_pitch_gain_taps(quant: u8) -> PitchGainTaps {
    let coef = forced_pitch_coef(quant);
    // Q6 centre tap: round the float coefficient onto the 1/64 grid the
    // §9.2 convolution expects.
    let g1 = (coef * FORCED_Q6_SCALING).round() as i16;
    let mut taps = [0i16; PITCH_GAIN_TAPS];
    taps[1] = g1;
    PitchGainTaps { taps }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode law is `0.066667 · quant`, exactly as `provenance/02`
    /// pins it; index 0 is silence and index 15 is the unit coefficient.
    #[test]
    fn coef_matches_decode_law() {
        for quant in 0u8..FORCED_PITCH_GAIN_LEVELS - 1 {
            let expected = 0.066667 * f32::from(quant);
            assert_eq!(forced_pitch_coef(quant), expected, "quant={quant}");
        }
        assert_eq!(forced_pitch_coef(0), 0.0);
        // 0.066667 · 15 = 1.000005, clamped to the measured 0.99 cap
        // (r450 — the reference keeps the loop just under unity).
        assert_eq!(forced_pitch_coef(15), 0.99);
    }

    /// The 4-bit field width masks stray high bits (the wire field is
    /// only 4 bits; a hand-built body with bit 4+ set must still resolve
    /// the low nibble).
    #[test]
    fn index_is_masked_to_four_bits() {
        for quant in 0u8..FORCED_PITCH_GAIN_LEVELS {
            assert_eq!(forced_pitch_coef(quant), forced_pitch_coef(quant | 0x10));
            assert_eq!(forced_pitch_coef(quant), forced_pitch_coef(quant | 0xF0));
        }
    }

    /// Index 0 reconstructs to the all-zero silence taps exactly.
    #[test]
    fn index_zero_is_silence() {
        assert_eq!(forced_pitch_gain_taps(0), PitchGainTaps::SILENCE);
        assert_eq!(forced_pitch_gain_taps(0).taps, [0, 0, 0]);
    }

    /// Only the centre tap (`g1`, the `e[n−T]` multiplier) is non-zero —
    /// the forced path is a single-tap predictor, not a 3-tap spread.
    #[test]
    fn only_centre_tap_is_nonzero() {
        for quant in 1u8..FORCED_PITCH_GAIN_LEVELS {
            let taps = forced_pitch_gain_taps(quant).taps;
            assert_eq!(taps[0], 0, "quant={quant} side tap g0 must be zero");
            assert_eq!(taps[2], 0, "quant={quant} side tap g2 must be zero");
            assert!(taps[1] > 0, "quant={quant} centre tap g1 must be positive");
        }
    }

    /// The Q6 centre tap is `round(0.066667 · quant · 64)`, and feeding it
    /// through the Q6 division (`/64`) recovers the float coefficient to
    /// within half a Q6 step.
    #[test]
    fn q6_tap_recovers_coefficient() {
        for quant in 0u8..FORCED_PITCH_GAIN_LEVELS {
            let coef = forced_pitch_coef(quant);
            let g1 = forced_pitch_gain_taps(quant).taps[1];
            assert_eq!(g1, (coef * 64.0).round() as i16, "quant={quant}");
            // Q6 division recovers the coefficient within half a Q6 step.
            let recovered = f32::from(g1) / 64.0;
            assert!(
                (recovered - coef).abs() <= 0.5 / 64.0 + 1e-6,
                "quant={quant}: {recovered} vs {coef}"
            );
        }
    }

    /// The centre tap is strictly monotone in the index (each distinct
    /// `quant` maps to a distinct Q6 tap — no collision on the grid).
    #[test]
    fn centre_tap_strictly_monotone() {
        let mut prev = i16::MIN;
        for quant in 0u8..FORCED_PITCH_GAIN_LEVELS {
            let g1 = forced_pitch_gain_taps(quant).taps[1];
            assert!(g1 > prev, "quant={quant}: {g1} not > {prev}");
            prev = g1;
        }
    }

    /// The unit-gain index (15) maps to a Q6 tap of ~64 (`1.0 · 64`), so
    /// the §9.2 division yields a unit pitch contribution.
    #[test]
    fn unit_index_is_q6_unity() {
        // Index 15 carries the measured 0.99 cap: Q6 round(0.99·64) = 63.
        let g1 = forced_pitch_gain_taps(15).taps[1];
        assert_eq!(g1, 63, "index 15 should be the capped Q6 63");
    }
}
