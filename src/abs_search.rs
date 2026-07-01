//! Analysis-by-synthesis primitives + closed-loop adaptive-codebook
//! (pitch) search (encoder, round r382 scope).
//!
//! Speex chooses its excitation parameters by *analysis-by-synthesis*:
//! it synthesises each candidate, filters the result into the
//! perceptually weighted domain, and keeps the candidate whose weighted
//! error against the (weighted) target is smallest (MAN §8 p.25 — "CELP
//! is based on … closed-loop analysis-by-synthesis"; companion §2). This
//! module lands the two encoder-side pieces that step requires:
//!
//! 1. [`weighted_synthesis_zero_state`] — filter an excitation vector
//!    through the cascade `1/A(z)` (synthesis, [`crate::synthesis`])
//!    then `W(z) = A(z/γ1)/A(z/γ2)` (weighting, [`crate::weighting`]),
//!    from zero initial state. This turns a candidate excitation into
//!    its contribution in the weighted-error domain.
//! 2. [`closed_loop_pitch_search`] — the §9.2 adaptive-codebook search:
//!    for every candidate integer pitch period `T`, build the three
//!    tap basis vectors from the past excitation (offsets `−T−1, −T,
//!    −T+1`, the [`crate::adaptive_codebook`] repeat rule), filter each
//!    through the weighted synthesis, and pick the `(T, gain-VQ index)`
//!    minimising the weighted squared error against the target.
//!
//! ## Domain
//!
//! Everything is `f64`, matching [`crate::synthesis`] /
//! [`crate::weighting`] / [`crate::lpc_analysis`]. The excitation
//! history is carried as `f64` (`history[len-1] = e[−1]`). The 3-tap
//! pitch gains come from the staged VQ codebook via
//! [`crate::pitch_gain`] and are interpreted in the crate's pinned **Q6**
//! pitch-gain domain (`gain = tap / 64`, [`crate::gain_scaled_pitch`]).

use crate::adaptive_codebook::{resolve_lookback, ADAPTIVE_CODEBOOK_TAPS, TAP_PITCH_OFFSETS};
use crate::gain_scaled_pitch::PITCH_GAIN_SCALING;
use crate::innovation::SUBFRAME_SAMPLES;
use crate::lsp_to_lpc::LPC_ORDER;
use crate::pitch_gain::{self, PitchGainTaps};
use crate::submode::PitchGainQuant;
use crate::synthesis::SynthesisFilter;
use crate::weighting::PerceptualWeighting;

/// Filter an excitation vector through the weighted synthesis cascade
/// `W(z)·(1/A(z))` from zero initial state, returning the filtered
/// vector (same length as `excitation`).
///
/// `a` is the sub-frame LPC in the crate convention; `gamma1` / `gamma2`
/// are the perceptual-weighting factors. Zero initial state is the
/// correct convention for a codebook *basis* vector, whose zero-input
/// (ringing) response is accounted for separately in the search target.
pub fn weighted_synthesis_zero_state(
    a: &[f64; LPC_ORDER],
    gamma1: f64,
    gamma2: f64,
    excitation: &[f64],
) -> Vec<f64> {
    let mut synth = SynthesisFilter::new();
    let mut syn_out = vec![0.0_f64; excitation.len()];
    synth.process(a, excitation, &mut syn_out);
    let mut weight = PerceptualWeighting::new(a, gamma1, gamma2);
    let mut out = vec![0.0_f64; excitation.len()];
    weight.process(&syn_out, &mut out);
    out
}

/// Build the adaptive-codebook basis vector for tap `tap_offset` at
/// pitch period `t`, over one 40-sample sub-frame.
///
/// For output sample `n`, the lookback is
/// `resolve_lookback(n − t + tap_offset, t)` (always `< 0`, applying the
/// §9.2 short-pitch repeat rule), read from `history` (`history[len-1]`
/// is `e[−1]`). Reaching past the start of `history` reads `0.0`
/// (stream-start silence).
fn adaptive_basis(history: &[f64], t: u16, tap_offset: i32) -> [f64; SUBFRAME_SAMPLES] {
    let hlen = history.len();
    let mut basis = [0.0_f64; SUBFRAME_SAMPLES];
    for (n, slot) in basis.iter_mut().enumerate() {
        let k = resolve_lookback(n as i32 - i32::from(t) + tap_offset, t);
        // k < 0: history index is hlen + k. Out-of-range → 0.
        let idx = hlen as i32 + k;
        if idx >= 0 {
            *slot = history[idx as usize];
        }
    }
    basis
}

/// One candidate's filtered 3-tap basis, cached during the search.
struct FilteredTaps {
    /// Weighted-synthesis of each tap's basis vector (`y_{-1}, y_0, y_{+1}`).
    y: [[f64; SUBFRAME_SAMPLES]; ADAPTIVE_CODEBOOK_TAPS],
}

impl FilteredTaps {
    fn build(history: &[f64], t: u16, a: &[f64; LPC_ORDER], gamma1: f64, gamma2: f64) -> Self {
        let mut y = [[0.0_f64; SUBFRAME_SAMPLES]; ADAPTIVE_CODEBOOK_TAPS];
        for (j, slot) in y.iter_mut().enumerate() {
            let basis = adaptive_basis(history, t, TAP_PITCH_OFFSETS[j]);
            let filt = weighted_synthesis_zero_state(a, gamma1, gamma2, &basis);
            slot.copy_from_slice(&filt);
        }
        Self { y }
    }

    /// Weighted squared error of gain triple `taps` (Q6) against the
    /// weighted `target`, plus the synthesised weighted contribution.
    fn error(&self, taps: &PitchGainTaps, target: &[f64; SUBFRAME_SAMPLES]) -> f64 {
        let g: [f64; ADAPTIVE_CODEBOOK_TAPS] = [
            f64::from(taps.taps[0]) / f64::from(PITCH_GAIN_SCALING),
            f64::from(taps.taps[1]) / f64::from(PITCH_GAIN_SCALING),
            f64::from(taps.taps[2]) / f64::from(PITCH_GAIN_SCALING),
        ];
        let mut err = 0.0_f64;
        for n in 0..SUBFRAME_SAMPLES {
            let s = g[0] * self.y[0][n] + g[1] * self.y[1][n] + g[2] * self.y[2][n];
            let d = target[n] - s;
            err += d * d;
        }
        err
    }
}

/// Result of a closed-loop adaptive-codebook (pitch) search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosedLoopPitch {
    /// Winning integer pitch period `T`.
    pub period: u16,
    /// Winning 3-tap gain-VQ index (into the active [`PitchGainQuant`]
    /// codebook). `0` when `quant == None`.
    pub gain_index: u8,
    /// The reconstructed Q6 tap coefficients for the winning index.
    pub taps: PitchGainTaps,
    /// The minimised weighted squared error.
    pub error: f64,
}

/// The number of gain-VQ codebook entries for a regime.
fn gain_codebook_len(quant: PitchGainQuant) -> u32 {
    match quant {
        PitchGainQuant::None => 1,
        PitchGainQuant::Vq5Bit => 32,
        PitchGainQuant::Vq7Bit => 128,
    }
}

/// Closed-loop adaptive-codebook (pitch) search over `[t_min, t_max]`.
///
/// For each candidate period the three tap basis vectors are filtered
/// through the weighted synthesis cascade and every gain-VQ codebook
/// entry is scored against `target` (the weighted target signal for the
/// sub-frame, zero-input response already removed by the caller). The
/// `(period, gain-VQ index)` minimising the weighted squared error wins.
///
/// * `quant == None` → returns the silent pitch (all-zero taps), period
///   `t_min`, no codebook consulted.
/// * `history` is the past excitation in `f64` (`history[len-1] = e[−1]`).
pub fn closed_loop_pitch_search(
    target: &[f64; SUBFRAME_SAMPLES],
    history: &[f64],
    a: &[f64; LPC_ORDER],
    gamma1: f64,
    gamma2: f64,
    t_min: u16,
    t_max: u16,
    quant: PitchGainQuant,
) -> ClosedLoopPitch {
    if matches!(quant, PitchGainQuant::None) {
        let taps = PitchGainTaps::SILENCE;
        // Error of the zero contribution = target energy.
        let error = target.iter().map(|&x| x * x).sum();
        return ClosedLoopPitch {
            period: t_min,
            gain_index: 0,
            taps,
            error,
        };
    }

    let n_entries = gain_codebook_len(quant);
    let mut best = ClosedLoopPitch {
        period: t_min,
        gain_index: 0,
        taps: PitchGainTaps::SILENCE,
        error: f64::INFINITY,
    };
    for t in t_min..=t_max {
        let ft = FilteredTaps::build(history, t, a, gamma1, gamma2);
        for idx in 0..n_entries {
            let Some(taps) = pitch_gain::reconstruct(idx as u8, quant) else {
                continue;
            };
            let err = ft.error(&taps, target);
            if err < best.error {
                best = ClosedLoopPitch {
                    period: t,
                    gain_index: idx as u8,
                    taps,
                    error: err,
                };
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weighting::{WEIGHT_GAMMA1, WEIGHT_GAMMA2};

    fn zero_lpc() -> [f64; LPC_ORDER] {
        [0.0; LPC_ORDER]
    }

    #[test]
    fn weighted_synthesis_zero_lpc_is_identity() {
        // A(z)=1 → 1/A=1 and W=1, so the cascade passes the input.
        let a = zero_lpc();
        let exc = [1.0_f64, -2.0, 3.0, 0.5, -4.0];
        let out = weighted_synthesis_zero_state(&a, WEIGHT_GAMMA1, WEIGHT_GAMMA2, &exc);
        for i in 0..exc.len() {
            assert!((out[i] - exc[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn silent_quant_returns_target_energy() {
        let target = [2.0_f64; SUBFRAME_SAMPLES];
        let hist = vec![0.0_f64; 200];
        let a = zero_lpc();
        let r = closed_loop_pitch_search(
            &target,
            &hist,
            &a,
            WEIGHT_GAMMA1,
            WEIGHT_GAMMA2,
            17,
            17,
            PitchGainQuant::None,
        );
        assert_eq!(r.taps, PitchGainTaps::SILENCE);
        let expect: f64 = target.iter().map(|&x| x * x).sum();
        assert!((r.error - expect).abs() < 1e-9);
    }

    #[test]
    fn recovers_planted_pitch_period() {
        // Build a history that, at period T=40 with a unit centre tap,
        // reconstructs the target exactly (zero LPC → basis = target).
        let t = 40u16;
        // history holds a recognisable pattern in its last T samples.
        let mut hist = vec![0.0_f64; 200];
        let hlen = hist.len();
        let pattern: Vec<f64> = (0..SUBFRAME_SAMPLES)
            .map(|n| ((n * 7 % 13) as f64) - 6.0)
            .collect();
        // Place the pattern so that e[n-T] for n in 0..40 == pattern[n].
        // e[n-T] index in history = hlen - T + n.
        for n in 0..SUBFRAME_SAMPLES {
            hist[hlen - t as usize + n] = pattern[n];
        }
        // Target = pattern with a unit centre-tap gain. Q6 gain 64 = 1.0.
        // With zero LPC, the weighted synthesis of the g1 basis = pattern.
        let mut target = [0.0_f64; SUBFRAME_SAMPLES];
        target.copy_from_slice(&pattern);
        let a = zero_lpc();
        let r = closed_loop_pitch_search(
            &target,
            &hist,
            &a,
            WEIGHT_GAMMA1,
            WEIGHT_GAMMA2,
            35,
            45,
            PitchGainQuant::Vq5Bit,
        );
        // The best period should be the planted 40 (the only lag whose
        // basis matches the target).
        assert_eq!(r.period, 40, "planted pitch not recovered");
        assert!(r.error < target.iter().map(|&x| x * x).sum::<f64>());
    }

    #[test]
    fn search_reduces_error_below_zero_gain() {
        let t = 40u16;
        let mut hist = vec![0.0_f64; 200];
        let hlen = hist.len();
        for n in 0..SUBFRAME_SAMPLES {
            hist[hlen - t as usize + n] = (n as f64 * 0.1).sin() * 10.0;
        }
        let mut target = [0.0_f64; SUBFRAME_SAMPLES];
        for n in 0..SUBFRAME_SAMPLES {
            target[n] = (n as f64 * 0.1).sin() * 10.0;
        }
        let a = zero_lpc();
        let zero_energy: f64 = target.iter().map(|&x| x * x).sum();
        let r = closed_loop_pitch_search(
            &target,
            &hist,
            &a,
            WEIGHT_GAMMA1,
            WEIGHT_GAMMA2,
            38,
            42,
            PitchGainQuant::Vq7Bit,
        );
        assert!(r.error < zero_energy, "search did not improve on silence");
        assert!((38..=42).contains(&r.period));
    }

    #[test]
    fn adaptive_basis_reads_history_tail() {
        let mut hist = vec![0.0_f64; 200];
        let hlen = hist.len();
        // e[-1] = 5.0 (history[hlen-1]).
        hist[hlen - 1] = 5.0;
        // For T=40, tap offset +1 (g2), n=39: k = resolve_lookback(39-40+1, 40)
        //   = resolve_lookback(0,40) = -40. history[hlen-40] should be read.
        let basis = adaptive_basis(&hist, 40, 1);
        // n where k = -1 → n - 40 + 1 = -1 → n = 38.
        assert_eq!(basis[38], 5.0);
    }
}
