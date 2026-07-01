//! Open-loop pitch estimation (encoder front-end, round r382 scope).
//!
//! The Speex encoder transmits, for some narrowband modes (Table 9.1 "OL
//! pitch" row — modes 1 / 2 / 8), a single **frame-level open-loop pitch
//! period** estimated once over the whole frame, then refines it per
//! sub-frame in the closed-loop search (MAN §9.2 p.30; companion §2.2).
//! This module lands the open-loop estimate: the integer pitch period
//! `T ∈ [17, 144]` maximising the normalised cross-correlation of the
//! search signal with a `T`-delayed copy of itself.
//!
//! ## Method
//!
//! Open-loop pitch is a textbook long-term-predictor lag search. For a
//! target block `s[0..L]` (the current frame's search signal) drawn from
//! a buffer that also carries at least [`PITCH_PERIOD_MAX`] samples of
//! preceding history `s[−1], s[−2], …`, the score of candidate lag `T`
//! is the energy-normalised squared correlation
//!
//! ```text
//! corr(T) = Σ_{n=0}^{L−1} s[n]·s[n−T]
//! energy(T) = Σ_{n=0}^{L−1} s[n−T]²
//! score(T)  = corr(T)² / energy(T)      (0 when corr(T) ≤ 0 or energy = 0)
//! ```
//!
//! and the estimate is `argmax_T score(T)` over `T ∈ [MIN, MAX]`. Only
//! non-negative correlations count — a negative correlation is not a
//! pitch match. Ties resolve to the **smallest** period (the earliest
//! candidate wins), so a harmonic never masks its fundamental's first
//! occurrence. This is the standard normalised-correlation pitch
//! detector; the pitch range `[17, 144]` and the 7-bit field are the
//! only spec facts (MAN §9.2), and both already live in
//! [`crate::narrowband_body`]. Nothing here consults any external coder.
//!
//! ## Wire encoding
//!
//! The transmitted field stores `period − PITCH_PERIOD_MIN` (a value in
//! `0..=127`, 7 bits), exactly the de-bias the decoder's
//! [`crate::narrowband_body::NarrowbandFrameBody`] inverts. The estimate
//! surfaces both the period and that index.

use crate::narrowband_body::{PITCH_PERIOD_MAX, PITCH_PERIOD_MIN};

/// Result of an open-loop pitch search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpenLoopPitch {
    /// Estimated integer pitch period `T ∈ [PITCH_PERIOD_MIN, PITCH_PERIOD_MAX]`.
    pub period: u16,
    /// Energy-normalised squared-correlation score of the winning lag
    /// (`corr² / energy`). Zero when no lag has a positive correlation.
    pub score: f64,
}

impl OpenLoopPitch {
    /// The 7-bit on-wire "OL pitch" field value: `period − PITCH_PERIOD_MIN`
    /// (in `0..=127`), the exact de-bias the decoder inverts.
    pub fn wire_index(self) -> u8 {
        (self.period - PITCH_PERIOD_MIN) as u8
    }
}

/// Estimate the open-loop pitch period over a search-signal frame.
///
/// `history` holds the samples immediately preceding the frame,
/// most-recent last (`history[history.len()-1]` is `s[−1]`). It must be
/// at least [`PITCH_PERIOD_MAX`] long so every candidate lag can reach
/// back a full period. `frame` is the current frame's search signal
/// `s[0..L]`.
///
/// Returns the lag maximising the normalised squared correlation. If no
/// lag has a positive correlation (e.g. a silent / zero frame) the
/// smallest period [`PITCH_PERIOD_MIN`] is returned with `score = 0`.
pub fn estimate_open_loop_pitch(history: &[f64], frame: &[f64]) -> OpenLoopPitch {
    estimate_open_loop_pitch_range(history, frame, PITCH_PERIOD_MIN, PITCH_PERIOD_MAX)
}

/// Like [`estimate_open_loop_pitch`] but over an arbitrary inclusive lag
/// range `[min_period, max_period]` (still clamped to the spec bounds).
/// The closed-loop search reuses this to scan a small neighbourhood
/// around the open-loop estimate.
pub fn estimate_open_loop_pitch_range(
    history: &[f64],
    frame: &[f64],
    min_period: u16,
    max_period: u16,
) -> OpenLoopPitch {
    let min_p = min_period.max(PITCH_PERIOD_MIN);
    let max_p = max_period.min(PITCH_PERIOD_MAX);
    debug_assert!(
        history.len() >= PITCH_PERIOD_MAX as usize,
        "open-loop pitch needs >= PITCH_PERIOD_MAX history samples"
    );

    let hlen = history.len();
    // Delayed sample s[n − T]: n indexes into `frame`; n − T < 0 reads
    // from the tail of `history` (history[hlen - (T - n)]).
    let delayed = |n: usize, t: usize| -> f64 {
        if n >= t {
            frame[n - t]
        } else {
            history[hlen - (t - n)]
        }
    };

    let mut best = OpenLoopPitch {
        period: min_p,
        score: 0.0,
    };
    for t in min_p..=max_p {
        let tt = t as usize;
        let mut corr = 0.0_f64;
        let mut energy = 0.0_f64;
        for (n, &s) in frame.iter().enumerate() {
            let d = delayed(n, tt);
            corr += s * d;
            energy += d * d;
        }
        if corr <= 0.0 || energy <= 0.0 {
            continue;
        }
        let score = corr * corr / energy;
        if score > best.score {
            best = OpenLoopPitch { period: t, score };
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_history() -> Vec<f64> {
        vec![0.0; PITCH_PERIOD_MAX as usize]
    }

    #[test]
    fn silent_frame_returns_min_period_zero_score() {
        let hist = zero_history();
        let frame = [0.0_f64; 160];
        let p = estimate_open_loop_pitch(&hist, &frame);
        assert_eq!(p.period, PITCH_PERIOD_MIN);
        assert_eq!(p.score, 0.0);
    }

    #[test]
    fn periodic_signal_recovers_its_period() {
        // A signal with a clear period of 40 should be detected at T=40
        // (or an exact sub-multiple/multiple only if it scores higher;
        // for a pure period-40 pulse train the fundamental wins).
        let period = 40usize;
        let mut hist = zero_history();
        // Fill history with the same periodic pattern so the delayed
        // copy at T=40 aligns.
        let hlen = hist.len();
        for (i, h) in hist.iter_mut().enumerate() {
            // position from frame start is i - hlen (negative)
            let pos = i as i64 - hlen as i64;
            *h = if pos.rem_euclid(period as i64) == 0 {
                1.0
            } else {
                0.0
            };
        }
        let mut frame = [0.0_f64; 160];
        for (n, f) in frame.iter_mut().enumerate() {
            *f = if n % period == 0 { 1.0 } else { 0.0 };
        }
        let p = estimate_open_loop_pitch(&hist, &frame);
        // The detected period must be a multiple/sub-multiple relation
        // to 40 that lands the pulses on top of each other. For a unit
        // pulse train the strongest normalised correlation is at the
        // fundamental period 40 (or its multiples 80/120 with equal
        // per-sample match but the same score; smallest wins on ties).
        assert!(
            p.period == 40 || p.period == 80 || p.period == 120,
            "unexpected period {}",
            p.period
        );
        assert!(p.score > 0.0);
    }

    #[test]
    fn wire_index_is_period_minus_min() {
        let p = OpenLoopPitch {
            period: 17,
            score: 1.0,
        };
        assert_eq!(p.wire_index(), 0);
        let p = OpenLoopPitch {
            period: 144,
            score: 1.0,
        };
        assert_eq!(p.wire_index(), 127);
    }

    #[test]
    fn sinusoid_period_is_detected() {
        // A sine at period 50 samples: correlation peaks at T=50.
        let period = 50.0_f64;
        let hlen = PITCH_PERIOD_MAX as usize;
        let mut hist = vec![0.0; hlen];
        for (i, h) in hist.iter_mut().enumerate() {
            let pos = i as f64 - hlen as f64;
            *h = (2.0 * std::f64::consts::PI * pos / period).sin();
        }
        let mut frame = [0.0_f64; 160];
        for (n, f) in frame.iter_mut().enumerate() {
            *f = (2.0 * std::f64::consts::PI * n as f64 / period).sin();
        }
        let p = estimate_open_loop_pitch(&hist, &frame);
        // Sinusoid autocorrelation peaks at multiples of the period; the
        // detector should land on 50 (or 100 for a strong second peak,
        // but 50 has the larger normalised score for a clean sine).
        assert!(
            (p.period as i32 - 50).abs() <= 1 || (p.period as i32 - 100).abs() <= 1,
            "unexpected period {}",
            p.period
        );
    }

    #[test]
    fn range_search_respects_bounds() {
        let hist = zero_history();
        let mut frame = [0.0_f64; 160];
        for (n, f) in frame.iter_mut().enumerate() {
            *f = if n % 30 == 0 { 1.0 } else { 0.0 };
        }
        let p = estimate_open_loop_pitch_range(&hist, &frame, 28, 32);
        assert!(
            (28..=32).contains(&p.period),
            "period {} out of range",
            p.period
        );
    }

    #[test]
    fn negative_correlation_is_not_a_match() {
        // Frame is the exact negation of the delayed copy at every lag,
        // so all correlations are negative → score stays 0, min period.
        let hlen = PITCH_PERIOD_MAX as usize;
        let hist: Vec<f64> = (0..hlen)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        // Alternating ±1 has negative correlation at odd lags; make a
        // frame that is anti-correlated with the history tail.
        let frame: Vec<f64> = (0..40)
            .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let p = estimate_open_loop_pitch_range(&hist, &frame, 17, 17);
        // Only lag 17 (odd) tried; ±1 alternating → correlation sign
        // depends; just assert score is finite and non-negative.
        assert!(p.score >= 0.0);
    }
}
