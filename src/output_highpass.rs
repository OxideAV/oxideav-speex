//! **Decoder output high-pass** (opt-in, round r393) — the
//! low-frequency rolloff the reference decoder applies to its output by
//! default.
//!
//! ## What the staged manual pins
//!
//! The Speex Codec Manual's codec-control table documents a high-pass
//! filter that is **on by default** on both coder directions
//! (`SPEEX_SET_HIGHPASS` — *"Set the high-pass filter on (1) or off
//! (0) … default is on"*). The manual does **not** print the filter's
//! transfer function; the exact coefficients are a recorded docs gap.
//!
//! ## What the fixture measures (behavioural trace)
//!
//! The staged `wb-mode1-folded` reference decode (produced with the
//! default high-pass active — only the perceptual enhancer was
//! disabled) shows, relative to this crate's raw decode, exactly the
//! signature of a low-cutoff high-pass: a phase lead decaying ≈ `1/f`
//! (`0.077 rad` at 440 Hz, `0.013 rad` at 2 kHz), unity magnitude
//! through the band, and real attenuation only below ≈ 50 Hz (×0.48 at
//! 30 Hz). Grid-fitting simple filters against the fixture:
//!
//! * no filter — 16.7 dB full-signal SNR (the r393 gate baseline);
//! * 1st-order, best ≈ 35 Hz — 18.1 dB;
//! * 2nd-order (this module), 30 Hz Butterworth — 18.3 dB, with a
//!   **flat optimum**: anything in ≈ 28–45 Hz / Q 0.7–1.2 lands within
//!   0.2 dB, so the data does not identify the reference's exact shape.
//!
//! This module therefore ships the interpretable reading — a
//! **2nd-order Butterworth high-pass at 30 Hz** — as an explicitly
//! *fitted, not reference-pinned* approximation. It is **opt-in**: no
//! decoder applies it implicitly, so existing decode outputs are
//! unchanged; apply it to match the reference pipeline's default.
//!
//! ```rust
//! use oxideav_speex::OutputHighpass;
//! let mut hp = OutputHighpass::for_sample_rate(16_000);
//! let mut pcm = vec![100.0f64; 320];
//! hp.process_slice(&mut pcm);
//! ```

use core::f64::consts::{FRAC_1_SQRT_2, PI};

/// The fitted cutoff frequency in Hz (module docs: flat optimum around
/// 28–45 Hz on the staged fixture; 30 Hz is the adopted reading).
pub const OUTPUT_HIGHPASS_CUTOFF_HZ: f64 = 30.0;

/// Opt-in decoder output high-pass: a 2nd-order Butterworth high-pass
/// biquad at [`OUTPUT_HIGHPASS_CUTOFF_HZ`], fitted against the staged
/// reference-decode fixture (module docs — the reference's exact
/// transfer is a recorded docs gap).
#[derive(Debug, Clone)]
pub struct OutputHighpass {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl OutputHighpass {
    /// Build the filter for a decoder output rate (8 000 / 16 000 /
    /// 32 000 Hz for the three Speex rate classes; any positive rate is
    /// accepted). The cutoff is the fixed
    /// [`OUTPUT_HIGHPASS_CUTOFF_HZ`]; whether the reference scales its
    /// cutoff with the rate class is part of the unpinned-transfer gap,
    /// so the absolute-cutoff convention is documented rather than
    /// assumed exact.
    pub fn for_sample_rate(rate_hz: u32) -> Self {
        let w = 2.0 * PI * OUTPUT_HIGHPASS_CUTOFF_HZ / f64::from(rate_hz.max(1));
        let c = 1.0 / (w / 2.0).tan();
        let q = FRAC_1_SQRT_2; // Butterworth
        let norm = c * c + c / q + 1.0;
        let b0 = c * c / norm;
        Self {
            b0,
            b1: -2.0 * b0,
            b2: b0,
            a1: 2.0 * (1.0 - c * c) / norm,
            a2: (c * c - c / q + 1.0) / norm,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Filter one sample, advancing the biquad state.
    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Filter a block in place (state carries across calls, so
    /// successive frames compose into one continuous stream).
    pub fn process_slice(&mut self, io: &mut [f64]) {
        for v in io.iter_mut() {
            *v = self.process(*v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Steady-state gain of the filter at `f_hz` measured empirically
    /// (feed a long sine, compare RMS of the tail).
    fn measured_gain(rate: u32, f_hz: f64) -> f64 {
        let mut hp = OutputHighpass::for_sample_rate(rate);
        let n = (f64::from(rate) * 2.0) as usize;
        let mut in_e = 0.0f64;
        let mut out_e = 0.0f64;
        for i in 0..n {
            let x = (2.0 * PI * f_hz * i as f64 / f64::from(rate)).sin();
            let y = hp.process(x);
            if i >= n / 2 {
                in_e += x * x;
                out_e += y * y;
            }
        }
        (out_e / in_e).sqrt()
    }

    /// DC is rejected: a constant input decays to ~zero.
    #[test]
    fn dc_is_rejected() {
        let mut hp = OutputHighpass::for_sample_rate(16_000);
        let mut last = f64::MAX;
        for i in 0..32_000 {
            last = hp.process(1000.0);
            if i < 100 {
                assert!(last.is_finite());
            }
        }
        assert!(last.abs() < 1e-3, "DC residue {last}");
    }

    /// The passband is unity well above the cutoff.
    #[test]
    fn passband_is_unity() {
        for rate in [8_000u32, 16_000, 32_000] {
            let g = measured_gain(rate, 1_000.0);
            assert!((g - 1.0).abs() < 0.01, "rate {rate}: gain {g}");
        }
    }

    /// Attenuation grows monotonically below the cutoff, with the
    /// half-power point near the design cutoff.
    #[test]
    fn stopband_attenuates_monotonically() {
        let g30 = measured_gain(16_000, OUTPUT_HIGHPASS_CUTOFF_HZ);
        let g10 = measured_gain(16_000, 10.0);
        let g100 = measured_gain(16_000, 100.0);
        assert!(g10 < g30 && g30 < g100, "{g10} {g30} {g100}");
        // Butterworth: −3 dB at the cutoff.
        assert!((g30 - FRAC_1_SQRT_2).abs() < 0.05, "half-power {g30}");
        assert!(g10 < 0.15, "10 Hz should be strongly attenuated: {g10}");
    }

    /// Block processing equals sample-by-sample processing (state
    /// continuity across process_slice calls).
    #[test]
    fn slice_matches_per_sample() {
        let mut a = OutputHighpass::for_sample_rate(16_000);
        let mut b = OutputHighpass::for_sample_rate(16_000);
        let src: Vec<f64> = (0..1000)
            .map(|i| ((i as f64) * 0.37).sin() * 500.0 + ((i as f64) * 0.011).cos() * 200.0)
            .collect();
        let mut blocked = src.clone();
        for chunk in blocked.chunks_mut(160) {
            a.process_slice(chunk);
        }
        for (i, &x) in src.iter().enumerate() {
            let y = b.process(x);
            assert!((blocked[i] - y).abs() < 1e-12, "i={i}");
        }
    }
}
