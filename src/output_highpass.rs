//! **Decoder output high-pass** (opt-in) — the low-frequency rolloff
//! the reference decoder applies to its output by default, with the
//! transfer **measured exactly** in round r450.
//!
//! ## What the staged manual pins
//!
//! The Speex Codec Manual's codec-control table documents a high-pass
//! filter that is **on by default** on both coder directions
//! (`SPEEX_SET_HIGHPASS` — *"Set the high-pass filter on (1) or off
//! (0) … default is on"*). The manual does **not** print the filter's
//! transfer function.
//!
//! ## The r450 measurement
//!
//! With the narrowband innovation path verified reference-exact to
//! 0.1 % (crafted pure-innovation streams,
//! `tests/fixtures/hb-gain-probes/NOTES.md`), the reference-vs-crate
//! cross-spectral transfer isolates the output high-pass directly.
//! Welch estimates over five crafted 8 kHz streams and two 16 kHz
//! streams, complex-LS-fitted to a biquad with a double zero at DC:
//!
//! * **8 kHz**: `fc ≈ 80.7 Hz, Q ≈ 0.870` (bilinear 2nd-order
//!   high-pass; poles r = 0.9642 at ±66 Hz; ≈ 5.7 % peaking around
//!   150 Hz; fit residual < 0.01 across 27 Hz–1.5 kHz);
//! * **16 kHz**: `fc ≈ 41.5 Hz, Q ≈ 1.118` (peaking ≈ 23 % around
//!   55 Hz);
//! * **32 kHz**: the measured response matches the 16 kHz filter's
//!   absolute-Hz response (half-power ≈ 30 Hz, peak ≈ 1.26 at 55 Hz),
//!   so the 16 kHz design carries over unchanged in Hz.
//!
//! The r393 30 Hz-Butterworth reading (a behavioural fit against one
//! fixture with a flat optimum) is superseded by these direct
//! measurements. The filter stays **opt-in**: no decoder applies it
//! implicitly, so existing decode outputs are unchanged; apply it to
//! match the reference pipeline's default.
//!
//! ```rust
//! use oxideav_speex::OutputHighpass;
//! let mut hp = OutputHighpass::for_sample_rate(16_000);
//! let mut pcm = vec![100.0f64; 320];
//! hp.process_slice(&mut pcm);
//! ```

use core::f64::consts::PI;

/// Measured cutoff of the 8 kHz (narrowband) output high-pass
/// (module docs — r450 cross-spectral measurement).
pub const OUTPUT_HIGHPASS_CUTOFF_HZ_8K: f64 = 80.7;
/// Measured resonance of the 8 kHz output high-pass.
pub const OUTPUT_HIGHPASS_Q_8K: f64 = 0.870;
/// Measured cutoff of the 16/32 kHz output high-pass biquad section.
pub const OUTPUT_HIGHPASS_CUTOFF_HZ: f64 = 41.75;
/// Measured resonance of the 16/32 kHz output high-pass biquad section.
pub const OUTPUT_HIGHPASS_Q: f64 = 1.38;
/// Measured cutoff of the extra first-order section the 16/32 kHz
/// response carries (the wide-band output rolls off at third order —
/// log-magnitude fit residual 1.2 % vs 3.1 % for a bare biquad).
pub const OUTPUT_HIGHPASS_FIRST_ORDER_HZ: f64 = 33.0;

/// Opt-in decoder output high-pass: the r450-measured bilinear
/// 2nd-order high-pass biquad (per-rate cutoff/Q constants above;
/// module docs).
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
    /// Optional first-order section (16/32 kHz — see
    /// [`OUTPUT_HIGHPASS_FIRST_ORDER_HZ`]): `(g, a, x_prev, y_prev)`.
    fo: Option<(f64, f64, f64, f64)>,
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
        let (fc, q, first_order) = if rate_hz <= 8_000 {
            (OUTPUT_HIGHPASS_CUTOFF_HZ_8K, OUTPUT_HIGHPASS_Q_8K, None)
        } else {
            (
                OUTPUT_HIGHPASS_CUTOFF_HZ,
                OUTPUT_HIGHPASS_Q,
                Some(OUTPUT_HIGHPASS_FIRST_ORDER_HZ),
            )
        };
        let w = 2.0 * PI * fc / f64::from(rate_hz.max(1));
        let c = 1.0 / (w / 2.0).tan();
        let norm = c * c + c / q + 1.0;
        let b0 = c * c / norm;
        let fo = first_order.map(|f1| {
            let k = (PI * f1 / f64::from(rate_hz.max(1))).tan();
            ((1.0 / (1.0 + k)), (1.0 - k) / (1.0 + k), 0.0, 0.0)
        });
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
            fo,
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
        match self.fo.as_mut() {
            Some((g, a, xp, yp)) => {
                let y1 = *g * (y - *xp) + *a * *yp;
                *xp = y;
                *yp = y1;
                y1
            }
            None => y,
        }
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

    /// The measured response points pin the per-rate designs: 8 kHz
    /// half-power near 59 Hz with ≈ 5.7 % peaking near 150 Hz; 16 kHz
    /// half-power near 30 Hz with ≈ 23 % peaking near 55 Hz.
    #[test]
    fn measured_response_points() {
        let g59 = measured_gain(8_000, 59.0);
        assert!((g59 - 0.53).abs() < 0.08, "8k 59 Hz {g59}");
        let g150 = measured_gain(8_000, 150.0);
        assert!((g150 - 1.057).abs() < 0.02, "8k 150 Hz peak {g150}");
        let g30 = measured_gain(16_000, 30.0);
        assert!((g30 - 0.47).abs() < 0.08, "16k 30 Hz {g30}");
        let g55 = measured_gain(16_000, 55.0);
        assert!((g55 - 1.23).abs() < 0.05, "16k 55 Hz peak {g55}");
        let g10 = measured_gain(16_000, 10.0);
        assert!(g10 < 0.15, "10 Hz strongly attenuated: {g10}");
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
