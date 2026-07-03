//! Encoder-side LPC analysis front-end — windowing, autocorrelation,
//! and the Levinson-Durbin recursion that turns a frame of input speech
//! into the order-10 short-term LPC predictor.
//!
//! This is the first stage of the Speex encoder (the inverse direction of
//! the decoder's [`crate::lsp_to_lpc`] / [`crate::synthesis`] path). The
//! decoder consumed quantised LSP indices and reconstructed the
//! synthesis filter `1/A(z)`; the encoder must first *find* `A(z)` from
//! the raw signal before quantising it. Per *The Speex Codec Manual*
//! §8.2 ("Linear Prediction") the steps are:
//!
//! 1. **Window** the analysis buffer with the asymmetric 200-sample
//!    analysis window centred on the 4th sub-frame (§9.1, the
//!    `lpc_analysis_window` data staged in [`crate::codebooks`]).
//! 2. **Autocorrelate** the windowed signal to order `N = 10`:
//!    `R(m) = Σ x[i]·x[i−m]` (§8.2, the Toeplitz right-hand side).
//! 3. **Stabilise** the autocorrelation: multiply `R(0)` by a number
//!    slightly above one (`1.0001`, the manual's worked value — "adding
//!    noise to the signal"), then apply the 11-tap autocorrelation lag
//!    window to `R(0..=10)` ("filtering in the frequency domain,
//!    reducing sharp resonances"). Both the floor and the lag window
//!    keep the recovered filter stable under finite precision.
//! 4. **Levinson-Durbin**: solve the symmetric Toeplitz system
//!    `R·a = r` for the order-10 LPC coefficients in `O(N²)` (§8.2,
//!    "Because R is Hermitian Toeplitz, the Levinson-Durbin algorithm
//!    can be used").
//!
//! The output is the ten predictor coefficients `a[0..10]` of the
//! analysis filter `A(z) = 1 − Σ aᵢ z⁻ⁱ` — the same convention the
//! decoder's [`crate::lsp_to_lpc`] produces, so an encoder built on this
//! front-end is round-trippable against the existing synthesis path
//! (window → autocorrelate → Levinson → [eventually quantise LSPs] →
//! decoder reconstructs the same envelope).
//!
//! Clean-room note: the windowing / autocorrelation / Levinson-Durbin
//! recursion are the textbook linear-prediction primitives the manual
//! §8.2 names and describes structurally; this module implements that
//! prose against the staged analysis + lag window **data**
//! ([`crate::codebooks::lpc_analysis_window_float`] /
//! [`crate::codebooks::lpc_lag_window_float`]). No external library
//! source is consulted. The white-noise floor `1.0001` is the manual's
//! own worked example value (§8.2 p.26).

use crate::codebooks::{
    lpc_analysis_window_float, lpc_lag_window_float, LPC_ANALYSIS_WINDOW_LEN, LPC_LAG_WINDOW_LEN,
};
use crate::lsp_to_lpc::LPC_ORDER;

/// The white-noise floor the manual applies to `R(0)` before the
/// Levinson-Durbin recursion: *"we multiply R(0) by a number slightly
/// above one (such as 1.0001), which is equivalent to adding noise to
/// the signal"* (manual §8.2 p.26). Keeps the recovered filter stable
/// under finite precision.
pub const LPC_NOISE_FLOOR: f64 = 1.0001;

/// Number of autocorrelation lags computed for an order-`LPC_ORDER`
/// analysis: `R(0)` through `R(LPC_ORDER)`, i.e. `LPC_ORDER + 1`
/// entries (matches the 11-tap lag window length).
pub const AUTOCORR_LAGS: usize = LPC_ORDER + 1;

/// Errors from the LPC analysis front-end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpcAnalysisError {
    /// The analysis input slice was shorter than
    /// [`LPC_ANALYSIS_WINDOW_LEN`] (200 samples) — the window cannot be
    /// applied.
    InputTooShort {
        /// The length the caller supplied.
        got: usize,
        /// The length the window requires.
        need: usize,
    },
}

impl core::fmt::Display for LpcAnalysisError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LpcAnalysisError::InputTooShort { got, need } => write!(
                f,
                "lpc analysis input too short: got {got} samples, need {need}"
            ),
        }
    }
}

impl std::error::Error for LpcAnalysisError {}

/// Apply the staged 200-sample asymmetric analysis window to the first
/// [`LPC_ANALYSIS_WINDOW_LEN`] samples of `input`.
///
/// Returns the windowed signal `w[i] = input[i] · window[i]`. The
/// asymmetric window is centred on the 4th sub-frame per manual §9.1 so
/// the LPC envelope best matches the part of the frame being coded.
pub fn apply_analysis_window(
    input: &[f64],
) -> Result<[f64; LPC_ANALYSIS_WINDOW_LEN], LpcAnalysisError> {
    if input.len() < LPC_ANALYSIS_WINDOW_LEN {
        return Err(LpcAnalysisError::InputTooShort {
            got: input.len(),
            need: LPC_ANALYSIS_WINDOW_LEN,
        });
    }
    let win = lpc_analysis_window_float();
    let mut out = [0.0f64; LPC_ANALYSIS_WINDOW_LEN];
    for (o, (&x, &w)) in out.iter_mut().zip(input.iter().zip(win.iter())) {
        *o = x * w;
    }
    Ok(out)
}

/// Compute the autocorrelation `R(0..=LPC_ORDER)` of a windowed signal:
/// `R(m) = Σ_{i=m}^{L-1} w[i]·w[i−m]` (manual §8.2 — the Toeplitz
/// system's coefficients).
///
/// `signal` is the windowed analysis buffer (length
/// [`LPC_ANALYSIS_WINDOW_LEN`]); the result is the `AUTOCORR_LAGS`-entry
/// autocorrelation sequence.
pub fn autocorrelate(signal: &[f64; LPC_ANALYSIS_WINDOW_LEN]) -> [f64; AUTOCORR_LAGS] {
    let mut r = [0.0f64; AUTOCORR_LAGS];
    autocorrelate_general(signal, &mut r);
    r
}

/// Order-generic autocorrelation core: fills `out[m] = R(m)` for
/// `m = 0..out.len()` over the whole `signal` slice.
fn autocorrelate_general(signal: &[f64], out: &mut [f64]) {
    for (m, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for i in m..signal.len() {
            acc += signal[i] * signal[i - m];
        }
        *slot = acc;
    }
}

/// Stabilise an autocorrelation sequence in place: apply the white-noise
/// floor to `R(0)` (`R(0) *= 1.0001`) then multiply every lag by the
/// staged lag window (manual §8.2 — the two finite-precision stability
/// safeguards).
pub fn stabilise_autocorrelation(r: &mut [f64; AUTOCORR_LAGS]) {
    debug_assert_eq!(lpc_lag_window_float().len(), LPC_LAG_WINDOW_LEN);
    stabilise_general(r);
}

/// Order-generic stabilisation core: the `R(0)` white-noise floor plus
/// the staged per-lag window applied to the leading `r.len()` lags (the
/// lag window is a per-lag multiplier, so a shorter — lower-order —
/// autocorrelation consumes its leading taps).
fn stabilise_general(r: &mut [f64]) {
    r[0] *= LPC_NOISE_FLOOR;
    let lag = lpc_lag_window_float();
    for (slot, &w) in r.iter_mut().zip(lag.iter()) {
        *slot *= w;
    }
}

/// The result of the Levinson-Durbin recursion: the LPC predictor
/// coefficients plus the residual prediction error energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LpcCoefficients {
    /// The `LPC_ORDER` predictor coefficients `a[0..LPC_ORDER]` of the
    /// analysis filter `A(z) = 1 − Σ a[i]·z⁻⁽ⁱ⁺¹⁾`.
    pub a: [f64; LPC_ORDER],
    /// The final residual prediction-error energy (always ≥ 0 for a
    /// valid autocorrelation; the energy left after the order-`N`
    /// predictor).
    pub error: f64,
}

/// Solve the symmetric Toeplitz system `R·a = r` for the order-`LPC_ORDER`
/// LPC coefficients via the Levinson-Durbin recursion (manual §8.2).
///
/// `r` is the (stabilised) autocorrelation sequence. Returns the
/// predictor coefficients and the residual error. A degenerate
/// all-silence frame (`R(0) == 0`) yields the all-zero predictor and
/// zero error — the natural "no spectral envelope" result.
pub fn levinson_durbin(r: &[f64; AUTOCORR_LAGS]) -> LpcCoefficients {
    let mut a = [0.0f64; LPC_ORDER];
    let error = levinson_durbin_general(r, &mut a);
    LpcCoefficients { a, error }
}

/// Order-generic Levinson-Durbin core: solves for `a.len()` predictor
/// coefficients from the `a.len() + 1`-lag autocorrelation `r`,
/// returning the residual error. Shared by the narrowband (order-10)
/// and high-band (order-8) analysis paths.
fn levinson_durbin_general(r: &[f64], a: &mut [f64]) -> f64 {
    let order = a.len();
    debug_assert!(r.len() > order);
    let mut error = r[0];

    if error <= 0.0 {
        // Silent / degenerate frame: no predictor.
        return 0.0;
    }

    for i in 0..order {
        // Reflection coefficient k = -(r[i+1] + Σ a[j]·r[i-j]) / error.
        let mut acc = r[i + 1];
        for j in 0..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / error;

        // Update the predictor with the new reflection coefficient.
        a[i] = k;
        // a[j] += k · a[i-1-j] for the lower half (use a snapshot to
        // avoid aliasing the in-place update).
        let half = i / 2;
        for j in 0..half {
            let tmp = a[j];
            a[j] += k * a[i - 1 - j];
            a[i - 1 - j] += k * tmp;
        }
        if i % 2 == 1 {
            a[half] += k * a[half];
        }

        error *= 1.0 - k * k;
        if error <= 0.0 {
            // Numerically exhausted — clamp to zero and stop refining.
            return 0.0;
        }
    }

    error
}

/// Run the complete LPC analysis front-end on a 200-sample analysis
/// buffer: window → autocorrelate → stabilise → Levinson-Durbin.
///
/// `input` must be at least [`LPC_ANALYSIS_WINDOW_LEN`] samples; only
/// the first 200 are windowed. Returns the order-10 [`LpcCoefficients`].
pub fn analyse(input: &[f64]) -> Result<LpcCoefficients, LpcAnalysisError> {
    let windowed = apply_analysis_window(input)?;
    let mut r = autocorrelate(&windowed);
    stabilise_autocorrelation(&mut r);
    Ok(levinson_durbin(&r))
}

/// Number of autocorrelation lags for the order-8 **high-band** analysis:
/// `R(0)` through `R(8)`.
pub const HB_AUTOCORR_LAGS: usize = crate::codebooks::HB_LPC_ORDER + 1;

/// The result of the order-8 high-band Levinson-Durbin recursion — the
/// high-band counterpart of [`LpcCoefficients`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HbLpcCoefficients {
    /// The eight predictor coefficients `a[0..8]` of the high-band
    /// analysis filter `A_hb(z) = 1 − Σ a[i]·z⁻⁽ⁱ⁺¹⁾`.
    pub a: [f64; crate::codebooks::HB_LPC_ORDER],
    /// The final residual prediction-error energy.
    pub error: f64,
}

/// Run the complete **high-band** (order-8) LPC analysis front-end on a
/// 200-sample analysis buffer of the 8 kHz high-band half-band signal:
/// window → autocorrelate (to order 8) → stabilise → Levinson-Durbin
/// (round r385 scope).
///
/// Per *The Speex Codec Manual* §10.1 the high-band linear prediction is
/// *"very similar to narrowband"* at the high-band LPC order 8
/// ([`crate::codebooks::HB_LPC_ORDER`]); the same staged 200-sample
/// analysis window and per-lag stabilisation window apply (the lag
/// window is a per-lag multiplier, so the order-8 path consumes its
/// leading nine taps). Returns the order-8 [`HbLpcCoefficients`] in the
/// same `A(z) = 1 − Σ aᵢ z⁻ⁱ⁻¹` convention as the narrowband path — the
/// input convention [`crate::lpc_to_lsp::hb_lpc_to_lsp`] consumes.
pub fn analyse_hb(input: &[f64]) -> Result<HbLpcCoefficients, LpcAnalysisError> {
    let windowed = apply_analysis_window(input)?;
    let mut r = [0.0f64; HB_AUTOCORR_LAGS];
    autocorrelate_general(&windowed, &mut r);
    stabilise_general(&mut r);
    let mut a = [0.0f64; crate::codebooks::HB_LPC_ORDER];
    let error = levinson_durbin_general(&r, &mut a);
    Ok(HbLpcCoefficients { a, error })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(len: usize) -> Vec<f64> {
        (0..len).map(|i| (i as f64) * 0.01).collect()
    }

    #[test]
    fn window_rejects_short_input() {
        let short = vec![0.0f64; LPC_ANALYSIS_WINDOW_LEN - 1];
        assert_eq!(
            apply_analysis_window(&short),
            Err(LpcAnalysisError::InputTooShort {
                got: LPC_ANALYSIS_WINDOW_LEN - 1,
                need: LPC_ANALYSIS_WINDOW_LEN,
            })
        );
    }

    #[test]
    fn window_multiplies_elementwise() {
        let input = vec![2.0f64; LPC_ANALYSIS_WINDOW_LEN + 10];
        let w = apply_analysis_window(&input).unwrap();
        let win = lpc_analysis_window_float();
        for (o, &g) in w.iter().zip(win.iter()) {
            assert!((o - 2.0 * g).abs() < 1e-12);
        }
    }

    #[test]
    fn autocorr_zero_signal_is_zero() {
        let sig = [0.0f64; LPC_ANALYSIS_WINDOW_LEN];
        let r = autocorrelate(&sig);
        assert!(r.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn autocorr_r0_is_signal_energy() {
        let mut sig = [0.0f64; LPC_ANALYSIS_WINDOW_LEN];
        for (i, s) in sig.iter_mut().enumerate() {
            *s = (i as f64).sin();
        }
        let r = autocorrelate(&sig);
        let energy: f64 = sig.iter().map(|&x| x * x).sum();
        assert!((r[0] - energy).abs() < 1e-9);
    }

    #[test]
    fn autocorr_is_symmetric_definition() {
        // R(m) computed by the forward sum equals the manual definition.
        let mut sig = [0.0f64; LPC_ANALYSIS_WINDOW_LEN];
        for (i, s) in sig.iter_mut().enumerate() {
            *s = ((i % 7) as f64) - 3.0;
        }
        let r = autocorrelate(&sig);
        for m in 0..AUTOCORR_LAGS {
            let mut want = 0.0f64;
            for i in m..LPC_ANALYSIS_WINDOW_LEN {
                want += sig[i] * sig[i - m];
            }
            assert!((r[m] - want).abs() < 1e-9, "lag {m}");
        }
    }

    #[test]
    fn stabilise_applies_floor_and_lag_window() {
        let mut r = [1.0f64; AUTOCORR_LAGS];
        stabilise_autocorrelation(&mut r);
        let lag = lpc_lag_window_float();
        // R(0) carries floor × lag[0] (lag[0] == 1.0).
        assert!((r[0] - LPC_NOISE_FLOOR * lag[0]).abs() < 1e-12);
        for i in 1..AUTOCORR_LAGS {
            assert!((r[i] - lag[i]).abs() < 1e-12, "lag {i}");
        }
    }

    #[test]
    fn levinson_silent_frame_yields_zero_predictor() {
        let r = [0.0f64; AUTOCORR_LAGS];
        let c = levinson_durbin(&r);
        assert!(c.a.iter().all(|&x| x == 0.0));
        assert_eq!(c.error, 0.0);
    }

    #[test]
    fn levinson_residual_error_is_non_increasing_and_bounded() {
        // For a real signal the residual error must be in [0, R(0)].
        let sig = ramp(LPC_ANALYSIS_WINDOW_LEN);
        let windowed = apply_analysis_window(&sig).unwrap();
        let mut r = autocorrelate(&windowed);
        stabilise_autocorrelation(&mut r);
        let c = levinson_durbin(&r);
        assert!(c.error >= 0.0, "error must be non-negative");
        assert!(c.error <= r[0] + 1e-9, "error must not exceed R(0)");
    }

    #[test]
    fn levinson_recovers_known_ar_process_filter() {
        // Generate a signal from a known order-2 AR process
        // x[n] = a1·x[n-1] + a2·x[n-2] + impulse, then check the
        // recovered predictor approximates (-a1, -a2) in the leading
        // coefficients (A(z) = 1 - a1 z^-1 - a2 z^-2 → predictor a[0]
        // = -a1_true under our sign convention).
        let a1 = 0.6f64;
        let a2 = -0.3f64;
        let n = LPC_ANALYSIS_WINDOW_LEN + 50;
        let mut x = vec![0.0f64; n];
        // Excite with a deterministic pseudo-random innovation.
        let mut seed = 12345u64;
        for i in 2..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let innov = ((seed >> 40) as f64 / (1u64 << 24) as f64) - 0.5;
            x[i] = a1 * x[i - 1] + a2 * x[i - 2] + innov;
        }
        // Use the steady-state tail as the analysis buffer.
        let buf = &x[50..];
        let c = analyse(buf).unwrap();
        // The recovered predictor's leading two coefficients should be
        // close to -a1, -a2 (the AR coefficients enter A(z) negated).
        assert!(
            (c.a[0] + a1).abs() < 0.2,
            "a[0]={} should be near {}",
            c.a[0],
            -a1
        );
        assert!(
            (c.a[1] + a2).abs() < 0.2,
            "a[1]={} should be near {}",
            c.a[1],
            -a2
        );
    }

    #[test]
    fn analyse_full_pipeline_runs_on_real_length_buffer() {
        let sig = ramp(LPC_ANALYSIS_WINDOW_LEN + 5);
        let c = analyse(&sig).unwrap();
        assert_eq!(c.a.len(), LPC_ORDER);
        assert!(c.error.is_finite() && c.error >= 0.0);
        assert!(c.a.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn analyse_propagates_short_input_error() {
        let sig = ramp(LPC_ANALYSIS_WINDOW_LEN - 1);
        assert!(analyse(&sig).is_err());
    }

    #[test]
    fn autocorr_lags_matches_lag_window_length() {
        assert_eq!(AUTOCORR_LAGS, LPC_LAG_WINDOW_LEN);
        assert_eq!(AUTOCORR_LAGS, LPC_ORDER + 1);
    }

    // ---- Order-8 high-band analysis ----

    #[test]
    fn hb_autocorr_lags_is_order_8_plus_one() {
        assert_eq!(HB_AUTOCORR_LAGS, crate::codebooks::HB_LPC_ORDER + 1);
        assert_eq!(HB_AUTOCORR_LAGS, 9);
    }

    #[test]
    fn analyse_hb_full_pipeline_runs_on_real_length_buffer() {
        let sig = ramp(LPC_ANALYSIS_WINDOW_LEN + 5);
        let c = analyse_hb(&sig).unwrap();
        assert_eq!(c.a.len(), crate::codebooks::HB_LPC_ORDER);
        assert!(c.error.is_finite() && c.error >= 0.0);
        assert!(c.a.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn analyse_hb_silent_input_yields_zero_predictor() {
        let sig = vec![0.0f64; LPC_ANALYSIS_WINDOW_LEN];
        let c = analyse_hb(&sig).unwrap();
        assert!(c.a.iter().all(|&x| x == 0.0));
        assert_eq!(c.error, 0.0);
    }

    #[test]
    fn analyse_hb_propagates_short_input_error() {
        let sig = ramp(LPC_ANALYSIS_WINDOW_LEN - 1);
        assert!(analyse_hb(&sig).is_err());
    }

    #[test]
    fn analyse_hb_recovers_known_ar_process_filter() {
        // Same AR(2) recovery pin as the order-10 test, at order 8: the
        // recovered leading coefficients approximate the generating
        // filter under the shared sign convention.
        let a1 = 0.6f64;
        let a2 = -0.3f64;
        let n = LPC_ANALYSIS_WINDOW_LEN + 50;
        let mut x = vec![0.0f64; n];
        let mut seed = 12345u64;
        for i in 2..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let innov = ((seed >> 40) as f64 / (1u64 << 24) as f64) - 0.5;
            x[i] = a1 * x[i - 1] + a2 * x[i - 2] + innov;
        }
        let c = analyse_hb(&x[50..]).unwrap();
        assert!(
            (c.a[0] + a1).abs() < 0.2,
            "a[0]={} should be near {}",
            c.a[0],
            -a1
        );
        assert!(
            (c.a[1] + a2).abs() < 0.2,
            "a[1]={} should be near {}",
            c.a[1],
            -a2
        );
    }

    #[test]
    fn analyse_hb_matches_analyse_on_leading_reflection_structure() {
        // Both orders analyse the same buffer through the same window /
        // stabilisation; the order-8 predictor is the order-10 recursion
        // stopped two steps earlier, so its prediction error must be
        // >= the order-10 error (each Levinson step never increases it).
        let n = LPC_ANALYSIS_WINDOW_LEN + 20;
        let mut x = vec![0.0f64; n];
        let mut seed = 777u64;
        for i in 2..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let innov = ((seed >> 40) as f64 / (1u64 << 24) as f64) - 0.5;
            x[i] = 0.5 * x[i - 1] - 0.2 * x[i - 2] + innov;
        }
        let c10 = analyse(&x).unwrap();
        let c8 = analyse_hb(&x).unwrap();
        assert!(
            c8.error >= c10.error - 1e-9,
            "order-8 error {} must be >= order-10 error {}",
            c8.error,
            c10.error
        );
    }
}
