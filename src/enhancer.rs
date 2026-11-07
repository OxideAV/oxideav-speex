//! **Narrowband perceptual enhancer** — the decoder-side pitch comb on
//! the narrowband excitation (`docs/audio/speex/tracer-2026-09.md` §2).
//!
//! The enhancer is not a formant postfilter. It runs once per 80-sample
//! *block* of narrowband excitation, two blocks per frame — `e[−40..39]`
//! and `e[40..119]` relative to the frame, which is why the narrowband
//! synthesis runs one sub-frame behind the excitation decode
//! ([`crate::NarrowbandDecoder`]). Each block is filtered by a two-tap
//! fractional-delay comb
//!
//! ```text
//! ê[n] = κ · ( e[n] + a₀·y₀[n] + a₁·y₁[n] )
//! ```
//!
//! where `y₀` is the excitation interpolated at a delay searched around
//! the frame's enhancer period `T` (§2 "Period"), `y₁` the same around
//! `2T` (`T > 40`) or the look-ahead delay `−T`, `a₀/a₁` the
//! correlation-driven tap gains and `κ ≤ 1` the energy renormalisation
//! (the block never gains energy). The comb strength γ is per
//! narrowband submode ([`enhancer_comb_gain`], staged
//! `tables/enh-comb-gain.csv`); submode 1 has none.
//!
//! The enhanced excitation feeds **only** the synthesis filter: the
//! stored excitation (pitch memory, the high-band gain base `e_L`) is
//! never modified.

/// Excitation samples per enhancer block (two blocks per 160-sample
/// narrowband frame; tracer §2 "Blocks").
pub const ENHANCER_BLOCK_SAMPLES: usize = 80;

/// Taps of each fractional-delay interpolator row.
const FRAC_TAPS: usize = 7;

/// The three 7-tap fractional-delay interpolators (staged
/// `tables/enh-frac-delay-float.csv`): row 1 is a half-sample delay,
/// rows 2 and 3 quarter-sample delays either side. Row index `r` of
/// the tracer's §2 notation is `r = 1..=3` → `ENH_FRAC_DELAY[r − 1]`.
const ENH_FRAC_DELAY: [[f64; FRAC_TAPS]; 3] = [
    [
        -0.011915, 0.046995, -0.152373, 0.614108, 0.614108, -0.152373, 0.046995,
    ],
    [
        -0.0324855, 0.0859768, -0.2042986, 0.9640297, 0.2086420, -0.0302054, -0.0063646,
    ],
    [
        -0.0063646, -0.0302054, 0.2086420, 0.9640297, -0.2042986, 0.0859768, -0.0324855,
    ],
];

/// Per-submode comb strength γ (staged `tables/enh-comb-gain.csv`).
///
/// Returns `None` for the submodes whose enhancer is off (submode 1,
/// the vocoder) and for those that carry no CELP excitation (0, and the
/// non-audio ids).
pub fn enhancer_comb_gain(nb_submode: u8) -> Option<f64> {
    match nb_submode {
        2 => Some(0.60),
        3 => Some(0.55),
        4 => Some(0.45),
        5 => Some(0.25),
        6 => Some(0.15),
        7 => Some(0.05),
        8 => Some(0.50),
        _ => None,
    }
}

/// The frame's enhancer period `T` from the four sub-frames' pitch
/// lags and 3-tap gains `(b₀, b₁, b₂)` (tracer §2 "Period"), `b₁`
/// being the centre (lag-`p`) tap.
pub fn enhancer_period(subframes: &[(i32, [f64; 3]); 4]) -> i32 {
    let mut t = 40i32;
    let mut g = 0.0f64;
    for &(p, b) in subframes {
        let side = |v: f64| if v > 0.0 { v } else { -0.5 * v };
        let tap = b[1].abs() + side(b[0]) + side(b[2]);
        // p is (close to) a multiple of the current T …
        let p_multiple_of_t =
            (2 * t - p).abs() < 3 || (3 * t - p).abs() < 4 || (4 * t - p).abs() < 5;
        // … or the current T is (close to) a multiple of p.
        let t_multiple_of_p =
            (t - 2 * p).abs() < 3 || (t - 3 * p).abs() < 4 || (t - 4 * p).abs() < 5;
        if (tap > g && !p_multiple_of_t)
            || (tap > 0.6 * g && t_multiple_of_p)
            || (0.67 * tap > g && p_multiple_of_t)
        {
            t = p;
            if tap > g {
                g = tap;
            }
        }
    }
    t
}

/// What the comb chose for one block (diagnostics / trace gates; the
/// columns of the staged `fixtures/nb-enhancer-q4/enhancer-trace.csv`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EnhancerBlock {
    /// Integer lag of each tap (`lag1 < 0` = look-ahead tap).
    pub lag: [i32; 2],
    /// Fractional-delay row per tap (0 = integer delay, 1..=3 = the
    /// interpolator rows).
    pub phase: [u8; 2],
    /// Normalised correlations `p₀`, `p₁` in `[0, 1]`.
    pub norm_corr: [f64; 2],
    /// Tap gains `a₀`, `a₁`.
    pub tap_gain: [f64; 2],
    /// Final block renormalisation `κ = min(ρ_o, ρ_n)/ρ_n`.
    pub energy_renorm: f64,
}

/// Enhance one 80-sample block in place.
///
/// `exc(i)` returns the excitation sample at offset `i` from the block
/// start (negative = earlier history; samples past the decoded frame
/// read as `0`, the tracer's look-ahead convention). `period` is the
/// frame's [`enhancer_period`] and `gamma` the submode's comb strength.
pub fn enhance_block(
    exc: &dyn Fn(i32) -> f64,
    period: i32,
    gamma: f64,
    out: &mut [f64; ENHANCER_BLOCK_SAMPLES],
) -> EnhancerBlock {
    const N: i32 = ENHANCER_BLOCK_SAMPLES as i32;
    let delays = [period, if period > 40 { 2 * period } else { -period }];
    let weights = if period > 40 { [0.7, 0.3] } else { [0.6, 0.6] };
    let c1 = 0.4 * gamma + 0.07;
    let c2 = 0.5 + 1.72 * (c1 - 0.07);
    let e_energy: f64 = (0..N).map(|n| exc(n) * exc(n)).sum();
    let m_e = (1.0 + e_energy).sqrt();

    let mut taps = [[0.0f64; ENHANCER_BLOCK_SAMPLES]; 2];
    let mut info = EnhancerBlock {
        lag: [0; 2],
        phase: [0; 2],
        norm_corr: [0.0; 2],
        tap_gain: [0.0; 2],
        energy_renorm: 1.0,
    };
    for (ti, &d) in delays.iter().enumerate() {
        // Seven raw correlations at lags D+3−i, i = 0..6.
        let mut c = [0.0f64; FRAC_TAPS];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (0..N).map(|n| exc(n) * exc(n - d - 3 + i as i32)).sum();
        }
        // Best (row, phase) over the integer and the three interpolated
        // correlation curves; first strict maximum wins.
        let (mut best_r, mut best_j, mut best_v) = (0usize, 0i32, c[0]);
        for r in 0..4usize {
            for j in 0..FRAC_TAPS as i32 {
                let v = if r == 0 {
                    c[j as usize]
                } else {
                    let lo = (3 - j).max(0);
                    let hi = (9 - j).min(6);
                    (lo..=hi)
                        .map(|k| ENH_FRAC_DELAY[r - 1][k as usize] * c[(j + k - 3) as usize])
                        .sum()
                };
                if v > best_v {
                    best_v = v;
                    best_r = r;
                    best_j = j;
                }
            }
        }
        let lag = d - best_j + 3;
        let y = &mut taps[ti];
        for (n, slot) in y.iter_mut().enumerate() {
            let n = n as i32;
            *slot = if best_r == 0 {
                exc(n - lag)
            } else {
                (0..FRAC_TAPS)
                    .map(|k| ENH_FRAC_DELAY[best_r - 1][k] * exc(n - lag + k as i32 - 3))
                    .sum()
            };
        }
        let m_i = (1000.0 + y.iter().map(|v| v * v).sum::<f64>()).sqrt();
        let rho = (0..N).map(|n| y[n as usize] * exc(n)).sum::<f64>().max(0.0);
        let p = if rho > m_i * m_e {
            1.0
        } else {
            rho / (m_e * m_i)
        };
        let g = c1 / c1.max(1.0 - c2 * p * p);
        info.lag[ti] = lag;
        info.phase[ti] = best_r as u8;
        info.norm_corr[ti] = p;
        info.tap_gain[ti] = weights[ti] * g * m_e / m_i;
    }

    let mut z_energy = 0.0f64;
    for (n, slot) in out.iter_mut().enumerate() {
        let v = exc(n as i32) + info.tap_gain[0] * taps[0][n] + info.tap_gain[1] * taps[1][n];
        z_energy += v * v;
        *slot = v;
    }
    let rho_o = (0.1 + e_energy / f64::from(N)).sqrt().max(1.0);
    let rho_n = (0.1 + z_energy / f64::from(N)).sqrt().max(1.0);
    info.energy_renorm = rho_o.min(rho_n) / rho_n;
    for v in out.iter_mut() {
        *v *= info.energy_renorm;
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comb_gain_table() {
        assert_eq!(enhancer_comb_gain(1), None);
        assert_eq!(enhancer_comb_gain(3), Some(0.55));
        assert_eq!(enhancer_comb_gain(7), Some(0.05));
        assert_eq!(enhancer_comb_gain(0), None);
    }

    #[test]
    fn period_takes_strongest_non_multiple() {
        let sf = [
            (60, [0.0, 0.5, 0.0]),
            (60, [0.0, 0.6, 0.0]),
            (120, [0.0, 0.9, 0.0]),
            (61, [0.0, 0.7, 0.0]),
        ];
        // 120 ≈ 2·60 needs 0.67·t > G (0.603 > 0.6): taken; 61 is not a
        // multiple of 120 but 120 ≈ 2·61 → rule (b), taken.
        assert_eq!(enhancer_period(&sf), 61);
    }

    #[test]
    fn silent_block_is_unchanged() {
        let mut out = [1.0f64; ENHANCER_BLOCK_SAMPLES];
        let info = enhance_block(&|_| 0.0, 50, 0.55, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        assert_eq!(info.energy_renorm, 1.0);
    }

    #[test]
    fn periodic_block_never_gains_energy() {
        // A pulse train of period 50: the comb locks onto the integer lag.
        let f = |i: i32| if i.rem_euclid(50) == 7 { 1000.0 } else { 0.0 };
        let mut out = [0.0f64; ENHANCER_BLOCK_SAMPLES];
        let info = enhance_block(&f, 50, 0.55, &mut out);
        let e_in: f64 = (0..80).map(|n| f(n) * f(n)).sum();
        let e_out: f64 = out.iter().map(|v| v * v).sum();
        // Up to the +0.1 floor under the two RMS terms.
        assert!(e_out <= e_in * 1.000_01);
        assert_eq!(info.lag[0], 50);
        assert!(info.norm_corr[0] > 0.99);
    }
}
