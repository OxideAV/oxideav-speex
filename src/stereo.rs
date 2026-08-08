//! **Intensity stereo** — in-band (mode-14 / code-9) channel
//! reconstruction and the encoder mono fold.
//!
//! A stereo Speex stream is a **mono bitstream plus a per-frame in-band
//! message**: code 9 of Manual Table 5.1 carries an 8-bit payload in
//! front of every audio frame, and the decoder turns that payload into a
//! left/right gain pair applied to the single decoded mono signal. There
//! is no second coded channel. The layout, reconstruction law and
//! encoder fold are the clean-room result of
//! `docs/audio/speex/intensity-stereo.md` (docs issue #325), pinned by
//! black-box observation of the reference codec.
//!
//! ## Payload (8 bits, MSB-first)
//!
//! ```text
//!  bit 0    : balance sign   0 = left louder, 1 = right louder
//!  bits 1-5 : balance index  0 .. 31
//!  bits 6-7 : e_ratio index  0 .. 3
//! ```
//!
//! ## Reconstruction ([`stereo_gains`])
//!
//! ```text
//!  b  = exp(bal / 8)
//!  F  = sqrt(0.5 / e_ratio[e]),  e_ratio = {0.25, 0.315, 0.397, 0.5}
//!  gL = sqrt(2) * F * b / sqrt(1 + b*b)
//!  gR = sqrt(2) * F     / sqrt(1 + b*b)
//!  if sign == 1: swap(gL, gR)
//!  L[n] = gL * m[n]  ;  R[n] = gR * m[n]
//! ```
//!
//! Equivalently `gL/gR = exp(bal/8)` (balance) and
//! `gL² + gR² = 1/e_ratio[e]` (total two-channel power).
//!
//! ## Intra-frame interpolation ([`StereoDecoder`])
//!
//! When the payload changes, the reference blends the new gains with the
//! previous frame's over one output block, weight on the **current**
//! frame's gains largest at the block's **first** sample:
//!
//! ```text
//!  g(i) = g_new * (1 - a^(N-i))  +  g_prev * a^(N-i),   a = 0.980
//! ```
//!
//! The block-phase offset the reference file carries (§4.1 of the note —
//! narrowband exact, wideband/ultra-wideband a fixed negative offset) is
//! a `speexdec`-pipeline detail; this decoder applies the interpolation
//! aligned to the frame and does not reproduce that sub-frame phase, so
//! byte-exactness against `expected.pcm` is bounded by it while the
//! per-sample gains are reference-correct.

/// The `e_ratio` ladder (docs `intensity-stereo.md` §3.1): geometric,
/// ratio ≈ 2^(1/3), fitting all three sampling modes uniformly.
pub const STEREO_E_RATIO: [f64; 4] = [0.25, 0.315, 0.397, 0.5];

/// Per-output-sample interpolation decay of the intra-frame gain blend
/// (docs §4): the same `a = 0.980` for every frame size.
pub const STEREO_INTERP_A: f64 = 0.980;

/// A reconstructed left/right gain pair for one frame's steady state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoGains {
    /// Left-channel scalar applied to the mono signal.
    pub gl: f64,
    /// Right-channel scalar applied to the mono signal.
    pub gr: f64,
}

/// Reconstruct the steady-state `(gL, gR)` from the 8-bit code-9 payload
/// (docs `intensity-stereo.md` §2 / §3).
pub fn stereo_gains(payload: u8) -> StereoGains {
    let sign = (payload >> 7) & 1;
    let bal = f64::from((payload >> 2) & 0x1F);
    let e = (payload & 0x03) as usize;
    let b = (bal / 8.0).exp();
    let f = (0.5 / STEREO_E_RATIO[e]).sqrt();
    let denom = (1.0 + b * b).sqrt();
    let gl = std::f64::consts::SQRT_2 * f * b / denom;
    let gr = std::f64::consts::SQRT_2 * f / denom;
    if sign == 1 {
        StereoGains { gl: gr, gr: gl }
    } else {
        StereoGains { gl, gr }
    }
}

/// Quantise an amplitude ratio `max/min` into the encoder's balance
/// index (docs §5): `clamp(round(8·ln(ratio)), 0, 31)`.
pub fn quantise_balance(ratio: f64) -> u8 {
    let idx = (8.0 * ratio.max(1.0).ln()).round();
    idx.clamp(0.0, 31.0) as u8
}

/// Choose the `e_ratio` index whose `1/e_ratio` total power is closest
/// to the input's measured `gL² + gR²` (docs §3/§5). `e = 3` is the
/// neutral unit-power point.
pub fn quantise_e_ratio(total_power: f64) -> u8 {
    let mut best = 3u8;
    let mut best_d = f64::INFINITY;
    for (i, &er) in STEREO_E_RATIO.iter().enumerate() {
        let d = (1.0 / er - total_power).abs();
        if d < best_d {
            best_d = d;
            best = i as u8;
        }
    }
    best
}

/// Pack `(sign, bal, e)` into the 8-bit code-9 payload (docs §2 layout).
pub fn pack_stereo_payload(sign: u8, bal: u8, e: u8) -> u8 {
    ((sign & 1) << 7) | ((bal & 0x1F) << 2) | (e & 0x03)
}

/// Stateful intensity-stereo **decoder**: applies the per-frame gains
/// with the §4 intra-frame interpolation, producing interleaved L/R.
#[derive(Debug, Clone)]
pub struct StereoDecoder {
    prev: StereoGains,
    seeded: bool,
}

impl Default for StereoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StereoDecoder {
    /// Fresh decoder; the pre-stream gains are the neutral `1.0` the
    /// start-up transient pulls toward (docs §3.1).
    pub fn new() -> Self {
        Self {
            prev: StereoGains { gl: 1.0, gr: 1.0 },
            seeded: false,
        }
    }

    /// Interleave one mono frame into L/R `i16` using the payload's
    /// reconstructed gains, blended from the previous frame's gains over
    /// the block (docs §4). Returns `2 * mono.len()` interleaved samples.
    pub fn interleave_frame(&mut self, mono: &[i16], payload: u8) -> Vec<i16> {
        let target = stereo_gains(payload);
        // First frame: no transient, use the target directly as prev so
        // the block starts at steady state.
        if !self.seeded {
            self.prev = target;
            self.seeded = true;
        }
        let n = mono.len();
        let mut out = Vec::with_capacity(n * 2);
        for (i, &s) in mono.iter().enumerate() {
            // Weight on the previous frame's gains: a^(N-i).
            let w_prev = STEREO_INTERP_A.powi((n - i) as i32);
            let gl = target.gl * (1.0 - w_prev) + self.prev.gl * w_prev;
            let gr = target.gr * (1.0 - w_prev) + self.prev.gr * w_prev;
            let sv = f64::from(s);
            out.push(clamp_i16(gl * sv));
            out.push(clamp_i16(gr * sv));
        }
        self.prev = target;
        out
    }
}

#[inline]
fn clamp_i16(v: f64) -> i16 {
    v.round().clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16
}

/// Encoder mono fold (docs §5): the transmitted signal is the plain
/// arithmetic mean of the two input channels.
#[inline]
pub fn downmix_mean(l: i16, r: i16) -> i16 {
    ((i32::from(l) + i32::from(r)) / 2) as i16
}

/// Choose the code-9 payload for a frame from its per-channel magnitude
/// measures (docs §5): balance from the amplitude ratio, e_ratio from
/// the total power `gL²+gR²`. `mag_l` / `mag_r` are any consistent
/// per-frame magnitude measures (RMS or mean-abs — the exact estimator
/// is not pinned, only the quantiser grid and clamp).
pub fn encode_stereo_payload(mag_l: f64, mag_r: f64) -> u8 {
    let (hi, lo, sign) = if mag_l >= mag_r {
        (mag_l, mag_r, 0u8)
    } else {
        (mag_r, mag_l, 1u8)
    };
    let ratio = if lo > 0.0 { hi / lo } else { f64::INFINITY };
    let bal = quantise_balance(ratio);
    // Reconstruct the ratio the decoder will see, then the total power so
    // the e_ratio quantiser matches the reconstruction domain.
    let b = f64::from(bal) / 8.0;
    let b = b.exp();
    // Total input power normalised so equal channels give ~unit gains.
    let denom = (mag_l * mag_l + mag_r * mag_r).max(1e-9);
    let total = 2.0 * (mag_l.max(mag_r).powi(2)) / denom * (1.0 + b * b) / (1.0 + b * b);
    let e = quantise_e_ratio(total.max(1.0));
    pack_stereo_payload(sign, bal, e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_payload_is_unit_gain() {
        // bal = 0, e = 3 -> gL = gR = 1.
        let g = stereo_gains(pack_stereo_payload(0, 0, 3));
        assert!((g.gl - 1.0).abs() < 1e-9);
        assert!((g.gr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn balance_sets_ratio_only() {
        for bal in [1u8, 5, 12, 22, 31] {
            let g = stereo_gains(pack_stereo_payload(0, bal, 3));
            let ratio = g.gl / g.gr;
            assert!(
                (ratio - (f64::from(bal) / 8.0).exp()).abs() < 1e-9,
                "bal={bal}"
            );
            // e = 3 -> total power 2.
            assert!((g.gl * g.gl + g.gr * g.gr - 2.0).abs() < 1e-9, "bal={bal}");
        }
    }

    #[test]
    fn sign_swaps_channels() {
        let a = stereo_gains(pack_stereo_payload(0, 16, 1));
        let b = stereo_gains(pack_stereo_payload(1, 16, 1));
        assert!((a.gl - b.gr).abs() < 1e-12);
        assert!((a.gr - b.gl).abs() < 1e-12);
    }

    #[test]
    fn e_ratio_sets_total_power() {
        for (e, er) in STEREO_E_RATIO.iter().enumerate() {
            let g = stereo_gains(pack_stereo_payload(0, 0, e as u8));
            assert!((g.gl * g.gl + g.gr * g.gr - 1.0 / er).abs() < 1e-9, "e={e}");
        }
    }

    #[test]
    fn payload_layout_round_trips() {
        for sign in 0..2u8 {
            for bal in 0..32u8 {
                for e in 0..4u8 {
                    let p = pack_stereo_payload(sign, bal, e);
                    assert_eq!((p >> 7) & 1, sign);
                    assert_eq!((p >> 2) & 0x1F, bal);
                    assert_eq!(p & 3, e);
                }
            }
        }
    }

    #[test]
    fn interpolation_starts_at_steady_state_on_first_frame() {
        let mut d = StereoDecoder::new();
        let mono = vec![1000i16; 160];
        // bal=6 -> left louder.
        let payload = pack_stereo_payload(0, 6, 3);
        let out = d.interleave_frame(&mono, payload);
        assert_eq!(out.len(), 320);
        let g = stereo_gains(payload);
        // First frame is steady (prev seeded to target): every sample uses g.
        assert_eq!(out[0], (g.gl * 1000.0).round() as i16);
        assert_eq!(out[1], (g.gr * 1000.0).round() as i16);
    }

    #[test]
    fn downmix_is_mean() {
        assert_eq!(downmix_mean(1000, 2000), 1500);
        assert_eq!(downmix_mean(-1000, 1000), 0);
    }

    #[test]
    fn balance_quantiser_matches_staged_ladder() {
        // docs §5: ratio 2 -> 6, 4 -> 11, 8 -> 17, 16 -> 22.
        assert_eq!(quantise_balance(2.0), 6);
        assert_eq!(quantise_balance(4.0), 11);
        assert_eq!(quantise_balance(8.0), 17);
        assert_eq!(quantise_balance(16.0), 22);
    }
}
