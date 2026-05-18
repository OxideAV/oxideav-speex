//! Speex intensity-stereo side channel (float path).
//!
//! Speex encodes stereo as a mono CELP frame plus a tiny 8-bit
//! intensity side channel delivered as an in-band request packet
//! (Table 5.1 code 9, "Intensity stereo information" — Speex manual
//! §5.5). The decoder expands the mono output to left/right by
//! multiplying each mono sample with a pair of smoothed gains
//! derived from:
//!   * `balance` — the left/right energy ratio, coded as a signed 5-bit
//!     exponent (`exp(sign * 0.25 * dexp)`),
//!   * `e_ratio` — the total-vs-sum-of-sides energy coherence, 2 bits
//!     indexing into `{0.25, 0.315, 0.397, 0.5}`.
//!
//! The float-mode reconstruction is:
//!   * `e_right = 1 / sqrt(e_ratio · (1 + balance))`
//!   * `e_left  = sqrt(balance) · e_right`
//! with per-sample smoothing `smooth ← 0.98·smooth + 0.02·e_*` applied
//! from the last sample backwards so the iteration can operate on an
//! in-place `data[i]` → `data[2i], data[2i+1]` expansion.
//!
//! The encoder direction is the inverse: from per-frame `eL`, `eR`, `eM`
//! energies the encoder picks (sign, dexp, e_ratio_idx) and writes the
//! 8-bit payload, then mixes L/R down to a mono frame that the standard
//! CELP encoder consumes. See [`StereoSideChannel::from_lr`] and
//! [`StereoSideChannel::mix_to_mono`].

use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{Error, Result};

/// 4-bit in-band request id assigned to "intensity stereo information"
/// in Speex manual §5.5 Table 5.1. Follows the 4-bit `m=14` marker (and
/// the 1-bit wideband prefix `0`) that introduces every in-band
/// request.
pub const SPEEX_INBAND_STEREO: u32 = 9;

/// Float-path energy-ratio quantisation table — `e_ratio_quant` in
/// `stereo.c`. The 2-bit `e_ratio` field indexes into this array.
const E_RATIO_QUANT: [f32; 4] = [0.25, 0.315, 0.397, 0.5];

/// Per-stream intensity-stereo state — mirrors `RealSpeexStereoState`.
/// Carried across frames so the smoothing filter has the history the
/// reference expects (a hard reset would audibly click between frames).
#[derive(Clone, Copy, Debug)]
pub struct StereoState {
    pub balance: f32,
    pub e_ratio: f32,
    pub smooth_left: f32,
    pub smooth_right: f32,
}

impl Default for StereoState {
    fn default() -> Self {
        Self::new()
    }
}

impl StereoState {
    /// Neutral state: balance = 1 (equal L/R energy), e_ratio = 0.5
    /// (centre of the quantiser), smooth gains primed to unity. Matches
    /// `speex_stereo_state_reset` (float path).
    pub const fn new() -> Self {
        Self {
            balance: 1.0,
            e_ratio: 0.5,
            smooth_left: 1.0,
            smooth_right: 1.0,
        }
    }

    /// Consume one intensity-stereo payload from the bitstream. The
    /// caller is expected to have already read the 4-bit `m=14` marker
    /// AND the 4-bit `id=9` request tag; this reads the remaining 8
    /// bits (`1-bit sign + 5-bit dexp + 2-bit e_ratio_idx`).
    pub fn read_side_channel(&mut self, br: &mut BitReader<'_>) -> Result<()> {
        let sign_bit = br.read_u32(1)?;
        let sign: f32 = if sign_bit != 0 { -1.0 } else { 1.0 };
        let dexp = br.read_u32(5)? as f32;
        self.balance = (sign * 0.25 * dexp).exp();
        let idx = br.read_u32(2)? as usize;
        // `read_u32(2)` is in [0, 3]; indexing is always in bounds.
        self.e_ratio = E_RATIO_QUANT[idx];
        Ok(())
    }

    /// Expand a mono buffer of `frame_size` float samples in-place into
    /// `2·frame_size` interleaved (L, R, L, R, …) samples using the
    /// current intensity-stereo parameters. `data` must be at least
    /// `2·frame_size` long; the first `frame_size` entries hold the mono
    /// input (the tail is overwritten).
    ///
    /// Iteration runs backwards so an in-place expansion is valid — same
    /// trick `speex_decode_stereo` uses to avoid a scratch buffer. Each
    /// iteration advances the smoothing filter by one sample, so the
    /// smoothed L/R gains converge to the per-frame `e_left` / `e_right`
    /// target over the first ~70 samples.
    pub fn expand_mono_in_place(&mut self, data: &mut [f32], frame_size: usize) -> Result<()> {
        if data.len() < 2 * frame_size {
            return Err(Error::invalid(format!(
                "Speex stereo: buffer length {} < 2 * {frame_size}",
                data.len()
            )));
        }
        // `e_right = 1 / sqrt(e_ratio · (1 + balance))`, `e_left
        //   = sqrt(balance) · e_right`. Clamp the argument so a
        // mis-behaving payload (balance ≤ -1 is impossible from the
        // `exp(·)` decode but e_ratio can legitimately approach 0.25) can
        // never divide by zero.
        let arg = self.e_ratio * (1.0 + self.balance);
        let e_right = if arg > 1e-12 { 1.0 / arg.sqrt() } else { 1.0 };
        let e_left = self.balance.max(0.0).sqrt() * e_right;

        for i in (0..frame_size).rev() {
            let tmp = data[i];
            self.smooth_left = 0.98 * self.smooth_left + 0.02 * e_left;
            self.smooth_right = 0.98 * self.smooth_right + 0.02 * e_right;
            data[2 * i] = self.smooth_left * tmp;
            data[2 * i + 1] = self.smooth_right * tmp;
        }
        Ok(())
    }
}

/// Advance past a non-stereo in-band request payload. The skip ladder
/// matches the size column of Speex manual §5.5 Table 5.1 — every
/// in-band request advertises a fixed payload width so an unrecognised
/// code can be opaquely skipped without desynchronising the CELP
/// reader. `id` is the 4-bit request tag already read from the
/// bitstream.
pub fn inband_skip_bits(id: u32) -> u32 {
    if id < 2 {
        1
    } else if id < 8 {
        4
    } else if id < 10 {
        8
    } else if id < 12 {
        16
    } else if id < 14 {
        32
    } else {
        64
    }
}

/// One frame's worth of intensity-stereo side-channel parameters,
/// pre-quantised to the on-wire layout (sign, dexp, e_ratio_idx) so the
/// encoder can write the 8-bit payload bit-for-bit without re-running
/// the quantiser. Round-trips through `read_side_channel` exactly when
/// the same triple is fed back in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StereoSideChannel {
    /// 1-bit sign — `0` means left-loud (`balance > 1`), `1` means
    /// right-loud (`balance < 1`).
    pub sign: u32,
    /// 5-bit unsigned exponent in `[0, 31]`. `balance = exp(sign * 0.25
    /// * dexp)` (sign is mapped to ±1.0 before scaling).
    pub dexp: u32,
    /// 2-bit index into [`E_RATIO_QUANT`].
    pub e_ratio_idx: u32,
}

/// Encoder-side energy-ratio quantiser — same table as
/// [`E_RATIO_QUANT`], exposed for callers that want to compute their
/// own `e_ratio_idx` from a precomputed `eM / (eL + eR)` ratio. The
/// decoder always indexes through the same table so any value in this
/// list round-trips exactly.
pub const E_RATIO_QUANT_VALUES: [f32; 4] = E_RATIO_QUANT;

impl StereoSideChannel {
    /// Quantise per-frame energies into the on-wire 8-bit payload.
    ///
    /// `e_left`, `e_right`, `e_mono` are the **summed-squared-sample**
    /// energies (i.e. `Σ x[n]²`, not RMS) of the L channel, R channel,
    /// and downmixed mono signal respectively. They must be ≥ 0.
    ///
    /// The mapping is:
    ///   * `balance = e_left / e_right` ⇒ `dexp = round(4·|ln(balance)|)`
    ///     clamped to `[0, 31]`; `sign = (balance < 1) as u32`.
    ///   * `e_ratio = e_mono / (e_left + e_right)` ⇒ nearest entry in
    ///     [`E_RATIO_QUANT_VALUES`].
    ///
    /// Silent frames (`e_left == e_right == 0`) return the neutral
    /// payload `(sign=0, dexp=0, e_ratio_idx=3)` — matches the decoder's
    /// `StereoState::new` (balance=1, e_ratio=0.5).
    pub fn from_lr(e_left: f64, e_right: f64, e_mono: f64) -> Self {
        // Silent or near-silent frame: emit the neutral payload
        // matching `StereoState::new` so the decoder doesn't see a
        // step-change in balance/e_ratio when audio resumes.
        const SILENCE_EPS: f64 = 1e-6;
        if e_left + e_right < SILENCE_EPS {
            return Self {
                sign: 0,
                dexp: 0,
                e_ratio_idx: 3,
            };
        }

        // Floor at a tiny positive value so log/divide are always
        // well-defined for asymmetric near-silence on one channel only.
        const EPS: f64 = 1e-12;
        let el = e_left.max(EPS);
        let er = e_right.max(EPS);
        let em = e_mono.max(0.0);

        // |ln(balance)| with balance = el/er.
        let ln_bal = (el / er).ln();
        let abs_ln = ln_bal.abs();
        // 0.25·dexp ≈ |ln(balance)|  ⇒  dexp ≈ 4·|ln(balance)|.
        let dexp = (abs_ln * 4.0).round().clamp(0.0, 31.0) as u32;
        let sign: u32 = if ln_bal < 0.0 { 1 } else { 0 };

        // e_ratio target = e_mono / (e_left + e_right).
        let e_ratio = (em / (el + er)).clamp(0.0, 1.0) as f32;
        let mut best_idx = 0u32;
        let mut best_diff = f32::INFINITY;
        for (i, q) in E_RATIO_QUANT.iter().enumerate() {
            let d = (e_ratio - q).abs();
            if d < best_diff {
                best_diff = d;
                best_idx = i as u32;
            }
        }
        Self {
            sign,
            dexp,
            e_ratio_idx: best_idx,
        }
    }

    /// Write the in-band intensity-stereo request to `bw`:
    /// `wb=0 (1) || m=14 (4) || id=9 (4) || sign (1) || dexp (5) ||
    /// e_ratio_idx (2)` — 17 bits total.
    ///
    /// The wideband bit is fixed at 0 because every CELP frame
    /// (narrowband, wideband-low-band, ultra-wideband-low-band)
    /// introduces a request packet with a 0-prefix and the 4-bit
    /// `m=14` marker after that 1-bit narrowband indicator. The
    /// decoder in [`crate::nb_decoder`] expects exactly this layout
    /// when scanning for in-band markers ahead of the CELP frame
    /// proper.
    pub fn write_inband(&self, bw: &mut BitWriter) {
        // 5-bit prefix: wb=0 + m=14, packed MSB-first as 0b01110 (= 14).
        bw.write_bits(14, 5);
        bw.write_bits(SPEEX_INBAND_STEREO, 4);
        bw.write_bits(self.sign & 1, 1);
        bw.write_bits(self.dexp & 0x1f, 5);
        bw.write_bits(self.e_ratio_idx & 0x3, 2);
    }

    /// Reconstruct the quantised state this payload will produce on the
    /// decoder side — handy for end-to-end sanity tests.
    pub fn to_state(self) -> StereoState {
        let sign_f: f32 = if self.sign != 0 { -1.0 } else { 1.0 };
        let balance = (sign_f * 0.25 * self.dexp as f32).exp();
        let e_ratio = E_RATIO_QUANT[(self.e_ratio_idx & 0x3) as usize];
        StereoState {
            balance,
            e_ratio,
            smooth_left: 1.0,
            smooth_right: 1.0,
        }
    }

    /// Down-mix interleaved L/R samples (i16) to a single mono f32
    /// buffer of length `frame_size`. Uses the additive mix `M = (L+R)/2`
    /// — the same downmix the manual §9 perceptual model assumes for
    /// the side-channel's energy-coherence parameter.
    ///
    /// `lr` must be 2·`frame_size` long (L, R, L, R, …).
    pub fn mix_to_mono(lr: &[i16], frame_size: usize) -> Result<Vec<f32>> {
        if lr.len() < 2 * frame_size {
            return Err(Error::invalid(format!(
                "Speex stereo encode: input length {} < 2 * {frame_size}",
                lr.len()
            )));
        }
        let mut out = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            let l = lr[2 * i] as f32;
            let r = lr[2 * i + 1] as f32;
            out[i] = 0.5 * (l + r);
        }
        Ok(out)
    }

    /// Compute per-channel energies for a single L/R frame. Returns
    /// `(e_left, e_right, e_mono)` as `f64` sums-of-squares.
    pub fn energies(lr: &[i16], frame_size: usize) -> Result<(f64, f64, f64)> {
        if lr.len() < 2 * frame_size {
            return Err(Error::invalid(format!(
                "Speex stereo encode: input length {} < 2 * {frame_size}",
                lr.len()
            )));
        }
        let mut el = 0.0f64;
        let mut er = 0.0f64;
        let mut em = 0.0f64;
        for i in 0..frame_size {
            let l = lr[2 * i] as f64;
            let r = lr[2 * i + 1] as f64;
            let m = 0.5 * (l + r);
            el += l * l;
            er += r * r;
            em += m * m;
        }
        Ok((el, er, em))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_neutral() {
        let s = StereoState::default();
        assert_eq!(s.balance, 1.0);
        assert_eq!(s.e_ratio, 0.5);
        assert_eq!(s.smooth_left, 1.0);
        assert_eq!(s.smooth_right, 1.0);
    }

    #[test]
    fn expand_with_neutral_state_approximately_doubles_buffer() {
        // balance=1, e_ratio=0.5 ⇒
        //   e_right = 1 / sqrt(0.5 * 2) = 1,
        //   e_left  = sqrt(1) * 1 = 1.
        // Smoothed gains start at 1 so output equals input on both
        // channels from the very first sample.
        let mut s = StereoState::default();
        let mut buf = vec![0.0f32; 32];
        for i in 0..16 {
            buf[i] = (i as f32) * 0.1;
        }
        s.expand_mono_in_place(&mut buf, 16).unwrap();
        for i in 0..16 {
            let expected = (i as f32) * 0.1;
            assert!((buf[2 * i] - expected).abs() < 1e-5, "L[{i}]");
            assert!((buf[2 * i + 1] - expected).abs() < 1e-5, "R[{i}]");
        }
    }

    #[test]
    fn side_channel_packet_updates_state() {
        // Pack a side channel with sign=+, dexp=4, idx=2 (e_ratio=0.397).
        // `exp(0.25 * 4) = e` ≈ 2.71828.
        // Bit layout (MSB-first): `0_00100_10` = 0b00010010 = 0x12.
        let mut br = BitReader::new(&[0x12]);
        let mut s = StereoState::default();
        s.read_side_channel(&mut br).unwrap();
        assert!((s.balance - std::f32::consts::E).abs() < 1e-5);
        assert!((s.e_ratio - 0.397).abs() < 1e-5);
    }

    #[test]
    fn asymmetric_balance_scales_channels_differently() {
        // Large balance (≫1) ⇒ e_left ≫ e_right ⇒ L is louder than R.
        let mut s = StereoState {
            balance: 9.0,
            e_ratio: 0.25,
            smooth_left: 1.0,
            smooth_right: 1.0,
        };
        // Drive the smoothing filter to convergence first by expanding a
        // long flat buffer; the 0.98/0.02 smoothing needs many samples.
        let mut warm = vec![0.0f32; 2048];
        for v in warm.iter_mut().take(1024) {
            *v = 1.0;
        }
        s.expand_mono_in_place(&mut warm, 1024).unwrap();
        // After 1024 samples the smoothed gains are very close to the
        // per-frame target values.
        // e_right = 1 / sqrt(0.25 * (1 + 9)) = 1 / sqrt(2.5)  ≈ 0.632
        // e_left  = sqrt(9) * e_right          = 3 * 0.632   ≈ 1.897
        // So the left channel should end up ≈ 3× the right channel.
        let ratio = s.smooth_left / s.smooth_right;
        assert!(
            (ratio - 3.0).abs() < 0.05,
            "L/R smoothed-gain ratio should be ≈3, got {ratio:.3}"
        );
    }

    #[test]
    fn side_channel_from_balanced_lr_is_neutral() {
        // Identical L/R energies + e_ratio targeted at 0.5 ⇒ payload
        // (sign=0, dexp=0, e_ratio_idx=3).
        let lr: Vec<i16> = (0..320)
            .map(|i| (1000.0 * ((i % 32) as f32 - 16.0)) as i16)
            .collect();
        // L=R copy.
        let (el, er, em) = StereoSideChannel::energies(&lr, 160).unwrap();
        let p = StereoSideChannel::from_lr(el, er, em);
        assert_eq!(p.sign, 0);
        assert_eq!(p.dexp, 0);
        // balanced L/R ⇒ e_ratio = (e_mono)/(2·e_left). With L=R,
        // mono=(L+R)/2=L so e_mono = e_left ⇒ ratio = 0.5 ⇒ idx=3.
        assert_eq!(p.e_ratio_idx, 3);
    }

    #[test]
    fn side_channel_sign_picks_louder_channel() {
        // L 3× louder than R ⇒ balance ≈ 9, ln(9) ≈ 2.197,
        // dexp ≈ 4·2.197 ≈ 8.79 → 9. sign=0.
        let mut lr = vec![0i16; 320];
        for i in 0..160 {
            lr[2 * i] = 3000;
            lr[2 * i + 1] = 1000;
        }
        let (el, er, em) = StereoSideChannel::energies(&lr, 160).unwrap();
        let p = StereoSideChannel::from_lr(el, er, em);
        assert_eq!(p.sign, 0, "left-loud ⇒ sign=0");
        assert!(
            (p.dexp as i32 - 9).abs() <= 1,
            "dexp ≈ 9 expected, got {}",
            p.dexp
        );

        // R louder ⇒ sign=1.
        let mut lr = vec![0i16; 320];
        for i in 0..160 {
            lr[2 * i] = 1000;
            lr[2 * i + 1] = 3000;
        }
        let (el, er, em) = StereoSideChannel::energies(&lr, 160).unwrap();
        let p = StereoSideChannel::from_lr(el, er, em);
        assert_eq!(p.sign, 1, "right-loud ⇒ sign=1");
    }

    #[test]
    fn side_channel_round_trips_through_writer_reader() {
        // Picking a known triple, encode it, then decode and confirm
        // we get back the same quantised state.
        let p = StereoSideChannel {
            sign: 1,
            dexp: 12,
            e_ratio_idx: 2,
        };
        let mut bw = BitWriter::new();
        p.write_inband(&mut bw);
        // 5 + 4 + 1 + 5 + 2 = 17 bits → 3 bytes after padding.
        let bytes = bw.finish();
        assert!(!bytes.is_empty());

        // Parse: skip wb-bit + m=14 + id=9 manually (matches what the
        // crate's CELP decoders do when they see m=14 + id=9), then ask
        // StereoState to consume the 8-bit payload.
        let mut br = BitReader::new(&bytes);
        let wb = br.read_u32(1).unwrap();
        let m = br.read_u32(4).unwrap();
        let id = br.read_u32(4).unwrap();
        assert_eq!(wb, 0);
        assert_eq!(m, 14);
        assert_eq!(id, SPEEX_INBAND_STEREO);
        let mut s = StereoState::default();
        s.read_side_channel(&mut br).unwrap();
        let want = p.to_state();
        assert!(
            (s.balance - want.balance).abs() < 1e-5,
            "balance mismatch: got {}, want {}",
            s.balance,
            want.balance
        );
        assert!((s.e_ratio - want.e_ratio).abs() < 1e-6, "e_ratio mismatch");
    }

    #[test]
    fn mix_to_mono_averages_l_and_r() {
        let lr: Vec<i16> = vec![100, 200, -300, 400, 1000, 0];
        let mono = StereoSideChannel::mix_to_mono(&lr, 3).unwrap();
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 150.0).abs() < 1e-3);
        assert!((mono[1] - 50.0).abs() < 1e-3);
        assert!((mono[2] - 500.0).abs() < 1e-3);
    }

    #[test]
    fn from_lr_handles_silent_frame() {
        let p = StereoSideChannel::from_lr(0.0, 0.0, 0.0);
        // Neutral payload — decoder side will see balance=1, e_ratio=0.5.
        assert_eq!(p.sign, 0);
        assert_eq!(p.dexp, 0);
        assert_eq!(p.e_ratio_idx, 3);
    }

    #[test]
    fn inband_skip_table_matches_reference_ladder() {
        // Mirrors the `speex_inband_handler` skip ladder.
        assert_eq!(inband_skip_bits(0), 1);
        assert_eq!(inband_skip_bits(1), 1);
        assert_eq!(inband_skip_bits(2), 4);
        assert_eq!(inband_skip_bits(7), 4);
        assert_eq!(inband_skip_bits(8), 8);
        assert_eq!(inband_skip_bits(9), 8);
        assert_eq!(inband_skip_bits(10), 16);
        assert_eq!(inband_skip_bits(12), 32);
        assert_eq!(inband_skip_bits(14), 64);
    }
}
