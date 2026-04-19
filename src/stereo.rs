//! Speex intensity-stereo side channel (float path).
//!
//! Port of `libspeex/stereo.c` (`speex_decode_stereo` +
//! `speex_std_stereo_request_handler`). Speex encodes stereo as a mono
//! CELP frame plus a tiny 8-bit intensity side channel delivered as an
//! in-band request packet (ID 9). The decoder expands the mono output to
//! left/right by multiplying each mono sample with a pair of smoothed
//! gains derived from:
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
//! in-place `data[i]` → `data[2i], data[2i+1]` expansion — mirrors the
//! reference byte-for-byte.
//!
//! Encoder direction is intentionally not implemented here (the crate's
//! focus is the decoder + a first-cut mono encoder); the reference's
//! `speex_encode_stereo_int` can be added alongside this without
//! touching the decoder path.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// `SPEEX_INBAND_STEREO` from `speex_callbacks.h` — the 4-bit request id
/// that follows an `m=14` in-band marker when the encoder attaches an
/// intensity-stereo side-channel payload.
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
    ///
    /// Mirrors `speex_std_stereo_request_handler`.
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

/// Advance past a non-stereo in-band request payload. The reference's
/// `speex_inband_handler` skips `adv` bits when no callback is
/// registered; we use the same ladder so a stream carrying
/// mode-change / VBR-quality / bitrate in-band requests doesn't
/// desynchronise the CELP reader.
///
/// `id` is the 4-bit request tag already read from the bitstream.
pub fn inband_skip_bits(id: u32) -> u32 {
    // Matches the ladder in libspeex/speex_callbacks.c.
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
