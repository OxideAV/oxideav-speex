//! Speex ultra-wideband (sub-band CELP) decoder — float path.
//!
//! UWB stacks a second SB-CELP layer on top of a full wideband (WB)
//! decoder. The WB output is the "low band" of the UWB QMF (0–8 kHz
//! of the 32 kHz output); the UWB extension layer supplies the 8–16
//! kHz "high band" via an 8th-order LPC synthesis filter + spectral-
//! folding excitation drawn from the WB high-band innovation.
//!
//! Mirrors the `sb_uwb_mode` configuration in `libspeex/modes.c` /
//! `sb_celp.c`: `lpcSize = 8`, `subframeSize = 40`, `frameSize = 320`
//! (at the decimated 16 kHz rate), `subframes = 8`, and only WB
//! sub-mode 1 (spectral folding, 36 bits/UWB frame) is defined.
//!
//! Bitstream layout (after the WB payload):
//!
//! ```text
//!   +---- WB payload (NB + WB extension, variable bits) ----+
//!   | 1 bit   uwb-bit                                        |
//!   | 3 bit   UWB sub-mode id (only id=1 is defined)         |
//!   | 12 bit  high-band LSP VQ (2×6)                         |
//!   | 4 × 5 bit  folding gain per UWB sub-frame              |
//!   +--------------------------------------------------------+
//! ```
//!
//! 36 bits of UWB overhead total — exactly the `bits=36` recorded in
//! the reference's `wb_submode1` struct (reused verbatim for UWB). The
//! UWB layer uses 4 sub-frames of 80 samples each at the 16 kHz
//! decimated rate, matching `sb_uwb_mode.subframes = 4`.
//!
//! A `uwb_bit` of 0 means "no high-band extension this frame" — the
//! decoder zero-pads the UWB high band and upsamples the WB output
//! through QMF, which is exactly what the reference does for silent /
//! DTX frames. This path round-trips cleanly with our
//! [`crate::uwb_encoder`].
//!
//! The full folding path is also implemented here and exercised by the
//! integration tests. Sub-modes 2/3/4 (stochastic) are **not** defined
//! by the Speex UWB reference at all; UWB is folding-only.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::lsp::{lsp_interpolate, lsp_to_lpc, lsp_unquant_high};
use crate::nb_decoder::iir_mem16;
use crate::qmf::{qmf_synth, H0_PROTOTYPE, QMF_ORDER};
use crate::wb_decoder::{
    WbDecoder, FOLDING_GAIN, LSP_MARGIN_HIGH, WB_FRAME_SIZE, WB_FULL_FRAME_SIZE, WB_LPC_ORDER,
};

/// Number of UWB sub-frames per frame (`sb_uwb_mode.subframes = 4`).
pub const UWB_NB_SUBFRAMES: usize = 4;
/// UWB sub-frame size at the decimated 16 kHz rate — 320/4 = 80 samples.
pub const UWB_SUBFRAME_SIZE: usize = 80;
/// UWB frame size at the decimated 16 kHz rate (= WB full-frame size).
pub const UWB_FRAME_SIZE: usize = WB_FULL_FRAME_SIZE;
/// UWB full-band frame size at 32 kHz.
pub const UWB_FULL_FRAME_SIZE: usize = 640;

/// Only sub-mode id defined for UWB (spectral folding).
pub const UWB_SUBMODE_FOLDING: u32 = 1;

pub struct UwbDecoder {
    /// Wideband sub-decoder that handles the 0–16 kHz half of the
    /// output (NB + WB extension).
    wb: WbDecoder,
    /// Previous frame's quantised high-band (8–16 kHz) LSPs.
    old_qlsp: [f32; WB_LPC_ORDER],
    /// Interpolated LPC from the last sub-frame — feeds the 1-subframe
    /// delay line.
    interp_qlpc: [f32; WB_LPC_ORDER],
    /// LPC synthesis filter memory for the UWB high band.
    mem_sp: [f32; WB_LPC_ORDER],
    /// Previous sub-frame's excitation (1-sub-frame delay, 80 samples
    /// for UWB's bigger sub-frame).
    prev_exc: [f32; UWB_SUBFRAME_SIZE],
    /// QMF synthesis memory for the low branch (WB output).
    g0_mem: [f32; QMF_ORDER],
    /// QMF synthesis memory for the high branch (UWB extension).
    g1_mem: [f32; QMF_ORDER],
    first: bool,
}

impl Default for UwbDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl UwbDecoder {
    pub fn new() -> Self {
        let mut old_qlsp = [0.0f32; WB_LPC_ORDER];
        for i in 0..WB_LPC_ORDER {
            old_qlsp[i] = std::f32::consts::PI * (i as f32 + 1.0) / (WB_LPC_ORDER as f32 + 1.0);
        }
        Self {
            wb: WbDecoder::new(),
            old_qlsp,
            interp_qlpc: [0.0; WB_LPC_ORDER],
            mem_sp: [0.0; WB_LPC_ORDER],
            prev_exc: [0.0; UWB_SUBFRAME_SIZE],
            g0_mem: [0.0; QMF_ORDER],
            g1_mem: [0.0; QMF_ORDER],
            first: true,
        }
    }

    /// Intensity-stereo state forwarded from the embedded WB decoder
    /// (which forwards from the NB decoder). The UWB bitstream layers
    /// don't introduce a separate stereo payload; the `m=14, id=9`
    /// packet lives in the NB portion of the frame and applies to the
    /// full-band mono output.
    pub fn stereo_state(&self) -> &crate::stereo::StereoState {
        self.wb.stereo_state()
    }

    /// Mutable access — used by the top-level [`crate::decoder`] to
    /// expand the mono UWB output into L/R via
    /// [`crate::stereo::StereoState::expand_mono_in_place`].
    pub fn stereo_state_mut(&mut self) -> &mut crate::stereo::StereoState {
        self.wb.stereo_state_mut()
    }

    /// Decode one 640-sample UWB frame (32 kHz).
    pub fn decode_frame(&mut self, br: &mut BitReader, out: &mut [f32]) -> Result<()> {
        debug_assert_eq!(out.len(), UWB_FULL_FRAME_SIZE);

        // ---- Low band: full WB decode (320 samples at 16 kHz). ------
        let mut low = [0.0f32; UWB_FRAME_SIZE];
        self.wb.decode_frame(br, &mut low)?;

        // ---- UWB sub-mode selector ----------------------------------
        let mut submode_id = 0u32;
        if br.bits_remaining() >= 1 {
            let uwb_bit = br.read_u32(1)?;
            if uwb_bit == 1 {
                if br.bits_remaining() < 3 {
                    return Err(Error::invalid("Speex UWB: truncated sub-mode selector"));
                }
                submode_id = br.read_u32(3)?;
            }
        }

        // ---- High band: 320-sample buffer at 16 kHz (= half of 640) -
        let mut high = [0.0f32; UWB_FRAME_SIZE];

        if submode_id == 0 {
            // Null extension — silent high band, just QMF upsample.
            self.first = true;
        } else if submode_id == UWB_SUBMODE_FOLDING {
            self.decode_folding_layer(br, &mut high)?;
        } else {
            // UWB reference only defines sub-mode 1; reject the rest.
            return Err(Error::invalid(format!(
                "Speex UWB: unsupported sub-mode id {submode_id} (reference \
                 defines folding / id=1 only)"
            )));
        }

        // ---- QMF synthesis: low (WB) + high (UWB) → 640 samples ----
        qmf_synth(
            &low,
            &high,
            &H0_PROTOTYPE,
            out,
            UWB_FULL_FRAME_SIZE,
            QMF_ORDER,
            &mut self.g0_mem,
            &mut self.g1_mem,
        );

        Ok(())
    }

    fn decode_folding_layer(
        &mut self,
        br: &mut BitReader,
        high: &mut [f32; UWB_FRAME_SIZE],
    ) -> Result<()> {
        // ---- High-band LSPs (same codebook as WB high-band LSPs) ---
        let mut qlsp = [0.0f32; WB_LPC_ORDER];
        lsp_unquant_high(&mut qlsp, WB_LPC_ORDER, br)?;
        if self.first {
            self.old_qlsp.copy_from_slice(&qlsp);
        }

        // Pull context from the WB sub-decoder. The UWB folding source
        // is the WB high-band excitation (`wb.hb_innov()`) — that's
        // `SPEEX_SET_INNOVATION_SAVE` wired one level down.
        let wb_hb_innov = *self.wb.hb_innov();
        let wb_pi_gain = *self.wb.nb().pi_gain();

        // ---- 4 sub-frames × 80 samples = 320-sample UWB high band --
        let mut interp_qlsp = [0.0f32; WB_LPC_ORDER];
        let mut ak = [0.0f32; WB_LPC_ORDER];
        for sub in 0..UWB_NB_SUBFRAMES {
            let offset = UWB_SUBFRAME_SIZE * sub;

            lsp_interpolate(
                &self.old_qlsp,
                &qlsp,
                &mut interp_qlsp,
                WB_LPC_ORDER,
                sub,
                UWB_NB_SUBFRAMES,
                LSP_MARGIN_HIGH,
            );
            lsp_to_lpc(&interp_qlsp, &mut ak, WB_LPC_ORDER);

            // Filter-ratio between the "below" (WB) and UWB high-band
            // LPC filters at ω=π. UWB and WB's NB both have 4 sub-frames
            // so indices line up 1:1.
            let mut rh = 1.0f32;
            let mut k = 0;
            while k + 1 < WB_LPC_ORDER {
                rh += ak[k + 1] - ak[k];
                k += 2;
            }
            let rl_idx = sub.min(wb_pi_gain.len() - 1);
            let rl = wb_pi_gain[rl_idx];
            let filter_ratio = (rl + 0.01) / (rh + 0.01);

            // ---- Folding excitation (5-bit gain quant) -------------
            let quant = br.read_u32(5)? as i32;
            let g = (0.125f32 * (quant - 10) as f32).exp();
            let eps = 1e-6f32;
            let g = g / filter_ratio.max(eps);

            // Fold the WB high-band innovation into the UWB high band.
            // WB innov is 160 samples (4 × 40); UWB sub-frame is 80
            // samples so we consume two WB sub-frame slabs per UWB
            // sub-frame (80/40 = 2). Indexing walks `offset/2 ..
            // offset/2 + 80` in the WB innov, with alternating sign
            // producing the spectral-folding pattern.
            let src_off = (offset / 2) & !1;
            let mut exc = [0.0f32; UWB_SUBFRAME_SIZE];
            let mut i = 0;
            while i + 1 < UWB_SUBFRAME_SIZE {
                let src_i = (src_off + i / 2) % WB_FRAME_SIZE;
                let src_j = (src_off + i / 2 + 1) % WB_FRAME_SIZE;
                let s0 = wb_hb_innov[src_i];
                let s1 = wb_hb_innov[src_j];
                exc[i] = FOLDING_GAIN * s0 * g;
                exc[i + 1] = -FOLDING_GAIN * s1 * g;
                i += 2;
            }
            // Sanitise — exp() on the 5-bit grid gives a large dynamic
            // range; wb innov is already clamped but the product can
            // still explode at the extremes.
            for v in &mut exc {
                if !v.is_finite() {
                    *v = 0.0;
                }
                *v = v.clamp(-32000.0, 32000.0);
            }

            // 1-subframe-delayed synthesis filter (same pipeline trick
            // the WB decoder uses).
            let sp_slice = &mut high[offset..offset + UWB_SUBFRAME_SIZE];
            iir_mem16(
                &self.prev_exc,
                &self.interp_qlpc,
                sp_slice,
                UWB_SUBFRAME_SIZE,
                WB_LPC_ORDER,
                &mut self.mem_sp,
            );

            self.prev_exc.copy_from_slice(&exc);
            self.interp_qlpc.copy_from_slice(&ak);
        }

        self.old_qlsp.copy_from_slice(&qlsp);
        self.first = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_initialises_state_defaults() {
        let d = UwbDecoder::new();
        assert!(d.first);
        assert!(d.old_qlsp[0] > 0.0);
        assert!(d.old_qlsp[WB_LPC_ORDER - 1] < std::f32::consts::PI);
    }

    #[test]
    fn null_submode_silent_layer_runs_qmf() {
        // Decode a synthetic UWB "frame" whose wb payload is a valid
        // NB null submode (leading 1-bit wideband flag = 0 → NB submode
        // id=15 terminator). For this sanity check we just make sure
        // the decoder path that writes the UWB sub-mode selector=0
        // then runs QMF produces finite output.
        let mut d = UwbDecoder::new();
        // Feed an all-zero bitstream so NB decodes a null-submode frame
        // (wideband bit=0, submode=0). WB sees no wb-bit (truncated ok
        // — the reference returns after NB), then UWB sees no uwb-bit.
        let data = vec![0u8; 64];
        let mut br = BitReader::new(&data);
        let mut out = [0.0f32; UWB_FULL_FRAME_SIZE];
        let r = d.decode_frame(&mut br, &mut out);
        // Either cleanly decodes the null path (Ok) or hits a truncation
        // from the inner NB decoder — both acceptable for this sanity
        // check; crucially, no panic.
        assert!(r.is_ok() || matches!(r, Err(Error::Eof) | Err(Error::InvalidData(_))));
        for v in &out {
            assert!(v.is_finite(), "UWB output must be finite");
        }
    }
}
