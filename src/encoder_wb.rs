//! Top-level wideband (sub-band CELP) encoder (round r385 scope) — the
//! encode-direction mirror of [`crate::WidebandDecoder`].
//!
//! Per *The Speex Codec Manual* §10, the wideband encoder splits the
//! 16 kHz input into two 8 kHz half-bands with the QMF analysis
//! filterbank, encodes the low band with the **embedded** narrowband
//! CELP encoder (§2.2 — a wideband frame's low band *is* a narrowband
//! frame), and encodes the high band with the simplified sub-band CELP
//! of §10.1–10.3: an order-8 LPC envelope quantised through the 12-bit
//! two-stage MSVQ, and a per-sub-frame excitation of `gain × innovation`
//! with **no pitch prediction** (§10.2). §10.4 fixes the packing: the
//! entire narrowband frame first (with the wideband flag set), then the
//! Table 10.1 high-band frame.
//!
//! ## Pipeline (per 320-sample 16 kHz frame)
//!
//! 1. **QMF split** ([`crate::QmfAnalysis`]) → low band `x_lb[0..160]` +
//!    high band `x_hb[0..160]`, both at 8 kHz.
//! 2. **Low band** — the r382 [`NarrowbandEncoder`] encodes `x_lb` to a
//!    Table 9.1 body (shared narrowband state, so a stream mixing NB and
//!    WB frames stays continuous).
//! 3. **High-band envelope** — 200-sample analysis buffer (previous 40
//!    high-band samples + current 160) → order-8 LPC
//!    ([`crate::lpc_analysis::analyse_hb`]) → LSP
//!    ([`crate::lpc_to_lsp::hb_lpc_to_lsp`]) → Q10 − base → 2-stage MSVQ
//!    ([`crate::hb_lsp::quantise_q10`]) → packed 12-bit `lsp_index`.
//!    The **quantised** LSPs drive the per-sub-frame LPC via the §9.1
//!    four-way interpolation ([`crate::HbSubFrameLsp`] →
//!    [`crate::hb_subframe_lpc_set_with_base`]), exactly as the decoder
//!    reconstructs them, so encoder and decoder share one envelope.
//! 4. **High-band excitation** — per sub-frame: the order-8 LPC residual
//!    `r[n] = A_hb(z)·x_hb` (continuous analysis history), then the
//!    open-loop gain quantised against the staged HB gain grid
//!    ([`crate::gain_reconstruction::quantise_hb_exc_gain`]) and — for
//!    the modes that carry an excitation-VQ field — the fixed-codebook
//!    search ([`crate::hb_innovation_search`]) at the *reconstructed*
//!    gain, so the chosen indices decode to the matched magnitude.
//! 5. **Pack** — [`crate::hb_encode::encode_wideband_frame`] emits the
//!    §10.4 embedded layout the [`crate::WidebandDecoder`] walks.
//!
//! ## Fidelity
//!
//! Functional, not bit-exact — the same posture as the r382 narrowband
//! encoder: the reference gain normalisation is part of the documented
//! gain-Q-format gap, so gains are chosen by direct magnitude matching
//! against the staged reconstruction levels. Supported high-band modes:
//! 0 (silence), 1 (gain-only), 2 and 3 (documented innovation). Mode 4's
//! innovation-codebook binding is the recorded docs gap and is rejected,
//! as are the reserved high-rate IDs 5..=7.

use crate::encoder_nb::{EncodeError, NarrowbandEncoder, NB_FRAME_SAMPLES};
use crate::frame::FrameError;
use crate::gain_reconstruction::{quantise_hb_exc_gain, reconstruct_hb_exc_gain};
use crate::hb_encode::encode_wideband_frame;
use crate::hb_innovation::{HbInnovationCodebook, HbInnovationMapping, HB_SUBFRAME_SAMPLES};
use crate::hb_innovation_search::search_hb_innovation;
use crate::hb_lsp::{pack_hb_lsp_index, quantise_q10 as quantise_hb_lsp_q10, reconstruct_q10};
use crate::hb_lsp_interp::HbSubFrameLsp;
use crate::lpc_analysis::analyse_hb;
use crate::lpc_to_lsp::hb_lpc_to_lsp;
use crate::lsp_base::hb_lsp_base_q10;
use crate::lsp_to_lpc::{hb_subframe_lpc_set_with_base, radians_to_lsp_q10};
use crate::narrowband_body::NarrowbandFrameBody;
use crate::narrowband_decoder::saturate_i16;
use crate::qmf::{QmfAnalysis, QMF_WIDEBAND_FRAME};
use crate::submode::NarrowbandSubmode;
use crate::wideband::{
    HighBandSubFrameIndices, WidebandHighBandBody, WidebandHighBandSubmode,
    HIGH_BAND_SUBFRAMES_PER_FRAME,
};

use crate::codebooks::HB_LPC_ORDER;

/// Number of 40-sample sub-frames in a high-band frame.
const HB_SUBFRAMES: usize = HIGH_BAND_SUBFRAMES_PER_FRAME as usize;
/// High-band frame length in samples (= the low band's).
const HB_FRAME: usize = HB_SUBFRAMES * HB_SUBFRAME_SAMPLES;
/// Analysis look-back for the high-band 200-sample window: the previous
/// 40 high-band samples followed by the current 160.
const HB_ANALYSIS_LOOKBACK: usize = 40;

/// Errors from [`WidebandEncoder::encode_frame`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WbEncodeError {
    /// The embedded narrowband encode failed (unknown / undocumented NB
    /// mode, or packing).
    Narrowband(EncodeError),
    /// The requested high-band mode ID is not a documented Table 10.1
    /// column (IDs 5..=7 are the reserved high-rate slots; ≥ 8 is out of
    /// the 3-bit range).
    UnknownHbMode(u8),
    /// The high-band mode's innovation-codebook binding is the recorded
    /// docs gap (mode 4).
    UndocumentedHbInnovation(u8),
    /// Frame packing failed (bit-budget overflow — should not happen for
    /// documented modes).
    Pack(FrameError),
}

impl core::fmt::Display for WbEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WbEncodeError::Narrowband(e) => write!(f, "wideband encode: low band: {e}"),
            WbEncodeError::UnknownHbMode(m) => {
                write!(f, "wideband encode: high-band mode {m} not in Table 10.1")
            }
            WbEncodeError::UndocumentedHbInnovation(m) => write!(
                f,
                "wideband encode: high-band mode {m} innovation binding is a docs gap"
            ),
            WbEncodeError::Pack(e) => write!(f, "wideband encode: frame packing failed: {e}"),
        }
    }
}

impl std::error::Error for WbEncodeError {}

impl From<EncodeError> for WbEncodeError {
    fn from(e: EncodeError) -> Self {
        WbEncodeError::Narrowband(e)
    }
}

impl From<FrameError> for WbEncodeError {
    fn from(e: FrameError) -> Self {
        WbEncodeError::Pack(e)
    }
}

/// The two encoded layers of one wideband frame — the embedded
/// narrowband body plus the high-band body, with their resolved
/// sub-modes' IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WidebandFrameBodies {
    /// The embedded narrowband (low-band) frame body.
    pub nb_body: NarrowbandFrameBody,
    /// The narrowband sub-mode ID the body was encoded with.
    pub nb_mode: u8,
    /// The high-band frame body.
    pub hb_body: WidebandHighBandBody,
    /// The high-band sub-mode ID the body was encoded with.
    pub hb_mode: u8,
}

/// Stateful wideband (sub-band CELP) encoder. One instance encodes a
/// continuous stream of 320-sample 16 kHz frames, carrying the QMF
/// analysis history, the embedded narrowband encoder state, and the
/// high-band envelope / analysis state across frames.
#[derive(Debug, Clone)]
pub struct WidebandEncoder {
    /// QMF analysis filterbank state (full-rate FIR history).
    qmf: QmfAnalysis,
    /// Embedded narrowband encoder (low-band state).
    low_band: NarrowbandEncoder,
    /// Previous 40 high-band samples for the 200-sample analysis window.
    prev_hb_tail: [f64; HB_ANALYSIS_LOOKBACK],
    /// Previous frame's reconstructed (quantised) high-band delta-Q10
    /// LSPs — the §9.1 interpolation partner, mirroring the decoder's
    /// carried state. `None` at stream start and after high-band silence
    /// frames (which transmit no LSP).
    prev_hb_lsp_delta_q10: Option<[i32; HB_LPC_ORDER]>,
    /// High-band analysis (prediction-error) filter input history,
    /// most-recent-last.
    hb_analysis_hist: [f64; HB_LPC_ORDER],
}

impl Default for WidebandEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl WidebandEncoder {
    /// A fresh encoder with zero history (stream start).
    pub fn new() -> Self {
        Self {
            qmf: QmfAnalysis::new(),
            low_band: NarrowbandEncoder::new(),
            prev_hb_tail: [0.0; HB_ANALYSIS_LOOKBACK],
            prev_hb_lsp_delta_q10: None,
            hb_analysis_hist: [0.0; HB_LPC_ORDER],
        }
    }

    /// Encode one 320-sample 16 kHz wideband frame, returning the packed
    /// §10.4 frame bytes (embedded narrowband frame with the wideband
    /// flag set, then the high-band frame) — the exact layout
    /// [`crate::WidebandDecoder::decode_packet`] consumes.
    pub fn encode_frame(
        &mut self,
        pcm: &[i16; QMF_WIDEBAND_FRAME],
        nb_mode: u8,
        hb_mode: u8,
    ) -> Result<Vec<u8>, WbEncodeError> {
        let bodies = self.encode_frame_bodies(pcm, nb_mode, hb_mode)?;
        let nb_submode = NarrowbandSubmode::for_id(bodies.nb_mode)
            .expect("mode validated by encode_frame_bodies");
        let hb_submode = WidebandHighBandSubmode::for_id(bodies.hb_mode)
            .expect("mode validated by encode_frame_bodies");
        let writer =
            encode_wideband_frame(&bodies.nb_body, &nb_submode, &bodies.hb_body, &hb_submode)?;
        Ok(writer.into_bytes())
    }

    /// Encode a whole **packet** of consecutive 320-sample 16 kHz frames:
    /// each frame packs as the §10.4 embedded layout (narrowband layer
    /// with the wideband flag set, then the high band), the frames are
    /// concatenated back-to-back (§5.5), closed with the mode-15
    /// terminator, and zero-padded to the byte boundary. The result is
    /// directly consumable by [`crate::SpeexDecoder::decode_packet`].
    pub fn encode_packet(
        &mut self,
        frames: &[[i16; QMF_WIDEBAND_FRAME]],
        nb_mode: u8,
        hb_mode: u8,
    ) -> Result<Vec<u8>, WbEncodeError> {
        let mut writer = crate::bitreader::BitWriter::new();
        for pcm in frames {
            let bodies = self.encode_frame_bodies(pcm, nb_mode, hb_mode)?;
            let nb_submode = NarrowbandSubmode::for_id(bodies.nb_mode)
                .expect("mode validated by encode_frame_bodies");
            let hb_submode = WidebandHighBandSubmode::for_id(bodies.hb_mode)
                .expect("mode validated by encode_frame_bodies");
            let header = crate::frame::NarrowbandFrameHeader::new(true, nb_submode.mode_id)
                .map_err(WbEncodeError::Pack)?;
            header.write(&mut writer).map_err(WbEncodeError::Pack)?;
            crate::nb_encode::write_narrowband_body(&mut writer, &bodies.nb_body, &nb_submode)
                .map_err(|e| WbEncodeError::Pack(FrameError::from(e)))?;
            crate::hb_encode::write_high_band_frame(&mut writer, &bodies.hb_body, &hb_submode)
                .map_err(|e| WbEncodeError::Pack(FrameError::from(e)))?;
        }
        crate::nb_encode::write_packet_terminator(&mut writer).map_err(WbEncodeError::Pack)?;
        Ok(writer.into_bytes())
    }

    /// Encode one frame, returning the two intermediate frame bodies
    /// (the quantised indices) instead of packed bytes. Useful for tests
    /// and for callers assembling multi-frame packets themselves.
    pub fn encode_frame_bodies(
        &mut self,
        pcm: &[i16; QMF_WIDEBAND_FRAME],
        nb_mode: u8,
        hb_mode: u8,
    ) -> Result<WidebandFrameBodies, WbEncodeError> {
        let hb_submode = WidebandHighBandSubmode::for_id(hb_mode)
            .ok_or(WbEncodeError::UnknownHbMode(hb_mode))?;

        // --- QMF split: 16 kHz → two 8 kHz half-bands. ---
        let mut input = [0.0f64; QMF_WIDEBAND_FRAME];
        for (slot, &s) in input.iter_mut().zip(pcm.iter()) {
            *slot = f64::from(s);
        }
        let (lb, hb) = self.qmf.split_frame(&input);

        // --- Low band: the embedded narrowband encode. ---
        let mut lb_i16 = [0i16; NB_FRAME_SAMPLES];
        for (slot, &s) in lb_i16.iter_mut().zip(lb.iter()) {
            *slot = saturate_i16(s);
        }
        let nb_body = self.low_band.encode_frame_body(&lb_i16, nb_mode)?;

        // --- High band: envelope + excitation. ---
        let hb_body = self.encode_high_band(&hb, &hb_submode)?;

        Ok(WidebandFrameBodies {
            nb_body,
            nb_mode,
            hb_body,
            hb_mode,
        })
    }

    /// Encode the high-band layer of one frame.
    fn encode_high_band(
        &mut self,
        hb: &[f64; HB_FRAME],
        submode: &WidebandHighBandSubmode,
    ) -> Result<WidebandHighBandBody, WbEncodeError> {
        // Mode 0 (silence): a zero-bit body; the high-band envelope /
        // analysis state is left untouched (mirrors the decoder, whose
        // silence frames leave the previous LSP envelope alone), but the
        // analysis look-back still advances with the signal.
        if submode.lsp_bits == 0 {
            self.prev_hb_tail
                .copy_from_slice(&hb[HB_FRAME - HB_ANALYSIS_LOOKBACK..]);
            return Ok(WidebandHighBandBody {
                lsp_index: 0,
                subframes: [HighBandSubFrameIndices::default(); HB_SUBFRAMES],
            });
        }

        // Innovation dispatch up-front so an undocumented mode fails
        // before any state is advanced.
        let mapping = HbInnovationMapping::for_mode(submode);
        let (codebook, count) = match mapping {
            HbInnovationMapping::Silence => (None, 0u8),
            HbInnovationMapping::Documented { codebook, count } => (Some(codebook), count),
            HbInnovationMapping::Undocumented => {
                return Err(WbEncodeError::UndocumentedHbInnovation(submode.mode_id))
            }
        };

        // --- Envelope: order-8 LPC analysis → 2-stage MSVQ. ---
        let (lsp_index, active_lsp) = self.encode_hb_envelope(hb);

        // Per-sub-frame quantised LPC via the §9.1 interpolation —
        // matches the decoder exactly.
        let sub_lsp = match self.prev_hb_lsp_delta_q10 {
            Some(prev) => HbSubFrameLsp::new(&prev, &active_lsp),
            None => HbSubFrameLsp::first_frame(&active_lsp),
        };
        let lpc_sets = hb_subframe_lpc_set_with_base(&sub_lsp);

        // --- Excitation: per sub-frame gain (+ innovation). ---
        let gain_bits = submode.excitation_gain_bits;
        let mut subframes = [HighBandSubFrameIndices::default(); HB_SUBFRAMES];
        for (sf, slot) in subframes.iter_mut().enumerate() {
            let block: &[f64] = &hb[sf * HB_SUBFRAME_SAMPLES..(sf + 1) * HB_SUBFRAME_SAMPLES];
            let residual = self.hb_analysis_residual(block, &lpc_sets[sf]);

            let r_rms =
                (residual.iter().map(|&x| x * x).sum::<f64>() / HB_SUBFRAME_SAMPLES as f64).sqrt();

            match codebook {
                Some(cb) => {
                    // Gain guess: match the residual RMS to the codebook
                    // row RMS, quantise, then search the codebook at the
                    // *reconstructed* gain so the indices decode to the
                    // matched magnitude.
                    let cb_rms = hb_codebook_rms(cb).max(1e-9);
                    let g_guess = r_rms / cb_rms;
                    let gain_idx = quantise_hb_exc_gain(g_guess as f32, gain_bits);
                    let g_q = gain_idx
                        .map(|i| f64::from(reconstruct_hb_exc_gain(i)))
                        .unwrap_or(0.0);
                    let choice = search_hb_innovation(&residual, g_q.max(1e-9), cb, count);
                    slot.excitation_gain_index = gain_idx.and_then(|i| i.raw_index()).unwrap_or(0);
                    slot.excitation_vq_index = choice.packed;
                }
                None => {
                    // Mode 1: gain-only — the field conveys the high-band
                    // energy envelope (no VQ field on the wire).
                    let gain_idx = quantise_hb_exc_gain(r_rms as f32, gain_bits);
                    slot.excitation_gain_index = gain_idx.and_then(|i| i.raw_index()).unwrap_or(0);
                    slot.excitation_vq_index = 0;
                }
            }
        }

        // Advance the analysis look-back and commit the envelope state.
        self.prev_hb_tail
            .copy_from_slice(&hb[HB_FRAME - HB_ANALYSIS_LOOKBACK..]);
        self.prev_hb_lsp_delta_q10 = Some(active_lsp);

        Ok(WidebandHighBandBody {
            lsp_index,
            subframes,
        })
    }

    /// High-band envelope encode: returns `(packed lsp_index,
    /// active_delta_q10)` where the active vector is the **quantised**
    /// reconstruction (what the decoder will see).
    fn encode_hb_envelope(&mut self, hb: &[f64; HB_FRAME]) -> (u16, [i32; HB_LPC_ORDER]) {
        let fallback = |prev: Option<[i32; HB_LPC_ORDER]>| prev.unwrap_or([0; HB_LPC_ORDER]);

        // 200-sample analysis buffer: previous 40 + current 160.
        let mut buf = Vec::with_capacity(HB_ANALYSIS_LOOKBACK + HB_FRAME);
        buf.extend_from_slice(&self.prev_hb_tail);
        buf.extend_from_slice(hb);

        let analysed = match analyse_hb(&buf) {
            Ok(c) => c,
            Err(_) => return (0, fallback(self.prev_hb_lsp_delta_q10)),
        };
        let lsp_rad = match hb_lpc_to_lsp(&analysed.a) {
            Ok(r) => r,
            Err(_) => return (0, fallback(self.prev_hb_lsp_delta_q10)),
        };

        // Radians → Q10 absolute → subtract the pinned HB base vector.
        let base = hb_lsp_base_q10();
        let mut delta = [0i32; HB_LPC_ORDER];
        for i in 0..HB_LPC_ORDER {
            delta[i] = radians_to_lsp_q10(lsp_rad[i]) - base[i];
        }

        let stages = quantise_hb_lsp_q10(&delta);
        let lsp_index = pack_hb_lsp_index(&stages);
        let active = reconstruct_q10(stages).unwrap_or(delta);
        (lsp_index, active)
    }

    /// High-band residual `r[n] = x[n] − Σ a[i]·x[n−1−i]` (order-8
    /// analysis / prediction-error filter), advancing the continuous
    /// analysis history.
    fn hb_analysis_residual(
        &mut self,
        block: &[f64],
        lpc: &[f64; HB_LPC_ORDER],
    ) -> [f64; HB_SUBFRAME_SAMPLES] {
        let mut out = [0.0f64; HB_SUBFRAME_SAMPLES];
        for (slot, &x) in out.iter_mut().zip(block.iter()) {
            let mut r = x;
            for (i, &c) in lpc.iter().enumerate() {
                r -= c * self.hb_analysis_hist[HB_LPC_ORDER - 1 - i];
            }
            self.hb_analysis_hist.rotate_left(1);
            self.hb_analysis_hist[HB_LPC_ORDER - 1] = x;
            *slot = r;
        }
        out
    }

    /// Read-only view of the embedded narrowband encoder (diagnostics).
    pub fn low_band_encoder(&self) -> &NarrowbandEncoder {
        &self.low_band
    }
}

/// Mean per-sample RMS of a high-band codebook's rows (scales the
/// innovation gain guess) — the high-band analogue of the narrowband
/// encoder's codebook RMS helper.
fn hb_codebook_rms(codebook: HbInnovationCodebook) -> f64 {
    let sv_len = codebook.sub_vector_len();
    let n = codebook.entries();
    let mut acc = 0.0f64;
    let mut samples = 0usize;
    for idx in 0..n {
        if let Some(row) = crate::hb_innovation::hb_innovation_sub_vector(codebook, idx) {
            for &v in row {
                acc += f64::from(v) * f64::from(v);
            }
            samples += sv_len;
        }
    }
    if samples == 0 {
        0.0
    } else {
        (acc / samples as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::frame::NarrowbandFrameHeader;
    use crate::wideband::WidebandHighBandFrameHeader;

    /// A 16 kHz test frame mixing a voiced low-band structure with a
    /// high-band tone so both bands carry energy.
    fn wideband_frame(amp: f64) -> [i16; QMF_WIDEBAND_FRAME] {
        let mut f = [0i16; QMF_WIDEBAND_FRAME];
        for (n, s) in f.iter_mut().enumerate() {
            let t = n as f64;
            let low = (2.0 * std::f64::consts::PI * t * 300.0 / 16_000.0).sin();
            let high = 0.5 * (2.0 * std::f64::consts::PI * t * 6_000.0 / 16_000.0).sin();
            let v = amp * (low + high);
            *s = v.round().clamp(-32768.0, 32767.0) as i16;
        }
        f
    }

    #[test]
    fn encodes_documented_hb_modes_to_valid_frames() {
        for hb_mode in [0u8, 1, 2, 3] {
            let mut enc = WidebandEncoder::new();
            let frame = wideband_frame(6000.0);
            let bytes = enc
                .encode_frame(&frame, 3, hb_mode)
                .unwrap_or_else(|e| panic!("hb mode {hb_mode}: {e}"));
            assert!(!bytes.is_empty(), "hb mode {hb_mode} produced empty frame");
        }
    }

    #[test]
    fn hb_mode_4_rejected_as_docs_gap() {
        let mut enc = WidebandEncoder::new();
        let frame = wideband_frame(4000.0);
        assert_eq!(
            enc.encode_frame(&frame, 3, 4),
            Err(WbEncodeError::UndocumentedHbInnovation(4))
        );
    }

    #[test]
    fn reserved_and_out_of_range_hb_modes_rejected() {
        let mut enc = WidebandEncoder::new();
        let frame = wideband_frame(4000.0);
        for m in [5u8, 6, 7, 8, 255] {
            assert_eq!(
                enc.encode_frame(&frame, 3, m),
                Err(WbEncodeError::UnknownHbMode(m)),
                "mode {m}"
            );
        }
    }

    #[test]
    fn narrowband_error_propagates() {
        let mut enc = WidebandEncoder::new();
        let frame = wideband_frame(4000.0);
        assert_eq!(
            enc.encode_frame(&frame, 9, 2),
            Err(WbEncodeError::Narrowband(EncodeError::UnknownMode(9)))
        );
    }

    #[test]
    fn encoded_frame_reparses_to_same_bodies() {
        // encode_frame's bytes must walk back through the exact reader
        // chain to the bodies encode_frame_bodies produced.
        let frame = wideband_frame(5000.0);
        let (nb_mode, hb_mode) = (3u8, 2u8);

        let mut enc_a = WidebandEncoder::new();
        let bodies = enc_a.encode_frame_bodies(&frame, nb_mode, hb_mode).unwrap();
        let mut enc_b = WidebandEncoder::new();
        let bytes = enc_b.encode_frame(&frame, nb_mode, hb_mode).unwrap();

        let nb_submode = NarrowbandSubmode::for_id(nb_mode).unwrap();
        let hb_submode = WidebandHighBandSubmode::for_id(hb_mode).unwrap();
        let mut r = BitReader::new(&bytes);
        let nb_header = NarrowbandFrameHeader::parse(&mut r).unwrap();
        assert!(nb_header.wideband, "NB prefix carries the wideband flag");
        assert_eq!(nb_header.mode_id, nb_mode);
        let nb_parsed = NarrowbandFrameBody::parse(&mut r, &nb_submode).unwrap();
        assert_eq!(nb_parsed, bodies.nb_body);

        let hb_header = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        assert!(hb_header.wideband);
        assert_eq!(hb_header.mode_id, hb_mode);
        let hb_parsed = WidebandHighBandBody::parse(&mut r, &hb_submode).unwrap();
        assert_eq!(hb_parsed, bodies.hb_body);
    }

    #[test]
    fn hb_lsp_index_is_stable_for_stationary_input() {
        // A stationary high-band signal should quantise to the same LSP
        // envelope frame after frame (the quantised active LSP is the
        // decoder-visible reconstruction, deterministic per frame).
        let frame = wideband_frame(5000.0);
        let mut enc = WidebandEncoder::new();
        let b1 = enc.encode_frame_bodies(&frame, 3, 2).unwrap();
        let b2 = enc.encode_frame_bodies(&frame, 3, 2).unwrap();
        let b3 = enc.encode_frame_bodies(&frame, 3, 2).unwrap();
        // Frames 2 and 3 see identical (steady-state) analysis windows.
        assert_eq!(b2.hb_body.lsp_index, b3.hb_body.lsp_index);
        let _ = b1;
    }

    #[test]
    fn silent_input_encodes_zero_gain_indices() {
        // A silent frame's high-band residual is zero → the quantiser
        // drives every excitation-gain field to its floor (raw 0).
        let frame = [0i16; QMF_WIDEBAND_FRAME];
        let mut enc = WidebandEncoder::new();
        let bodies = enc.encode_frame_bodies(&frame, 3, 2).unwrap();
        for sf in &bodies.hb_body.subframes {
            assert_eq!(sf.excitation_gain_index, 0);
        }
    }
}
