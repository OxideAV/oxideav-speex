//! Top-level narrowband CELP encoder (round r382 scope).
//!
//! Integrates the round-r372 envelope chain (LPC analysis → LSP VQ) and
//! the round-r382 excitation encoders (pitch search, innovation search,
//! gain quantisation, frame packing) into a single [`NarrowbandEncoder`]
//! that turns a 160-sample 8 kHz PCM frame into a decodable Speex
//! narrowband frame.
//!
//! ## Pipeline (per frame, MAN §9)
//!
//! 1. **Envelope** — build the 200-sample analysis buffer (previous 40
//!    input samples + current 160), run [`crate::lpc_analysis`] → order-10
//!    LPC, convert to LSP ([`crate::lpc_to_lsp`]), quantise through the
//!    multi-stage VQ ([`crate::lsp_quant`]), and pack the `lsp_index`.
//!    The **quantised** LSPs (reconstructed exactly as the decoder will)
//!    drive per-sub-frame LPC via the §9.1 interpolation
//!    ([`crate::lsp_interp`] → [`crate::lsp_to_lpc::subframe_lpc_set_with_base`]),
//!    so encoder and decoder share one envelope.
//! 2. **Excitation** — per sub-frame: compute the LPC residual `r[n]`
//!    (the input through the analysis filter `A(z)`, the exact inverse of
//!    the decoder's synthesis), search the adaptive codebook for the
//!    pitch period + 3-tap gain, subtract the pitch contribution, and
//!    quantise the remainder against the innovation codebook
//!    ([`crate::innovation_search`]). The reconstructed excitation
//!    `e[n] = p[n] + g·c[n]` is pushed into the encoder's excitation
//!    history — the same signal the decoder reconstructs — so the next
//!    sub-frame's pitch predictor sees live history.
//! 3. **Pack** — assemble the quantised indices into a
//!    [`NarrowbandFrameBody`] and emit the frame
//!    ([`crate::nb_encode::encode_narrowband_frame`]).
//!
//! ## Fidelity
//!
//! This is a **functional** encoder: it produces valid, decodable frames
//! whose reconstruction tracks the input, but it is not bit-exact against
//! the reference encoder. The reference's exact gain normalisation
//! (the mapping between residual magnitude and the `exp(qe/3.5)` OL-gain
//! domain) is part of the crate's documented gain-Q-format gap; this
//! encoder chooses gains by direct magnitude matching against the staged
//! reconstruction levels. Supported modes are the documented-innovation
//! ones (2 / 3 / 4 / 5 / 6 / 8); modes 1 / 7 (undocumented innovation)
//! and mode 0 (pure silence) are handled as their degenerate cases.

use crate::adaptive_codebook::{resolve_lookback, TAP_PITCH_OFFSETS};
use crate::frame::FrameError;
use crate::gain_reconstruction::{
    quantise_frame_ol_exc_gain, quantise_subframe_gain_correction, reconstruct_frame_ol_exc_gain,
    reconstruct_subframe_gain_correction,
};
use crate::innovation::{sub_vector, InnovationCodebook, InnovationMapping, SUBFRAME_SAMPLES};
use crate::innovation_search::search_innovation;
use crate::lpc_to_lsp::lpc_to_lsp;
use crate::lsp_base::nb_lsp_base_q10;
use crate::lsp_interp::NbSubFrameLsp;
use crate::lsp_quant::{pack_lsp_index, quantise_lsp_q10};
use crate::lsp_to_lpc::{lsp_vector_radians_to_q10, subframe_lpc_set_with_base, LPC_ORDER};
use crate::narrowband_body::{
    NarrowbandFrameBody, NarrowbandSubFrameIndices, PITCH_PERIOD_MAX, PITCH_PERIOD_MIN,
};
use crate::nb_encode::encode_narrowband_frame;
use crate::pitch_gain;
use crate::submode::{LspQuant, NarrowbandSubmode, PitchGainQuant};

/// Number of 40-sample sub-frames in a narrowband frame.
const SUBFRAMES: usize = 4;
/// Narrowband frame length in samples.
pub const NB_FRAME_SAMPLES: usize = SUBFRAMES * SUBFRAME_SAMPLES;
/// Analysis look-back: the 200-sample window is the previous 40 input
/// samples followed by the current 160.
const ANALYSIS_LOOKBACK: usize = 40;
/// Length of the retained excitation history (>= PITCH_PERIOD_MAX).
const EXC_HIST_LEN: usize = PITCH_PERIOD_MAX as usize + SUBFRAME_SAMPLES;

/// Errors from [`NarrowbandEncoder::encode_frame`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// The requested mode ID is not a documented narrowband sub-mode.
    UnknownMode(u8),
    /// The mode's innovation handling is not documented (modes 1 / 7).
    UndocumentedInnovation(u8),
    /// Frame packing failed (bit-budget overflow — should not happen for
    /// documented modes).
    Pack(FrameError),
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EncodeError::UnknownMode(m) => write!(f, "unknown narrowband mode {m}"),
            EncodeError::UndocumentedInnovation(m) => {
                write!(f, "mode {m} has undocumented innovation handling")
            }
            EncodeError::Pack(e) => write!(f, "frame packing failed: {e}"),
        }
    }
}

impl std::error::Error for EncodeError {}

impl From<FrameError> for EncodeError {
    fn from(e: FrameError) -> Self {
        EncodeError::Pack(e)
    }
}

/// Stateful narrowband CELP encoder. One instance encodes a continuous
/// stream of 160-sample frames, carrying LSP / excitation / analysis
/// state across frames.
#[derive(Debug, Clone)]
pub struct NarrowbandEncoder {
    /// Previous 40 input samples for the analysis window.
    prev_input_tail: [f64; ANALYSIS_LOOKBACK],
    /// Previous frame's reconstructed (quantised) delta-Q10 LSPs.
    prev_lsp_q10: Option<[i32; LPC_ORDER]>,
    /// Excitation history (`f64`), most-recent last.
    exc_hist: Vec<f64>,
    /// Analysis (prediction-error) filter input history, most-recent last.
    analysis_hist: [f64; LPC_ORDER],
}

impl Default for NarrowbandEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl NarrowbandEncoder {
    /// A fresh encoder with zero history (stream start).
    pub fn new() -> Self {
        Self {
            prev_input_tail: [0.0; ANALYSIS_LOOKBACK],
            prev_lsp_q10: None,
            exc_hist: vec![0.0; EXC_HIST_LEN],
            analysis_hist: [0.0; LPC_ORDER],
        }
    }

    /// Encode one 160-sample narrowband frame in the given mode, returning
    /// the packed frame bytes.
    pub fn encode_frame(
        &mut self,
        pcm: &[i16; NB_FRAME_SAMPLES],
        mode: u8,
    ) -> Result<Vec<u8>, EncodeError> {
        let submode = NarrowbandSubmode::for_id(mode).ok_or(EncodeError::UnknownMode(mode))?;
        let body = self.build_body(pcm, &submode)?;
        let writer = encode_narrowband_frame(&body, &submode)?;
        Ok(writer.into_bytes())
    }

    /// Encode a whole **packet** of consecutive 160-sample frames in the
    /// given mode: the frames are packed back-to-back (§5.5 — *"it is
    /// desirable to pack more than one frame per packet"*), closed with
    /// the mode-15 terminator, and zero-padded to the byte boundary.
    /// The result is directly consumable by
    /// [`crate::SpeexDecoder::decode_packet`].
    pub fn encode_packet(
        &mut self,
        frames: &[[i16; NB_FRAME_SAMPLES]],
        mode: u8,
    ) -> Result<Vec<u8>, EncodeError> {
        let submode = NarrowbandSubmode::for_id(mode).ok_or(EncodeError::UnknownMode(mode))?;
        let mut writer = crate::bitreader::BitWriter::new();
        for pcm in frames {
            let body = self.build_body(pcm, &submode)?;
            let header = crate::frame::NarrowbandFrameHeader::new(false, submode.mode_id)
                .map_err(EncodeError::Pack)?;
            header.write(&mut writer).map_err(EncodeError::Pack)?;
            crate::nb_encode::write_narrowband_body(&mut writer, &body, &submode)
                .map_err(|e| EncodeError::Pack(FrameError::from(e)))?;
        }
        crate::nb_encode::write_packet_terminator(&mut writer).map_err(EncodeError::Pack)?;
        Ok(writer.into_bytes())
    }

    /// Encode one frame, returning the intermediate [`NarrowbandFrameBody`]
    /// (the quantised indices) instead of packed bytes. Useful for tests
    /// and for callers that assemble multi-frame packets themselves.
    pub fn encode_frame_body(
        &mut self,
        pcm: &[i16; NB_FRAME_SAMPLES],
        mode: u8,
    ) -> Result<NarrowbandFrameBody, EncodeError> {
        let submode = NarrowbandSubmode::for_id(mode).ok_or(EncodeError::UnknownMode(mode))?;
        self.build_body(pcm, &submode)
    }

    fn build_body(
        &mut self,
        pcm: &[i16; NB_FRAME_SAMPLES],
        submode: &NarrowbandSubmode,
    ) -> Result<NarrowbandFrameBody, EncodeError> {
        let input: Vec<f64> = pcm.iter().map(|&s| f64::from(s)).collect();

        // --- Envelope: LPC analysis → LSP VQ. ---
        let (lsp_index, active_lsp) = self.encode_envelope(&input, submode);

        // Per-sub-frame quantised LPC (matches the decoder exactly).
        let prev = self.prev_lsp_q10.unwrap_or(active_lsp);
        let sub_lsp = NbSubFrameLsp::new(&prev, &active_lsp);
        let lpc_sets = subframe_lpc_set_with_base(&sub_lsp);

        // Frame-level OL excitation gain: a magnitude estimate from the
        // whole-frame residual (computed with a scratch analysis filter so
        // the real per-sub-frame residual pass below stays continuous).
        let frame_gain_target = self.frame_gain_estimate(&input, &lpc_sets, submode);
        let frame_gain_idx = quantise_frame_ol_exc_gain(frame_gain_target as f32);
        let g_frame = f64::from(reconstruct_frame_ol_exc_gain(frame_gain_idx));

        // --- Excitation: per sub-frame pitch + innovation. ---
        let mapping = InnovationMapping::for_mode(submode);
        let (codebook, count) = match mapping {
            InnovationMapping::Silence => (None, 0u8),
            InnovationMapping::Documented { codebook, count } => (Some(codebook), count),
            InnovationMapping::Undocumented => {
                return Err(EncodeError::UndocumentedInnovation(submode.mode_id))
            }
        };

        let pitch_quant = submode.pitch_gain;
        let has_fine_pitch = submode.fine_pitch_bits > 0;
        let mut subframes = [NarrowbandSubFrameIndices::default(); SUBFRAMES];

        for (sf, slot) in subframes.iter_mut().enumerate() {
            let block = &input[sf * SUBFRAME_SAMPLES..(sf + 1) * SUBFRAME_SAMPLES];
            let lpc = &lpc_sets[sf];

            // Residual r[n] = A(z)·input (analysis filter, continuous state).
            let residual = self.analysis_residual(block, lpc);

            // Pitch search in the excitation domain.
            let (period, pitch_gain_idx, taps) = self.search_pitch(&residual, pitch_quant);
            let pitch = self.pitch_contribution(period, &taps);

            // Innovation on the pitch-removed residual.
            let mut r2 = [0.0_f64; SUBFRAME_SAMPLES];
            for n in 0..SUBFRAME_SAMPLES {
                r2[n] = residual[n] - pitch[n];
            }

            let (innovation_gain_idx, innovation_vq, exc) = match codebook {
                Some(cb) => self.encode_innovation(&r2, &pitch, g_frame, cb, count, submode),
                None => (0u8, 0u128, pitch),
            };

            // Push the reconstructed excitation into history.
            self.push_excitation(&exc);

            slot.pitch_index = if has_fine_pitch {
                (period - PITCH_PERIOD_MIN) as u8
            } else {
                0
            };
            slot.pitch_gain_index = pitch_gain_idx;
            slot.innovation_gain_index = innovation_gain_idx;
            slot.innovation_vq_index = innovation_vq;
        }

        // Frame-level OL pitch field (modes 1 / 2 / 8): reuse the first
        // sub-frame's period estimate as the frame value.
        let ol_pitch_index = if submode.ol_pitch_bits > 0 {
            let (period, _, _) = self.search_pitch(&[0.0; SUBFRAME_SAMPLES], pitch_quant);
            (period.saturating_sub(PITCH_PERIOD_MIN)) as u8
        } else {
            0
        };

        // Advance analysis-window look-back and commit envelope state.
        self.prev_input_tail
            .copy_from_slice(&input[NB_FRAME_SAMPLES - ANALYSIS_LOOKBACK..]);
        self.commit_lsp(active_lsp);

        Ok(NarrowbandFrameBody {
            lsp_index,
            ol_pitch_index,
            ol_pitch_gain_index: 0,
            ol_exc_gain_index: frame_gain_field(frame_gain_idx),
            subframes,
        })
    }

    /// Envelope encode: returns `(lsp_index, active_delta_q10)`.
    fn encode_envelope(
        &mut self,
        input: &[f64],
        submode: &NarrowbandSubmode,
    ) -> (u32, [i32; LPC_ORDER]) {
        let regime = submode.lsp;
        if matches!(regime, LspQuant::None) {
            let active = self.prev_lsp_q10.unwrap_or([0; LPC_ORDER]);
            return (0, active);
        }

        // Build the 200-sample analysis buffer.
        let mut buf = Vec::with_capacity(ANALYSIS_LOOKBACK + input.len());
        buf.extend_from_slice(&self.prev_input_tail);
        buf.extend_from_slice(input);

        let analysed = match crate::lpc_analysis::analyse(&buf) {
            Ok(c) => c,
            Err(_) => {
                let active = self.prev_lsp_q10.unwrap_or([0; LPC_ORDER]);
                return (0, active);
            }
        };
        let lsp_rad = match lpc_to_lsp(&analysed.a) {
            Ok(r) => r,
            Err(_) => {
                let active = self.prev_lsp_q10.unwrap_or([0; LPC_ORDER]);
                return (0, active);
            }
        };
        let absolute = lsp_vector_radians_to_q10(&lsp_rad);
        let base = nb_lsp_base_q10();
        let mut delta = [0i32; LPC_ORDER];
        for i in 0..LPC_ORDER {
            delta[i] = absolute[i] - base[i];
        }

        let Some(stages) = quantise_lsp_q10(&delta, regime) else {
            let active = self.prev_lsp_q10.unwrap_or(delta);
            return (0, active);
        };
        let lsp_index = pack_lsp_index(&stages);
        let active = crate::lsp::reconstruct_q10(stages).unwrap_or(delta);
        (lsp_index, active)
    }

    /// Estimate a frame-level excitation-gain magnitude from the residual
    /// energy (uses a scratch analysis filter so the real pass is not
    /// perturbed).
    fn frame_gain_estimate(
        &self,
        input: &[f64],
        lpc_sets: &[[f64; LPC_ORDER]; SUBFRAMES],
        submode: &NarrowbandSubmode,
    ) -> f64 {
        let mut hist = self.analysis_hist;
        let mut energy = 0.0_f64;
        for (sf, lpc) in lpc_sets.iter().enumerate() {
            let block = &input[sf * SUBFRAME_SAMPLES..(sf + 1) * SUBFRAME_SAMPLES];
            for &x in block {
                let mut r = x;
                for (i, &c) in lpc.iter().enumerate() {
                    r -= c * hist[LPC_ORDER - 1 - i];
                }
                hist.rotate_left(1);
                hist[LPC_ORDER - 1] = x;
                energy += r * r;
            }
        }
        let rms = (energy / NB_FRAME_SAMPLES as f64).sqrt();
        let cb_rms = InnovationMapping::for_mode(submode)
            .documented_codebook()
            .map(codebook_rms)
            .unwrap_or(1.0);
        if cb_rms > 0.0 {
            rms / cb_rms
        } else {
            rms
        }
    }

    /// Residual `r[n] = x[n] − Σ a[i]·x[n−1−i]` (analysis / prediction-error
    /// filter), advancing the continuous analysis history.
    fn analysis_residual(
        &mut self,
        block: &[f64],
        lpc: &[f64; LPC_ORDER],
    ) -> [f64; SUBFRAME_SAMPLES] {
        let mut out = [0.0_f64; SUBFRAME_SAMPLES];
        for (slot, &x) in out.iter_mut().zip(block.iter()) {
            let mut r = x;
            for (i, &c) in lpc.iter().enumerate() {
                r -= c * self.analysis_hist[LPC_ORDER - 1 - i];
            }
            self.analysis_hist.rotate_left(1);
            self.analysis_hist[LPC_ORDER - 1] = x;
            *slot = r;
        }
        out
    }

    /// Excitation-domain adaptive-codebook (pitch) search: pick the period
    /// and 3-tap gain-VQ index minimising `Σ (r − Σ g_j·e[n−T+off_j])²`.
    fn search_pitch(
        &self,
        residual: &[f64; SUBFRAME_SAMPLES],
        quant: PitchGainQuant,
    ) -> (u16, u8, pitch_gain::PitchGainTaps) {
        if matches!(quant, PitchGainQuant::None) {
            return (PITCH_PERIOD_MIN, 0, pitch_gain::PitchGainTaps::SILENCE);
        }
        let n_entries: u16 = match quant {
            PitchGainQuant::Vq5Bit => 32,
            PitchGainQuant::Vq7Bit => 128,
            PitchGainQuant::None => 1,
        };
        let mut best = (
            PITCH_PERIOD_MIN,
            0u8,
            pitch_gain::PitchGainTaps::SILENCE,
            f64::INFINITY,
        );
        for t in PITCH_PERIOD_MIN..=PITCH_PERIOD_MAX {
            let basis = self.pitch_basis(t);
            for idx in 0..n_entries {
                let Some(taps) = pitch_gain::reconstruct(idx as u8, quant) else {
                    continue;
                };
                let g = [
                    f64::from(taps.taps[0]) / 64.0,
                    f64::from(taps.taps[1]) / 64.0,
                    f64::from(taps.taps[2]) / 64.0,
                ];
                let mut err = 0.0_f64;
                for (n, &rv) in residual.iter().enumerate() {
                    let p = g[0] * basis[0][n] + g[1] * basis[1][n] + g[2] * basis[2][n];
                    let d = rv - p;
                    err += d * d;
                }
                if err < best.3 {
                    best = (t, idx as u8, taps, err);
                }
            }
        }
        (best.0, best.1, best.2)
    }

    /// Three adaptive-codebook basis vectors (offsets −T−1, −T, −T+1) for
    /// pitch period `t`.
    fn pitch_basis(&self, t: u16) -> [[f64; SUBFRAME_SAMPLES]; 3] {
        let mut basis = [[0.0_f64; SUBFRAME_SAMPLES]; 3];
        let hlen = self.exc_hist.len();
        for (row, off) in basis.iter_mut().zip(TAP_PITCH_OFFSETS.iter()) {
            for (n, slot) in row.iter_mut().enumerate() {
                let k = resolve_lookback(n as i32 - i32::from(t) + off, t);
                let idx = hlen as i32 + k;
                if idx >= 0 {
                    *slot = self.exc_hist[idx as usize];
                }
            }
        }
        basis
    }

    /// The reconstructed pitch contribution `p[n] = Σ (g_j/64)·basis_j[n]`.
    fn pitch_contribution(
        &self,
        period: u16,
        taps: &pitch_gain::PitchGainTaps,
    ) -> [f64; SUBFRAME_SAMPLES] {
        let basis = self.pitch_basis(period);
        let g = [
            f64::from(taps.taps[0]) / 64.0,
            f64::from(taps.taps[1]) / 64.0,
            f64::from(taps.taps[2]) / 64.0,
        ];
        let mut p = [0.0_f64; SUBFRAME_SAMPLES];
        for n in 0..SUBFRAME_SAMPLES {
            p[n] = g[0] * basis[0][n] + g[1] * basis[1][n] + g[2] * basis[2][n];
        }
        p
    }

    /// Innovation encode: choose the codebook indices + sub-frame gain
    /// correction and return `(innovation_gain_index, packed_vq, excitation)`.
    fn encode_innovation(
        &self,
        r2: &[f64; SUBFRAME_SAMPLES],
        pitch: &[f64; SUBFRAME_SAMPLES],
        g_frame: f64,
        codebook: InnovationCodebook,
        count: u8,
        submode: &NarrowbandSubmode,
    ) -> (u8, u128, [f64; SUBFRAME_SAMPLES]) {
        let cb_rms = codebook_rms(codebook).max(1e-9);
        let r2_rms = (r2.iter().map(|&x| x * x).sum::<f64>() / SUBFRAME_SAMPLES as f64).sqrt();
        let g_guess = if r2_rms > 0.0 {
            r2_rms / cb_rms
        } else {
            g_frame.max(1e-3)
        };

        let choice = search_innovation(r2, g_guess, codebook, count);

        // Reconstruct the concatenated codebook shape.
        let sv_len = codebook.sub_vector_len();
        let mut cb = [0.0_f64; SUBFRAME_SAMPLES];
        for (sv, &idx) in choice.indices.iter().enumerate() {
            if let Some(row) = sub_vector(codebook, idx) {
                let base = sv * sv_len;
                for (k, &v) in row.iter().enumerate() {
                    cb[base + k] = f64::from(v);
                }
            }
        }

        // Sub-frame gain correction: quantise α/g_frame into the field.
        let dot: f64 = r2.iter().zip(cb.iter()).map(|(&a, &b)| a * b).sum();
        let energy: f64 = cb.iter().map(|&b| b * b).sum::<f64>().max(1e-9);
        let alpha = (dot / energy).max(0.0);
        let correction_target = if g_frame > 0.0 {
            alpha / g_frame
        } else {
            alpha
        };
        let bits = submode.innovation_gain_bits;
        let correction = quantise_subframe_gain_correction(correction_target as f32, bits)
            .unwrap_or(crate::fixed_codebook_gain::SubFrameInnovationGainCorrection::Absent);
        let g_subf = f64::from(reconstruct_subframe_gain_correction(correction));
        let gain = g_frame * g_subf;

        let mut exc = [0.0_f64; SUBFRAME_SAMPLES];
        for n in 0..SUBFRAME_SAMPLES {
            exc[n] = pitch[n] + gain * cb[n];
        }
        (correction_field(correction), choice.packed, exc)
    }

    /// Push a 40-sample excitation block into the history ring.
    fn push_excitation(&mut self, exc: &[f64; SUBFRAME_SAMPLES]) {
        self.exc_hist.extend_from_slice(exc);
        let len = self.exc_hist.len();
        if len > EXC_HIST_LEN {
            self.exc_hist.drain(0..len - EXC_HIST_LEN);
        }
        // Commit the reconstructed LSP state for the next frame after the
        // full frame is processed (done by the caller via finish()).
    }

    /// Commit the frame's reconstructed LSP as the previous-frame state.
    fn commit_lsp(&mut self, active: [i32; LPC_ORDER]) {
        self.prev_lsp_q10 = Some(active);
    }
}

/// Extract the 5-bit OL exc-gain field value from a quantised index.
fn frame_gain_field(idx: crate::fixed_codebook_gain::FrameInnovationGainIndex) -> u8 {
    match idx {
        crate::fixed_codebook_gain::FrameInnovationGainIndex::Silence => 0,
        crate::fixed_codebook_gain::FrameInnovationGainIndex::Indexed(i) => i,
    }
}

/// Extract the innovation-gain-correction field value.
fn correction_field(c: crate::fixed_codebook_gain::SubFrameInnovationGainCorrection) -> u8 {
    use crate::fixed_codebook_gain::SubFrameInnovationGainCorrection as C;
    match c {
        C::Absent => 0,
        C::OneBit(i) => i,
        C::ThreeBit(i) => i,
    }
}

/// Mean per-sample RMS of a codebook's rows (used to scale the innovation
/// gain guess).
fn codebook_rms(codebook: InnovationCodebook) -> f64 {
    let sv_len = codebook.sub_vector_len();
    let n = codebook.entries();
    let mut acc = 0.0_f64;
    let mut samples = 0usize;
    for idx in 0..n {
        if let Some(row) = sub_vector(codebook, idx) {
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

impl InnovationMapping {
    /// The documented codebook shape, if this mapping is documented.
    fn documented_codebook(self) -> Option<InnovationCodebook> {
        match self {
            InnovationMapping::Documented { codebook, .. } => Some(codebook),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voiced_frame(period: usize, amp: f64) -> [i16; NB_FRAME_SAMPLES] {
        let mut f = [0i16; NB_FRAME_SAMPLES];
        for (n, s) in f.iter_mut().enumerate() {
            // A pitch-periodic pulse train shaped by a decaying formant.
            let phase = (n % period) as f64;
            let v =
                amp * (-(phase) * 0.1).exp() * (2.0 * std::f64::consts::PI * n as f64 / 12.0).sin();
            *s = v.round().clamp(-32768.0, 32767.0) as i16;
        }
        f
    }

    #[test]
    fn encodes_documented_modes_to_valid_frames() {
        for mode in [2u8, 3, 4, 5, 6, 8] {
            let mut enc = NarrowbandEncoder::new();
            let frame = voiced_frame(40, 4000.0);
            let bytes = enc.encode_frame(&frame, mode).expect("encode");
            assert!(!bytes.is_empty(), "mode {mode} produced empty frame");
        }
    }

    #[test]
    fn undocumented_modes_are_rejected() {
        let mut enc = NarrowbandEncoder::new();
        let frame = voiced_frame(50, 2000.0);
        assert_eq!(
            enc.encode_frame(&frame, 7),
            Err(EncodeError::UndocumentedInnovation(7))
        );
    }

    #[test]
    fn encoded_frame_reparses_to_same_body() {
        use crate::bitreader::BitReader;
        use crate::frame::NarrowbandFrameHeader;
        let mut enc = NarrowbandEncoder::new();
        let frame = voiced_frame(45, 5000.0);
        let mode = 5u8;
        let body = enc.encode_frame_body(&frame, mode).expect("body");

        // Re-encode the *same* body (a fresh encoder would diverge on
        // state, so encode the body directly) and parse it back.
        let submode = NarrowbandSubmode::for_id(mode).unwrap();
        let writer = encode_narrowband_frame(&body, &submode).unwrap();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        let header = NarrowbandFrameHeader::parse(&mut reader).unwrap();
        assert_eq!(header.mode_id, mode);
        let parsed = NarrowbandFrameBody::parse(&mut reader, &submode).unwrap();
        assert_eq!(parsed, body);
    }

    #[test]
    fn unknown_mode_rejected() {
        let mut enc = NarrowbandEncoder::new();
        let frame = [0i16; NB_FRAME_SAMPLES];
        assert_eq!(
            enc.encode_frame(&frame, 9),
            Err(EncodeError::UnknownMode(9))
        );
    }

    #[test]
    fn voiced_frame_picks_matching_pitch() {
        // Prime the excitation history so that e[n − 40] == residual[n]
        // for the whole sub-frame (the last 40 history samples ARE the
        // residual pattern). The pitch search must then lock onto 40.
        let mut enc = NarrowbandEncoder::new();
        let t = 40usize;
        let pattern: Vec<f64> = (0..SUBFRAME_SAMPLES)
            .map(|n| ((n * 7 % 13) as f64) - 6.0)
            .collect();
        let hlen = enc.exc_hist.len();
        for (n, &pv) in pattern.iter().enumerate() {
            enc.exc_hist[hlen - t + n] = pv;
        }
        let mut residual = [0.0_f64; SUBFRAME_SAMPLES];
        residual.copy_from_slice(&pattern);
        let (period, _idx, _taps) = enc.search_pitch(&residual, PitchGainQuant::Vq7Bit);
        assert_eq!(
            period, 40,
            "pitch search did not recover the planted period"
        );
    }
}
