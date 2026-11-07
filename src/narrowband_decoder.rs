//! **Narrowband CELP decoder** — one parsed [`crate::NarrowbandFrameBody`]
//! to 160 samples of PCM, structured the way the reference decoder's
//! float path is (`docs/audio/speex/tracer-2026-09.md`, Manual §8–§9):
//!
//! ```text
//!   for each 20 ms frame:
//!     LSPs: linear base + staged VQ stages          (Manual §9.1)
//!     per sub-frame s: ω_s = (1−w)·ω_{f−1} + w·ω_f,  w = (s+1)/4,
//!                      0.002-rad margin, LSP→LPC     (tracer §1.1, §3)
//!     per sub-frame:  e = p + c   (3-tap / forced pitch + innovation)
//!                     or the submode-1 vocoder mix   (tracer §4)
//!     enhancer on e[−40..119] (optional)             (tracer §2)
//!     synthesis 1/A(z) of e[−40..119]                (one sub-frame behind)
//!     output high-pass                               (on by default)
//! ```
//!
//! ## One sub-frame behind
//!
//! The synthesis runs one 40-sample sub-frame behind the excitation
//! decode: frame `f` outputs the synthesis of excitation samples
//! `−40..119` (the previous frame's last sub-frame, then this frame's
//! first three), each through its own sub-frame's LPC. That look-ahead
//! is what lets the perceptual enhancer's 80-sample blocks straddle the
//! frame boundary, and it is the reference's output alignment: this
//! decoder's PCM lines up with the reference decode sample for sample.
//!
//! ## Arithmetic
//!
//! Float, in the reference float build's order where it matters:
//!
//! * the stage scalings are the decimal constants `0.0039062`,
//!   `0.0019531`, `0.00097656` (not the exact powers of two) — measured,
//!   +0.8 dB against the staged speech fixtures;
//! * the LSP cosine is the staged even polynomial (tracer §3,
//!   `tables/lsp-cos-poly.csv`), not `cos`;
//! * the synthesis filter is the **transposed** direct form, whose
//!   state carries partial sums formed with the *previous* sub-frame's
//!   coefficients across an LPC change;
//! * the excitation memory the adaptive codebook reads is the float
//!   excitation, not a 16-bit rounding of it.
//!
//! The float reference is itself not bit-exact across compilers and
//! hosts (the fixtures' ±2 LSB note); this decoder reaches ±1 LSB on
//! the staged narrowband fixtures.

use crate::codebooks::{nb_lsp_high1, nb_lsp_high2, nb_lsp_low1, nb_lsp_low2, nb_lsp_stage0};
use crate::enhancer::{ENHANCER_BLOCK_SAMPLES, enhance_block, enhancer_comb_gain, enhancer_period};
use crate::fixed_codebook_gain::FixedCodebookGainIndices;
use crate::innovation::SUBFRAME_SAMPLES;
use crate::lsp_to_lpc::LPC_ORDER;
use crate::narrowband_body::{NarrowbandFrameBody, PITCH_PERIOD_MAX, PITCH_PERIOD_MIN};
use crate::pitch_gain::PitchGainTaps;
use crate::submode::{NarrowbandSubmode, PitchGainQuant, SUBFRAMES_PER_FRAME};
use core::fmt;

/// Number of PCM samples a narrowband frame decodes to
/// (`SUBFRAMES_PER_FRAME × SUBFRAME_SAMPLES` = 4 × 40 = 160; §9.1).
pub const NARROWBAND_FRAME_SAMPLES: usize = SUBFRAMES_PER_FRAME * SUBFRAME_SAMPLES;

/// Past excitation kept ahead of the current frame: the deepest read is
/// the enhancer's second tap (`2·144 + 3` lags, plus its 3-tap
/// interpolator reach) from a block that starts 40 samples before the
/// frame.
const EXC_HISTORY: usize = 384;

/// Narrowband LSP stage scalings of the reference float path (stage 0,
/// the 1/512 split stages, the 1/1024 split stages).
const LSP_STAGE_SCALE: [f64; 3] = [0.0039062, 0.0019531, 0.00097656];

/// Narrowband LSP stability margin in radians (tracer §1.1 step 1).
const NB_LSP_MARGIN: f64 = 0.002;

/// Reference decoder **output high-pass** at 8 kHz, measured: a biquad
/// `H(z) = (b₀ + b₁z⁻¹ + b₂z⁻²)/(1 + a₁z⁻¹ + a₂z⁻²)` with poles at
/// 66.6 Hz (`r = 0.96474`), a zero pair near DC and unit gain at
/// Nyquist. Recovered by output-error identification of crafted
/// constant-envelope streams (pitch taps zero, random innovation —
/// where the synthesis path is linear time-invariant) against the
/// black-box reference decode: 81 dB fit, ±1 LSB, identical across
/// submodes 2/3/4/5/6. Coefficients as `[b₀, b₁, b₂, a₁, a₂]`.
pub const NB_OUTPUT_HIGHPASS: [f64; 5] = [
    0.964_445_821,
    -1.928_749_302,
    0.964_433_377,
    -1.926_831_997,
    0.930_713_134,
];

/// The output high-pass the reference applies to the **wideband**
/// stream's narrowband layer, at 8 kHz before the QMF synthesis — a
/// different design from [`NB_OUTPUT_HIGHPASS`]: poles at 38.2 Hz
/// (`r = 0.98657`), a near-DC zero pair, unit gain at Nyquist. Measured
/// as [`NB_OUTPUT_HIGHPASS`] was, through the crate's QMF synthesis
/// against the reference wideband decode of a crafted stream whose
/// high band is mode 0 (silent): 85 dB, ±2 LSB on speech.
pub const WB_LOW_BAND_OUTPUT_HIGHPASS: [f64; 5] = [
    0.986_448_905,
    -1.972_768_951,
    0.986_450_153,
    -1.972_261_770,
    0.973_321_785,
];

/// Gain of the uniform unit-variance noise draw (tracer §4,
/// `r = 3.4642·std·(u − 1.5)` for `u ∈ [1, 2)`).
const NOISE_SCALE: f64 = 3.4642;

/// Seed of the decoder's noise generator (tracer §4).
const NOISE_SEED: u32 = 1000;

/// Bandwidth expansion of the last LPC for null-frame comfort noise
/// (`a_k · 0.93^k`, tracer §4).
const NULL_FRAME_BANDWIDTH: f64 = 0.93;

/// Level of the mode-1 comfort-noise excitation relative to the frame
/// gain. Retained for API compatibility: the vocoder excitation is now
/// the exact noise-generator + pulse-train law of tracer §4, whose
/// unit-variance draw is scaled by the frame gain directly.
pub const NB_MODE1_NOISE_SCALE: f32 = 1.0;

/// Errors from [`NarrowbandDecoder::decode_frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrowbandDecodeError {
    /// A sub-mode that transmits an LSP field produced no reconstructed
    /// LSP vector — should not happen for the nine documented modes.
    LspUnavailable,
    /// The fixed-codebook gain index pair could not be composed (a
    /// non-conforming hand-built sub-mode).
    GainIndicesUnavailable,
    /// The innovation sub-vector for a sub-frame could not be decoded.
    InnovationUndocumented { mode_id: u8 },
    /// A resolved pitch period fell outside the §9.2 `[17, 144]` range
    /// (a corrupt stream or a non-conforming sub-mode).
    PitchOutOfRange { period: u16 },
}

impl fmt::Display for NarrowbandDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NarrowbandDecodeError::LspUnavailable => {
                write!(f, "narrowband decode: sub-mode transmits no LSP field")
            }
            NarrowbandDecodeError::GainIndicesUnavailable => write!(
                f,
                "narrowband decode: fixed-codebook gain indices could not be composed"
            ),
            NarrowbandDecodeError::InnovationUndocumented { mode_id } => write!(
                f,
                "narrowband decode: innovation codebook binding for mode {mode_id} is a docs gap"
            ),
            NarrowbandDecodeError::PitchOutOfRange { period } => write!(
                f,
                "narrowband decode: resolved pitch period {period} outside [17, 144]"
            ),
        }
    }
}

impl std::error::Error for NarrowbandDecodeError {}

/// Biquad state of the output high-pass.
#[derive(Debug, Clone, Copy, Default)]
struct Highpass {
    x: [f64; 2],
    y: [f64; 2],
}

impl Highpass {
    #[inline]
    fn step(&mut self, c: &[f64; 5], v: f64) -> f64 {
        let y =
            c[0] * v + c[1] * self.x[0] + c[2] * self.x[1] - c[3] * self.y[0] - c[4] * self.y[1];
        self.x = [v, self.x[0]];
        self.y = [y, self.y[0]];
        y
    }
}

/// State of the submode-1 vocoder excitation (tracer §4).
#[derive(Debug, Clone, Copy, Default)]
struct VocoderState {
    /// Persistent pulse offset `o` (carries across sub-frames).
    pulse_offset: i32,
    /// `P[−1]`, `i[−1]` and the running mean `μ`.
    last_pulse: f64,
    last_noise: f64,
    mean: f64,
}

/// Stateful narrowband CELP decoder carrying the synthesis, excitation,
/// high-pass, enhancer and noise-generator state across frames.
///
/// Construct with [`NarrowbandDecoder::new`] (stream start), then call
/// [`NarrowbandDecoder::decode_frame`] once per 20 ms narrowband frame.
/// The perceptual enhancer and the output high-pass are **on** by
/// default, as in the reference decoder; [`Self::set_enhancer`] /
/// [`Self::set_output_highpass`] turn them off.
#[derive(Debug, Clone)]
pub struct NarrowbandDecoder {
    /// Excitation timeline: `EXC_HISTORY` past samples followed by the
    /// current frame's 160.
    exc: Vec<f64>,
    /// Transposed-form synthesis state.
    synth: [f64; LPC_ORDER],
    /// `A(z) = 1 + Σ a_k z^{−k}` of the previous frame's last sub-frame
    /// (synthesises the pending excitation `e[−40..−1]`).
    pending_lpc: [f64; LPC_ORDER],
    /// Previous frame's decoded LSPs (radians); `None` restarts the
    /// interpolation (stream start, after a null frame).
    prev_lsp: Option<[f64; LPC_ORDER]>,
    highpass: Highpass,
    highpass_on: bool,
    highpass_coeffs: [f64; 5],
    enhancer_on: bool,
    noise_state: u32,
    vocoder: VocoderState,
    last_frame_excitation: [f32; NARROWBAND_FRAME_SAMPLES],
    last_frame_innovation: [f32; NARROWBAND_FRAME_SAMPLES],
    last_crossover_response: [f64; 4],
}

impl Default for NarrowbandDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl NarrowbandDecoder {
    /// A fresh decoder at stream start (zero history everywhere,
    /// enhancer and output high-pass on).
    pub fn new() -> Self {
        Self {
            exc: vec![0.0; EXC_HISTORY + NARROWBAND_FRAME_SAMPLES],
            synth: [0.0; LPC_ORDER],
            pending_lpc: [0.0; LPC_ORDER],
            prev_lsp: None,
            highpass: Highpass::default(),
            highpass_on: true,
            highpass_coeffs: NB_OUTPUT_HIGHPASS,
            enhancer_on: true,
            noise_state: NOISE_SEED,
            vocoder: VocoderState::default(),
            last_frame_excitation: [0.0; NARROWBAND_FRAME_SAMPLES],
            last_frame_innovation: [0.0; NARROWBAND_FRAME_SAMPLES],
            last_crossover_response: [1.0; 4],
        }
    }

    /// Turn the perceptual enhancer (tracer §2) on or off. The
    /// reference decoder's default — and this decoder's — is on; the
    /// in-band signalling code 0 (Manual Table 5.1) toggles it.
    pub fn set_enhancer(&mut self, on: bool) {
        self.enhancer_on = on;
    }

    /// Whether the perceptual enhancer is on.
    pub fn enhancer(&self) -> bool {
        self.enhancer_on
    }

    /// Turn the output high-pass ([`NB_OUTPUT_HIGHPASS`]) on or off (on
    /// by default, as in the reference decoder).
    pub fn set_output_highpass(&mut self, on: bool) {
        self.highpass_on = on;
    }

    /// Select the output high-pass design for the stream's rate class:
    /// [`NB_OUTPUT_HIGHPASS`] for a narrowband stream,
    /// [`WB_LOW_BAND_OUTPUT_HIGHPASS`] when this decoder is the
    /// narrowband layer of a wideband / ultra-wideband stream.
    pub(crate) fn set_highpass_layer(&mut self, wideband_layer: bool) {
        self.highpass_coeffs = if wideband_layer {
            WB_LOW_BAND_OUTPUT_HIGHPASS
        } else {
            NB_OUTPUT_HIGHPASS
        };
    }

    /// Whether the output high-pass is on.
    pub fn output_highpass(&self) -> bool {
        self.highpass_on
    }

    /// The composed excitation `e[n]` of the most recently decoded
    /// frame (before the enhancer): adaptive plus fixed codebook, the
    /// vocoder mix for submode 1, the comfort noise for a null frame.
    /// This is the high band's gain base `e_L` (tracer §1.1 step 5).
    pub fn last_frame_excitation(&self) -> &[f32; NARROWBAND_FRAME_SAMPLES] {
        &self.last_frame_excitation
    }

    /// Per-sub-frame `|A(−1)|` of the most recently decoded frame's
    /// interpolated quantised LPC — the low band's response at the
    /// 4 kHz QMF crossover (tracer §5). Persists across null frames;
    /// `1.0` before the first envelope-carrying frame.
    pub fn last_crossover_response(&self) -> &[f64; 4] {
        &self.last_crossover_response
    }

    /// Fixed-codebook part of the most recent frame's excitation (the
    /// gained innovation; the gained noise for submode 1) — the wideband
    /// fold source (tracer §5).
    pub fn last_frame_innovation(&self) -> &[f32; NARROWBAND_FRAME_SAMPLES] {
        &self.last_frame_innovation
    }

    /// Resolve the pitch period for sub-frame `sf_idx`: per-sub-frame
    /// fine pitch (modes 3..=7) or the frame-level open-loop period
    /// (modes 1, 2, 8); `None` for silence.
    fn pitch_period(
        body: &NarrowbandFrameBody,
        submode: &NarrowbandSubmode,
        sf_idx: usize,
    ) -> Option<u16> {
        if submode.fine_pitch_bits != 0 {
            Some(u16::from(body.subframes[sf_idx].pitch_index) + PITCH_PERIOD_MIN)
        } else if submode.ol_pitch_bits != 0 {
            Some(u16::from(body.ol_pitch_index) + PITCH_PERIOD_MIN)
        } else {
            None
        }
    }

    /// The pitch taps for sub-frame `sf_idx` (Q6, `+32` bias applied):
    /// the per-sub-frame 3-tap VQ (modes 2..=7), the frame-level forced
    /// centre tap (modes 1 / 8), or silence.
    fn pitch_taps(
        body: &NarrowbandFrameBody,
        submode: &NarrowbandSubmode,
        sf_idx: usize,
    ) -> PitchGainTaps {
        match submode.pitch_gain {
            PitchGainQuant::Vq5Bit | PitchGainQuant::Vq7Bit => body.subframes[sf_idx]
                .pitch_gain_taps(submode)
                .unwrap_or(PitchGainTaps::SILENCE),
            PitchGainQuant::None => {
                if submode.ol_pitch_gain_bits != 0 {
                    crate::forced_pitch_gain::forced_pitch_gain_taps(body.ol_pitch_gain_index)
                } else {
                    PitchGainTaps::SILENCE
                }
            }
        }
    }

    /// One noise-generator draw (tracer §4): the 32-bit LCG advanced
    /// once, mapped to a uniform unit-variance sample.
    fn draw_noise(&mut self) -> f64 {
        self.noise_state = self
            .noise_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        let u = 1.0 + f64::from(self.noise_state & 0x007F_FFFF) / f64::from(1u32 << 23);
        NOISE_SCALE * (u - 1.5)
    }

    /// Decode one narrowband frame to `f64` PCM, advancing all decoder
    /// state. Returns [`NARROWBAND_FRAME_SAMPLES`] = 160 samples, aligned
    /// with the reference decoder's output.
    // The sub-frame index addresses five parallel per-sub-frame arrays.
    #[allow(clippy::needless_range_loop)]
    pub fn decode_frame(
        &mut self,
        body: &NarrowbandFrameBody,
        submode: &NarrowbandSubmode,
    ) -> Result<[f64; NARROWBAND_FRAME_SAMPLES], NarrowbandDecodeError> {
        // Advance the excitation timeline by one frame.
        self.exc.copy_within(NARROWBAND_FRAME_SAMPLES.., 0);
        let cur = EXC_HISTORY;

        let Some(lsp) = nb_lsp_radians(body, submode) else {
            return Ok(self.decode_null_frame());
        };
        let lpcs = nb_subframe_lpcs(self.prev_lsp, &lsp);
        for (slot, lpc) in self.last_crossover_response.iter_mut().zip(lpcs.iter()) {
            *slot = lpc_at_minus_one(lpc).abs();
        }

        let mut enh_taps = [(40i32, [0.0f64; 3]); 4];
        for sf in 0..SUBFRAMES_PER_FRAME {
            let base = cur + sf * SUBFRAME_SAMPLES;
            let gain_indices = FixedCodebookGainIndices::from_body(body, submode, sf)
                .ok_or(NarrowbandDecodeError::GainIndicesUnavailable)?;
            let g_frame = (f64::from(body.ol_exc_gain_index) / 3.5).exp();
            let period = Self::pitch_period(body, submode, sf);
            if let Some(t) = period
                && !(PITCH_PERIOD_MIN..=PITCH_PERIOD_MAX).contains(&t)
            {
                return Err(NarrowbandDecodeError::PitchOutOfRange { period: t });
            }

            if submode.mode_id == 1 {
                // Vocoder (tracer §4): noise × frame gain plus a pulse
                // train at the open-loop period; the decoded adaptive
                // contribution is discarded.
                let t = i32::from(period.unwrap_or(PITCH_PERIOD_MIN));
                let beta = f64::from(crate::forced_pitch_gain::FORCED_PITCH_GAIN_STEP)
                    * f64::from(body.ol_pitch_gain_index);
                let voicing = (1.5 * (beta - 0.2)).clamp(0.0, 1.0);
                let mut noise = [0.0f64; SUBFRAME_SAMPLES];
                for v in noise.iter_mut() {
                    *v = self.draw_noise() * g_frame;
                }
                let mut pulses = [0.0f64; SUBFRAME_SAMPLES];
                let height = (2.0 * f64::from(t)).sqrt() * voicing * g_frame;
                let st = &mut self.vocoder;
                while st.pulse_offset < SUBFRAME_SAMPLES as i32 {
                    pulses[st.pulse_offset as usize] = height;
                    st.pulse_offset += t;
                }
                st.pulse_offset -= SUBFRAME_SAMPLES as i32;
                for n in 0..SUBFRAME_SAMPLES {
                    let pp = if n > 0 { pulses[n - 1] } else { st.last_pulse };
                    let ip = if n > 0 { noise[n - 1] } else { st.last_noise };
                    let v = 0.7 * pulses[n] + 0.3 * pp + (1.0 - 0.85 * voicing) * noise[n]
                        - 0.15 * voicing * ip;
                    st.mean = 0.8 * st.mean + 0.2 * v;
                    self.exc[base + n] = v - st.mean;
                    self.last_frame_innovation[sf * SUBFRAME_SAMPLES + n] = noise[n] as f32;
                }
                st.last_pulse = pulses[SUBFRAME_SAMPLES - 1];
                st.last_noise = noise[SUBFRAME_SAMPLES - 1];
                enh_taps[sf] = (t, [0.0; 3]);
                continue;
            }

            // Fixed-codebook contribution c[n] = g·shape/32.
            let shape = crate::innovation::decode_subframe_f32(
                submode,
                body.subframes[sf].innovation_vq_index,
            )
            .map_err(|_| NarrowbandDecodeError::InnovationUndocumented {
                mode_id: submode.mode_id,
            })?;
            let g_sub = f64::from(
                crate::gain_reconstruction::reconstruct_subframe_gain_correction(
                    gain_indices.subframe,
                ),
            );
            let g = g_frame * g_sub;

            // Adaptive-codebook contribution from the float history.
            let mut p = [0.0f64; SUBFRAME_SAMPLES];
            let forced = submode.pitch_gain == PitchGainQuant::None;
            let b = if forced {
                // Forced open-loop gain: the float coefficient on the
                // centre tap alone.
                let coef = crate::forced_pitch_gain::forced_pitch_coef(body.ol_pitch_gain_index);
                [0.0, f64::from(coef), 0.0]
            } else {
                let taps = Self::pitch_taps(body, submode, sf);
                [
                    f64::from(taps.taps[0]) / 64.0,
                    f64::from(taps.taps[1]) / 64.0,
                    f64::from(taps.taps[2]) / 64.0,
                ]
            };
            if let Some(t) = period {
                let t = t as i32;
                enh_taps[sf] = (t, b);
                for n in 0..SUBFRAME_SAMPLES as i32 {
                    if forced {
                        // Forced open-loop gain: a centre-tap recursion
                        // over the pitch partial.
                        let k = n - t;
                        p[n as usize] = b[1]
                            * if k >= 0 {
                                p[k as usize]
                            } else {
                                self.exc[(base as i32 + k) as usize]
                            };
                    } else {
                        // 3-tap predictor; a lag reaching into the
                        // sub-frame folds back by one period once and
                        // never further (nb-pitch-short-lag.md).
                        let mut acc = 0.0;
                        for (j, off) in [-1i32, 0, 1].iter().enumerate() {
                            let mut k = n - t + off;
                            if k >= 0 {
                                k -= t;
                            }
                            if k < 0 {
                                acc += b[j] * self.exc[(base as i32 + k) as usize];
                            }
                        }
                        p[n as usize] = acc;
                    }
                }
            }
            for n in 0..SUBFRAME_SAMPLES {
                let c = g * f64::from(shape[n]) / 32.0;
                self.exc[base + n] = p[n] + c;
                self.last_frame_innovation[sf * SUBFRAME_SAMPLES + n] = c as f32;
            }
        }
        for (o, &e) in self
            .last_frame_excitation
            .iter_mut()
            .zip(&self.exc[cur..cur + NARROWBAND_FRAME_SAMPLES])
        {
            *o = e as f32;
        }

        // The synthesised excitation e[−40..119], enhanced or not.
        let mut syn = [0.0f64; NARROWBAND_FRAME_SAMPLES];
        let gamma = enhancer_comb_gain(submode.mode_id).filter(|_| self.enhancer_on);
        match gamma {
            Some(gamma) => {
                let period = enhancer_period(&enh_taps);
                for half in 0..2 {
                    let start = cur as i32 - 40 + (half * ENHANCER_BLOCK_SAMPLES) as i32;
                    let end = (cur + NARROWBAND_FRAME_SAMPLES) as i32;
                    let exc = &self.exc;
                    let read = |i: i32| -> f64 {
                        let k = start + i;
                        if k >= end { 0.0 } else { exc[k as usize] }
                    };
                    let mut block = [0.0f64; ENHANCER_BLOCK_SAMPLES];
                    enhance_block(&read, period, gamma, &mut block);
                    syn[half * ENHANCER_BLOCK_SAMPLES..(half + 1) * ENHANCER_BLOCK_SAMPLES]
                        .copy_from_slice(&block);
                }
            }
            None => syn.copy_from_slice(&self.exc[cur - 40..cur + 120]),
        }

        let mut pcm = [0.0f64; NARROWBAND_FRAME_SAMPLES];
        for (n, (o, &v)) in pcm.iter_mut().zip(syn.iter()).enumerate() {
            let a = if n < SUBFRAME_SAMPLES {
                &self.pending_lpc
            } else {
                &lpcs[n / SUBFRAME_SAMPLES - 1]
            };
            *o = synth_step(&mut self.synth, a, v);
        }
        self.pending_lpc = lpcs[SUBFRAMES_PER_FRAME - 1];
        self.prev_lsp = Some(lsp);
        if self.highpass_on {
            for v in pcm.iter_mut() {
                *v = self.highpass.step(&self.highpass_coeffs, *v);
            }
        }
        Ok(pcm)
    }

    /// A null (submode-0) frame: 160 comfort-noise draws at the previous
    /// frame's excitation level, synthesised directly through the last
    /// LPC bandwidth-expanded by `0.93^k`, without the output high-pass;
    /// the next frame restarts the LSP interpolation (tracer §4).
    fn decode_null_frame(&mut self) -> [f64; NARROWBAND_FRAME_SAMPLES] {
        let cur = EXC_HISTORY;
        let prev = &self.exc[cur - NARROWBAND_FRAME_SAMPLES..cur];
        let level = (0.1 + prev.iter().map(|v| v * v).sum::<f64>() / prev.len() as f64).sqrt();
        let mut lpc = self.pending_lpc;
        let mut f = NULL_FRAME_BANDWIDTH;
        for a in lpc.iter_mut() {
            *a *= f;
            f *= NULL_FRAME_BANDWIDTH;
        }
        let mut pcm = [0.0f64; NARROWBAND_FRAME_SAMPLES];
        for (n, o) in pcm.iter_mut().enumerate() {
            let v = level * self.draw_noise();
            self.exc[cur + n] = v;
            self.last_frame_excitation[n] = v as f32;
            *o = synth_step(&mut self.synth, &lpc, v);
        }
        self.last_frame_innovation = [0.0; NARROWBAND_FRAME_SAMPLES];
        self.prev_lsp = None;
        pcm
    }

    /// Decode one narrowband frame to rounded, saturated `i16` PCM.
    pub fn decode_frame_i16(
        &mut self,
        body: &NarrowbandFrameBody,
        submode: &NarrowbandSubmode,
    ) -> Result<[i16; NARROWBAND_FRAME_SAMPLES], NarrowbandDecodeError> {
        let pcm = self.decode_frame(body, submode)?;
        let mut out = [0i16; NARROWBAND_FRAME_SAMPLES];
        for (o, &s) in out.iter_mut().zip(pcm.iter()) {
            *o = saturate_i16(s);
        }
        Ok(out)
    }

    /// Read-only view of the synthesis filter state (diagnostics).
    pub fn synthesis_history(&self) -> &[f64; LPC_ORDER] {
        &self.synth
    }
}

/// One sample of the transposed-direct-form all-pole synthesis
/// `y = x − Σ a_k y[n−k]`.
#[inline]
pub(crate) fn synth_step(mem: &mut [f64; LPC_ORDER], a: &[f64; LPC_ORDER], x: f64) -> f64 {
    let y = x + mem[0];
    for j in 0..LPC_ORDER - 1 {
        mem[j] = mem[j + 1] - a[j] * y;
    }
    mem[LPC_ORDER - 1] = -a[LPC_ORDER - 1] * y;
    y
}

/// The frame's ten decoded narrowband LSPs in radians (linear base
/// `0.25·(i+1)` plus the scaled VQ stages), or `None` for a frame that
/// carries no LSP field.
fn nb_lsp_radians(body: &NarrowbandFrameBody, submode: &NarrowbandSubmode) -> Option<[f64; 10]> {
    nb_lsp_radians_from_stages(body.lsp_stages(submode)?)
}

/// [`nb_lsp_radians`] from already-split VQ stage indices (the encoder's
/// decoder mirror).
pub(crate) fn nb_lsp_radians_from_stages(st: crate::lsp::NbLspStages) -> Option<[f64; 10]> {
    let mut l = [0.0f64; 10];
    for (i, v) in l.iter_mut().enumerate() {
        *v = 0.25 * i as f64 + 0.25;
    }
    let s0 = nb_lsp_stage0().get(usize::from(st.stage0))?;
    for (v, &c) in l.iter_mut().zip(s0.iter()) {
        *v += f64::from(c) * LSP_STAGE_SCALE[0];
    }
    let low1 = nb_lsp_low1().get(usize::from(st.low1))?;
    for (v, &c) in l[..5].iter_mut().zip(low1.iter()) {
        *v += f64::from(c) * LSP_STAGE_SCALE[1];
    }
    if let Some(i) = st.low2 {
        for (v, &c) in l[..5]
            .iter_mut()
            .zip(nb_lsp_low2().get(usize::from(i))?.iter())
        {
            *v += f64::from(c) * LSP_STAGE_SCALE[2];
        }
    }
    let high1 = nb_lsp_high1().get(usize::from(st.high1))?;
    for (v, &c) in l[5..].iter_mut().zip(high1.iter()) {
        *v += f64::from(c) * LSP_STAGE_SCALE[1];
    }
    if let Some(i) = st.high2 {
        for (v, &c) in l[5..]
            .iter_mut()
            .zip(nb_lsp_high2().get(usize::from(i))?.iter())
        {
            *v += f64::from(c) * LSP_STAGE_SCALE[2];
        }
    }
    Some(l)
}

/// The decoder's four per-sub-frame LPC sets (`A(z) = 1 + Σ a_k z^{−k}`)
/// for a frame's decoded LSPs and the previous frame's (`None` = stream
/// start / interpolation restart) — shared with the encoder's decoder
/// mirror so both run the same synthesis.
pub(crate) fn nb_subframe_lpcs(
    prev: Option<[f64; LPC_ORDER]>,
    lsp: &[f64; LPC_ORDER],
) -> [[f64; LPC_ORDER]; SUBFRAMES_PER_FRAME] {
    let prev = prev.unwrap_or(*lsp);
    let mut lpcs = [[0.0f64; LPC_ORDER]; SUBFRAMES_PER_FRAME];
    for (sf, lpc) in lpcs.iter_mut().enumerate() {
        let w = (sf as f64 + 1.0) / 4.0;
        let mut l = [0.0f64; LPC_ORDER];
        for i in 0..LPC_ORDER {
            l[i] = (1.0 - w) * prev[i] + w * lsp[i];
        }
        enforce_interpolated_margin(&mut l, NB_LSP_MARGIN);
        *lpc = lsp_to_lpc_poly(&l);
    }
    lpcs
}

/// The stability margin of the interpolated LSPs, in index order
/// (tracer §1.1 step 1): `ω₀ ≥ m`, `ω_{p−1} ≤ π − m`, then for
/// `i = 1..p−2` raise `ω_i` to `ω_{i−1} + m` and, if it crowds
/// `ω_{i+1}`, replace it by the mean of itself and `ω_{i+1} − m`.
#[doc(hidden)]
pub fn enforce_interpolated_margin(l: &mut [f64], m: f64) {
    let p = l.len();
    if p < 2 {
        return;
    }
    if l[0] < m {
        l[0] = m;
    }
    if l[p - 1] > core::f64::consts::PI - m {
        l[p - 1] = core::f64::consts::PI - m;
    }
    for i in 1..p - 1 {
        if l[i] < l[i - 1] + m {
            l[i] = l[i - 1] + m;
        }
        if l[i] > l[i + 1] - m {
            l[i] = 0.5 * (l[i] + l[i + 1] - m);
        }
    }
}

/// The reference float LSP cosine: an even polynomial in the angle with
/// a π/2 breakpoint (tracer §3, `tables/lsp-cos-poly.csv`).
#[doc(hidden)]
pub fn lsp_cos_poly(x: f64) -> f64 {
    const K: [f64; 4] = [
        0.999_993_294_6,
        -0.499_912_437_6,
        0.041_487_747_2,
        -0.001_271_209_5,
    ];
    let even = |y: f64| {
        let u = y * y;
        K[0] + u * (K[1] + u * (K[2] + u * K[3]))
    };
    if x < core::f64::consts::FRAC_PI_2 {
        even(x)
    } else {
        -even(core::f64::consts::PI - x)
    }
}

/// LSP → LPC: `A(z) = ½(P(z) + Q(z))` with `P` the product of the
/// even-indexed sections `1 − 2cos(ω)z⁻¹ + z⁻²` times `(1 + z⁻¹)` and
/// `Q` the odd-indexed ones times `(1 − z⁻¹)`. Returns `a₁..a_p` of
/// `A(z) = 1 + Σ a_k z^{−k}`.
#[doc(hidden)]
pub fn lsp_to_lpc_poly<const P: usize>(lsp: &[f64; P]) -> [f64; P] {
    // Product polynomials of degree P (P + 1 coefficients).
    const MAX: usize = 17;
    assert!(P < MAX, "LPC order");
    let mut pp = [0.0f64; MAX];
    let mut qq = [0.0f64; MAX];
    pp[0] = 1.0;
    qq[0] = 1.0;
    let (mut np, mut nq) = (1usize, 1usize);
    for (i, &w) in lsp.iter().enumerate() {
        let c = -2.0 * lsp_cos_poly(w);
        let (t, n) = if i % 2 == 0 {
            (&mut pp, &mut np)
        } else {
            (&mut qq, &mut nq)
        };
        // Multiply by (1 + c·z⁻¹ + z⁻²).
        let len = *n + 2;
        for k in (0..len).rev() {
            let mut v = t[k];
            if k >= 1 {
                v += c * t[k - 1];
            }
            if k >= 2 {
                v += t[k - 2];
            }
            t[k] = v;
        }
        *n = len;
    }
    let mut a = [0.0f64; P];
    for k in 1..=P {
        a[k - 1] = 0.5 * (pp[k] + pp[k - 1] + qq[k] - qq[k - 1]);
    }
    a
}

/// `A(−1) = 1 + Σ (−1)^k a_k` for `A(z) = 1 + Σ a_k z^{−k}`.
fn lpc_at_minus_one(a: &[f64; LPC_ORDER]) -> f64 {
    let mut v = 1.0;
    for (k, &ak) in a.iter().enumerate() {
        if k % 2 == 0 {
            v -= ak;
        } else {
            v += ak;
        }
    }
    v
}

/// Round to nearest and saturate an `f64` into the signed-16-bit range.
///
/// The single rounding convention shared by every `*_i16` PCM
/// convenience in the crate: round to nearest (ties away from zero, per
/// [`f64::round`]) then clamp to `[i16::MIN, i16::MAX]`.
#[inline]
pub fn saturate_i16(v: f64) -> i16 {
    let r = v.round();
    if r >= f64::from(i16::MAX) {
        i16::MAX
    } else if r <= f64::from(i16::MIN) {
        i16::MIN
    } else {
        r as i16
    }
}
