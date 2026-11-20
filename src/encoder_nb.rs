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
//! reconstruction levels.
//!
//! All nine Table 9.1 modes encode (r438; the staged
//! `nb-innovation-binding.md` pinned the last two):
//!
//! * modes 2 / 3 / 4 / 5 / 6 / 8 — single-stage innovation VQ;
//! * mode 7 — **two-stage** innovation (stage 2 quantises the residual
//!   stage 1 leaves, binding doc §3);
//! * mode 1 — the vocoder mode: no innovation vector; the excitation is
//!   the frame-level forced pitch path alone, and the four inert 1-bit
//!   innovation-gain fields are written as `0` (the value the binding
//!   doc §4 observed on every reference-encoded frame);
//! * mode 0 — pure silence.
//!
//! Modes with a frame-level open-loop pitch field (1 / 2 / 8) estimate
//! it by normalised correlation over the frame residual against the
//! excitation history ([`crate::estimate_open_loop_pitch`]); the
//! forced-gain modes (1 / 8) quantise the correlation coefficient
//! through the staged `provenance/02` law (0.9 damping, `15·coef`
//! clamped to the 4-bit grid).

use crate::adaptive_codebook::TAP_PITCH_OFFSETS;
use crate::frame::FrameError;
use crate::gain_reconstruction::{
    quantise_frame_ol_exc_gain_exact, quantise_subframe_gain_correction,
    reconstruct_frame_ol_exc_gain, reconstruct_subframe_gain_correction,
};
use crate::innovation::{InnovationCodebook, InnovationMapping, SUBFRAME_SAMPLES, sub_vector};
use crate::innovation_search::search_innovation;
use crate::lpc_to_lsp::lpc_to_lsp;
use crate::lsp_base::nb_lsp_base_q10;
use crate::lsp_interp::NbSubFrameLsp;
use crate::lsp_quant::pack_lsp_index;
use crate::lsp_to_lpc::{LPC_ORDER, lsp_vector_radians_to_q10, subframe_lpc_set_with_base};
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
/// The shortest pitch lag the encoder transmits. Lags 17..=19 are
/// legal on the wire, but a sub-frame's tail beyond two periods of
/// such a lag is content decoders are not known to agree on
/// (`nb-pitch-short-lag.md` §2.4), so the searches leave them out —
/// every stream then means the same thing to every decoder.
const SHORTEST_SEARCHED_LAG: u16 = 20;
/// Length of the retained excitation history (>= PITCH_PERIOD_MAX).
const EXC_HIST_LEN: usize = PITCH_PERIOD_MAX as usize + SUBFRAME_SAMPLES;

/// Errors from [`NarrowbandEncoder::encode_frame`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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

/// Frame-level pitch decisions shared by all four sub-frames — the
/// encode-side counterpart of the two frame-level Table 9.1 pitch rows.
#[derive(Debug, Clone, Copy, Default)]
struct PitchPlan {
    /// Frame-level open-loop pitch period (modes 1 / 2 / 8). Also the
    /// fixed per-sub-frame lag for modes without a fine-pitch field
    /// (the wire transmits no per-sub-frame period there).
    ol_period: Option<u16>,
    /// Forced (frame-level) pitch-gain taps for the `OL pitch gain`
    /// modes (1 / 8) — the same single-centre-tap reconstruction the
    /// decoder applies ([`crate::forced_pitch_gain_taps`]).
    forced_taps: Option<pitch_gain::PitchGainTaps>,
    /// The forced-gain field itself (modes 1 / 8): the decoder applies
    /// its float coefficient, not the Q6-rounded tap.
    forced_q: Option<u8>,
}

/// Per-sub-frame search tables that depend only on the sub-frame's
/// decoder LPC, shared by every frame-gain candidate of a frame.
#[derive(Debug, Clone)]
struct SubframeTables {
    /// Impulse response of the search-domain filter: the decoder's
    /// synthesis `1/A(z)`, through the noise-weighting filter when one
    /// is configured.
    h: [f64; SUBFRAME_SAMPLES],
    /// Every codebook row through `h`, in the decode-law scale:
    /// `fs[idx·N + m] = scale · Σ_k h[m−k]·row[k]`.
    fs: Vec<f64>,
}

impl SubframeTables {
    fn build(
        a: &[f64; LPC_ORDER],
        weights: Option<&WeightingTaps>,
        cb: Option<InnovationCodebook>,
    ) -> Self {
        const N: usize = SUBFRAME_SAMPLES;
        let mut h = [0.0f64; N];
        {
            let mut m = [0.0f64; LPC_ORDER];
            let (mut fir, mut iir) = ([0.0f64; LPC_ORDER], [0.0f64; LPC_ORDER]);
            for (n, v) in h.iter_mut().enumerate() {
                let y = crate::narrowband_decoder::synth_step(
                    &mut m,
                    a,
                    if n == 0 { 1.0 } else { 0.0 },
                );
                *v = match weights {
                    Some(w) => weighting_step(w, &mut fir, &mut iir, y),
                    None => y,
                };
            }
        }
        let Some(cb) = cb else {
            return Self { h, fs: Vec::new() };
        };
        let scale = f64::from(crate::gain_scaled_innovation::INNOVATION_CODEBOOK_SCALE);
        let entries = cb.entries() as usize;
        let mut fs = vec![0.0f64; entries * N];
        for (idx, out) in fs.as_chunks_mut::<N>().0.iter_mut().enumerate() {
            let Some(row) = sub_vector(cb, idx as u32) else {
                continue;
            };
            for (m, slot) in out.iter_mut().enumerate() {
                let mut f = 0.0;
                for (k, &rv) in row.iter().enumerate() {
                    if k <= m {
                        f += h[m - k] * f64::from(rv);
                    }
                }
                *slot = f * scale;
            }
        }
        Self { h, fs }
    }
}

/// The two tap sets of the noise-weighting filter
/// `W(z) = A(z/γ1)/A(z/γ2)` (manual Eq. 8.1) for one sub-frame's LPC in
/// the decoder's `A(z) = 1 + Σ a_k z^{−k}` convention.
#[derive(Debug, Clone, Copy)]
struct WeightingTaps {
    num: [f64; LPC_ORDER],
    den: [f64; LPC_ORDER],
}

fn weighting_taps(a: &[f64; LPC_ORDER], gamma1: f64, gamma2: f64) -> WeightingTaps {
    let mut w = WeightingTaps {
        num: [0.0; LPC_ORDER],
        den: [0.0; LPC_ORDER],
    };
    let (mut p1, mut p2) = (gamma1, gamma2);
    for k in 0..LPC_ORDER {
        w.num[k] = a[k] * p1;
        w.den[k] = a[k] * p2;
        p1 *= gamma1;
        p2 *= gamma2;
    }
    w
}

/// One sample through `W(z)`; `fir` / `iir` hold the previous inputs /
/// outputs, most recent first.
#[inline]
fn weighting_step(
    w: &WeightingTaps,
    fir: &mut [f64; LPC_ORDER],
    iir: &mut [f64; LPC_ORDER],
    x: f64,
) -> f64 {
    let mut y = x;
    for k in 0..LPC_ORDER {
        y += w.num[k] * fir[k] - w.den[k] * iir[k];
    }
    fir.copy_within(0..LPC_ORDER - 1, 1);
    fir[0] = x;
    iir.copy_within(0..LPC_ORDER - 1, 1);
    iir[0] = y;
    y
}

/// How deep the encoder's searches go — the knobs behind the 1..=10
/// complexity setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SearchEffort {
    /// Frame-gain levels tried on each side of the open-loop estimate.
    pub gain_span: u8,
    /// Pitch lags kept from the open-loop pre-selection for the
    /// exhaustive gain-VQ search.
    pub lags: u8,
    /// Adaptive-codebook candidates carried through the innovation
    /// search (the pair with the lowest final error wins).
    pub pitch_candidates: u8,
    /// Partial paths kept by the innovation sub-vector search.
    pub innovation_paths: u8,
    /// Innovation passes: 2 re-searches at the quantised sub-frame gain.
    pub gain_passes: u8,
    /// Candidates kept per stage by the LSP quantiser.
    pub lsp_paths: u8,
    /// Centre the frame-gain candidates by a probe pass with
    /// unquantised innovation gains (otherwise on the open-loop
    /// estimate).
    pub gain_probe: bool,
}

impl SearchEffort {
    /// The search depths of complexity `c` (clamped to 1..=10).
    pub(crate) fn for_complexity(c: u8) -> Self {
        let c = c.clamp(1, 10);
        Self {
            gain_span: match c {
                1..=2 => 0,
                3..=7 => 1,
                _ => 2,
            },
            lags: [2, 4, 8, 8, 8, 10, 12, 12, 14, 16][usize::from(c) - 1],
            pitch_candidates: [1, 1, 1, 2, 2, 2, 3, 3, 3, 4][usize::from(c) - 1],
            innovation_paths: [1, 1, 3, 3, 4, 5, 5, 6, 8, 10][usize::from(c) - 1],
            gain_passes: if c == 1 { 1 } else { 2 },
            lsp_paths: [1, 1, 3, 3, 4, 4, 5, 6, 8, 8][usize::from(c) - 1],
            gain_probe: c >= 2,
        }
    }
}

/// The complexity a fresh encoder searches at.
pub const DEFAULT_COMPLEXITY: u8 = 3;

/// One adaptive-codebook candidate: (pitch-only search error, period,
/// gain-VQ index, contribution).
type PitchCandidate = (f64, u16, u8, [f64; SUBFRAME_SAMPLES]);

/// What one frame's sub-frame searches share across the frame-gain
/// candidates: the per-sub-frame tables (built on first use), the first
/// sub-frame's adaptive-codebook candidates (its history and target do
/// not depend on the frame gain), and the innovation search's working
/// buffers.
#[derive(Debug, Default)]
struct FrameScratch {
    tables: [Option<SubframeTables>; SUBFRAMES],
    first_pitch: Option<Vec<PitchCandidate>>,
    /// Gain-scaled row responses and their running energies.
    scaled: Vec<f64>,
    energy: Vec<f64>,
    /// Centre-tap synthesis responses by pitch lag.
    centre: Vec<[f64; SUBFRAME_SAMPLES]>,
    /// Probe pass: innovation gains are left unquantised and recorded
    /// in `ideal` instead of being coded.
    probe: bool,
    /// Per-sub-frame `(least-squares innovation gain, energy of the
    /// shape's synthesis response)` from the probe pass, once it ran.
    ideal: Option<[(f64, f64); SUBFRAMES]>,
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
    /// Most recent frame's innovation-only excitation `g·c[n]` (the
    /// encoder-side mirror of
    /// [`crate::NarrowbandDecoder::last_frame_innovation`]) — the r450
    /// mode-1 fold source.
    last_inn: [f64; NB_FRAME_SAMPLES],
    /// Most recent frame's per-sub-frame `|A_lb(π)|` from the quantised
    /// envelope (mirror of
    /// [`crate::NarrowbandDecoder::last_crossover_response`]).
    last_api: [f64; 4],
    /// Excitation history (`f64`), most-recent last.
    exc_hist: Vec<f64>,
    /// The input's own LPC-residual history, most-recent last — the
    /// open-loop pitch reference of the vocoder mode, whose decoded
    /// excitation is generated rather than matched.
    res_hist: Vec<f64>,
    /// Analysis (prediction-error) filter input history, most-recent last.
    analysis_hist: [f64; LPC_ORDER],
    /// Decoder mirror: the decoder's transposed-form synthesis state
    /// (`A(z) = 1 + Σ a_k z^{−k}` convention) driven by the reconstructed
    /// excitation, for the synthesis-domain searches (r483).
    synth_mem: [f64; LPC_ORDER],
    /// Decoder mirror: the previous frame's decoded LSPs on the decoder's
    /// own float path (`None` = interpolation restart).
    prev_lsp_dec: Option<[f64; LPC_ORDER]>,
    /// The decoder itself, fed every encoded frame: the source of truth
    /// for the state after frames whose excitation the decoder generates
    /// (vocoder and null frames).
    mirror: crate::narrowband_decoder::NarrowbandDecoder,
    /// Search depths (the complexity setting).
    effort: SearchEffort,
    /// The complexity setting `effort` was derived from.
    complexity: u8,
    /// Noise-weighting factors `(γ1, γ2)` of the search domain (the
    /// manual's by default); `None` searches on the plain waveform
    /// error.
    weighting: Option<(f64, f64)>,
    /// Noise-weighting filter memory over the coding error (inputs /
    /// outputs, most recent first).
    w_fir: [f64; LPC_ORDER],
    w_iir: [f64; LPC_ORDER],
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
            last_inn: [0.0; NB_FRAME_SAMPLES],
            last_api: [1.0; 4],
            exc_hist: vec![0.0; EXC_HIST_LEN],
            res_hist: vec![0.0; EXC_HIST_LEN],
            analysis_hist: [0.0; LPC_ORDER],
            synth_mem: [0.0; LPC_ORDER],
            prev_lsp_dec: None,
            mirror: {
                let mut m = crate::narrowband_decoder::NarrowbandDecoder::new();
                m.set_enhancer(false);
                m
            },
            effort: SearchEffort::for_complexity(DEFAULT_COMPLEXITY),
            complexity: DEFAULT_COMPLEXITY,
            weighting: Some((
                crate::weighting::WEIGHT_GAMMA1,
                crate::weighting::WEIGHT_GAMMA2,
            )),
            w_fir: [0.0; LPC_ORDER],
            w_iir: [0.0; LPC_ORDER],
        }
    }

    /// Set the search complexity, 1..=10 (out-of-range values clamp).
    /// Higher settings search deeper — more pitch lags, several
    /// innovation paths, joint pitch / innovation candidates, a wider
    /// frame-gain neighbourhood — for a lower coding error at a higher
    /// CPU cost; the stream format is unaffected.
    /// [`DEFAULT_COMPLEXITY`] is the default.
    pub fn set_complexity(&mut self, complexity: u8) {
        self.complexity = complexity.clamp(1, 10);
        self.effort = SearchEffort::for_complexity(self.complexity);
    }

    /// The current search complexity.
    pub fn complexity(&self) -> u8 {
        self.complexity
    }

    /// The search depths in force (shared with the high-band layer).
    pub(crate) fn effort(&self) -> SearchEffort {
        self.effort
    }

    /// Choose the search domain: `Some((γ1, γ2))` searches through the
    /// noise-weighting filter `W(z) = A(z/γ1)/A(z/γ2)` of manual §8.5
    /// (Eq. 8.1), which pushes coding noise under the formants; `None`
    /// minimises the plain waveform error. The default is the manual's
    /// pair, [`crate::WEIGHT_GAMMA1`] / [`crate::WEIGHT_GAMMA2`]
    /// (0.9 / 0.6) — on speech it measures better than the plain search
    /// on waveform SNR as well as on weighted SNR (README, "Encoder
    /// quality").
    pub fn set_noise_weighting(&mut self, gammas: Option<(f64, f64)>) {
        self.weighting = gammas.map(|(g1, g2)| (g1.clamp(0.0, 1.0), g2.clamp(0.0, 1.0)));
        self.w_fir = [0.0; LPC_ORDER];
        self.w_iir = [0.0; LPC_ORDER];
    }

    /// The noise-weighting factors in use, if any.
    pub fn noise_weighting(&self) -> Option<(f64, f64)> {
        self.weighting
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
    /// desirable to pack more than one frame per packet"*) and padded
    /// to the octet boundary as RFC 5574 §3.3 prescribes.
    /// The result is directly consumable by
    /// [`crate::SpeexDecoder::decode_packet`].
    pub fn encode_packet(
        &mut self,
        frames: &[[i16; NB_FRAME_SAMPLES]],
        mode: u8,
    ) -> Result<Vec<u8>, EncodeError> {
        let mut writer = crate::bitreader::BitWriter::new();
        for pcm in frames {
            self.encode_frame_into(&mut writer, pcm, mode)?;
        }
        crate::nb_encode::write_packet_terminator(&mut writer).map_err(EncodeError::Pack)?;
        Ok(writer.into_bytes())
    }

    /// Encode one frame in the given mode and append it (prefix + body,
    /// no padding) to `writer` — the building block for packets that
    /// carry several frames or in-band messages. Close the packet with
    /// [`crate::write_packet_terminator`].
    pub fn encode_frame_into(
        &mut self,
        writer: &mut crate::bitreader::BitWriter,
        pcm: &[i16; NB_FRAME_SAMPLES],
        mode: u8,
    ) -> Result<(), EncodeError> {
        let submode = NarrowbandSubmode::for_id(mode).ok_or(EncodeError::UnknownMode(mode))?;
        let body = self.build_body(pcm, &submode)?;
        let header = crate::frame::NarrowbandFrameHeader::new(false, submode.mode_id)
            .map_err(EncodeError::Pack)?;
        header.write(writer).map_err(EncodeError::Pack)?;
        crate::nb_encode::write_narrowband_body(writer, &body, &submode)
            .map_err(|e| EncodeError::Pack(FrameError::from(e)))
    }

    /// Encode a packet at a §2.1 quality setting (0..=10), selecting the
    /// narrowband sub-mode through the Table 9.2 quality ladder
    /// ([`crate::nb_mode_for_quality`]).
    ///
    /// Every quality 0..=10 encodes (r438 — the staged
    /// `nb-innovation-binding.md` bound the vocoder mode 1 behind
    /// quality 0 and the two-stage mode 7 behind qualities 9..=10's
    /// mode ladder tail).
    pub fn encode_packet_quality(
        &mut self,
        frames: &[[i16; NB_FRAME_SAMPLES]],
        quality: u8,
    ) -> Result<Vec<u8>, EncodeError> {
        let mode = crate::quality::nb_mode_for_quality(quality)
            .ok_or(EncodeError::UnknownMode(quality))?;
        self.encode_packet(frames, mode)
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
        for (slot, lpc) in self.last_api.iter_mut().zip(lpc_sets.iter()) {
            let mut a_pi = 1.0f64;
            for (j, &aj) in lpc.iter().enumerate() {
                let sgn = if (j + 1) % 2 == 0 { 1.0 } else { -1.0 };
                a_pi -= aj * sgn;
            }
            *slot = a_pi.abs();
        }

        // The decoder's exact per-sub-frame LPC for the transmitted LSPs
        // (its float path), for the synthesis-domain searches.
        let dec_lsp = crate::lsp::NbLspStages::from_packed(lsp_index, submode.lsp)
            .and_then(crate::narrowband_decoder::nb_lsp_radians_from_stages);
        let dec_lpcs =
            dec_lsp.map(|l| crate::narrowband_decoder::nb_subframe_lpcs(self.prev_lsp_dec, &l));
        let abs_lpcs = dec_lpcs.filter(|_| submode.mode_id >= 2);

        // Frame-level OL excitation gain: a magnitude estimate from the
        // whole-frame residual (computed with a scratch analysis filter so
        // the real per-sub-frame residual pass below stays continuous),
        // then refined closed-loop over the estimate's neighbourhood
        // (round r389 — see `refine_frame_gain`).
        let scratch = self.scratch_frame_residual(&input, &lpc_sets);
        let frame_gain_target = self.frame_gain_estimate(&scratch, submode);
        // r440: the exact staged float-build quantiser law
        // (qe = floor(0.5 + 3.5*ln g), provenance/02) replaces the
        // Q15 threshold walk for the first estimate; the closed-loop
        // refinement below is unchanged.
        let frame_gain_est = quantise_frame_ol_exc_gain_exact(frame_gain_target as f32);

        // --- Excitation: per sub-frame pitch + innovation. ---
        let mapping = InnovationMapping::for_mode(submode);
        let (codebook, count, stages) = match mapping {
            InnovationMapping::Silence => (None, 0u8, 0u8),
            InnovationMapping::Documented { codebook, count } => (Some(codebook), count, 1),
            InnovationMapping::DocumentedTwoStage { codebook, count } => (Some(codebook), count, 2),
            InnovationMapping::Undocumented => {
                return Err(EncodeError::UndocumentedInnovation(submode.mode_id));
            }
        };

        // Frame-level OL pitch fields (modes 1 / 2 / 8), estimated on
        // the whole-frame scratch residual before the real pass.
        let (ol_pitch_index, ol_pitch_gain_index, plan) = self.plan_ol_pitch(&scratch, submode);
        self.res_hist.drain(..NB_FRAME_SAMPLES);
        self.res_hist.extend_from_slice(&scratch);

        let (frame_gain_idx, subframes, _err) = self.refine_frame_gain(
            &input,
            &lpc_sets,
            abs_lpcs.as_ref(),
            frame_gain_est,
            codebook,
            count,
            stages,
            &plan,
            submode,
        );

        // Advance analysis-window look-back and commit envelope state.
        self.prev_input_tail
            .copy_from_slice(&input[NB_FRAME_SAMPLES - ANALYSIS_LOOKBACK..]);
        self.commit_lsp(active_lsp);
        if dec_lsp.is_some() {
            self.prev_lsp_dec = dec_lsp;
        } else {
            self.prev_lsp_dec = None;
        }

        let body = NarrowbandFrameBody {
            lsp_index,
            ol_pitch_index,
            ol_pitch_gain_index,
            ol_exc_gain_index: frame_gain_field(frame_gain_idx),
            subframes,
        };
        // Keep the decoder mirror in step. A frame the searches did not
        // synthesise themselves (the vocoder's generated excitation, a
        // null frame's comfort noise) hands its decoded state back, so
        // the next frame's adaptive codebook and zero-input response are
        // the decoder's.
        if self.mirror.decode_frame(&body, submode).is_ok() && abs_lpcs.is_none() {
            self.exc_hist.clear();
            self.exc_hist
                .extend_from_slice(self.mirror.excitation_tail(EXC_HIST_LEN));
            self.synth_mem = self.mirror.synthesis_state_ahead();
            for (d, &s) in self
                .last_inn
                .iter_mut()
                .zip(self.mirror.last_frame_innovation())
            {
                *d = f64::from(s);
            }
            self.w_fir = [0.0; LPC_ORDER];
            self.w_iir = [0.0; LPC_ORDER];
        }
        Ok(body)
    }

    /// Estimate the frame-level open-loop pitch fields (Table 9.1 `OL
    /// pitch` / `OL pitch gain` rows, modes 1 / 2 / 8) on the scratch
    /// whole-frame residual, and build the [`PitchPlan`] the sub-frame
    /// pass consumes.
    ///
    /// The period is the normalised-correlation open-loop estimate
    /// ([`estimate_open_loop_pitch`]) of the residual against the
    /// excitation history. For the forced-gain modes (1 / 8) the pitch
    /// coefficient at that lag is quantised through the staged
    /// `provenance/02` law: `coef = 0.9 · corr/energy` (the documented
    /// damping), clamped to the `0.99` synthesis bound, encoded as
    /// `15 · coef` on the 4-bit grid.
    fn plan_ol_pitch(
        &self,
        residual: &[f64; NB_FRAME_SAMPLES],
        submode: &NarrowbandSubmode,
    ) -> (u8, u8, PitchPlan) {
        if submode.ol_pitch_bits == 0 {
            return (0, 0, PitchPlan::default());
        }
        // The vocoder's excitation is generated by the decoder, so its
        // pitch is measured on the input's own residual; the other
        // modes predict from the decoded excitation.
        let hist = if submode.mode_id == 1 {
            &self.res_hist
        } else {
            &self.exc_hist
        };
        let ol = crate::ol_pitch::estimate_open_loop_pitch_range(
            hist,
            residual,
            SHORTEST_SEARCHED_LAG,
            PITCH_PERIOD_MAX,
        );
        let (gain_quant, forced_taps) = if submode.ol_pitch_gain_bits > 0 {
            let q = forced_gain_quant(hist, residual, ol.period);
            (q, Some(crate::forced_pitch_gain::forced_pitch_gain_taps(q)))
        } else {
            (0, None)
        };
        (
            ol.wire_index(),
            gain_quant,
            PitchPlan {
                ol_period: Some(ol.period),
                forced_taps,
                forced_q: (submode.ol_pitch_gain_bits > 0).then_some(gain_quant),
            },
        )
    }

    /// Encode the four sub-frames at a fixed reconstructed frame gain
    /// `g_frame`, advancing the analysis-filter + excitation state, and
    /// return the quantised indices plus the total **decoded-excitation
    /// error** `Σ_sf Σ_n (r[n] − ê[n])²` (the residual each sub-frame's
    /// reconstructed excitation `ê = p + g·c` fails to match).
    #[allow(clippy::too_many_arguments)]
    fn encode_subframes(
        &mut self,
        input: &[f64],
        lpc_sets: &[[f64; LPC_ORDER]; SUBFRAMES],
        dec_lpcs: Option<&[[f64; LPC_ORDER]; SUBFRAMES]>,
        g_frame: f64,
        codebook: Option<InnovationCodebook>,
        count: u8,
        stages: u8,
        plan: &PitchPlan,
        submode: &NarrowbandSubmode,
        scratch: &mut FrameScratch,
    ) -> ([NarrowbandSubFrameIndices; SUBFRAMES], f64) {
        let pitch_quant = submode.pitch_gain;
        let has_fine_pitch = submode.fine_pitch_bits > 0;
        let mut subframes = [NarrowbandSubFrameIndices::default(); SUBFRAMES];
        let mut err = 0.0_f64;

        for (sf, slot) in subframes.iter_mut().enumerate() {
            let block = &input[sf * SUBFRAME_SAMPLES..(sf + 1) * SUBFRAME_SAMPLES];
            let lpc = &lpc_sets[sf];

            // Residual r[n] = A(z)·input (analysis filter, continuous state).
            let residual = self.analysis_residual(block, lpc);

            // Synthesis-domain (analysis-by-synthesis) search through the
            // decoder's own LPC and synthesis state whenever the frame is
            // a CELP mode with a transmitted envelope (r483); the
            // excitation-domain match otherwise.
            if let Some(dl) = dec_lpcs {
                let a = &dl[sf];
                let (fields, exc, sf_err) = self.abs_subframe(
                    block, a, g_frame, codebook, count, stages, plan, submode, scratch, sf,
                );
                let pitch = fields.3;
                let base = sf * SUBFRAME_SAMPLES;
                for n in 0..SUBFRAME_SAMPLES {
                    self.last_inn[base + n] = exc[n] - pitch[n];
                }
                err += sf_err;
                self.push_excitation(&exc);
                slot.pitch_index = if has_fine_pitch {
                    (fields.0 - PITCH_PERIOD_MIN) as u8
                } else {
                    0
                };
                slot.pitch_gain_index = fields.1;
                slot.innovation_gain_index = fields.2.0;
                slot.innovation_vq_index = fields.2.1;
                continue;
            }

            // Pitch, dispatched exactly along the decoder's Table 9.1
            // rows so every transmitted field is the one the decoder
            // reads back:
            // * fine-pitch modes (3..=7): full per-sub-frame lag + VQ
            //   gain search;
            // * OL-period VQ mode (2): the wire carries no per-sub-frame
            //   lag, so the 3-tap gain VQ is searched at the frame's OL
            //   period;
            // * forced-gain modes (1 / 8): the frame-level forced taps
            //   at the frame's OL period, no per-sub-frame search;
            // * silence (0): no pitch contribution.
            let (period, pitch_gain_idx, taps) = match pitch_quant {
                PitchGainQuant::None => {
                    let period = plan.ol_period.unwrap_or(PITCH_PERIOD_MIN);
                    let taps = plan
                        .forced_taps
                        .unwrap_or(pitch_gain::PitchGainTaps::SILENCE);
                    (period, 0u8, taps)
                }
                _ if has_fine_pitch => self.search_pitch(&residual, pitch_quant),
                _ => {
                    let period = plan.ol_period.unwrap_or(PITCH_PERIOD_MIN);
                    let (idx, taps) = self.search_pitch_gains_at(&residual, pitch_quant, period);
                    (period, idx, taps)
                }
            };
            let pitch = self.pitch_contribution(period, &taps);

            // Innovation on the pitch-removed residual.
            let mut r2 = [0.0_f64; SUBFRAME_SAMPLES];
            for n in 0..SUBFRAME_SAMPLES {
                r2[n] = residual[n] - pitch[n];
            }

            let (innovation_gain_idx, innovation_vq, exc) = match codebook {
                Some(cb) => {
                    self.encode_innovation(&r2, &pitch, g_frame, cb, count, stages, submode)
                }
                None => (0u8, 0u128, pitch),
            };
            // Innovation-only part (r450 mode-1 fold source mirror).
            {
                let base = sf * SUBFRAME_SAMPLES;
                for n in 0..SUBFRAME_SAMPLES {
                    self.last_inn[base + n] = exc[n] - pitch[n];
                }
            }

            // Decoded-excitation error for this sub-frame.
            for n in 0..SUBFRAME_SAMPLES {
                let d = residual[n] - exc[n];
                err += d * d;
            }

            // Push the reconstructed excitation into history and advance
            // the decoder mirror's synthesis state.
            self.push_excitation(&exc);
            let mut a = [0.0f64; LPC_ORDER];
            for (d, &c) in a.iter_mut().zip(lpc.iter()) {
                *d = -c;
            }
            for &v in &exc {
                crate::narrowband_decoder::synth_step(&mut self.synth_mem, &a, v);
            }

            slot.pitch_index = if has_fine_pitch {
                (period - PITCH_PERIOD_MIN) as u8
            } else {
                0
            };
            slot.pitch_gain_index = pitch_gain_idx;
            slot.innovation_gain_index = innovation_gain_idx;
            slot.innovation_vq_index = innovation_vq;
        }

        (subframes, err)
    }

    /// **Adaptive (closed-loop) frame-gain refinement** (round r389):
    /// evaluate the magnitude estimate's quantised neighbourhood
    /// (`{est−1, est, est+1}` on the staged 32-level `ol_gain` grid) by
    /// running the full sub-frame encode at each candidate's
    /// *reconstructed* gain and keeping the one whose decoded
    /// excitation matches the residual best.
    ///
    /// The open-loop estimate maps residual RMS onto the `exp(qe/3.5)`
    /// grid by magnitude alone (the reference's exact normalisation is
    /// the documented gain-Q-format gap); because the per-sub-frame
    /// innovation-gain *correction* is only 1 or 3 bits wide, a
    /// one-level frame-gain misestimate is often unrecoverable
    /// downstream. Trying the neighbourhood closed-loop is pure
    /// encoder-side search freedom — the decode law is untouched — and
    /// is never worse than the single-pass estimate (the estimate is
    /// one of the candidates; pinned by the module tests). Each trial
    /// runs on a scratch clone; the winner's advanced state is
    /// committed to `self`.
    #[allow(clippy::too_many_arguments)]
    fn refine_frame_gain(
        &mut self,
        input: &[f64],
        lpc_sets: &[[f64; LPC_ORDER]; SUBFRAMES],
        dec_lpcs: Option<&[[f64; LPC_ORDER]; SUBFRAMES]>,
        estimate: crate::fixed_codebook_gain::FrameInnovationGainIndex,
        codebook: Option<InnovationCodebook>,
        count: u8,
        stages: u8,
        plan: &PitchPlan,
        submode: &NarrowbandSubmode,
    ) -> (
        crate::fixed_codebook_gain::FrameInnovationGainIndex,
        [NarrowbandSubFrameIndices; SUBFRAMES],
        f64,
    ) {
        use crate::fixed_codebook_gain::FrameInnovationGainIndex as Idx;

        let mut scratch = FrameScratch::default();
        // Where to centre the candidates. The open-loop estimate reads
        // the frame gain off the residual level, but what the frame
        // gain has to do is put every sub-frame's innovation gain
        // within reach of the sub-frame correction grid. So a probe
        // pass codes the frame with unquantised innovation gains, and
        // the frame level whose grid then quantises those gains with
        // the least (shape-energy-weighted) error becomes the centre.
        let mut centre = estimate;
        if let (Some(_), Some(_), Idx::Indexed(_), true) =
            (dec_lpcs, codebook, estimate, self.effort.gain_probe)
        {
            scratch.probe = true;
            scratch.ideal = Some([(0.0, 0.0); SUBFRAMES]);
            let mut trial = self.clone();
            trial.encode_subframes(
                input,
                lpc_sets,
                dec_lpcs,
                f64::from(reconstruct_frame_ol_exc_gain(estimate)),
                codebook,
                count,
                stages,
                plan,
                submode,
                &mut scratch,
            );
            scratch.probe = false;
            let levels = correction_levels(submode.innovation_gain_bits);
            let ideal = scratch.ideal.unwrap_or([(0.0, 0.0); SUBFRAMES]);
            let mut best = (f64::INFINITY, 0u8);
            for i in 0..=31u8 {
                let g = f64::from(reconstruct_frame_ol_exc_gain(Idx::Indexed(i)));
                let cost: f64 = ideal
                    .iter()
                    .map(|&(alpha, weight)| {
                        weight
                            * levels
                                .iter()
                                .map(|&c| (alpha - g * c).powi(2))
                                .fold(f64::INFINITY, f64::min)
                    })
                    .sum();
                if cost < best.0 {
                    best = (cost, i);
                }
            }
            centre = Idx::Indexed(best.1);
        }

        // Candidate set: the centre plus its quantiser neighbours
        // (clamped to the 5-bit grid). Silence (mode 0 / zero residual)
        // has no meaningful neighbourhood.
        let span = i32::from(if dec_lpcs.is_some() {
            self.effort.gain_span
        } else {
            1
        });
        let mut candidates: Vec<Idx> = Vec::with_capacity(6);
        match centre {
            Idx::Silence => candidates.push(Idx::Silence),
            Idx::Indexed(i) => {
                for d in -span..=span {
                    let c = i32::from(i) + d;
                    if (0..=31).contains(&c) {
                        candidates.push(Idx::Indexed(c as u8));
                    }
                }
            }
        }
        // The probe's least-squares gains shrink where the codebook
        // matches the target poorly, so the open-loop estimate stays in
        // the running.
        if !candidates.contains(&estimate) {
            candidates.push(estimate);
        }

        let mut best: Option<(f64, Idx, [NarrowbandSubFrameIndices; SUBFRAMES], Self)> = None;
        for cand in candidates {
            let g = f64::from(reconstruct_frame_ol_exc_gain(cand));
            let mut trial = self.clone();
            let (subframes, err) = trial.encode_subframes(
                input,
                lpc_sets,
                dec_lpcs,
                g,
                codebook,
                count,
                stages,
                plan,
                submode,
                &mut scratch,
            );
            if best.as_ref().is_none_or(|(e, _, _, _)| err < *e) {
                best = Some((err, cand, subframes, trial));
            }
        }
        let (err, idx, subframes, winner) =
            best.expect("candidate set is non-empty by construction");
        *self = winner;
        (idx, subframes, err)
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

        // Spectral sensitivity is highest where two LSPs sit close (a
        // formant), so each coefficient's error is weighted by its
        // closeness to its nearer neighbour.
        let weights = lsp_error_weights(&absolute);
        let Some(stages) = crate::lsp_quant::quantise_lsp_q10_search(
            &delta,
            regime,
            usize::from(self.effort.lsp_paths),
            &weights,
        ) else {
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
        residual: &[f64; NB_FRAME_SAMPLES],
        submode: &NarrowbandSubmode,
    ) -> f64 {
        let energy: f64 = residual.iter().map(|&r| r * r).sum();
        let rms = (energy / NB_FRAME_SAMPLES as f64).sqrt();
        // The decode law scales codebook rows by INNOVATION_CODEBOOK_SCALE
        // (Q5 signed-char rows, see `gain_scaled_innovation`), so the
        // transmitted-gain domain divides the row RMS by the same factor.
        // No innovation codebook (mode 1, the vocoder): the decoder's
        // innovation is unit-variance noise times the frame gain
        // (`tracer-2026-09.md` §4), so the gain targets the residual RMS
        // directly.
        let Some(cb) = InnovationMapping::for_mode(submode).documented_codebook() else {
            return rms;
        };
        let cb_rms =
            codebook_rms(cb) * f64::from(crate::gain_scaled_innovation::INNOVATION_CODEBOOK_SCALE);
        if cb_rms > 0.0 { rms / cb_rms } else { rms }
    }

    /// Whole-frame residual through a **scratch** analysis filter (the
    /// real per-sub-frame pass keeps its continuous state untouched) —
    /// shared by the frame-gain estimate and the open-loop pitch plan.
    fn scratch_frame_residual(
        &self,
        input: &[f64],
        lpc_sets: &[[f64; LPC_ORDER]; SUBFRAMES],
    ) -> [f64; NB_FRAME_SAMPLES] {
        let mut hist = self.analysis_hist;
        let mut out = [0.0_f64; NB_FRAME_SAMPLES];
        for (sf, lpc) in lpc_sets.iter().enumerate() {
            let block = &input[sf * SUBFRAME_SAMPLES..(sf + 1) * SUBFRAME_SAMPLES];
            for (n, &x) in block.iter().enumerate() {
                let mut r = x;
                for (i, &c) in lpc.iter().enumerate() {
                    r -= c * hist[LPC_ORDER - 1 - i];
                }
                hist.rotate_left(1);
                hist[LPC_ORDER - 1] = x;
                out[sf * SUBFRAME_SAMPLES + n] = r;
            }
        }
        out
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

    /// Gain-only adaptive-codebook search at a **fixed** pitch period —
    /// for modes whose wire format carries a frame-level OL period and
    /// per-sub-frame 3-tap gain VQ but no per-sub-frame lag (mode 2).
    fn search_pitch_gains_at(
        &self,
        residual: &[f64; SUBFRAME_SAMPLES],
        quant: PitchGainQuant,
        period: u16,
    ) -> (u8, pitch_gain::PitchGainTaps) {
        let n_entries: u16 = match quant {
            PitchGainQuant::Vq5Bit => 32,
            PitchGainQuant::Vq7Bit => 128,
            PitchGainQuant::None => return (0, pitch_gain::PitchGainTaps::SILENCE),
        };
        let basis = self.pitch_basis(period);
        let mut best = (0u8, pitch_gain::PitchGainTaps::SILENCE, f64::INFINITY);
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
            if err < best.2 {
                best = (idx as u8, taps, err);
            }
        }
        (best.0, best.1)
    }

    /// One sub-frame by **analysis-by-synthesis** against the decoder's
    /// own synthesis: the target is the input minus the decoder mirror's
    /// zero-input response (through the noise-weighting filter when one
    /// is configured), every candidate excitation is scored through the
    /// sub-frame's impulse response, and the chosen excitation advances
    /// the mirror. Returns `((period, pitch-gain index, (gain field,
    /// packed VQ), pitch contribution), excitation, search-domain
    /// squared error)`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn abs_subframe(
        &mut self,
        block: &[f64],
        a: &[f64; LPC_ORDER],
        g_frame: f64,
        codebook: Option<InnovationCodebook>,
        count: u8,
        stages: u8,
        plan: &PitchPlan,
        submode: &NarrowbandSubmode,
        scratch: &mut FrameScratch,
        sf: usize,
    ) -> (
        (u16, u8, (u8, u128), [f64; SUBFRAME_SAMPLES]),
        [f64; SUBFRAME_SAMPLES],
        f64,
    ) {
        use crate::narrowband_decoder::synth_step;
        const N: usize = SUBFRAME_SAMPLES;
        let effort = self.effort;
        let weights = self.weighting.map(|(g1, g2)| weighting_taps(a, g1, g2));
        let FrameScratch {
            tables,
            first_pitch,
            scaled: fg,
            energy,
            centre,
            probe,
            ideal,
        } = scratch;
        let tabs =
            tables[sf].get_or_insert_with(|| SubframeTables::build(a, weights.as_ref(), codebook));
        let h = tabs.h;
        // Target: the input less the mirror's zero-input response, in the
        // search domain.
        let mut target = [0.0f64; N];
        {
            let mut m = self.synth_mem;
            for (n, t) in target.iter_mut().enumerate() {
                *t = block[n] - synth_step(&mut m, a, 0.0);
            }
            if let Some(w) = &weights {
                let (mut fir, mut iir) = (self.w_fir, self.w_iir);
                for t in target.iter_mut() {
                    *t = weighting_step(w, &mut fir, &mut iir, *t);
                }
            }
        }
        let filt = |x: &[f64; N]| -> [f64; N] {
            let mut y = [0.0f64; N];
            for (n, yv) in y.iter_mut().enumerate() {
                let mut acc = 0.0;
                for k in 0..=n {
                    acc += h[k] * x[n - k];
                }
                *yv = acc;
            }
            y
        };
        let dot = |x: &[f64; N], y: &[f64; N]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();

        // --- Adaptive codebook: the candidates to carry forward. ---
        let has_fine = submode.fine_pitch_bits > 0;
        let ol_period = plan.ol_period.unwrap_or(PITCH_PERIOD_MIN);
        // The first sub-frame's search does not depend on the frame
        // gain, so the frame's gain candidates share it.
        let cands: Vec<PitchCandidate> = if let (0, Some(shared)) = (sf, first_pitch.as_ref()) {
            shared.clone()
        } else {
            let mut cands: Vec<PitchCandidate> = Vec::new();
            match submode.pitch_gain {
                PitchGainQuant::None => {
                    let mut pitch = [0.0f64; N];
                    if let Some(q) = plan.forced_q {
                        // The decoder's forced path: a centre-tap recursion
                        // over the pitch partial at the float coefficient.
                        let coef = f64::from(crate::forced_pitch_gain::forced_pitch_coef(q));
                        let hlen = self.exc_hist.len() as i32;
                        let t = i32::from(ol_period);
                        for n in 0..N as i32 {
                            let k = n - t;
                            pitch[n as usize] = coef
                                * if k >= 0 {
                                    pitch[k as usize]
                                } else if hlen + k >= 0 {
                                    self.exc_hist[(hlen + k) as usize]
                                } else {
                                    0.0
                                };
                        }
                    }
                    cands.push((0.0, ol_period, 0, pitch));
                }
                quant => {
                    let n_entries: u16 = if quant == PitchGainQuant::Vq7Bit {
                        128
                    } else {
                        32
                    };
                    // Centre-tap synthesis responses for every lag whose three
                    // taps read the history without folding (T ≥ 41): the
                    // response at lag T+1 is the response at T delayed by one
                    // sample plus the newly exposed history sample's.
                    const FIRST: usize = SUBFRAME_SAMPLES;
                    const LAST: usize = PITCH_PERIOD_MAX as usize + 1;
                    centre.clear();
                    if has_fine {
                        let hlen = self.exc_hist.len();
                        let mut y = filt(&self.pitch_tap(FIRST as u16, 0));
                        centre.reserve(LAST - FIRST + 1);
                        centre.push(y);
                        for t in FIRST..LAST {
                            let x = self.exc_hist[hlen - t - 1];
                            for n in (1..N).rev() {
                                y[n] = y[n - 1] + h[n] * x;
                            }
                            y[0] = h[0] * x;
                            centre.push(y);
                        }
                    }
                    // Shorter lags read the last T history samples twice
                    // (once folded back by a period) and nothing after:
                    // their centre response is the response `z` of those
                    // T samples alone plus itself delayed by T, and `z`
                    // obeys the same one-sample recursion from T = 0.
                    let mut short = [[0.0f64; N]; FIRST];
                    if has_fine {
                        let hlen = self.exc_hist.len();
                        let mut z = [0.0f64; N];
                        for t in 0..FIRST - 1 {
                            let x = self.exc_hist[hlen - t - 1];
                            for n in (1..N).rev() {
                                z[n] = z[n - 1] + h[n] * x;
                            }
                            z[0] = h[0] * x;
                            // `z` is now the response at lag t + 1.
                            let lag = t + 1;
                            let folded = &mut short[lag];
                            *folded = z;
                            for n in lag..N {
                                folded[n] += z[n - lag];
                            }
                        }
                    }
                    let taps_of = |t: u16| -> [[f64; N]; 3] {
                        let ti = usize::from(t);
                        if has_fine && ti > FIRST && ti < LAST {
                            // Tap j reads the history at lag T − offset_j.
                            [
                                centre[ti + 1 - FIRST],
                                centre[ti - FIRST],
                                centre[ti - 1 - FIRST],
                            ]
                        } else {
                            let b = self.pitch_basis(t);
                            [filt(&b[0]), filt(&b[1]), filt(&b[2])]
                        }
                    };
                    let lags: Vec<u16> = if has_fine {
                        // Pre-select by the centre tap's normalised
                        // synthesis-domain correlation, then search the
                        // gain VQ exhaustively on the best few.
                        let mut scored: Vec<(f64, u16)> = (SHORTEST_SEARCHED_LAG
                            ..=PITCH_PERIOD_MAX)
                            .map(|t| {
                                let ti = usize::from(t);
                                let y = if ti >= FIRST {
                                    &centre[ti - FIRST]
                                } else {
                                    &short[ti]
                                };
                                let c = dot(&target, y);
                                let e = dot(y, y).max(1e-9);
                                (if c > 0.0 { c * c / e } else { 0.0 }, t)
                            })
                            .collect();
                        scored.sort_by(|x, y| {
                            y.0.partial_cmp(&x.0).unwrap_or(core::cmp::Ordering::Equal)
                        });
                        let mut l: Vec<u16> = scored
                            .iter()
                            .take(usize::from(effort.lags.max(1)))
                            .map(|&(_, t)| t)
                            .collect();
                        l.sort_unstable();
                        l.dedup();
                        l
                    } else {
                        vec![ol_period]
                    };
                    let keep = usize::from(effort.pitch_candidates.max(1));
                    // (error, period, gain index), best first.
                    let mut best: Vec<(f64, u16, u8)> = Vec::with_capacity(keep + 1);
                    for t in lags {
                        let y = taps_of(t);
                        let c = [
                            dot(&target, &y[0]),
                            dot(&target, &y[1]),
                            dot(&target, &y[2]),
                        ];
                        let mut r = [[0.0f64; 3]; 3];
                        for i in 0..3 {
                            for j in i..3 {
                                r[i][j] = dot(&y[i], &y[j]);
                                r[j][i] = r[i][j];
                            }
                        }
                        for idx in 0..n_entries {
                            let Some(taps) = pitch_gain::reconstruct(idx as u8, quant) else {
                                continue;
                            };
                            let g = [
                                f64::from(taps.taps[0]) / 64.0,
                                f64::from(taps.taps[1]) / 64.0,
                                f64::from(taps.taps[2]) / 64.0,
                            ];
                            let mut e = 0.0;
                            for i in 0..3 {
                                e -= 2.0 * g[i] * c[i];
                                for j in 0..3 {
                                    e += g[i] * g[j] * r[i][j];
                                }
                            }
                            if best.len() < keep || e < best[best.len() - 1].0 {
                                let at = best.partition_point(|b| b.0 <= e);
                                best.insert(at, (e, t, idx as u8));
                                best.truncate(keep);
                            }
                        }
                    }
                    for (e, t, idx) in best {
                        let b = self.pitch_basis(t);
                        let g = pitch_gain::reconstruct(idx, quant)
                            .map(|taps| {
                                [
                                    f64::from(taps.taps[0]) / 64.0,
                                    f64::from(taps.taps[1]) / 64.0,
                                    f64::from(taps.taps[2]) / 64.0,
                                ]
                            })
                            .unwrap_or([0.0; 3]);
                        let mut pitch = [0.0f64; N];
                        for n in 0..N {
                            pitch[n] = g[0] * b[0][n] + g[1] * b[1][n] + g[2] * b[2][n];
                        }
                        cands.push((e, t, idx, pitch));
                    }
                    if cands.is_empty() {
                        cands.push((0.0, ol_period, 0, [0.0; N]));
                    }
                }
            }
            if sf == 0 {
                *first_pitch = Some(cands.clone());
            }
            cands
        };

        // --- Fixed codebook, once per carried adaptive candidate. ---
        let scale = f64::from(crate::gain_scaled_innovation::INNOVATION_CODEBOOK_SCALE);
        let w2 = f64::from(crate::innovation::NB_MODE7_STAGE2_WEIGHT);
        let survivors = usize::from(effort.innovation_paths.max(1));
        let mut chosen: Option<(f64, u16, u8, (u8, u128), [f64; N], [f64; N])> = None;
        let mut chosen_probe = (0.0f64, 0.0f64);
        for (_, period, pg_idx, pitch) in cands {
            let yp = filt(&pitch);
            let mut t2 = [0.0f64; N];
            for n in 0..N {
                t2[n] = target[n] - yp[n];
            }
            let mut fields = (0u8, 0u128);
            let mut innov = [0.0f64; N];
            let mut resid = dot(&t2, &t2);
            let mut probed = (0.0f64, 0.0f64);
            if let Some(cb) = codebook {
                let sv_len = cb.sub_vector_len();
                let bits = u32::from(cb.index_bits());
                let entries = cb.entries() as usize;
                // One pass of the sub-vector search at frame-level gain
                // `g`: `survivors` best partial paths are kept after
                // every sub-vector (one = the greedy search); stage 2 of
                // a two-stage mode continues the same paths on the
                // remainder at its weight. Returns (packed indices,
                // decode-law shape).
                fg.clear();
                fg.resize(entries * N, 0.0);
                energy.clear();
                energy.resize(entries * (N + 1), 0.0);
                let mut run = |g: f64| -> (u128, [f64; N]) {
                    // (metric, remainder, packed, shape)
                    let mut paths: Vec<(f64, [f64; N], u128, [f64; N])> =
                        vec![(0.0, t2, 0u128, [0.0f64; N])];
                    for stage in 0..stages {
                        let (gs, ws) = if stage == 0 { (g, 1.0) } else { (g * w2, w2) };
                        for (dst, &f) in fg.iter_mut().zip(&tabs.fs) {
                            *dst = f * gs;
                        }
                        for (row, pe) in fg
                            .as_chunks::<N>()
                            .0
                            .iter()
                            .zip(energy.as_chunks_mut::<{ N + 1 }>().0.iter_mut())
                        {
                            let mut e = 0.0f64;
                            for (m, &f) in row.iter().enumerate() {
                                e += f * f;
                                pe[m + 1] = e;
                            }
                        }
                        for sv in 0..usize::from(count) {
                            let o = sv * sv_len;
                            // (metric, parent path, index), best first.
                            let mut next: Vec<(f64, usize, u32)> =
                                Vec::with_capacity(survivors + 1);
                            for (pi, path) in paths.iter().enumerate() {
                                for (idx, (row, pe)) in fg
                                    .as_chunks::<N>()
                                    .0
                                    .iter()
                                    .zip(energy.as_chunks::<{ N + 1 }>().0.iter())
                                    .enumerate()
                                {
                                    let mut c = 0.0f64;
                                    for (&tv, &f) in path.1[o..].iter().zip(row) {
                                        c += tv * f;
                                    }
                                    let m = path.0 + pe[N - o] - 2.0 * c;
                                    if next.len() < survivors || m < next[next.len() - 1].0 {
                                        let at = next.partition_point(|b| b.0 <= m);
                                        next.insert(at, (m, pi, idx as u32));
                                        next.truncate(survivors);
                                    }
                                }
                            }
                            paths = next
                                .into_iter()
                                .map(|(m, pi, idx)| {
                                    let (_, mut t, packed, mut shape) = paths[pi];
                                    let row = &fg[idx as usize * N..(idx as usize + 1) * N];
                                    for (tv, &f) in t[o..].iter_mut().zip(row) {
                                        *tv -= f;
                                    }
                                    if let Some(raw) = sub_vector(cb, idx) {
                                        for (k, &rv) in raw.iter().enumerate() {
                                            shape[o + k] += ws * f64::from(rv) * scale;
                                        }
                                    }
                                    (m, t, (packed << bits) | u128::from(idx), shape)
                                })
                                .collect();
                        }
                    }
                    let (_, _, packed, shape) = paths[0];
                    (packed, shape)
                };
                // The gain correction the shape wants (search-domain
                // least squares), quantised on the field's grid.
                let quantise = |shape: &[f64; N]| {
                    let y = filt(shape);
                    let alpha = (dot(&t2, &y) / dot(&y, &y).max(1e-9)).max(0.0);
                    let target = if g_frame > 0.0 {
                        alpha / g_frame
                    } else {
                        alpha
                    };
                    quantise_subframe_gain_correction(target as f32, submode.innovation_gain_bits)
                        .unwrap_or(
                            crate::fixed_codebook_gain::SubFrameInnovationGainCorrection::Absent,
                        )
                };
                let score = |s: &[f64; N], g: f64| {
                    let y = filt(s);
                    t2.iter()
                        .zip(&y)
                        .map(|(t, v)| (t - g * v).powi(2))
                        .sum::<f64>()
                };
                // The least-squares gain of a shape and the energy of
                // its synthesis response.
                let fit = |shape: &[f64; N]| -> (f64, f64) {
                    let y = filt(shape);
                    let yy = dot(&y, &y);
                    ((dot(&t2, &y) / yy.max(1e-9)).max(0.0), yy)
                };
                let (packed, shape, corr, g, e) = if *probe {
                    // Probe: the shape for the frame gain, then the
                    // shape for the gain that one asked for; both at
                    // their unquantised least-squares gains.
                    let (p1, s1) = run(g_frame.max(1e-6));
                    let (a1, yy1) = fit(&s1);
                    let e1 = score(&s1, a1);
                    let (p2, s2) = run(a1.max(1e-6));
                    let (a2, yy2) = fit(&s2);
                    let e2 = score(&s2, a2);
                    let absent =
                        crate::fixed_codebook_gain::SubFrameInnovationGainCorrection::Absent;
                    if e2 <= e1 {
                        probed = (a2, yy2);
                        (p2, s2, absent, a2, e2)
                    } else {
                        probed = (a1, yy1);
                        (p1, s1, absent, a1, e1)
                    }
                } else {
                    // Pass 1 at the frame gain times the correction the
                    // probe's gain maps to (the bare frame gain without
                    // a probe), pass 2 at the quantised gain the first
                    // shape asked for; keep the better of the two. (A
                    // second pass at an unchanged gain repeats the first.)
                    let g0 = match ideal.as_ref() {
                        Some(hint) if g_frame > 0.0 => {
                            let c = quantise_subframe_gain_correction(
                                (hint[sf].0 / g_frame) as f32,
                                submode.innovation_gain_bits,
                            )
                            .unwrap_or(
                                crate::fixed_codebook_gain::SubFrameInnovationGainCorrection::Absent,
                            );
                            g_frame * f64::from(reconstruct_subframe_gain_correction(c))
                        }
                        _ => g_frame,
                    };
                    let (p1, s1) = run(g0.max(1e-6));
                    let c1 = quantise(&s1);
                    let g1 = g_frame * f64::from(reconstruct_subframe_gain_correction(c1));
                    let e1 = score(&s1, g1);
                    if effort.gain_passes < 2 || g1.max(1e-6) == g0.max(1e-6) {
                        (p1, s1, c1, g1, e1)
                    } else {
                        let (p2, s2) = run(g1.max(1e-6));
                        let c2 = quantise(&s2);
                        let g2 = g_frame * f64::from(reconstruct_subframe_gain_correction(c2));
                        let e2 = score(&s2, g2);
                        if e2 <= e1 {
                            (p2, s2, c2, g2, e2)
                        } else {
                            (p1, s1, c1, g1, e1)
                        }
                    }
                };
                fields = (correction_field(corr), packed);
                for n in 0..N {
                    innov[n] = g * shape[n];
                }
                resid = e;
            }
            if chosen.as_ref().is_none_or(|c| resid < c.0) {
                chosen = Some((resid, period, pg_idx, fields, pitch, innov));
                chosen_probe = probed;
            }
        }
        if let (true, Some(ideal)) = (*probe, ideal.as_mut()) {
            ideal[sf] = chosen_probe;
        }
        let (_, period, pg_idx, fields, pitch, innov) =
            chosen.expect("at least one adaptive candidate");

        // Reconstruct, advance the mirror, score.
        let mut exc = [0.0f64; N];
        let mut err = 0.0f64;
        for n in 0..N {
            exc[n] = pitch[n] + innov[n];
            let y = synth_step(&mut self.synth_mem, a, exc[n]);
            let d = block[n] - y;
            let d = match &weights {
                Some(w) => weighting_step(w, &mut self.w_fir, &mut self.w_iir, d),
                None => d,
            };
            err += d * d;
        }
        ((period, pg_idx, fields, pitch), exc, err)
    }

    /// Three adaptive-codebook basis vectors (offsets −T−1, −T, −T+1) for
    /// pitch period `t`.
    fn pitch_basis(&self, t: u16) -> [[f64; SUBFRAME_SAMPLES]; 3] {
        [
            self.pitch_tap(t, TAP_PITCH_OFFSETS[0]),
            self.pitch_tap(t, TAP_PITCH_OFFSETS[1]),
            self.pitch_tap(t, TAP_PITCH_OFFSETS[2]),
        ]
    }

    /// One adaptive-codebook basis vector: the excitation history read
    /// at lag `t` shifted by the tap offset `off`.
    fn pitch_tap(&self, t: u16, off: i32) -> [f64; SUBFRAME_SAMPLES] {
        let mut row = [0.0_f64; SUBFRAME_SAMPLES];
        let hlen = self.exc_hist.len();
        for (n, slot) in row.iter_mut().enumerate() {
            // The decoder folds a lag reaching into the sub-frame back
            // by one period once, and reads nothing for a lag still
            // inside it (T ≤ 19 tail; nb-pitch-short-lag.md).
            let mut k = n as i32 - i32::from(t) + off;
            if k >= 0 {
                k -= i32::from(t);
            }
            let idx = hlen as i32 + k;
            if k < 0 && idx >= 0 {
                *slot = self.exc_hist[idx as usize];
            }
        }
        row
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
    ///
    /// `stages` is `1` for the single-stage modes and `2` for mode 7's
    /// two-pass quantisation (binding doc §3): the second stage searches
    /// the residual the first stage leaves, and the decoded innovation
    /// is the sum of both stages' rows — exactly what
    /// [`crate::innovation::decode_subframe`] reconstructs.
    #[allow(clippy::too_many_arguments)]
    fn encode_innovation(
        &self,
        r2: &[f64; SUBFRAME_SAMPLES],
        pitch: &[f64; SUBFRAME_SAMPLES],
        g_frame: f64,
        codebook: InnovationCodebook,
        count: u8,
        stages: u8,
        submode: &NarrowbandSubmode,
    ) -> (u8, u128, [f64; SUBFRAME_SAMPLES]) {
        // Work in the decode-law domain: codebook rows carry the Q5
        // normalisation (`gain_scaled_innovation`), so the effective row
        // is `row · scale` and the transmitted gain is 32× the raw-row
        // least-squares fit.
        let scale = f64::from(crate::gain_scaled_innovation::INNOVATION_CODEBOOK_SCALE);
        let cb_rms = codebook_rms(codebook).max(1e-9) * scale;
        let r2_rms = (r2.iter().map(|&x| x * x).sum::<f64>() / SUBFRAME_SAMPLES as f64).sqrt();
        let g_guess = if r2_rms > 0.0 {
            r2_rms / cb_rms
        } else {
            g_frame.max(1e-3)
        };

        // `search_innovation` scores raw rows, so it takes the effective
        // per-row multiplier `g_guess · scale`.
        let choice = search_innovation(r2, g_guess * scale, codebook, count);

        // Reconstruct the concatenated codebook shape in the decode-law
        // domain (rows normalised by `scale`).
        let sv_len = codebook.sub_vector_len();
        let mut cb = [0.0_f64; SUBFRAME_SAMPLES];
        for (sv, &idx) in choice.indices.iter().enumerate() {
            if let Some(row) = sub_vector(codebook, idx) {
                let base = sv * sv_len;
                for (k, &v) in row.iter().enumerate() {
                    cb[base + k] = f64::from(v) * scale;
                }
            }
        }
        let mut packed = choice.packed;

        if stages == 2 {
            // Stage 2 quantises the residual stage 1 leaves at the same
            // working gain; the decode law adds stage 2 at the measured
            // NB_MODE7_STAGE2_WEIGHT (r450), so the search sees that
            // effective scale.
            let w2 = f64::from(crate::innovation::NB_MODE7_STAGE2_WEIGHT);
            let mut r3 = [0.0_f64; SUBFRAME_SAMPLES];
            for n in 0..SUBFRAME_SAMPLES {
                r3[n] = r2[n] - g_guess * cb[n];
            }
            let choice2 = search_innovation(&r3, g_guess * scale * w2, codebook, count);
            for (sv, &idx) in choice2.indices.iter().enumerate() {
                if let Some(row) = sub_vector(codebook, idx) {
                    let base = sv * sv_len;
                    for (k, &v) in row.iter().enumerate() {
                        cb[base + k] += f64::from(v) * scale * w2;
                    }
                }
            }
            let stage_bits = u32::from(codebook.index_bits()) * u32::from(count);
            packed = (packed << stage_bits) | choice2.packed;
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
        (correction_field(correction), packed, exc)
    }

    /// The most recent frame's locally reconstructed excitation
    /// `e = p + g·c` (160 samples, most-recent last) — the encoder-side
    /// analysis-by-synthesis mirror of
    /// [`crate::NarrowbandDecoder::last_frame_excitation`]. The wideband
    /// encoder reads this as the fold source when choosing the
    /// high-band gain-only sub-mode's 5-bit gains (r393). Zero-padded at
    /// stream start.
    pub fn last_frame_excitation(&self) -> [f64; NB_FRAME_SAMPLES] {
        let mut out = [0.0f64; NB_FRAME_SAMPLES];
        let n = self.exc_hist.len().min(NB_FRAME_SAMPLES);
        let src = &self.exc_hist[self.exc_hist.len() - n..];
        out[NB_FRAME_SAMPLES - n..].copy_from_slice(src);
        out
    }

    /// The most recent frame's innovation-only excitation `g·c[n]` —
    /// the encoder-side mirror of
    /// [`crate::NarrowbandDecoder::last_frame_innovation`], the r450
    /// mode-1 fold source.
    pub fn last_frame_innovation(&self) -> &[f64; NB_FRAME_SAMPLES] {
        &self.last_inn
    }

    /// Per-sub-frame `|A_lb(π)|` of the most recent frame's quantised
    /// envelope — mirror of
    /// [`crate::NarrowbandDecoder::last_crossover_response`].
    pub fn last_crossover_response(&self) -> &[f64; 4] {
        &self.last_api
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

/// Per-coefficient LSP error weights (unit mean): the inverse square
/// root of each coefficient's distance to its nearer neighbour (the
/// full inverse over-commits to sharp formants), with the band edges
/// `0` and `π` (Q10) standing in for the missing neighbours.
fn lsp_error_weights(lsp_q10: &[i32; LPC_ORDER]) -> [f64; LPC_ORDER] {
    let top = core::f64::consts::PI * 1024.0;
    let mut w = [0.0f64; LPC_ORDER];
    for i in 0..LPC_ORDER {
        let below = if i == 0 {
            0.0
        } else {
            f64::from(lsp_q10[i - 1])
        };
        let above = if i + 1 == LPC_ORDER {
            top
        } else {
            f64::from(lsp_q10[i + 1])
        };
        let here = f64::from(lsp_q10[i]);
        let gap = (here - below).min(above - here).max(8.0);
        w[i] = 1.0 / gap.sqrt();
    }
    let mean = w.iter().sum::<f64>() / LPC_ORDER as f64;
    w.map(|v| v / mean)
}

/// Quantise the frame's forced (open-loop) pitch coefficient at lag
/// `period` into the 4-bit `OL pitch gain` field (modes 1 / 8).
///
/// The coefficient is the normalised correlation of the frame residual
/// against the `period`-delayed excitation history, damped by the
/// staged `provenance/02` factor `0.9` and clamped to the recorded
/// `0.99` synthesis bound, then encoded through the staged forward law
/// `15 · coef` clamped to `[0, 15]` (the exact inverse of the decoder's
/// `0.066667 · quant`). A non-positive correlation (no pitch match)
/// encodes as `0` — no pitch contribution.
fn forced_gain_quant(hist: &[f64], frame: &[f64], period: u16) -> u8 {
    let hlen = hist.len();
    let t = period as usize;
    let mut corr = 0.0_f64;
    let mut energy = 0.0_f64;
    for (n, &s) in frame.iter().enumerate() {
        let d = if n >= t {
            frame[n - t]
        } else {
            hist[hlen - (t - n)]
        };
        corr += s * d;
        energy += d * d;
    }
    if corr <= 0.0 || energy <= 0.0 {
        return 0;
    }
    let coef = (0.9 * corr / energy).clamp(0.0, 0.99);
    ((15.0 * coef).round() as i32).clamp(0, 15) as u8
}

/// Extract the 5-bit OL exc-gain field value from a quantised index.
fn frame_gain_field(idx: crate::fixed_codebook_gain::FrameInnovationGainIndex) -> u8 {
    match idx {
        crate::fixed_codebook_gain::FrameInnovationGainIndex::Silence => 0,
        crate::fixed_codebook_gain::FrameInnovationGainIndex::Indexed(i) => i,
    }
}

/// The reconstruction levels of the sub-frame innovation-gain correction
/// for a field of `bits` bits (a single unit level when the mode
/// carries no field).
fn correction_levels(bits: u8) -> Vec<f64> {
    use crate::fixed_codebook_gain::SubFrameInnovationGainCorrection as C;
    let level = |c: C| f64::from(reconstruct_subframe_gain_correction(c));
    match bits {
        1 => (0..2).map(|i| level(C::OneBit(i))).collect(),
        3 => (0..8).map(|i| level(C::ThreeBit(i))).collect(),
        _ => vec![level(C::Absent)],
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
    /// The documented codebook shape, if this mapping is documented
    /// (single- or two-stage; mode 7's stages share one codebook).
    fn documented_codebook(self) -> Option<InnovationCodebook> {
        match self {
            InnovationMapping::Documented { codebook, .. }
            | InnovationMapping::DocumentedTwoStage { codebook, .. } => Some(codebook),
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
        for mode in [1u8, 2, 3, 4, 5, 6, 7, 8] {
            let mut enc = NarrowbandEncoder::new();
            let frame = voiced_frame(40, 4000.0);
            let bytes = enc.encode_frame(&frame, mode).expect("encode");
            assert!(!bytes.is_empty(), "mode {mode} produced empty frame");
        }
    }

    #[test]
    fn mode_7_two_stage_frames_decode() {
        // Mode 7 (24.6 kbps): two-stage innovation. A multi-frame
        // packet must decode to finite, non-silent PCM.
        let mut enc = NarrowbandEncoder::new();
        let frames = [voiced_frame(45, 5000.0), voiced_frame(45, 5200.0)];
        let pkt = enc.encode_packet(&frames, 7).expect("mode 7 encodes");
        let mut dec = crate::SpeexDecoder::new();
        let pcm = dec.decode_packet_pcm_i16(&pkt).expect("mode 7 decodes");
        assert_eq!(pcm.len(), 2 * NB_FRAME_SAMPLES);
        assert!(
            pcm.iter().any(|&s| s != 0),
            "mode 7 PCM should be non-silent"
        );
    }

    #[test]
    fn mode_1_vocoder_frames_decode_with_forced_pitch() {
        // Mode 1 (2.15 kbps vocoder): no innovation vector; the audible
        // content is the forced pitch path. Warm the encoder with a
        // frame so the excitation history is live, then check the
        // second frame transmits a non-zero forced gain and that the
        // stream decodes.
        let mut enc = NarrowbandEncoder::new();
        let f0 = voiced_frame(40, 8000.0);
        let _ = enc.encode_frame_body(&f0, 1).expect("frame 0");
        let body = enc.encode_frame_body(&f0, 1).expect("frame 1");
        assert!(
            body.ol_pitch_gain_index > 0,
            "periodic input over a live history should transmit a forced pitch gain"
        );
        // The four inert 1-bit innovation-gain fields are written 0
        // (binding doc §4: the reference encoder writes all of them 0).
        for sf in &body.subframes {
            assert_eq!(sf.innovation_gain_index, 0);
        }
        let mut enc2 = NarrowbandEncoder::new();
        let pkt = enc2
            .encode_packet(&[f0, voiced_frame(40, 8000.0)], 1)
            .expect("mode 1 packet");
        let mut dec = crate::SpeexDecoder::new();
        let pcm = dec.decode_packet_pcm_i16(&pkt).expect("mode 1 decodes");
        assert_eq!(pcm.len(), 2 * NB_FRAME_SAMPLES);
        assert!(
            pcm.iter()
                .all(|&s| (-32768..=32767).contains(&i32::from(s)))
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
    fn frame_gain_refinement_never_worse_than_single_pass() {
        // The closed-loop neighbourhood selection must never decode
        // worse than the single-pass magnitude estimate (the estimate
        // is one of the candidates).
        let submode = NarrowbandSubmode::for_id(5).unwrap();
        let (codebook, count) = match InnovationMapping::for_mode(&submode) {
            InnovationMapping::Documented { codebook, count } => (Some(codebook), count),
            _ => unreachable!("mode 5 innovation is documented"),
        };
        for (amp, period) in [(900.0, 45usize), (4000.0, 60), (11000.0, 80)] {
            let mut enc = NarrowbandEncoder::new();
            // Warm one frame so the probe state is non-trivial.
            let _ = enc
                .encode_frame_body(&voiced_frame(period, amp), 5)
                .unwrap();
            let frame = voiced_frame(period, amp * 1.3);
            let input: Vec<f64> = frame.iter().map(|&s| f64::from(s)).collect();

            // Reproduce the pre-sub-frame pipeline on a probe clone.
            let mut probe = enc.clone();
            let (_, active) = probe.encode_envelope(&input, &submode);
            let prev = probe.prev_lsp_q10.unwrap_or(active);
            let sub_lsp = NbSubFrameLsp::new(&prev, &active);
            let lpc_sets = subframe_lpc_set_with_base(&sub_lsp);
            let est = quantise_frame_ol_exc_gain_exact(
                probe
                    .frame_gain_estimate(&probe.scratch_frame_residual(&input, &lpc_sets), &submode)
                    as f32,
            );

            // Single-pass baseline at the estimate's reconstructed gain.
            let g_est = f64::from(reconstruct_frame_ol_exc_gain(est));
            let plan = PitchPlan::default();
            let mut base = probe.clone();
            let (_, err_base) = base.encode_subframes(
                &input,
                &lpc_sets,
                None,
                g_est,
                codebook,
                count,
                1,
                &plan,
                &submode,
                &mut FrameScratch::default(),
            );

            // Closed-loop refinement.
            let mut refined = probe.clone();
            let (_, _, err_ref) = refined.refine_frame_gain(
                &input, &lpc_sets, None, est, codebook, count, 1, &plan, &submode,
            );
            assert!(
                err_ref <= err_base + 1e-9,
                "amp {amp}: refined {err_ref} worse than single-pass {err_base}"
            );
        }
    }

    #[test]
    fn quality_packets_match_ladder_budgets() {
        // encode_packet_quality wires the Table 9.2 ladder: for EVERY
        // quality 0..=10 (r438 — modes 1 and 7 are now bound), one
        // frame packs to exactly ceil(mode_bits / 8) bytes — the
        // RFC 5574 §3.3 padding adds no octet — and decodes.
        for q in 0..=10u8 {
            let mode = crate::quality::nb_mode_for_quality(q).unwrap();
            let submode = NarrowbandSubmode::for_id(mode).unwrap();
            let mut enc = NarrowbandEncoder::new();
            let frames = [voiced_frame(60, 6000.0)];
            let pkt = enc
                .encode_packet_quality(&frames, q)
                .unwrap_or_else(|e| panic!("quality {q}: {e}"));
            let bits = u32::from(submode.total_bits);
            assert_eq!(pkt.len(), bits.div_ceil(8) as usize, "quality {q}");
            let mut dec = crate::SpeexDecoder::new();
            assert_eq!(dec.decode_packet(&pkt).unwrap().len(), 1, "quality {q}");
        }
        // Out-of-range qualities are rejected.
        let mut enc = NarrowbandEncoder::new();
        let frames = [voiced_frame(60, 6000.0)];
        assert!(
            enc.encode_packet_quality(&frames, 11).is_err(),
            "q11 out of range"
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

    /// A second of frames whose pitch and level move, so searches have
    /// something to find.
    fn moving_frames(n: usize) -> Vec<[i16; NB_FRAME_SAMPLES]> {
        (0..n)
            .map(|k| voiced_frame(48 + 3 * (k % 7), 2500.0 + 600.0 * (k % 5) as f64))
            .collect()
    }

    /// SNR (dB) of encode → decode against the input, through the
    /// packet decoder without its output filters' help: the decode
    /// trails the input by one sub-frame.
    fn round_trip_snr(enc: &mut NarrowbandEncoder, mode: u8) -> f64 {
        let frames = moving_frames(24);
        let submode = NarrowbandSubmode::for_id(mode).unwrap();
        let mut dec = crate::narrowband_decoder::NarrowbandDecoder::new();
        dec.set_enhancer(false);
        dec.set_output_highpass(false);
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for f in &frames {
            let body = enc.encode_frame_body(f, mode).unwrap();
            x.extend(f.iter().map(|&v| f64::from(v)));
            y.extend(dec.decode_frame(&body, &submode).unwrap());
        }
        let (mut s, mut e) = (0.0f64, 0.0f64);
        for i in NB_FRAME_SAMPLES..x.len() - SUBFRAME_SAMPLES {
            s += x[i] * x[i];
            e += (x[i] - y[i + SUBFRAME_SAMPLES]).powi(2);
        }
        10.0 * (s / e).log10()
    }

    #[test]
    fn complexity_deepens_the_search() {
        assert_eq!(NarrowbandEncoder::new().complexity(), DEFAULT_COMPLEXITY);
        let mut snr = Vec::new();
        for c in [1u8, DEFAULT_COMPLEXITY, 8] {
            let mut enc = NarrowbandEncoder::new();
            enc.set_complexity(c);
            assert_eq!(enc.complexity(), c);
            snr.push(round_trip_snr(&mut enc, 5));
        }
        assert!(
            snr[0] < snr[1] && snr[1] < snr[2],
            "SNR by complexity 1 / default / 8: {snr:?}"
        );
        // The effort ladder never shrinks a search with the setting.
        for c in 2..=10u8 {
            let (lo, hi) = (
                SearchEffort::for_complexity(c - 1),
                SearchEffort::for_complexity(c),
            );
            assert!(hi.gain_span >= lo.gain_span && hi.lags >= lo.lags);
            assert!(hi.pitch_candidates >= lo.pitch_candidates);
            assert!(hi.innovation_paths >= lo.innovation_paths);
            assert!(hi.gain_passes >= lo.gain_passes && hi.lsp_paths >= lo.lsp_paths);
        }
        assert_eq!(
            SearchEffort::for_complexity(0),
            SearchEffort::for_complexity(1)
        );
        assert_eq!(
            SearchEffort::for_complexity(99),
            SearchEffort::for_complexity(10)
        );
    }

    #[test]
    fn noise_weighting_is_the_default_search_domain() {
        let mut weighted = NarrowbandEncoder::new();
        assert_eq!(
            weighted.noise_weighting(),
            Some((
                crate::weighting::WEIGHT_GAMMA1,
                crate::weighting::WEIGHT_GAMMA2
            ))
        );
        let weighted_snr = round_trip_snr(&mut weighted, 4);
        let mut plain = NarrowbandEncoder::new();
        plain.set_noise_weighting(None);
        assert_eq!(plain.noise_weighting(), None);
        let plain_snr = round_trip_snr(&mut plain, 4);
        // Two different criteria, two different streams — both track
        // the input.
        assert!((weighted_snr - plain_snr).abs() > 1e-3);
        assert!(
            weighted_snr > 8.0 && plain_snr > 8.0,
            "{weighted_snr} / {plain_snr}"
        );
        // Equal factors make W(z) = 1: the plain search.
        let mut unit = NarrowbandEncoder::new();
        unit.set_noise_weighting(Some((0.8, 0.8)));
        assert!((round_trip_snr(&mut unit, 4) - plain_snr).abs() < 1e-6);
        // Factors are clamped to the stable range.
        unit.set_noise_weighting(Some((1.5, -0.2)));
        assert_eq!(unit.noise_weighting(), Some((1.0, 0.0)));
    }

    #[test]
    fn the_mirror_stays_on_the_decoder_through_vocoder_and_null_frames() {
        // Vocoder and null frames carry excitation the decoder generates
        // itself; the frames after them must still predict from exactly
        // what the decoder holds.
        let frames = moving_frames(12);
        let modes = [5u8, 5, 1, 1, 0, 0, 5, 5, 1, 4, 0, 6];
        let mut enc = NarrowbandEncoder::new();
        let mut dec = crate::narrowband_decoder::NarrowbandDecoder::new();
        dec.set_enhancer(false);
        for (f, &mode) in frames.iter().zip(&modes) {
            let submode = NarrowbandSubmode::for_id(mode).unwrap();
            let body = enc.encode_frame_body(f, mode).unwrap();
            dec.decode_frame(&body, &submode).unwrap();
            let ours = enc.last_frame_excitation();
            let theirs = dec.last_frame_excitation();
            let scale = theirs.iter().fold(1.0f32, |m, v| m.max(v.abs()));
            for (n, (&a, &b)) in ours.iter().zip(theirs.iter()).enumerate() {
                assert!(
                    (a as f32 - b).abs() <= 1e-4 * scale,
                    "mode {mode} sample {n}: encoder {a} vs decoder {b}"
                );
            }
        }
    }
}
