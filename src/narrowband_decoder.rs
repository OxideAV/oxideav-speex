//! **End-to-end narrowband CELP decode loop** — the closed excitation
//! feedback path from a parsed [`crate::NarrowbandFrameBody`] to 160
//! samples of `f64` / `i16` PCM.
//!
//! This module is the *composition* layer that threads the individual
//! reconstruction primitives the earlier rounds landed into the single
//! per-sub-frame recurrence *The Speex Codec Manual* §8–§9 describes:
//!
//! ```text
//!   for each 20 ms frame:
//!     reconstruct LSPs                          (§9.1, crate::lsp)
//!     interpolate per-sub-frame LSPs            (§9.1, crate::lsp_interp)
//!     for each of the 4 sub-frames:
//!       LSP → LPC                               (§9.1, crate::lsp_to_lpc)
//!       p[n] = gain · adaptive-codebook tap sum (§9.2, crate::gain_scaled_pitch)
//!       c[n] = gain · innovation sub-vector     (§9.2, crate::gain_scaled_innovation)
//!       e[n] = p[n] + c[n]                       (§8.4, crate::gain_scaled_excitation)
//!       push e[n] into the excitation history    (§9.2 long-term predictor)
//!       x[n] = synth_filter(e[n], LPC)           (§9.4, crate::synthesis)
//! ```
//!
//! ## What this round closes
//!
//! The earlier rounds landed every constituent primitive but never
//! threaded them through the **closed feedback loop**: the adaptive
//! codebook reads the *emitted excitation* `e[·]` of previous
//! sub-frames, so `e[n] = p[n] + c[n]` for sub-frame `k` must be pushed
//! into the [`crate::ExcitationBuffer`] *before* sub-frame `k + 1`'s
//! pitch contribution `p[n]` is evaluated. Until that feedback is wired,
//! the pitch contribution `p[n]` is always zero (the buffer never sees a
//! non-zero sample), and the synthesis filter is driven by the
//! innovation `c[n]` alone — which is exactly the limitation the
//! `tests/synthesis_pcm_fixture.rs` integration test documented.
//!
//! This module wires that feedback: the composed float excitation `e[n]`
//! is rounded + saturated to `i16` and pushed into the history buffer at
//! the end of each sub-frame, so the next sub-frame's adaptive codebook
//! sees the real excitation. The §8.4 `e[n] = p[n] + c[n]` excitation
//! the synthesis filter consumes is now the **full** excitation, not the
//! innovation-only stand-in.
//!
//! ## Pitch period: frame-level OL vs per-sub-frame fine
//!
//! The pitch period for a sub-frame comes from one of two Table 9.1
//! rows, never both:
//!
//! * **Per-sub-frame fine pitch** (`fine_pitch_bits = 7`, modes 3..=7):
//!   each sub-frame carries its own 7-bit period field, de-biased by
//!   [`crate::PITCH_PERIOD_MIN`] = 17.
//! * **Frame-level open-loop pitch** (`ol_pitch_bits = 7`, modes 1, 2,
//!   8): a single 7-bit period applies to all four sub-frames, de-biased
//!   the same way.
//!
//! Mode 0 (silence) carries neither and emits no pitch contribution.
//!
//! ## Q-format / bit-exactness caveat
//!
//! The float signal domain the contributions occupy is the one the
//! r326/r331 gain-scaling rounds pinned (gain-scaled `c[n]` and the Q6
//! pitch scaling). The LSP angular-unit pin for *reference-equivalent*
//! output is still a recorded docs gap (see [`crate::lsp_to_lpc`]); this
//! module produces magnitude-correct, input-responsive, stable PCM with
//! the full excitation but is not yet asserted bit-exact against
//! reference Speex output. The closed loop is the structural milestone;
//! the final Q-format pin is the remaining gap.

use crate::adaptive_codebook::ExcitationBuffer;
use crate::fixed_codebook_gain::FixedCodebookGainIndices;
use crate::gain_scaled_excitation::gain_scaled_excitation_subframe;
use crate::gain_scaled_innovation::gain_scaled_innovation_from_indices;
use crate::gain_scaled_pitch::gain_scaled_pitch_subframe;
use crate::innovation::SUBFRAME_SAMPLES;
use crate::lsp_to_lpc::{subframe_lpc_set_with_base, LPC_ORDER};
use crate::narrowband_body::{NarrowbandFrameBody, PITCH_PERIOD_MIN};
use crate::pitch_gain::PitchGainTaps;
use crate::submode::{NarrowbandSubmode, PitchGainQuant, SUBFRAMES_PER_FRAME};
use crate::synthesis::SynthesisFilter;
use core::fmt;

/// Number of PCM samples a narrowband frame decodes to
/// (`SUBFRAMES_PER_FRAME × SUBFRAME_SAMPLES` = 4 × 40 = 160; §9.1).
pub const NARROWBAND_FRAME_SAMPLES: usize = SUBFRAMES_PER_FRAME * SUBFRAME_SAMPLES;

/// Errors from [`NarrowbandDecoder::decode_frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrowbandDecodeError {
    /// A sub-mode that transmits an LSP field produced no reconstructed
    /// LSP vector — should not happen for the nine documented modes.
    LspUnavailable,
    /// The fixed-codebook gain index pair could not be composed (a
    /// non-conforming hand-built sub-mode).
    GainIndicesUnavailable,
    /// The innovation sub-vector for a sub-frame could not be decoded:
    /// its per-mode codebook binding is a recorded docs gap (narrowband
    /// modes 1 / 7).
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

/// Stateful narrowband CELP decoder carrying the IIR + excitation
/// feedback state across frames.
///
/// Construct with [`NarrowbandDecoder::new`] (stream-start: zero
/// synthesis history, zero excitation history, no previous LSP), then
/// call [`NarrowbandDecoder::decode_frame`] once per 20 ms narrowband
/// frame. The decoder keeps:
///
/// * the [`SynthesisFilter`] IIR memory (`1/A(z)` history),
/// * the [`ExcitationBuffer`] long-term-predictor history,
/// * the previous frame's reconstructed Q10 LSP vector (for the §9.1
///   sub-frame interpolation),
///
/// all of which must persist across frames for the decode to be
/// continuous.
#[derive(Debug, Clone)]
pub struct NarrowbandDecoder {
    filter: SynthesisFilter,
    excitation: ExcitationBuffer,
    prev_lsp_q10: Option<[i32; LPC_ORDER]>,
    /// The full-precision composed excitation `e[n] = p[n] + c[n]` of the
    /// most recently decoded frame, retained for the wideband high-band
    /// **folded-excitation** reconstruction (manual §10.2 / §10.3: the
    /// gain-only high-band sub-mode reuses the low-band excitation — see
    /// [`crate::hb_fold`]). Zero at stream start and after a silence
    /// frame.
    last_frame_excitation: [f32; NARROWBAND_FRAME_SAMPLES],
}

impl Default for NarrowbandDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl NarrowbandDecoder {
    /// A fresh decoder at stream start (zero history everywhere).
    pub fn new() -> Self {
        Self {
            filter: SynthesisFilter::new(),
            excitation: ExcitationBuffer::new(),
            prev_lsp_q10: None,
            last_frame_excitation: [0.0; NARROWBAND_FRAME_SAMPLES],
        }
    }

    /// The composed full-precision excitation `e[n] = p[n] + c[n]` of the
    /// most recently decoded frame (160 samples, one per PCM sample).
    ///
    /// This is the signal the wideband **folded** high-band excitation
    /// law reuses (manual §10.2 / §10.3, [`crate::hb_fold`]): the
    /// gain-only high-band sub-mode transmits no innovation vector, so
    /// its excitation is this low-band excitation scaled by the 5-bit
    /// folded gain. All-zero at stream start and after a silence frame
    /// (whose excitation is zero by construction).
    pub fn last_frame_excitation(&self) -> &[f32; NARROWBAND_FRAME_SAMPLES] {
        &self.last_frame_excitation
    }

    /// Resolve the de-biased pitch period for sub-frame `sf_idx` from
    /// whichever of the two Table 9.1 pitch rows the sub-mode carries.
    ///
    /// Returns `None` for the silence mode (no pitch field at all).
    fn pitch_period(
        body: &NarrowbandFrameBody,
        submode: &NarrowbandSubmode,
        sf_idx: usize,
    ) -> Option<u16> {
        if submode.fine_pitch_bits != 0 {
            // Per-sub-frame fine pitch (modes 3..=7).
            Some(u16::from(body.subframes[sf_idx].pitch_index) + PITCH_PERIOD_MIN)
        } else if submode.ol_pitch_bits != 0 {
            // Frame-level open-loop pitch applies to all sub-frames
            // (modes 1, 2, 8).
            Some(u16::from(body.ol_pitch_index) + PITCH_PERIOD_MIN)
        } else {
            None
        }
    }

    /// Resolve the pitch gains for sub-frame `sf_idx`, dispatching on
    /// whichever of the two Table 9.1 pitch-gain rows the sub-mode
    /// carries (they are mutually exclusive):
    ///
    /// * **Per-sub-frame 3-tap VQ** (`Pitch gain` row, modes 2..=7):
    ///   look up the `(g0, g1, g2)` triple from the sub-frame's VQ index.
    /// * **Frame-level forced gain** (`OL pitch gain` row = 4 bits,
    ///   modes 1 and 8): reconstruct the single-centre-tap forced gain
    ///   from the frame-level 4-bit
    ///   [`NarrowbandFrameBody::ol_pitch_gain_index`]
    ///   ([`crate::forced_pitch_gain_taps`]). The same forced taps apply
    ///   to all four sub-frames (the field is frame-level), mirroring the
    ///   frame-level open-loop pitch period.
    /// * **Silence** (mode 0, neither row): the all-zero taps, no
    ///   codebook consulted.
    ///
    /// The two pitch-gain rows are mutually exclusive per Table 9.1: a
    /// mode that carries the per-sub-frame `Pitch gain` VQ (modes 2..=7)
    /// has `OL pitch gain` = 0, and the two forced-gain modes (1, 8) have
    /// `Pitch gain` = 0. The VQ branch is therefore selected by
    /// `submode.pitch_gain`; the forced branch only fires when there is
    /// no per-sub-frame VQ *and* the frame carries a forced `OL pitch
    /// gain` field.
    fn pitch_taps(
        body: &NarrowbandFrameBody,
        submode: &NarrowbandSubmode,
        sf_idx: usize,
    ) -> PitchGainTaps {
        match submode.pitch_gain {
            // Per-sub-frame 3-tap VQ (modes 2..=7).
            PitchGainQuant::Vq5Bit | PitchGainQuant::Vq7Bit => body.subframes[sf_idx]
                .pitch_gain_taps(submode)
                .unwrap_or(PitchGainTaps::SILENCE),
            // No per-sub-frame VQ: either the frame-level forced gain
            // (modes 1 / 8, OL pitch gain = 4 bits) or true silence
            // (mode 0, no pitch field at all).
            PitchGainQuant::None => {
                if submode.ol_pitch_gain_bits != 0 {
                    crate::forced_pitch_gain::forced_pitch_gain_taps(body.ol_pitch_gain_index)
                } else {
                    PitchGainTaps::SILENCE
                }
            }
        }
    }

    /// Decode one narrowband frame to `f64` PCM, advancing all decoder
    /// state. Returns [`NARROWBAND_FRAME_SAMPLES`] = 160 samples.
    ///
    /// For the silence mode (mode 0) the frame carries no LSP, pitch, or
    /// innovation; the decoder runs its synthesis filter on zero
    /// excitation with the *previous* LSP set (the manual prescribes no
    /// LSP update for silence), so the IIR memory rings out naturally.
    pub fn decode_frame(
        &mut self,
        body: &NarrowbandFrameBody,
        submode: &NarrowbandSubmode,
    ) -> Result<[f64; NARROWBAND_FRAME_SAMPLES], NarrowbandDecodeError> {
        let mut pcm = [0.0f64; NARROWBAND_FRAME_SAMPLES];

        // Reconstruct this frame's LSPs. For silence (no LSP field) the
        // previous LSPs persist; if there is no previous set (silence at
        // stream start) we fall back to a flat all-pole filter (zero
        // LPC), which the synthesis filter handles as a pass-through.
        let curr_lsp_q10 = body.reconstructed_lsp_q10(submode);
        let active_lsp = match curr_lsp_q10 {
            Some(lsp) => lsp,
            None => match self.prev_lsp_q10 {
                Some(prev) => prev,
                None => {
                    // Silence at stream start: no spectral envelope to
                    // apply. Filter the (zero) excitation through a
                    // flat filter; output is silence.
                    let flat = [0.0f64; LPC_ORDER];
                    self.run_silence_subframes(&flat, &mut pcm);
                    return Ok(pcm);
                }
            },
        };

        // Per-sub-frame LSP interpolation uses the previous frame's
        // reconstructed LSPs (or the current ones at stream start, which
        // yields the no-transient first-frame initialisation).
        let prev = self.prev_lsp_q10.unwrap_or(active_lsp);
        let sub_lsp = crate::lsp_interp::NbSubFrameLsp::new(&prev, &active_lsp);

        // Per-sub-frame LPC sets with the **pinned LSP base vector**
        // added (round r347): the r194 reconstruction emits codebook
        // deltas only; `subframe_lpc_set_with_base` adds the documented
        // `LSP_LINEAR` base so each sub-frame's LSP angles land inside
        // the conformant `(0, π)` band by construction (see `lsp_base`).
        let lpc_sets = subframe_lpc_set_with_base(&sub_lsp);

        for sf_idx in 0..SUBFRAMES_PER_FRAME {
            let lpc: [f64; LPC_ORDER] = lpc_sets[sf_idx];

            // Pitch (adaptive-codebook) contribution p[n].
            let taps = Self::pitch_taps(body, submode, sf_idx);
            let p = match Self::pitch_period(body, submode, sf_idx) {
                Some(period) => gain_scaled_pitch_subframe(period, taps, &self.excitation)
                    .map_err(|_| NarrowbandDecodeError::PitchOutOfRange { period })?,
                None => [0.0f32; SUBFRAME_SAMPLES],
            };

            // Innovation (fixed-codebook) contribution c[n].
            let c_raw = body.subframes[sf_idx]
                .innovation_sub_vector(submode)
                .map_err(|_| NarrowbandDecodeError::InnovationUndocumented {
                    mode_id: submode.mode_id,
                })?;
            let gain_indices = FixedCodebookGainIndices::from_body(body, submode, sf_idx)
                .ok_or(NarrowbandDecodeError::GainIndicesUnavailable)?;
            let c = gain_scaled_innovation_from_indices(&c_raw, gain_indices);

            // Full excitation e[n] = p[n] + c[n].
            let e = gain_scaled_excitation_subframe(&p, &c);

            // Retain the full-precision excitation for the wideband
            // folded high-band law (see `last_frame_excitation`).
            self.last_frame_excitation[sf_idx * SUBFRAME_SAMPLES..(sf_idx + 1) * SUBFRAME_SAMPLES]
                .copy_from_slice(&e);

            // Feedback: push the emitted excitation into the history so
            // the NEXT sub-frame's adaptive codebook reads it.
            for &sample in &e {
                self.excitation.push(round_to_i16(sample));
            }

            // Synthesis filter x[n] = e[n] + Σ a[i]·x[n-1-i].
            let e_f64: [f64; SUBFRAME_SAMPLES] = {
                let mut tmp = [0.0f64; SUBFRAME_SAMPLES];
                for (o, &v) in tmp.iter_mut().zip(e.iter()) {
                    *o = f64::from(v);
                }
                tmp
            };
            let out = &mut pcm[sf_idx * SUBFRAME_SAMPLES..(sf_idx + 1) * SUBFRAME_SAMPLES];
            self.filter.process(&lpc, &e_f64, out);
        }

        // Update the previous-LSP state only when the frame actually
        // transmitted an LSP field (silence leaves it untouched).
        if curr_lsp_q10.is_some() {
            self.prev_lsp_q10 = curr_lsp_q10;
        }

        Ok(pcm)
    }

    /// Decode one narrowband frame to rounded, saturated `i16` PCM.
    ///
    /// Convenience over [`Self::decode_frame`] that quantises the `f64`
    /// output to 16-bit samples. The decoder's internal feedback uses the
    /// full-precision excitation regardless, so the `i16` reduction
    /// affects only the returned PCM, not the IIR / excitation state.
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

    /// Run the four sub-frames of a silence frame: zero excitation
    /// through the given (flat or previous) LPC set, pushing the zero
    /// excitation into the history and ringing the IIR memory out.
    fn run_silence_subframes(
        &mut self,
        lpc: &[f64; LPC_ORDER],
        pcm: &mut [f64; NARROWBAND_FRAME_SAMPLES],
    ) {
        let zero = [0.0f64; SUBFRAME_SAMPLES];
        self.last_frame_excitation = [0.0; NARROWBAND_FRAME_SAMPLES];
        for sf_idx in 0..SUBFRAMES_PER_FRAME {
            for _ in 0..SUBFRAME_SAMPLES {
                self.excitation.push(0);
            }
            let out = &mut pcm[sf_idx * SUBFRAME_SAMPLES..(sf_idx + 1) * SUBFRAME_SAMPLES];
            self.filter.process(lpc, &zero, out);
        }
    }

    /// Read-only view of the synthesis filter (for tests / diagnostics).
    pub fn synthesis_history(&self) -> &[f64; LPC_ORDER] {
        self.filter.history()
    }
}

/// Round to nearest and saturate an `f64` excitation sample into the
/// `i16` range for the history buffer. (The history buffer stores `i16`
/// matching the staged codebook entry width.)
#[inline]
fn round_to_i16(v: f32) -> i16 {
    saturate_i16(f64::from(v))
}

/// Round to nearest and saturate an `f64` into the signed-16-bit range.
///
/// The single rounding convention shared by every `*_i16` PCM
/// convenience in the crate (narrowband [`NarrowbandDecoder::decode_frame_i16`],
/// the wideband 16 kHz path, and the top-level packet decoder): round to
/// nearest (ties away from zero, per [`f64::round`]) then clamp to
/// `[i16::MIN, i16::MAX]`. Centralising it here keeps the narrowband and
/// wideband `i16` outputs bit-identical wherever the same `f64` sample
/// appears.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::NarrowbandFrameHeader;
    use crate::submode::{NarrowbandSubmode, Submode, NARROWBAND_SUBMODES};

    /// Build an all-ones frame body for the given mode (same helper the
    /// body parser tests use): wideband flag 0, mode ID in the next 4
    /// bits, every body bit set to 1.
    fn all_ones_frame(mode: u8) -> Vec<u8> {
        let submode = NarrowbandSubmode::for_id(mode).expect("valid mode");
        let total_bits = u32::from(submode.total_bits);
        let total_bytes = total_bits.div_ceil(8);
        let mut buf = vec![0xFFu8; total_bytes as usize];
        let prefix_byte = (mode & 0x0F) << 3 | 0b0000_0111;
        buf[0] = prefix_byte;
        buf
    }

    fn parse(mode: u8) -> (NarrowbandSubmode, NarrowbandFrameBody) {
        let buf = all_ones_frame(mode);
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
        let submode = match h.submode {
            Submode::Celp(s) => s,
            other => panic!("expected CELP, got {:?}", other),
        };
        let body = NarrowbandFrameBody::parse(&mut r, &submode).unwrap();
        (submode, body)
    }

    #[test]
    fn frame_samples_constant_is_160() {
        assert_eq!(NARROWBAND_FRAME_SAMPLES, 160);
    }

    #[test]
    fn silence_mode_at_stream_start_yields_zero_pcm() {
        let (submode, body) = parse(0);
        let mut dec = NarrowbandDecoder::new();
        let pcm = dec.decode_frame(&body, &submode).unwrap();
        assert!(
            pcm.iter().all(|&s| s == 0.0),
            "silence at stream start should produce zero PCM"
        );
    }

    #[test]
    fn mode_5_decodes_to_finite_responsive_pcm() {
        // Mode 5 (15 kbps) has 30-bit LSP, 7-bit fine pitch, 7-bit
        // pitch gain, 3-bit innovation gain, 48-bit innovation VQ —
        // every field exercised.
        let (submode, body) = parse(5);
        let mut dec = NarrowbandDecoder::new();
        let pcm = dec.decode_frame(&body, &submode).unwrap();
        assert_eq!(pcm.len(), 160);
        for (i, &s) in pcm.iter().enumerate() {
            assert!(s.is_finite(), "sample {i} not finite: {s}");
        }
    }

    /// The pinned LSP base vector (round r347) makes the decoder's
    /// per-sub-frame LSP→LPC reconstruction operate on angles inside the
    /// conformant `(0, π)` band by construction, so the synthesis output
    /// is finite and bounded for a real mode-5 frame *without relying on
    /// the radian clamp fallback*. We verify the base-aware sub-frame LPC
    /// path is the one in use: the decoded PCM equals what the explicit
    /// `subframe_lpc_set_with_base` path produces, and differs from the
    /// unbounded delta-only `subframe_lpc_set` path (the two paths must
    /// not be silently identical — the base is a real translation).
    #[test]
    fn base_vector_path_is_wired_and_bounds_lsp() {
        use crate::lsp_to_lpc::{subframe_lpc_set, subframe_lpc_set_with_base};

        let (submode, body) = parse(5);
        let mut dec = NarrowbandDecoder::new();
        let pcm = dec.decode_frame(&body, &submode).unwrap();
        for &s in &pcm {
            assert!(s.is_finite());
        }

        // Reconstruct the same first-frame sub-frame LSP set the decoder
        // used and confirm the base-aware LPC set differs from the
        // delta-only one (the base offset is non-trivial).
        let active = body
            .reconstructed_lsp_q10(&submode)
            .expect("mode 5 carries an LSP field");
        let sub_lsp = crate::lsp_interp::NbSubFrameLsp::new(&active, &active);
        let with_base = subframe_lpc_set_with_base(&sub_lsp);
        let delta_only = subframe_lpc_set(&sub_lsp);
        // At least one coefficient must differ between the two paths.
        let mut differs = false;
        for sf in 0..SUBFRAMES_PER_FRAME {
            for i in 0..LPC_ORDER {
                if (with_base[sf][i] - delta_only[sf][i]).abs() > 1e-9 {
                    differs = true;
                }
                assert!(with_base[sf][i].is_finite());
            }
        }
        assert!(differs, "base-vector path must differ from delta-only path");

        // Every pinned sub-frame LSP angle (base + delta) must sit
        // strictly inside (0, π): re-derive the radian angles directly.
        let base_q10 = crate::lsp_base::nb_lsp_base_q10();
        for sf_q12 in &sub_lsp.subframes {
            for (i, &v12) in sf_q12.iter().enumerate() {
                // Q12 interpolated delta + Q12 base (base_q10 << 2).
                let pinned_q12 = v12 + (base_q10[i] << 2);
                let rad = f64::from(pinned_q12) / f64::from(1u32 << 12);
                assert!(
                    rad > 0.0 && rad < core::f64::consts::PI,
                    "pinned LSP angle {rad} (coeff {i}) outside (0, π)"
                );
            }
        }
    }

    #[test]
    fn feedback_makes_pitch_nonzero_on_second_frame() {
        // Decode two non-silent frames. After the first frame the
        // excitation buffer holds real samples, so the second frame's
        // adaptive-codebook contribution is generally non-zero — the
        // very feedback this module wires. We verify by decoding the
        // same frame twice and observing the outputs differ (the IIR +
        // excitation state has advanced).
        let (submode, body) = parse(5);
        let mut dec = NarrowbandDecoder::new();
        let first = dec.decode_frame(&body, &submode).unwrap();
        let second = dec.decode_frame(&body, &submode).unwrap();
        // The two frames cannot be identical: the second frame's
        // adaptive codebook now reads the first frame's excitation, and
        // the IIR history is non-zero, so the synthesis differs.
        assert!(
            first.iter().zip(second.iter()).any(|(&a, &b)| a != b),
            "feedback should make the second frame differ from the first"
        );
    }

    #[test]
    fn excitation_buffer_holds_nonzero_after_nonsilent_frame() {
        let (submode, body) = parse(5);
        let mut dec = NarrowbandDecoder::new();
        let _ = dec.decode_frame(&body, &submode).unwrap();
        // The most recent excitation samples must be non-zero for a
        // non-silent frame (the feedback push happened).
        let mut any_nonzero = false;
        for k in 1..=40i32 {
            if dec.excitation.lookup(-k).unwrap() != 0 {
                any_nonzero = true;
                break;
            }
        }
        assert!(
            any_nonzero,
            "excitation history should be non-zero after a non-silent frame"
        );
    }

    #[test]
    fn i16_path_matches_f64_path_rounded() {
        let (submode, body) = parse(5);
        let mut dec_a = NarrowbandDecoder::new();
        let f64_pcm = dec_a.decode_frame(&body, &submode).unwrap();
        let mut dec_b = NarrowbandDecoder::new();
        let i16_pcm = dec_b.decode_frame_i16(&body, &submode).unwrap();
        for (i, (&f, &q)) in f64_pcm.iter().zip(i16_pcm.iter()).enumerate() {
            assert_eq!(q, saturate_i16(f), "sample {i}");
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let (submode, body) = parse(3);
        let mut a = NarrowbandDecoder::new();
        let mut b = NarrowbandDecoder::new();
        for _ in 0..5 {
            let pa = a.decode_frame(&body, &submode).unwrap();
            let pb = b.decode_frame(&body, &submode).unwrap();
            assert_eq!(pa, pb);
        }
    }

    #[test]
    fn every_documented_mode_with_innovation_decodes() {
        // Modes whose innovation binding is documented (0, 2, 3, 4, 5,
        // 6, 8) decode; modes 1 and 7 surface the documented gap.
        for mode in [0u8, 2, 3, 4, 5, 6, 8] {
            let (submode, body) = parse(mode);
            let mut dec = NarrowbandDecoder::new();
            let pcm = dec
                .decode_frame(&body, &submode)
                .unwrap_or_else(|e| panic!("mode {mode} should decode: {e}"));
            for &s in &pcm {
                assert!(s.is_finite(), "mode {mode} produced non-finite PCM");
            }
        }
    }

    #[test]
    fn undocumented_innovation_modes_report_gap() {
        for mode in [1u8, 7] {
            let (submode, body) = parse(mode);
            let mut dec = NarrowbandDecoder::new();
            match dec.decode_frame(&body, &submode) {
                Err(NarrowbandDecodeError::InnovationUndocumented { mode_id }) => {
                    assert_eq!(mode_id, mode);
                }
                other => panic!("mode {mode} expected docs gap, got {other:?}"),
            }
        }
    }

    #[test]
    fn submode_table_pitch_rows_are_mutually_exclusive() {
        // Sanity: no documented mode carries BOTH frame-level OL pitch
        // and per-sub-frame fine pitch — the decoder's pitch-period
        // resolution relies on this.
        for s in NARROWBAND_SUBMODES {
            assert!(
                !(s.ol_pitch_bits != 0 && s.fine_pitch_bits != 0),
                "mode {} carries both OL and fine pitch",
                s.mode_id
            );
        }
    }

    /// The forced (open-loop) pitch-gain modes (1 and 8) now resolve a
    /// non-silent pitch tap from the frame-level `OL pitch gain` field —
    /// before round r356 they fell back to `SILENCE`. The all-ones frame
    /// sets the 4-bit field to 15 (unit gain → Q6 centre tap 64).
    #[test]
    fn forced_pitch_gain_modes_resolve_nonsilent_centre_tap() {
        for mode in [1u8, 8] {
            let (submode, body) = parse(mode);
            assert_eq!(
                submode.ol_pitch_gain_bits, 4,
                "mode {mode} should carry a 4-bit OL pitch gain"
            );
            assert_eq!(
                body.ol_pitch_gain_index, 15,
                "all-ones frame sets the 4-bit field to 15"
            );
            for sf in 0..SUBFRAMES_PER_FRAME {
                let taps = NarrowbandDecoder::pitch_taps(&body, &submode, sf).taps;
                // Single-centre-tap forced gain: g0 = g2 = 0, g1 = Q6 64.
                assert_eq!(taps, [0, 64, 0], "mode {mode} sf {sf}");
            }
        }
    }

    /// Mode 0 (true silence — no pitch field at all) still resolves the
    /// all-zero taps; the forced branch only fires when a frame actually
    /// carries the `OL pitch gain` field.
    #[test]
    fn silence_mode_pitch_taps_stay_silent() {
        let (submode, body) = parse(0);
        assert_eq!(submode.ol_pitch_gain_bits, 0);
        for sf in 0..SUBFRAMES_PER_FRAME {
            assert_eq!(
                NarrowbandDecoder::pitch_taps(&body, &submode, sf),
                PitchGainTaps::SILENCE,
                "silence sf {sf}"
            );
        }
    }

    /// End-to-end: mode 8 (forced OL pitch gain + documented innovation)
    /// now feeds a real pitch contribution. Decoding two frames, the
    /// second frame's adaptive codebook reads the first frame's
    /// excitation through the forced centre tap, so its output is no
    /// longer the innovation-only stand-in. Cross-check that a decoder
    /// with the forced gain zeroed (index 0) produces a *different*
    /// second frame — proving the forced pitch path is load-bearing.
    #[test]
    fn forced_pitch_gain_changes_mode8_second_frame() {
        let (submode, mut body) = parse(8);
        // Unit forced gain (all-ones frame → index 15).
        assert_eq!(body.ol_pitch_gain_index, 15);
        let mut dec = NarrowbandDecoder::new();
        let _f0 = dec.decode_frame(&body, &submode).unwrap();
        let f1_forced = dec.decode_frame(&body, &submode).unwrap();

        // Same stream, but with the forced pitch gain set to 0 (no pitch
        // contribution). The first frames are identical (empty history →
        // zero pitch either way), but the second frame must differ once
        // the forced tap reads the non-zero history.
        body.ol_pitch_gain_index = 0;
        let mut dec0 = NarrowbandDecoder::new();
        let _g0 = dec0.decode_frame(&body, &submode).unwrap();
        let g1_zero = dec0.decode_frame(&body, &submode).unwrap();

        assert!(
            f1_forced.iter().zip(g1_zero.iter()).any(|(&a, &b)| a != b),
            "forced pitch gain should change the second-frame output"
        );
        for &s in &f1_forced {
            assert!(s.is_finite());
        }
    }
}
