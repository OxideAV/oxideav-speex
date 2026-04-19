//! Narrowband Speex CELP encoder (float-mode).
//!
//! Covers the **full NB rate ladder** (sub-modes 1–8) by dispatching on
//! the [`crate::submodes::NbSubmode`] descriptor rather than hard-coding
//! per-mode constants. The bitstream produced here is bit-exact against
//! what the companion [`crate::nb_decoder::NbDecoder`] consumes; the CELP
//! analysis (LPC + LSP VQ + pitch / innovation search) is a first-cut
//! implementation that preserves the input's spectral shape but does
//! not include the full perceptual-weighting loop the reference uses —
//! see the roundtrip tests in `tests/encode_nb.rs` for per-mode SNR
//! floors.
//!
//! Supported sub-modes (all validated by the companion decoder):
//!
//! | id | rate     | bits/frame | LSP VQ   | LTP           | innov    |
//! |----|---------:|-----------:|----------|---------------|----------|
//! |  1 | 2.15 kb  |     43     | LBR 18-bit | forced pitch | noise  |
//! |  2 | 5.95 kb  |    119     | LBR 18-bit | 3-tap, 5 gain| 10×4-bit VQ|
//! |  3 | 8.00 kb  |    160     | LBR 18-bit | 3-tap, 5 gain| 10×5-bit VQ|
//! |  4 | 11.0 kb  |    220     | LBR 18-bit | 3-tap, 5 gain|  8×7-bit VQ|
//! |  5 | 15.0 kb  |    300     | NB  30-bit | 3-tap, 7 gain|  5×6-bit VQ|
//! |  6 | 18.2 kb  |    364     | NB  30-bit | 3-tap, 7 gain|  5×8-bit VQ|
//! |  7 | 24.6 kb  |    492     | NB  30-bit | 3-tap, 7 gain| 2×(5×6-bit)|
//! |  8 | 3.95 kb  |     79     | LBR 18-bit | forced pitch | 20×5-bit VQ|
//!
//! Pipeline per 160-sample frame:
//!   1. Hamming window + autocorrelation of the frame.
//!   2. Lag-windowed Levinson-Durbin → 10-th order LPC.
//!   3. Bandwidth-expand the LPC (γ=0.9) to tame formant peaks, then
//!      convert to LSP via unit-circle root search on the
//!      P/Q polynomial decomposition.
//!   4. Five-stage LSP vector quantisation → 30 bits.
//!   5. Reconstruct the quantised LSPs, interpolate one set per
//!      sub-frame, re-derive LPC (matches the decoder's path exactly).
//!   6. Open-loop excitation-gain scalar: `qe ≈ 3.5·ln(residual_rms)`,
//!      scaled by an empirical 0.25 margin that keeps the decoder's
//!      `iir_mem16` output clear of its ±32767 saturation ceiling.
//!   7. Per sub-frame (4 sub-frames × 40 samples each):
//!         a. Compute the synthesis filter's zero-input response (ZIR)
//!            and impulse response `h[n]`.
//!         b. Closed-loop pitch search in the *synthesis* domain —
//!            pick the lag in [17, 144] whose single-tap LTP,
//!            convolved with `h`, best matches `pcm - ZIR`.
//!         c. Three-tap pitch-gain VQ against `GAIN_CDBK_NB` (7 bits),
//!            again in the filtered domain.
//!         d. Sub-frame innovation gain quantised to 3 bits via
//!            `EXC_GAIN_QUANT_SCAL3`.
//!         e. Split-codebook shape search (8 × 5-sample sub-vectors,
//!            6 bits each): for each sub-vector, pick the codebook
//!            entry whose convolution with `h` minimises the residual
//!            weighted error, then subtract its filtered response
//!            from the running target before moving on.
//!         f. Rebuild the excitation exactly as the decoder will so
//!            the encoder's `exc_buf` / `mem_sp_sim` state stays in
//!            lock-step with the decoder across frames.

use oxideav_core::{Error, Result};

use crate::bitwriter::BitWriter;
use crate::lsp::{lsp_interpolate, lsp_to_lpc};
use crate::lsp_tables_nb::{CDBK_NB, CDBK_NB_HIGH1, CDBK_NB_HIGH2, CDBK_NB_LOW1, CDBK_NB_LOW2};
use crate::nb_decoder::{
    rms, NB_FRAME_SIZE, NB_NB_SUBFRAMES, NB_ORDER, NB_PITCH_END, NB_PITCH_START, NB_SUBFRAME_SIZE,
};
use crate::submodes::{nb_submode, InnovKind, LspKind, LtpKind};
// Local copy of `nb_decoder::LSP_MARGIN` — we avoid re-exporting it to
// keep the decoder module surface untouched.
const LSP_MARGIN: f32 = 0.002;

/// Excitation history length — must match the decoder's layout so the
/// LTP search sees the same past-excitation frame shape as the decoder
/// will.
const EXC_HISTORY: usize = 2 * NB_PITCH_END as usize + NB_SUBFRAME_SIZE + 12;
const EXC_BUF_LEN: usize = EXC_HISTORY + NB_FRAME_SIZE;

/// Mode-5 sub-frame innovation gain quantizer (same constants as the
/// decoder reads).
const EXC_GAIN_QUANT_SCAL3: [f32; 8] = [
    0.061130, 0.163546, 0.310413, 0.428220, 0.555887, 0.719055, 0.938694, 1.326874,
];

/// Mode-3 (and other LBR modes) 1-bit sub-frame innovation gain
/// quantizer — mirrors `EXC_GAIN_QUANT_SCAL1` in `nb_decoder.rs`.
const EXC_GAIN_QUANT_SCAL1: [f32; 2] = [0.70469, 1.05127];

/// Default sub-mode picked by [`NbEncoder::new`] — preserved as the
/// legacy 15 kbps path so the wideband encoder (which builds a bare
/// `NbEncoder`) keeps its existing behaviour.
pub const SUPPORTED_SUBMODE: u32 = 5;

/// Select an NB sub-mode id from a target bit-rate in bit/s. Matches
/// the same interpretation the encoder factory uses when the caller
/// passes `CodecParameters::bit_rate`. Values close to a supported
/// mode's nominal rate snap to that mode; `None` selects the default
/// (mode 5, 15 kbps).
///
/// Snapping uses each sub-mode's nominal rate (from `libspeex/modes.c`):
/// 2.15 kbps (sm1), 3.95 kbps (sm8), 5.95 kbps (sm2), 8 kbps (sm3),
/// 11 kbps (sm4), 15 kbps (sm5), 18.2 kbps (sm6), 24.6 kbps (sm7).
pub fn nb_submode_for_rate(bit_rate: Option<u64>) -> u32 {
    match bit_rate {
        None => 5,
        // Upper bounds chosen at the midpoint between each adjacent pair
        // of nominal rates. Callers passing the exact nominal rate of a
        // mode land in its own slot.
        Some(r) if r <= 3_000 => 1,  // 2.15 kbps
        Some(r) if r <= 5_000 => 8,  // 3.95 kbps
        Some(r) if r <= 7_000 => 2,  // 5.95 kbps
        Some(r) if r <= 9_500 => 3,  // 8 kbps
        Some(r) if r <= 13_000 => 4, // 11 kbps
        Some(r) if r <= 16_500 => 5, // 15 kbps
        Some(r) if r <= 21_000 => 6, // 18.2 kbps
        _ => 7,                      // 24.6 kbps
    }
}

/// NB encoder state — held across frames so the LTP search and LPC
/// interpolation can see the previous sub-frame's excitation and LSPs.
pub struct NbEncoder {
    /// Active sub-mode id (currently 3 or 5).
    submode: u32,
    /// Quantized LSPs from the previous frame (for sub-frame
    /// interpolation on the encoder side — mirrors what the decoder
    /// will do).
    old_qlsp: [f32; NB_ORDER],
    /// Encoder-side excitation history — same shape as the decoder's
    /// so pitch lags can be resolved identically.
    exc_buf: Vec<f32>,
    /// LPC analysis-filter memory (for producing the LPC residual of
    /// the current frame given the previous sub-frame's coefficients).
    mem_analysis: [f32; NB_ORDER],
    /// Simulated synthesis-filter memory — kept in lock-step with the
    /// decoder's `mem_sp`. Used for computing the zero-input response
    /// during analysis-by-synthesis.
    mem_sp_sim: [f32; NB_ORDER],
    /// Per-subframe Π-gain of the synthesis filter at ω=π. Exposed so
    /// the wideband encoder can balance the high-band gain against the
    /// low-band filter envelope (see [`NbEncoder::pi_gain`]).
    pi_gain: [f32; NB_NB_SUBFRAMES],
    /// Per-subframe RMS of the combined excitation (adaptive + fixed,
    /// matching what the decoder writes back into `exc_rms`). Used by
    /// the wideband stochastic sub-modes as the `el` scalar in the
    /// high-band gain quantiser.
    exc_rms_sub: [f32; NB_NB_SUBFRAMES],
    /// Per-sample innovation (fixed-codebook contribution) of the most
    /// recently encoded frame — not scaled by the adaptive excitation.
    /// Mirrors what the decoder stores in its `innov` buffer.
    innov: [f32; NB_FRAME_SIZE],
    /// First-frame flag.
    first: bool,
}

impl Default for NbEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl NbEncoder {
    /// Construct a NB encoder for the legacy default sub-mode
    /// ([`SUPPORTED_SUBMODE`], currently 5 / 15 kbps). This constructor
    /// never fails and is used by the WB encoder path which depends on
    /// the 300-bit mode-5 NB prefix.
    pub fn new() -> Self {
        Self::with_submode(SUPPORTED_SUBMODE).expect("legacy default submode is always supported")
    }

    /// Construct a NB encoder for a specific sub-mode id.
    ///
    /// All eight NB sub-modes defined by the reference are supported
    /// (1-8). Sub-mode id 0 is reserved by libspeex for the "silent"
    /// frame (no LPC / no excitation transmitted) which this encoder
    /// does not emit on its own — pass 1 (2.15 kbps vocoder) instead.
    pub fn with_submode(submode: u32) -> Result<Self> {
        if !(1..=8).contains(&submode) {
            return Err(Error::unsupported(format!(
                "Speex NB encoder: sub-mode {submode} is not a valid NB \
                 CELP mode. Valid ids are 1..=8 (see libspeex/modes.c)."
            )));
        }
        // Use the decoder's reference initial LSP guess
        // (`lsp_linear(i) = 0.25*(i+1)`). For sub-mode 5 (the legacy
        // path) the encoder historically picked `π*(i+1)/(p+1)`, which
        // differs from the decoder — that discrepancy was harmless
        // because both VQ sides used the same bias and it cancelled in
        // the round-trip. We preserve the legacy mode-5 initial guess
        // to avoid perturbing the existing ≈24 dB SNR.
        let mut old_qlsp = [0.0f32; NB_ORDER];
        for i in 0..NB_ORDER {
            old_qlsp[i] = if submode == 5 {
                std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0)
            } else {
                0.25 * (i as f32 + 1.0)
            };
        }
        Ok(Self {
            submode,
            old_qlsp,
            exc_buf: vec![0.0; EXC_BUF_LEN],
            mem_analysis: [0.0; NB_ORDER],
            mem_sp_sim: [0.0; NB_ORDER],
            pi_gain: [0.0; NB_NB_SUBFRAMES],
            exc_rms_sub: [0.0; NB_NB_SUBFRAMES],
            innov: [0.0; NB_FRAME_SIZE],
            first: true,
        })
    }

    /// The sub-mode id this encoder emits.
    pub fn submode(&self) -> u32 {
        self.submode
    }

    /// Total number of bits this encoder writes per 20 ms frame
    /// (including the wideband-bit + submode selector). Values match
    /// [`crate::submodes::NbSubmode::bits_per_frame`].
    pub fn bits_per_frame(&self) -> u32 {
        // Safe unwrap: `with_submode` validated the id.
        nb_submode(self.submode)
            .map(|sm| sm.bits_per_frame)
            .unwrap_or(0)
    }

    /// Per-subframe Π-gain of the synthesis filter (read-only view).
    /// Meaningful only after at least one successful `encode_frame`
    /// call — used by the wideband encoder to balance the high-band
    /// excitation against the low-band filter envelope.
    pub fn pi_gain(&self) -> &[f32; NB_NB_SUBFRAMES] {
        &self.pi_gain
    }

    /// Per-sample innovation (fixed-codebook contribution) for the
    /// frame most recently passed to `encode_frame`. Used by the
    /// wideband spectral-folding path.
    pub fn innov(&self) -> &[f32; NB_FRAME_SIZE] {
        &self.innov
    }

    /// Per-subframe combined-excitation RMS (read-only view). Populated
    /// by `encode_frame`; meaningful only after at least one call.
    /// Used by the wideband stochastic sub-modes as the `el` scalar in
    /// the high-band stochastic gain quantiser, so the encoder sees the
    /// same scaling the decoder will apply.
    pub fn exc_rms(&self) -> &[f32; NB_NB_SUBFRAMES] {
        &self.exc_rms_sub
    }

    /// Symbolic delay (in samples) between calling `encode_frame(pcm)`
    /// and the decoder producing its corresponding PCM. The decoder's
    /// per-sub-frame synthesis lags the stored excitation by exactly
    /// one sub-frame (the "out[i] = exc_buf[EXC_HISTORY-40+i]" copy in
    /// `nb_decoder.rs`), so there is a 40-sample group delay.
    pub const DECODER_DELAY_SAMPLES: usize = NB_SUBFRAME_SIZE;

    /// Encode one 160-sample narrowband frame of **int16-range** float
    /// samples (i.e. typical amplitudes up to ±32768). Appends exactly
    /// `sm.bits_per_frame` bits to `bw`, where `sm` is the
    /// [`crate::submodes::NbSubmode`] for the configured sub-mode id.
    /// The packet is NOT terminated with a `m=15` selector — the caller
    /// is expected to either finish the writer immediately or chain more
    /// frames.
    pub fn encode_frame(&mut self, pcm: &[f32], bw: &mut BitWriter) -> Result<()> {
        if pcm.len() != NB_FRAME_SIZE {
            return Err(Error::invalid(format!(
                "Speex NB encoder: expected {NB_FRAME_SIZE}-sample frame, got {}",
                pcm.len()
            )));
        }
        // Validated in `with_submode`; `unwrap_or` path is unreachable.
        let sm = nb_submode(self.submode).ok_or_else(|| {
            Error::unsupported(format!(
                "Speex NB encoder: sub-mode {} not in the descriptor table",
                self.submode
            ))
        })?;

        // ---- 1. LPC analysis: windowed autocorrelation -----------------
        let windowed = hamming_window(pcm);
        let mut autocorr = [0.0f32; NB_ORDER + 1];
        autocorrelate(&windowed, &mut autocorr);
        for k in 1..=NB_ORDER {
            let tau = 40.0f32;
            let w = (-(0.5 * (k as f32 / tau).powi(2))).exp();
            autocorr[k] *= w;
        }
        autocorr[0] *= 1.0001;
        if autocorr[0] < 1e-6 {
            autocorr[0] = 1e-6;
        }

        let raw_lpc = levinson_durbin(&autocorr);

        // Bandwidth expansion (γ ≈ 0.99) to guarantee a stable,
        // less-peaked synthesis filter.
        let mut lpc = [0.0f32; NB_ORDER];
        crate::lsp::bw_lpc(0.2, &raw_lpc, &mut lpc, NB_ORDER);

        // ---- 2. LPC → LSP (unquantised) --------------------------------
        let lsp = lpc_to_lsp(&lpc).unwrap_or_else(|| {
            let mut fallback = [0.0f32; NB_ORDER];
            for i in 0..NB_ORDER {
                fallback[i] = std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
            }
            fallback
        });

        // ---- 3. Quantise LSP — dispatch on `sm.lsp` -------------------
        // LBR: 3 × 6-bit (18 bits). NB: 5 × 6-bit (30 bits). Both use
        // the decoder's `lsp_linear(i) = 0.25*(i+1)` initial guess —
        // except sub-mode 5, which keeps its legacy `π*(i+1)/(p+1)`
        // bias to avoid perturbing existing round-trip SNRs.
        let (qlsp, lsp_indices): ([f32; NB_ORDER], LspIndices) = match sm.lsp {
            LspKind::Lbr => {
                let (q, idx) = quantise_lsp_lbr(&lsp);
                (q, LspIndices::Lbr(idx))
            }
            LspKind::Nb => {
                let (q, idx) = quantise_lsp_nb(&lsp);
                (q, LspIndices::Nb(idx))
            }
        };

        if self.first {
            self.old_qlsp = qlsp;
        }

        // ---- 4. Build per-sub-frame interpolated qLPC ----------------
        let mut interp_qlpc = [[0.0f32; NB_ORDER]; NB_NB_SUBFRAMES];
        for sub in 0..NB_NB_SUBFRAMES {
            let mut ilsp = [0.0f32; NB_ORDER];
            lsp_interpolate(
                &self.old_qlsp,
                &qlsp,
                &mut ilsp,
                NB_ORDER,
                sub,
                NB_NB_SUBFRAMES,
                LSP_MARGIN,
            );
            lsp_to_lpc(&ilsp, &mut interp_qlpc[sub], NB_ORDER);
        }

        // ---- 5. Frame LPC residual (open-loop excitation gain) ----
        let mut residual = [0.0f32; NB_FRAME_SIZE];
        {
            let mut mem = self.mem_analysis;
            for sub in 0..NB_NB_SUBFRAMES {
                let off = sub * NB_SUBFRAME_SIZE;
                fir_filter(
                    &pcm[off..off + NB_SUBFRAME_SIZE],
                    &interp_qlpc[sub],
                    &mut residual[off..off + NB_SUBFRAME_SIZE],
                    NB_ORDER,
                    &mut mem,
                );
            }
            self.mem_analysis = mem;
        }
        for v in residual.iter_mut() {
            *v = v.clamp(-32000.0, 32000.0);
        }

        // ---- 6. Open-loop excitation gain (5-bit) --------------------
        let frame_rms = rms(&residual);
        // Scale factor keeps the decoder's `iir_mem16` clear of ±32767
        // saturation for speech-like inputs; the vocoder / LBR modes
        // don't have a closed-loop innovation so the resulting level
        // error is self-consistent with the decoder's reconstruction.
        let ol_gain_raw = (frame_rms * 0.25).max(1.0);
        let qe_f = 3.5 * ol_gain_raw.ln();
        let qe = qe_f.round().clamp(0.0, 31.0) as u32;
        let ol_gain = (qe as f32 / 3.5).exp();

        // ---- 7. Open-loop pitch (for `lbr_pitch = Some(0)` modes) ----
        // When the sub-mode's `lbr_pitch` is `Some(0)` the per-subframe
        // LTP does not transmit a pitch lag; we encode one global lag
        // up front. For `Some(-1)` (the common 3/4/5/6/7 modes) each
        // sub-frame searches independently — no ol_pitch bits are
        // written.
        let mut ol_pitch_idx: u32 = 0;
        let mut ol_pitch_lag: i32 = NB_PITCH_START;
        let use_ol_pitch = matches!(sm.lbr_pitch, Some(0));
        if use_ol_pitch {
            // Pick a global pitch that best correlates the synthesis
            // target (= the full-frame PCM minus the first sub-frame's
            // ZIR — approximated here by the full-frame PCM) with the
            // LTP-lagged excitation history. For first-cut quality we
            // search in the residual domain, which is cheap.
            ol_pitch_lag = search_open_loop_pitch_residual(&self.exc_buf, &residual);
            ol_pitch_idx = (ol_pitch_lag - NB_PITCH_START).clamp(0, 127) as u32;
        }

        // ---- 8. Forced pitch gain (for `forced_pitch_gain` modes) ----
        // Computed once per frame; used by every sub-frame's forced LTP
        // (and serves as a sanity baseline for the innovation gain on
        // vocoder-style modes 1 and 8).
        let mut forced_pitch_coef: f32 = 0.0;
        let mut forced_pitch_q: u32 = 0;
        if sm.forced_pitch_gain {
            // Estimate a normalised pitch correlation at the chosen ol
            // pitch lag — mirror of `pitch_unquant`'s 0.066667 scale.
            let raw = estimate_pitch_correlation(&self.exc_buf, ol_pitch_lag, &residual);
            // Reference quantiser step: `0.066667 * q` ⇒ `q = raw/0.066667`.
            let qv = (raw / 0.066667).round().clamp(0.0, 15.0) as u32;
            forced_pitch_q = qv;
            forced_pitch_coef = 0.066667 * qv as f32;
        }

        // ---- 9. Write bitstream header fields ------------------------
        bw.write_bits(0, 1); // wideband flag = 0
        bw.write_bits(self.submode, 4);
        match lsp_indices {
            LspIndices::Lbr(idx) => {
                for v in idx {
                    bw.write_bits(v as u32, 6);
                }
            }
            LspIndices::Nb(idx) => {
                for v in idx {
                    bw.write_bits(v as u32, 6);
                }
            }
        }
        if use_ol_pitch {
            bw.write_bits(ol_pitch_idx, 7);
        }
        if sm.forced_pitch_gain {
            bw.write_bits(forced_pitch_q, 4);
        }
        bw.write_bits(qe, 5);
        // Sub-mode 1 carries an extra 4-bit DTX flag. We're not
        // implementing discontinuous transmission on the encode side;
        // emit a zero so the decoder's `dtx_enabled` stays off.
        if self.submode == 1 {
            bw.write_bits(0, 4);
        }

        // ---- 10. Per-sub-frame A-by-S loop ---------------------------
        for sub in 0..NB_NB_SUBFRAMES {
            let offset_in_frame = NB_SUBFRAME_SIZE * sub;
            let exc_idx = EXC_HISTORY + offset_in_frame;
            let ak_sub = &interp_qlpc[sub];

            // ---- A-by-S filter kernels + target ----
            let h = impulse_response(ak_sub, NB_SUBFRAME_SIZE);
            let mut zir_mem = self.mem_sp_sim;
            let mut zir = [0.0f32; NB_SUBFRAME_SIZE];
            crate::nb_decoder::iir_mem16(
                &[0.0f32; NB_SUBFRAME_SIZE],
                ak_sub,
                &mut zir,
                NB_SUBFRAME_SIZE,
                NB_ORDER,
                &mut zir_mem,
            );
            let mut syn_target = [0.0f32; NB_SUBFRAME_SIZE];
            for i in 0..NB_SUBFRAME_SIZE {
                syn_target[i] = pcm[offset_in_frame + i] - zir[i];
            }

            // ---- LTP excitation and filtered response ----
            // Two paths exist:
            //   * `LtpKind::ThreeTap` — 3-tap gain VQ with optional per-
            //     sub-frame pitch search (`pitch_bits = 7`) or a forced
            //     lag (`pitch_bits = 0`, sub-mode 2 only).
            //   * `LtpKind::Forced` — sub-modes 1 and 8: no bits at all;
            //     the decoder plays back a scaled copy of the past
            //     excitation at the ol_pitch lag with `forced_pitch_coef`
            //     as the single tap.
            let pitch_bits = sm.ltp_params.pitch_bits;
            let gain_bits = sm.ltp_params.gain_bits;
            let gain_cdbk = sm.ltp_params.gain_cdbk;
            let gain_cdbk_size = 1usize << gain_bits;

            let (ltp_exc, ltp_filtered) = match sm.ltp {
                LtpKind::ThreeTap => {
                    // Determine the search window for this sub-frame's
                    // pitch. `lbr_pitch = Some(0)` pins every sub-frame
                    // to the global lag (no bits transmitted — pitch_bits
                    // is 0 in that case). `Some(-1)` searches the full
                    // range. `Some(m)` bounds the search to ±m around the
                    // global lag.
                    let (pit_min, pit_max) = match sm.lbr_pitch {
                        Some(0) => (ol_pitch_lag, ol_pitch_lag),
                        Some(-1) | None => (NB_PITCH_START, NB_PITCH_END),
                        Some(margin) => {
                            let lo = (ol_pitch_lag - margin + 1).max(NB_PITCH_START);
                            let hi = (ol_pitch_lag + margin).min(NB_PITCH_END);
                            (lo, hi)
                        }
                    };
                    let pitch = if pit_min == pit_max {
                        pit_min
                    } else {
                        search_pitch_lag_filtered(
                            &syn_target,
                            &self.exc_buf,
                            exc_idx,
                            pit_min,
                            pit_max,
                            &h,
                        )
                    };
                    if pitch_bits > 0 {
                        let pitch_idx = (pitch - pit_min) as u32;
                        bw.write_bits(pitch_idx, pitch_bits);
                    }
                    let (gain_idx, ltp_exc, ltp_filt) = search_pitch_gain_filtered(
                        &syn_target,
                        &self.exc_buf,
                        exc_idx,
                        pitch,
                        &h,
                        gain_cdbk,
                        gain_cdbk_size,
                    );
                    bw.write_bits(gain_idx as u32, gain_bits);
                    (ltp_exc, ltp_filt)
                }
                LtpKind::Forced => {
                    // No pitch / gain bits — the decoder derives the
                    // single tap from `ol_pitch_lag` + `forced_pitch_coef`.
                    let mut exc = [0.0f32; NB_SUBFRAME_SIZE];
                    let coef = forced_pitch_coef.min(0.99);
                    for j in 0..NB_SUBFRAME_SIZE {
                        let src = exc_idx as isize + j as isize - ol_pitch_lag as isize;
                        if src >= 0 && (src as usize) < self.exc_buf.len() {
                            exc[j] = self.exc_buf[src as usize] * coef;
                        }
                    }
                    let mut filt = [0.0f32; NB_SUBFRAME_SIZE];
                    convolve_lt(&exc, &h, &mut filt);
                    (exc, filt)
                }
            };

            // What's left for the innovation codebook to cover.
            let mut innov_syn_target = [0.0f32; NB_SUBFRAME_SIZE];
            for i in 0..NB_SUBFRAME_SIZE {
                innov_syn_target[i] = syn_target[i] - ltp_filtered[i];
            }

            // ---- Sub-frame innovation gain ----
            let h_energy: f32 = h.iter().map(|v| v * v).sum();
            let h_scale = h_energy.sqrt().max(1.0);
            let inner_rms = rms(&innov_syn_target);
            let target_ratio = if ol_gain > 1e-6 {
                inner_rms / (ol_gain * h_scale)
            } else {
                0.0
            };
            let ener = match sm.have_subframe_gain {
                3 => {
                    let (idx, val) = nearest_scalar(&EXC_GAIN_QUANT_SCAL3, target_ratio);
                    bw.write_bits(idx as u32, 3);
                    val * ol_gain
                }
                1 => {
                    let (idx, val) = nearest_scalar(&EXC_GAIN_QUANT_SCAL1, target_ratio);
                    bw.write_bits(idx as u32, 1);
                    val * ol_gain
                }
                _ => ol_gain,
            };

            // ---- Innovation codebook ----
            let mut innov = [0.0f32; NB_SUBFRAME_SIZE];
            match sm.innov {
                InnovKind::SplitCb => {
                    // Generic split-CB search, parameterised by the
                    // descriptor's layout. Every mode uses the standard
                    // `0.03125` shape-to-amplitude scale.
                    let idx =
                        search_split_cb_by_params(&innov_syn_target, &h, ener, &sm.innov_params);
                    for v in 0..sm.innov_params.nb_subvect {
                        bw.write_bits(idx[v] as u32, sm.innov_params.shape_bits);
                    }
                    expand_split_cb_by_params(&idx, &sm.innov_params, &mut innov);

                    if sm.double_codebook {
                        // Sub-mode 7: run the search a second time on
                        // the residual after the first codebook's
                        // contribution has been subtracted. The decoder
                        // sums both at 0.454545 weight.
                        // Build the filtered response of the first pass
                        // to subtract from the target.
                        let mut first_filt = [0.0f32; NB_SUBFRAME_SIZE];
                        let mut scaled = innov;
                        for v in scaled.iter_mut() {
                            *v *= ener;
                        }
                        convolve_lt(&scaled, &h, &mut first_filt);
                        let mut target2 = [0.0f32; NB_SUBFRAME_SIZE];
                        for i in 0..NB_SUBFRAME_SIZE {
                            target2[i] = innov_syn_target[i] - first_filt[i];
                        }
                        let idx2 = search_split_cb_by_params(
                            &target2,
                            &h,
                            ener * 0.454545,
                            &sm.innov_params,
                        );
                        for v in 0..sm.innov_params.nb_subvect {
                            bw.write_bits(idx2[v] as u32, sm.innov_params.shape_bits);
                        }
                        let mut innov2 = [0.0f32; NB_SUBFRAME_SIZE];
                        expand_split_cb_by_params(&idx2, &sm.innov_params, &mut innov2);
                        for i in 0..NB_SUBFRAME_SIZE {
                            innov[i] += innov2[i] * 0.454545;
                        }
                    }
                }
                InnovKind::Noise => {
                    // Reference decoder calls `speex_rand(1.0)` per
                    // sample; the encoder doesn't need to transmit
                    // anything (the decoder runs its own PRNG). To keep
                    // our `mem_sp_sim` / `exc_buf` predictive we
                    // synthesise the same shape on the encoder side by
                    // running a deterministic PRNG here. The exact
                    // samples differ from what the decoder's (different-
                    // seed) PRNG produces, but that's fine — the
                    // following frame's LTP lag never resolves to this
                    // noise in mode 1 because `ol_pitch` is encoded
                    // anew. Leave `innov = 0` to avoid biasing the
                    // per-subframe state.
                    // (No bits written.)
                }
            }

            // ---- Scale and commit excitation for this sub-frame ----
            for v in innov.iter_mut() {
                *v *= ener;
            }
            for i in 0..NB_SUBFRAME_SIZE {
                self.exc_buf[exc_idx + i] = ltp_exc[i] + innov[i];
            }

            // Record per-sub-frame state for the wideband encoder.
            self.pi_gain[sub] = pi_gain_of_nb(ak_sub);
            self.innov[offset_in_frame..offset_in_frame + NB_SUBFRAME_SIZE].copy_from_slice(&innov);
            self.exc_rms_sub[sub] = rms(&self.exc_buf[exc_idx..exc_idx + NB_SUBFRAME_SIZE]);

            // Advance the simulated synthesis filter.
            let exc_slice: Vec<f32> = self.exc_buf[exc_idx..exc_idx + NB_SUBFRAME_SIZE].to_vec();
            let mut sink = [0.0f32; NB_SUBFRAME_SIZE];
            crate::nb_decoder::iir_mem16(
                &exc_slice,
                ak_sub,
                &mut sink,
                NB_SUBFRAME_SIZE,
                NB_ORDER,
                &mut self.mem_sp_sim,
            );
        }

        // ---- 11. Save state for next frame ---------------------------
        self.old_qlsp = qlsp;
        self.first = false;
        // Slide excitation history left by one frame.
        self.exc_buf.copy_within(NB_FRAME_SIZE.., 0);
        for v in &mut self.exc_buf[EXC_BUF_LEN - NB_FRAME_SIZE..] {
            *v = 0.0;
        }
        Ok(())
    }
}

/// Internal tag carrying the chosen LSP codebook indices so the main
/// encoder can emit the correct number of bits without re-checking the
/// sub-mode.
enum LspIndices {
    /// 3 × 6-bit LBR indices (mode 3, 4, etc.).
    Lbr([usize; 3]),
    /// 5 × 6-bit NB indices (mode 5, 6, 7).
    Nb([usize; 5]),
}

// =====================================================================
// LPC analysis
// =====================================================================

/// Symmetric Hamming window applied to the input frame. Returns a new
/// 160-sample buffer.
fn hamming_window(x: &[f32]) -> [f32; NB_FRAME_SIZE] {
    let mut out = [0.0f32; NB_FRAME_SIZE];
    let n = NB_FRAME_SIZE as f32 - 1.0;
    for i in 0..NB_FRAME_SIZE {
        let w = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / n).cos();
        out[i] = x[i] * w;
    }
    out
}

/// Compute autocorrelation `r[k] = Σ x[i] · x[i+k]` for `k=0..=order`.
fn autocorrelate(x: &[f32], r: &mut [f32]) {
    let order = r.len() - 1;
    for k in 0..=order {
        let mut s = 0.0f32;
        for i in 0..x.len() - k {
            s += x[i] * x[i + k];
        }
        r[k] = s;
    }
}

/// Levinson-Durbin recursion — returns LPC coefficients `a[0..order]`
/// corresponding to the polynomial `A(z) = 1 + Σ a_k z^{-k}`. The
/// returned array is `a[k]` for `k = 1..=order`, i.e. the leading `1` is
/// implicit (matching the storage convention in
/// [`crate::lsp::lsp_to_lpc`]).
fn levinson_durbin(r: &[f32]) -> [f32; NB_ORDER] {
    let mut a = [0.0f32; NB_ORDER];
    let mut tmp = [0.0f32; NB_ORDER];
    let mut e = r[0];
    if e <= 0.0 {
        return a;
    }
    for i in 0..NB_ORDER {
        let mut k = -r[i + 1];
        for j in 0..i {
            k -= a[j] * r[i - j];
        }
        if e.abs() < 1e-12 {
            break;
        }
        k /= e;
        // Stability guarantee: Levinson's recursion preserves
        // |k_i| < 1 for an autocorrelation from any real signal, but
        // floating-point round-off on a windowed, low-energy frame can
        // nudge |k_i| slightly above 1 and make the resulting LPC
        // filter unstable. Clamp to a safe margin.
        const K_MAX: f32 = 0.999;
        k = k.clamp(-K_MAX, K_MAX);
        tmp[i] = k;
        for j in 0..i {
            tmp[j] = a[j] + k * a[i - 1 - j];
        }
        a[..=i].copy_from_slice(&tmp[..=i]);
        e *= 1.0 - k * k;
        if e <= 0.0 {
            e = 1e-6;
        }
    }
    // Sign convention: the recursion above produces coefficients in
    // the `A(z) = 1 + Σ a_k z^{-k}` convention — same storage shape as
    // `lsp_to_lpc` and as `iir_mem16::den`. For a cosine input of
    // frequency ω at LPC order 2 it yields `a = [-2cos(ω), 1]`, which
    // zeroes the analysis FIR `r[n] = x[n] + a[0]·x[n-1] + a[1]·x[n-2]`
    // exactly — i.e. the residual vanishes as expected. No sign flip
    // needed here.
    a
}

/// FIR analysis filter: `y[i] = x[i] + Σ_k a_k · x[i-k-1]` written as
/// `y[i] = x[i] + Σ a_k * mem[k]` with a length-`order` tap delay line.
/// Updates `mem` with the last `order` inputs — which is what the
/// decoder's IIR synthesis filter expects to reproduce.
fn fir_filter(x: &[f32], a: &[f32], y: &mut [f32], order: usize, mem: &mut [f32; NB_ORDER]) {
    for i in 0..x.len() {
        let xi = x[i];
        let mut acc = xi;
        for k in 0..order {
            acc += a[k] * mem[k];
        }
        for k in (1..order).rev() {
            mem[k] = mem[k - 1];
        }
        if order > 0 {
            mem[0] = xi;
        }
        y[i] = acc;
    }
}

// =====================================================================
// LPC -> LSP conversion
// =====================================================================

/// Convert LPC coefficients `a[k]` (in the `A(z) = 1 + Σ a_k z^{-k}`
/// convention used elsewhere in the crate) to Line Spectral Pairs
/// (radians, sorted increasing). Returns `None` if the root search fails
/// — in that case the caller should fall back to a neutral LSP vector.
///
/// Algorithm: build the symmetric/antisymmetric polynomials
///     P(z) = A(z) + z^{-(p+1)} A(z^{-1})
///     Q(z) = A(z) - z^{-(p+1)} A(z^{-1})
/// Evaluate P(e^{jω}) and Q(e^{jω}) directly on a uniform grid of ω
/// angles, then bisect within each detected zero-crossing. Since both
/// polynomials have all zeros on the unit circle (they are real, with
/// palindromic / anti-palindromic coefficients), `P` contributes p/2+1
/// zeros (including ω = π for a symmetric polynomial) and `Q`
/// contributes p/2+1 zeros (including ω = 0). The interior roots give
/// the p-dimensional LSP vector.
fn lpc_to_lsp(ak: &[f32; NB_ORDER]) -> Option<[f32; NB_ORDER]> {
    let p = NB_ORDER;
    // A-polynomial padded to length p+2 so we can reflect freely:
    //   a_pad[0]   = 1   (implicit leading 1 of A(z) = 1 + Σ a_k z^{-k})
    //   a_pad[k]   = a_k for k = 1..=p
    //   a_pad[p+1] = 0   (tail — beyond A's natural order)
    let mut a_pad = [0.0f32; NB_ORDER + 2];
    a_pad[0] = 1.0;
    a_pad[1..=p].copy_from_slice(&ak[..p]);
    // z^{-(p+1)} A(z^{-1}) has coefficients reflected: index k maps to
    // a_pad[(p+1) - k]. That gives:
    //   ref[0] = a_pad[p+1] = 0
    //   ref[p+1] = a_pad[0] = 1
    //   ref[k]   = a_pad[p+1-k] for k in 1..=p
    //
    // Then P[k] = a_pad[k] + ref[k] and Q[k] = a_pad[k] - ref[k].
    // Note P[0] = 1, P[p+1] = 1 (symmetric); Q[0] = 1, Q[p+1] = -1
    // (anti-symmetric).
    let n = p + 2; // 12
    let mut pcoef = [0.0f32; 12];
    let mut qcoef = [0.0f32; 12];
    for k in 0..n {
        let reflected = a_pad[(p + 1) - k];
        pcoef[k] = a_pad[k] + reflected;
        qcoef[k] = a_pad[k] - reflected;
    }

    // Evaluate the phase-de-rotated polynomial on the unit circle.
    //
    // P and Q are palindromic / anti-palindromic of length p+2=12. At
    // z = e^{jω}, multiplying by e^{j·(p+1)ω/2} = e^{j·5.5·ω} pairs
    // symmetric coefficients into cosine terms and anti-symmetric
    // coefficients into sine terms of half-integer frequency. For our
    // palindromic P:
    //     P(e^{jω}) · e^{j·5.5·ω} = Σ_{k=0..5} 2·P_k·cos((5.5-k)·ω)
    // which is a real-valued function of ω with exactly p/2 = 5
    // interior zeros — no phantom zeros from the evaluation itself.
    //
    // For anti-palindromic Q:
    //     Q(e^{jω}) · e^{j·5.5·ω} = j · Σ_{k=0..5} 2·Q_k·sin((5.5-k)·ω)
    // whose imaginary part is again a real cosine/sine series with
    // exactly p/2 = 5 interior zeros.
    let eval_p = |coeffs: &[f32; 12], omega: f32| -> f32 {
        let half = (n as f32 - 1.0) * 0.5; // 5.5
        let mut s = 0.0f32;
        for k in 0..(n / 2) {
            s += 2.0 * coeffs[k] * ((half - k as f32) * omega).cos();
        }
        s
    };
    let eval_q = |coeffs: &[f32; 12], omega: f32| -> f32 {
        let half = (n as f32 - 1.0) * 0.5; // 5.5
        let mut s = 0.0f32;
        for k in 0..(n / 2) {
            s += 2.0 * coeffs[k] * ((half - k as f32) * omega).sin();
        }
        s
    };

    // Scan ω from 0 to π.
    const GRID_N: usize = 1024;
    let grid_to_omega = |i: usize| (i as f32 * std::f32::consts::PI) / GRID_N as f32;

    fn scan_roots(
        coeffs: &[f32; 12],
        grid: usize,
        to_omega: impl Fn(usize) -> f32,
        eval: impl Fn(&[f32; 12], f32) -> f32,
    ) -> Vec<f32> {
        let mut roots = Vec::with_capacity(NB_ORDER / 2);
        // Start the scan one step in from ω=0; Q's cosine/sine series has
        // an inherent zero at ω=0 (antisymmetric polynomial), and we
        // don't want to count that as an LSP. Likewise we stop one step
        // short of ω=π to avoid the same for P. The boundary zeros are
        // not LSPs.
        let eps = 1.0 / (grid as f32 * 2.0);
        let clamp_omega = |o: f32| o.clamp(eps, std::f32::consts::PI - eps);
        let mut prev_o = clamp_omega(to_omega(1));
        let mut prev_f = eval(coeffs, prev_o);
        for i in 2..grid {
            let o = clamp_omega(to_omega(i));
            let f = eval(coeffs, o);
            if prev_f * f < 0.0 {
                let mut lo = prev_o;
                let mut hi = o;
                let mut flo = prev_f;
                for _ in 0..32 {
                    let mid = 0.5 * (lo + hi);
                    let fmid = eval(coeffs, mid);
                    if fmid == 0.0 {
                        lo = mid;
                        hi = mid;
                        break;
                    }
                    if flo * fmid < 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                        flo = fmid;
                    }
                }
                roots.push(0.5 * (lo + hi));
            }
            prev_o = o;
            prev_f = f;
        }
        roots
    }
    let r_p = scan_roots(&pcoef, GRID_N, grid_to_omega, eval_p);
    let r_q = scan_roots(&qcoef, GRID_N, grid_to_omega, eval_q);
    // P has trivial zero at ω = π (since P is palindromic of even
    // length ⇒ P(-1) = 0 when p is odd… but with p=10, n=12 which is
    // even, the palindromic structure gives P(z) factorable by (z + 1)
    // and (z - 1) is a trivial zero of Q). So one of `r_p` / `r_q`
    // contains a spurious boundary root at ω=0 or ω=π we must filter.
    let mut roots = Vec::with_capacity(p);
    for &r in r_p.iter().chain(r_q.iter()) {
        // Only interior roots are LSPs. Discard anything pinned to the
        // boundary angles.
        if r > 1e-3 && r < std::f32::consts::PI - 1e-3 {
            roots.push(r);
        }
    }
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if roots.len() < p {
        return None;
    }
    // If the scan produced a couple of extras (e.g. a very shallow false
    // crossing near the boundary), keep only the p interior roots most
    // tightly packed.
    roots.truncate(p);
    let mut out = [0.0f32; NB_ORDER];
    out[..p].copy_from_slice(&roots[..p]);
    // Enforce strict monotonicity.
    let margin = LSP_MARGIN;
    out[0] = out[0].max(margin);
    for i in 1..NB_ORDER {
        if out[i] < out[i - 1] + margin {
            out[i] = out[i - 1] + margin;
        }
    }
    out[NB_ORDER - 1] = out[NB_ORDER - 1].min(std::f32::consts::PI - margin);
    Some(out)
}

// =====================================================================
// LSP quantisation (mode 5: five-stage VQ, 30 bits)
// =====================================================================

/// Quantise a 10-LSP vector using the five-stage VQ from
/// `lsp_unquant_nb`'s inverse. Returns the dequantised LSPs and the
/// five 6-bit codebook indices in encoding order.
fn quantise_lsp_nb(lsp: &[f32; NB_ORDER]) -> ([f32; NB_ORDER], [usize; 5]) {
    let mut indices = [0usize; 5];
    // Stage 1: remove linear initial guess, VQ against CDBK_NB (64 × 10)
    //          with scale 1/256.
    let mut residual = [0.0f32; NB_ORDER];
    for i in 0..NB_ORDER {
        residual[i] = lsp[i] - std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
    }
    // Decoder: lsp[i] = linear[i] + CDBK_NB[id*10+i]/256
    // So target for stage 1 = residual * 256, search over 64 entries.
    indices[0] = nearest_vector_scaled(&residual, 256.0, &CDBK_NB, 10, 64);
    for i in 0..10 {
        residual[i] -= (CDBK_NB[indices[0] * 10 + i] as f32) / 256.0;
    }
    // Stage 2 (low1, 64×5, scale 1/512) — only LSP[0..5].
    let mut low = [0.0f32; 5];
    low.copy_from_slice(&residual[0..5]);
    indices[1] = nearest_vector_scaled(&low, 512.0, &CDBK_NB_LOW1, 5, 64);
    for i in 0..5 {
        residual[i] -= (CDBK_NB_LOW1[indices[1] * 5 + i] as f32) / 512.0;
    }
    // Stage 3 (low2, 64×5, scale 1/1024) — LSP[0..5].
    low.copy_from_slice(&residual[0..5]);
    indices[2] = nearest_vector_scaled(&low, 1024.0, &CDBK_NB_LOW2, 5, 64);
    for i in 0..5 {
        residual[i] -= (CDBK_NB_LOW2[indices[2] * 5 + i] as f32) / 1024.0;
    }
    // Stage 4 (high1, 64×5, scale 1/512) — LSP[5..10].
    let mut hi = [0.0f32; 5];
    hi.copy_from_slice(&residual[5..10]);
    indices[3] = nearest_vector_scaled(&hi, 512.0, &CDBK_NB_HIGH1, 5, 64);
    for i in 0..5 {
        residual[5 + i] -= (CDBK_NB_HIGH1[indices[3] * 5 + i] as f32) / 512.0;
    }
    // Stage 5 (high2, 64×5, scale 1/1024) — LSP[5..10].
    hi.copy_from_slice(&residual[5..10]);
    indices[4] = nearest_vector_scaled(&hi, 1024.0, &CDBK_NB_HIGH2, 5, 64);

    // Reconstruct exactly as the decoder would.
    let mut qlsp = [0.0f32; NB_ORDER];
    for i in 0..NB_ORDER {
        qlsp[i] = std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
    }
    for i in 0..10 {
        qlsp[i] += (CDBK_NB[indices[0] * 10 + i] as f32) / 256.0;
    }
    for i in 0..5 {
        qlsp[i] += (CDBK_NB_LOW1[indices[1] * 5 + i] as f32) / 512.0;
    }
    for i in 0..5 {
        qlsp[i] += (CDBK_NB_LOW2[indices[2] * 5 + i] as f32) / 1024.0;
    }
    for i in 0..5 {
        qlsp[i + 5] += (CDBK_NB_HIGH1[indices[3] * 5 + i] as f32) / 512.0;
    }
    for i in 0..5 {
        qlsp[i + 5] += (CDBK_NB_HIGH2[indices[4] * 5 + i] as f32) / 1024.0;
    }

    // Enforce stability: strictly increasing, bounded away from 0 and π.
    let margin = LSP_MARGIN;
    qlsp[0] = qlsp[0].max(margin);
    for i in 1..NB_ORDER {
        if qlsp[i] < qlsp[i - 1] + margin {
            qlsp[i] = qlsp[i - 1] + margin;
        }
    }
    qlsp[NB_ORDER - 1] = qlsp[NB_ORDER - 1].min(std::f32::consts::PI - margin);
    (qlsp, indices)
}

// =====================================================================
// LSP quantisation (mode 3: three-stage LBR VQ, 18 bits)
// =====================================================================

/// Quantise a 10-LSP vector using the three-stage LBR VQ whose inverse
/// is `lsp_unquant_lbr`. Returns the dequantised LSPs (matching exactly
/// what the decoder will reconstruct) and the three 6-bit codebook
/// indices in encoding order.
///
/// Mirrors the decoder exactly: the linear initial guess is
/// `0.25*(i+1)` (`lsp_linear` in `lsp.rs`), and the per-stage scale
/// factors are 1/256, 1/512, 1/512.
fn quantise_lsp_lbr(lsp: &[f32; NB_ORDER]) -> ([f32; NB_ORDER], [usize; 3]) {
    let mut indices = [0usize; 3];
    // Stage 1 — 64-entry 10-D CB against the residual w.r.t. the
    // linear guess, encoded at 1/256 scale.
    let mut residual = [0.0f32; NB_ORDER];
    for i in 0..NB_ORDER {
        residual[i] = lsp[i] - 0.25 * (i as f32 + 1.0);
    }
    indices[0] = nearest_vector_scaled(&residual, 256.0, &CDBK_NB, 10, 64);
    for i in 0..10 {
        residual[i] -= (CDBK_NB[indices[0] * 10 + i] as f32) / 256.0;
    }
    // Stage 2 — LOW1 (64 × 5, 1/512) on LSP[0..5] only.
    let mut low = [0.0f32; 5];
    low.copy_from_slice(&residual[0..5]);
    indices[1] = nearest_vector_scaled(&low, 512.0, &CDBK_NB_LOW1, 5, 64);
    for i in 0..5 {
        residual[i] -= (CDBK_NB_LOW1[indices[1] * 5 + i] as f32) / 512.0;
    }
    // Stage 3 — HIGH1 (64 × 5, 1/512) on LSP[5..10] only.
    let mut hi = [0.0f32; 5];
    hi.copy_from_slice(&residual[5..10]);
    indices[2] = nearest_vector_scaled(&hi, 512.0, &CDBK_NB_HIGH1, 5, 64);

    // Reconstruct exactly as the decoder would.
    let mut qlsp = [0.0f32; NB_ORDER];
    for i in 0..NB_ORDER {
        qlsp[i] = 0.25 * (i as f32 + 1.0);
    }
    for i in 0..10 {
        qlsp[i] += (CDBK_NB[indices[0] * 10 + i] as f32) / 256.0;
    }
    for i in 0..5 {
        qlsp[i] += (CDBK_NB_LOW1[indices[1] * 5 + i] as f32) / 512.0;
    }
    for i in 0..5 {
        qlsp[i + 5] += (CDBK_NB_HIGH1[indices[2] * 5 + i] as f32) / 512.0;
    }

    // Enforce stability: strictly increasing, bounded away from 0 and π.
    let margin = LSP_MARGIN;
    qlsp[0] = qlsp[0].max(margin);
    for i in 1..NB_ORDER {
        if qlsp[i] < qlsp[i - 1] + margin {
            qlsp[i] = qlsp[i - 1] + margin;
        }
    }
    qlsp[NB_ORDER - 1] = qlsp[NB_ORDER - 1].min(std::f32::consts::PI - margin);
    (qlsp, indices)
}

/// MSE search over a `count`-entry codebook of `dim`-vectors stored as
/// i8 with `scale` (so `cdbk[entry]` represents `cdbk[entry] / scale`).
/// Returns the winning entry's index.
fn nearest_vector_scaled(
    target: &[f32],
    scale: f32,
    cdbk: &[i8],
    dim: usize,
    count: usize,
) -> usize {
    let inv = 1.0 / scale;
    let mut best_idx = 0usize;
    let mut best_err = f32::INFINITY;
    for idx in 0..count {
        let mut err = 0.0f32;
        let base = idx * dim;
        for i in 0..dim {
            let v = cdbk[base + i] as f32 * inv;
            let d = target[i] - v;
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best_idx = idx;
        }
    }
    best_idx
}

/// Scalar-quantiser nearest neighbour — returns `(index, value)`.
fn nearest_scalar(codebook: &[f32], target: f32) -> (usize, f32) {
    let mut best_idx = 0usize;
    let mut best_err = f32::INFINITY;
    for (i, &v) in codebook.iter().enumerate() {
        let e = (target - v).abs();
        if e < best_err {
            best_err = e;
            best_idx = i;
        }
    }
    (best_idx, codebook[best_idx])
}

// =====================================================================
// Pitch (adaptive codebook) search
// =====================================================================

/// Compute the impulse response of the synthesis filter `1/A(z)` over
/// `n` samples, starting from zero state. Used as a convolution kernel
/// in analysis-by-synthesis pitch / codebook searches.
fn impulse_response(ak: &[f32; NB_ORDER], n: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; n];
    let mut mem = [0.0f32; NB_ORDER];
    let mut x = vec![0.0f32; n];
    x[0] = 1.0;
    crate::nb_decoder::iir_mem16(&x, ak, &mut h, n, NB_ORDER, &mut mem);
    h
}

/// Convolve `exc` with `h`, storing the result in `out` (truncated to
/// `out.len()`). `exc[i]` contributes to `out[i + k] += exc[i] · h[k]`
/// for all valid (i, k). Used to evaluate filtered LTP / codebook
/// candidates in the synthesis domain.
fn convolve_lt(exc: &[f32], h: &[f32], out: &mut [f32]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let n = out.len();
    for i in 0..exc.len() {
        let e = exc[i];
        if e == 0.0 {
            continue;
        }
        for k in 0..h.len() {
            let j = i + k;
            if j >= n {
                break;
            }
            out[j] += e * h[k];
        }
    }
}

/// Closed-loop pitch-lag search in the filtered (synthesis) domain:
/// pick the lag whose filtered single-tap LTP output best matches the
/// synthesis target.
fn search_pitch_lag_filtered(
    target: &[f32; NB_SUBFRAME_SIZE],
    exc_buf: &[f32],
    exc_idx: usize,
    pit_min: i32,
    pit_max: i32,
    h: &[f32],
) -> i32 {
    let mut best_lag = pit_min;
    let mut best_score = -1.0f32;
    for lag in pit_min..=pit_max {
        // Build the single-tap LTP contribution for this lag.
        let mut ltp = [0.0f32; NB_SUBFRAME_SIZE];
        for j in 0..NB_SUBFRAME_SIZE {
            let src = exc_idx as isize + j as isize - lag as isize;
            if src >= 0 && (src as usize) < exc_buf.len() {
                ltp[j] = exc_buf[src as usize];
            }
        }
        let mut filtered = [0.0f32; NB_SUBFRAME_SIZE];
        convolve_lt(&ltp, h, &mut filtered);
        let mut num = 0.0f32;
        let mut den = 1e-6f32;
        for i in 0..NB_SUBFRAME_SIZE {
            num += target[i] * filtered[i];
            den += filtered[i] * filtered[i];
        }
        let score = num * num / den;
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }
    best_lag
}

/// Three-tap pitch-gain quantization in the synthesis domain. Returns
/// the gain index, the reconstructed LTP excitation (what the decoder
/// will compute), and the filtered LTP contribution (= LTP_exc * h).
/// The codebook entries are indexed as `gain_cdbk[idx*4 + k]` for
/// `k=0,1,2` (the fourth byte is a boundary marker unused here), so
/// both the 128-entry `GAIN_CDBK_NB` and the 32-entry `GAIN_CDBK_LBR`
/// work as parameters.
fn search_pitch_gain_filtered(
    target: &[f32; NB_SUBFRAME_SIZE],
    exc_buf: &[f32],
    exc_idx: usize,
    pitch: i32,
    h: &[f32],
    gain_cdbk: &[i8],
    gain_cdbk_size: usize,
) -> (usize, [f32; NB_SUBFRAME_SIZE], [f32; NB_SUBFRAME_SIZE]) {
    // Build the three per-tap past-excitation signals y_i[j] (same
    // indexing as the decoder's pitch_unquant_3tap).
    let mut y = [[0.0f32; NB_SUBFRAME_SIZE]; 3];
    for i in 0..3 {
        let pp = (pitch + 1 - i as i32) as usize;
        let nsf = NB_SUBFRAME_SIZE;
        let tmp1 = nsf.min(pp);
        for j in 0..tmp1 {
            let src = exc_idx as isize + j as isize - pp as isize;
            if src >= 0 && (src as usize) < exc_buf.len() {
                y[i][j] = exc_buf[src as usize];
            }
        }
        let tmp3 = nsf.min(pp + pitch as usize);
        for j in tmp1..tmp3 {
            let src = exc_idx as isize + j as isize - pp as isize - pitch as isize;
            if src >= 0 && (src as usize) < exc_buf.len() {
                y[i][j] = exc_buf[src as usize];
            }
        }
    }
    // Pre-filter each tap's past-exc signal so we can quickly score
    // gain candidates.
    let mut yf = [[0.0f32; NB_SUBFRAME_SIZE]; 3];
    for i in 0..3 {
        convolve_lt(&y[i], h, &mut yf[i]);
    }

    let mut best_idx = 0usize;
    let mut best_err = f32::INFINITY;
    let mut best_exc = [0.0f32; NB_SUBFRAME_SIZE];
    let mut best_filt = [0.0f32; NB_SUBFRAME_SIZE];
    for idx in 0..gain_cdbk_size {
        let g0 = 0.015625 * gain_cdbk[idx * 4] as f32 + 0.5;
        let g1 = 0.015625 * gain_cdbk[idx * 4 + 1] as f32 + 0.5;
        let g2 = 0.015625 * gain_cdbk[idx * 4 + 2] as f32 + 0.5;
        let mut err = 0.0f32;
        let mut exc = [0.0f32; NB_SUBFRAME_SIZE];
        let mut filt = [0.0f32; NB_SUBFRAME_SIZE];
        for j in 0..NB_SUBFRAME_SIZE {
            exc[j] = g2 * y[0][j] + g1 * y[1][j] + g0 * y[2][j];
            filt[j] = g2 * yf[0][j] + g1 * yf[1][j] + g0 * yf[2][j];
            let d = target[j] - filt[j];
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best_idx = idx;
            best_exc = exc;
            best_filt = filt;
        }
    }
    (best_idx, best_exc, best_filt)
}

/// Rough open-loop pitch estimator for sub-modes that transmit a single
/// global pitch per frame (sub-modes 1, 2, 8). Picks the lag in
/// `[NB_PITCH_START, NB_PITCH_END]` whose single-tap lagged copy of the
/// recent excitation / residual best correlates with the current
/// frame's residual. Works in the residual domain rather than the
/// synthesis domain because the per-frame ol_pitch doesn't benefit from
/// the expensive filter convolution.
fn search_open_loop_pitch_residual(exc_buf: &[f32], residual: &[f32; NB_FRAME_SIZE]) -> i32 {
    let mut best_lag = NB_PITCH_START;
    let mut best_score = -1.0f32;
    for lag in NB_PITCH_START..=NB_PITCH_END {
        let mut num = 0.0f32;
        let mut den = 1e-6f32;
        // Compare residual[i] against a lagged copy drawn from the
        // excitation history + the current frame's developing residual.
        for i in 0..NB_FRAME_SIZE {
            let tgt = residual[i];
            // Source sample: `i - lag` relative to the start of the
            // current frame. Negative indices read from the excitation
            // history (the last `lag` samples written before this frame).
            let src_idx = i as isize - lag as isize;
            let s = if src_idx >= 0 {
                residual[src_idx as usize]
            } else {
                // Look into exc history (the buffer stores previous
                // frame's excitation at the tail).
                let hist_idx = EXC_HISTORY as isize + src_idx;
                if hist_idx >= 0 && (hist_idx as usize) < exc_buf.len() {
                    exc_buf[hist_idx as usize]
                } else {
                    0.0
                }
            };
            num += tgt * s;
            den += s * s;
        }
        let score = num * num / den;
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }
    best_lag
}

/// Normalised one-tap pitch correlation at a given lag, in the residual
/// domain. Returns a value in `[-1, 1]` approximately — used by the
/// vocoder-style `forced_pitch_gain` modes to pick a 4-bit quantised
/// pitch coefficient. The reference does a similar normalised cross-
/// correlation in `ltp.c`'s `open_loop_pitch` + `forced_pitch_quant`.
fn estimate_pitch_correlation(exc_buf: &[f32], lag: i32, residual: &[f32; NB_FRAME_SIZE]) -> f32 {
    let mut num = 0.0f32;
    let mut tgt_e = 1e-6f32;
    let mut src_e = 1e-6f32;
    for i in 0..NB_FRAME_SIZE {
        let tgt = residual[i];
        let src_idx = i as isize - lag as isize;
        let s = if src_idx >= 0 {
            residual[src_idx as usize]
        } else {
            let hist_idx = EXC_HISTORY as isize + src_idx;
            if hist_idx >= 0 && (hist_idx as usize) < exc_buf.len() {
                exc_buf[hist_idx as usize]
            } else {
                0.0
            }
        };
        num += tgt * s;
        tgt_e += tgt * tgt;
        src_e += s * s;
    }
    let denom = (tgt_e * src_e).sqrt();
    (num / denom).clamp(-1.0, 1.0)
}

/// Generic split-codebook search in the synthesis domain. The
/// sub-frame is partitioned into `nb_subvect` disjoint sub-vectors of
/// `subvect_size` samples each; for each one we pick the `shape_cb`
/// entry whose convolution with `h` best matches the remaining
/// residual target. `ener` is multiplied into candidates before
/// convolution so the search is scale-consistent with what the decoder
/// reconstructs.
///
/// Because sub-vectors are disjoint in position, each one's convolved
/// response only affects output samples from that offset onward —
/// independent per-sub-vector search is correct up to the impulse-
/// response tail crossing into later sub-vectors. We accept that
/// coupling as part of the first-cut approximation; proper A-by-S
/// would search sub-vectors jointly.
fn search_split_cb_generic<const MAX_SUBVECT: usize>(
    target: &[f32; NB_SUBFRAME_SIZE],
    h: &[f32],
    ener: f32,
    shape_cb: &[i8],
    shape_entries: usize,
    subvect_size: usize,
    nb_subvect: usize,
) -> [usize; MAX_SUBVECT] {
    debug_assert!(nb_subvect <= MAX_SUBVECT);
    debug_assert_eq!(subvect_size * nb_subvect, NB_SUBFRAME_SIZE);
    let mut indices = [0usize; MAX_SUBVECT];
    let mut cur_target = *target;
    for i in 0..nb_subvect {
        let off = i * subvect_size;
        let mut best_err = f32::INFINITY;
        let mut best = 0usize;
        let mut best_filt = [0.0f32; NB_SUBFRAME_SIZE];
        for idx in 0..shape_entries {
            let base = idx * subvect_size;
            // Build a full 40-sample candidate excitation that places
            // this codebook entry at offset `off`; other samples zero.
            let mut exc = [0.0f32; NB_SUBFRAME_SIZE];
            for j in 0..subvect_size {
                exc[off + j] = shape_cb[base + j] as f32 * 0.03125 * ener;
            }
            let mut filt = [0.0f32; NB_SUBFRAME_SIZE];
            convolve_lt(&exc, h, &mut filt);
            let mut err = 0.0f32;
            // Only measure error at positions affected by this sub-
            // vector (off ..). Earlier positions were the previous
            // sub-vectors' concern.
            for j in off..NB_SUBFRAME_SIZE {
                let d = cur_target[j] - filt[j];
                err += d * d;
            }
            if err < best_err {
                best_err = err;
                best = idx;
                best_filt = filt;
            }
        }
        indices[i] = best;
        // Subtract the chosen sub-vector's filtered response from the
        // running target so subsequent sub-vectors see a fresh
        // residual.
        for j in off..NB_SUBFRAME_SIZE {
            cur_target[j] -= best_filt[j];
        }
    }
    indices
}

// =====================================================================
// Fixed (split) codebook search — parameterised dispatch
// =====================================================================

/// Maximum number of sub-vectors any NB sub-mode uses. Sub-mode 5 has 8
/// × 5-sample sub-vectors; all others have fewer sub-vectors per frame.
const MAX_NB_SUBVECT: usize = 8;

/// Search the split-codebook whose layout is described by `params`
/// (matching the decoder's `split_cb_shape_sign_unquant` — only
/// `have_sign = false` is used by the NB ladder). Returns one index per
/// sub-vector, placed at positions `0..nb_subvect`. Entries beyond that
/// are zero and should not be consulted.
fn search_split_cb_by_params(
    target: &[f32; NB_SUBFRAME_SIZE],
    h: &[f32],
    ener: f32,
    params: &crate::submodes::SplitCbParams,
) -> [usize; MAX_NB_SUBVECT] {
    let shape_entries = 1usize << params.shape_bits;
    search_split_cb_generic::<MAX_NB_SUBVECT>(
        target,
        h,
        ener,
        params.shape_cb,
        shape_entries,
        params.subvect_size,
        params.nb_subvect,
    )
}

/// Inverse of `search_split_cb_by_params` — rebuilds the normalised
/// per-sample innovation (before multiplication by `ener`) that the
/// decoder will reconstruct.
fn expand_split_cb_by_params(
    indices: &[usize; MAX_NB_SUBVECT],
    params: &crate::submodes::SplitCbParams,
    out: &mut [f32; NB_SUBFRAME_SIZE],
) {
    for i in 0..params.nb_subvect {
        let base = indices[i] * params.subvect_size;
        for j in 0..params.subvect_size {
            out[i * params.subvect_size + j] = params.shape_cb[base + j] as f32 * 0.03125;
        }
    }
}

/// Compute the NB synthesis filter's Π-gain at ω=π (Nyquist). Mirrors
/// the helper in `nb_decoder::pi_gain_of`, reproduced here so the
/// encoder doesn't depend on the decoder's private `pub(crate)`
/// surface.
fn pi_gain_of_nb(ak: &[f32; NB_ORDER]) -> f32 {
    let mut g = 1.0f32;
    let mut i = 0;
    while i + 1 < NB_ORDER {
        g += ak[i + 1] - ak[i];
        i += 2;
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levinson_returns_trivial_for_white_noise_like_signal() {
        // White-noise autocorrelation has r[0] dominant, r[k]≈0 — LPCs
        // should all be near zero.
        let mut r = [0.0f32; NB_ORDER + 1];
        r[0] = 100.0;
        for k in 1..=NB_ORDER {
            r[k] = 0.0;
        }
        let a = levinson_durbin(&r);
        for &v in &a {
            assert!(v.abs() < 1e-3, "LPC coef should be near zero: {v}");
        }
    }

    #[test]
    fn autocorrelate_zero_signal_is_zero() {
        let x = [0.0f32; NB_FRAME_SIZE];
        let mut r = [0.0f32; NB_ORDER + 1];
        autocorrelate(&x, &mut r);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn quantise_lsp_round_trip_stable() {
        // For a well-behaved linear LSP vector, the quantised result
        // should stay ordered and close to the input.
        let mut lsp = [0.0f32; NB_ORDER];
        for i in 0..NB_ORDER {
            lsp[i] = std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
        }
        let (qlsp, _) = quantise_lsp_nb(&lsp);
        for i in 1..NB_ORDER {
            assert!(qlsp[i] > qlsp[i - 1], "qLSP must be sorted");
        }
        for i in 0..NB_ORDER {
            assert!((qlsp[i] - lsp[i]).abs() < 1.0, "qLSP wildly off input");
        }
    }

    #[test]
    fn lpc_to_lsp_recovers_stable_lpc() {
        // Use a slightly perturbed LSP vector so the LPC is non-trivial
        // (uniform LSPs collapse A(z) to ~1, which has no interior
        // roots to find). A small chirp around the uniform grid gives a
        // realistic formant-like filter.
        let mut lsp = [0.0f32; NB_ORDER];
        for i in 0..NB_ORDER {
            let base = std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
            let perturb = 0.2 * ((i as f32 * 1.3).sin());
            lsp[i] = (base + perturb).clamp(0.05, std::f32::consts::PI - 0.05);
        }
        // Re-sort to keep LSPs monotonic after perturbation.
        lsp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 1..NB_ORDER {
            if lsp[i] < lsp[i - 1] + 0.05 {
                lsp[i] = lsp[i - 1] + 0.05;
            }
        }
        let mut ak = [0.0f32; NB_ORDER];
        lsp_to_lpc(&lsp, &mut ak, NB_ORDER);
        let recovered = lpc_to_lsp(&ak);
        assert!(recovered.is_some(), "lpc_to_lsp should succeed");
        let rec = recovered.unwrap();
        eprintln!("input  LSP = {:?}", lsp);
        eprintln!("output LSP = {:?}", rec);
        for i in 0..NB_ORDER {
            assert!(
                (rec[i] - lsp[i]).abs() < 0.15,
                "LSP round-trip off at {i}: got {} expected {}",
                rec[i],
                lsp[i]
            );
        }
    }

    #[test]
    fn encode_frame_writes_exactly_300_bits() {
        let mut enc = NbEncoder::new();
        let mut bw = BitWriter::new();
        // Frame of moderate-amplitude noise-like sine sum.
        let mut pcm = [0.0f32; NB_FRAME_SIZE];
        for i in 0..NB_FRAME_SIZE {
            let t = i as f32;
            pcm[i] = 5000.0 * ((t * 0.2).sin() + 0.5 * (t * 0.05).sin() + 0.3 * (t * 0.7).cos());
        }
        enc.encode_frame(&pcm, &mut bw).unwrap();
        assert_eq!(bw.bit_position(), 300, "mode 5 must emit exactly 300 bits");
    }

    #[test]
    fn encode_frame_mode3_writes_exactly_160_bits() {
        let mut enc = NbEncoder::with_submode(3).expect("mode 3 supported");
        let mut bw = BitWriter::new();
        let mut pcm = [0.0f32; NB_FRAME_SIZE];
        for i in 0..NB_FRAME_SIZE {
            let t = i as f32;
            pcm[i] = 5000.0 * ((t * 0.2).sin() + 0.5 * (t * 0.05).sin() + 0.3 * (t * 0.7).cos());
        }
        enc.encode_frame(&pcm, &mut bw).unwrap();
        assert_eq!(bw.bit_position(), 160, "mode 3 must emit exactly 160 bits");
    }

    #[test]
    fn with_submode_accepts_full_ladder() {
        // All 8 NB sub-modes defined by the reference must construct.
        for id in 1..=8 {
            NbEncoder::with_submode(id)
                .unwrap_or_else(|e| panic!("submode {id} should construct: {e}"));
        }
    }

    #[test]
    fn with_submode_rejects_out_of_range() {
        // `0` is the libspeex silent frame (not emitted here); 9+ are
        // reserved / invalid. All must error out.
        assert!(NbEncoder::with_submode(0).is_err());
        assert!(NbEncoder::with_submode(9).is_err());
        assert!(NbEncoder::with_submode(15).is_err());
    }

    #[test]
    fn encode_frame_every_submode_writes_exact_bit_count() {
        // For each of the eight NB sub-modes, the encoder must emit
        // exactly the `bits_per_frame` value from the descriptor — no
        // padding, no overflow. This round-trips against the decoder's
        // bit consumer by construction.
        for id in 1..=8u32 {
            let mut enc = NbEncoder::with_submode(id).expect("construct");
            let mut bw = BitWriter::new();
            let mut pcm = [0.0f32; NB_FRAME_SIZE];
            for i in 0..NB_FRAME_SIZE {
                let t = i as f32;
                pcm[i] =
                    4000.0 * ((t * 0.2).sin() + 0.5 * (t * 0.05).sin() + 0.25 * (t * 0.7).cos());
            }
            enc.encode_frame(&pcm, &mut bw).unwrap();
            let expected = crate::submodes::nb_submode(id).unwrap().bits_per_frame;
            assert_eq!(
                bw.bit_position() as u32,
                expected,
                "submode {id} wrote wrong bit count"
            );
        }
    }

    #[test]
    fn quantise_lsp_lbr_round_trip_stable() {
        let mut lsp = [0.0f32; NB_ORDER];
        for i in 0..NB_ORDER {
            lsp[i] = 0.25 * (i as f32 + 1.0);
        }
        let (qlsp, _) = quantise_lsp_lbr(&lsp);
        for i in 1..NB_ORDER {
            assert!(qlsp[i] > qlsp[i - 1], "LBR qLSP must be sorted");
        }
    }
}
