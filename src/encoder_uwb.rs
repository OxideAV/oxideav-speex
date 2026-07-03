//! Top-level ultra-wideband (32 kHz) encoder — the encode-direction
//! mirror of [`crate::UltraWidebandDecoder`] (round r389 scope).
//!
//! The §2.2 embedded recursion, encode side: the 32 kHz input splits
//! through an **outer** QMF analysis bank into two 16 kHz half-bands;
//! the low half (0–8 kHz) is encoded by the embedded r385
//! [`WidebandEncoder`] (which itself splits again and embeds the r382
//! narrowband encoder — the recursion bottoms out exactly as the
//! bit-stream nests), and the high half (8–16 kHz) is encoded as a
//! second Table 10.1 high-band layer. Per the RFC 5574 Table 2 pin
//! (see [`crate::quality`]), a conformant stream's second layer is the
//! 36-bit **mode-1** gain-only frame at every quality: a 12-bit LSP
//! MSVQ envelope plus four 5-bit excitation gains — exactly the fields
//! this encoder derives from the high half.
//!
//! ## Pipeline (per 640-sample 32 kHz frame)
//!
//! 1. **Outer QMF split** ([`crate::QmfAnalysis::split_slices`]) →
//!    low half `x_wb[0..320]` + high half `x_uwb[0..320]`, both 16 kHz.
//! 2. **Low half** — [`WidebandEncoder::encode_frame_bodies`] produces
//!    the embedded narrowband + first-high-band bodies (shared state,
//!    continuous across frames).
//! 3. **High half envelope** — order-8 LPC over the staged 200-sample
//!    analysis window aligned to the frame end (§10.1's *"very similar
//!    to narrowband"* at the only staged window length; the exact
//!    ultra-wideband analysis geometry is not pinned, so this is a
//!    documented encoder-side engineering choice, same functional
//!    posture as r382/r385) → LSP → Q10 − base → the same 2-stage
//!    12-bit MSVQ the wideband high band uses.
//! 4. **High half gains** — per 80-sample sub-frame (four per frame,
//!    pinned by the mode-1 bit budget): the order-8 prediction residual
//!    RMS, quantised through the staged 32-level 5-bit folded-gain
//!    grid ([`crate::gain_reconstruction::quantise_hb_exc_gain`]).
//! 5. **Pack** — the embedded wideband frame first (narrowband prefix
//!    with the wideband flag, Table 9.1 body, first high-band frame),
//!    then the second high-band frame through the same
//!    [`crate::write_high_band_frame`] writer; packets close with the
//!    §5.5 mode-15 terminator.
//!
//! ## Fidelity
//!
//! Functional, not bit-exact — the same posture as the narrowband and
//! wideband encoders. The decoder-side mode-1 folded-excitation source
//! is the recorded docs gap (#170), so the gains this encoder writes
//! describe the true 8–16 kHz energy envelope even though the current
//! decoder reconstructs that band as zero. Supported second-layer
//! modes: 0 (silence) and 1 (gain-only). The excitation-VQ modes 2..=4
//! have no pinned sub-frame geometry at the 16 kHz half-band and are
//! rejected, as are the reserved IDs 5..=7.

use crate::bitreader::BitWriter;
use crate::codebooks::HB_LPC_ORDER;
use crate::encoder_wb::{WbEncodeError, WidebandEncoder, WidebandFrameBodies};
use crate::frame::{FrameError, NarrowbandFrameHeader};
use crate::gain_reconstruction::quantise_hb_exc_gain;
use crate::hb_encode::write_high_band_frame;
use crate::hb_lsp::{pack_hb_lsp_index, quantise_q10 as quantise_hb_lsp_q10, reconstruct_q10};
use crate::lpc_analysis::analyse_hb;
use crate::lpc_to_lsp::hb_lpc_to_lsp;
use crate::lsp_base::hb_lsp_base_q10;
use crate::lsp_to_lpc::{lpc_from_hb_lsp_delta_q10, radians_to_lsp_q10};
use crate::narrowband_decoder::saturate_i16;
use crate::qmf::{QmfAnalysis, QMF_WIDEBAND_FRAME};
use crate::quality::uwb_modes_for_quality;
use crate::submode::NarrowbandSubmode;
use crate::uwb_decoder::{UWB_FRAME_SAMPLES, UWB_HALF_BAND_FRAME, UWB_HB_SUBFRAMES};
use crate::wideband::{HighBandSubFrameIndices, WidebandHighBandBody, WidebandHighBandSubmode};

/// Samples per second-layer sub-frame: 320 half-band samples over the
/// four gain slots the mode-1 budget pins.
pub const UWB_HB_SUBFRAME_SAMPLES: usize = UWB_HALF_BAND_FRAME / UWB_HB_SUBFRAMES;

/// Length of the staged LPC analysis window (the only staged length),
/// aligned to the frame end for the second-layer envelope.
const UWB_ANALYSIS_WINDOW: usize = 200;

/// Errors from [`UltraWidebandEncoder::encode_frame`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UwbEncodeError {
    /// The embedded wideband encode failed.
    Wideband(WbEncodeError),
    /// The requested second-layer mode is not a documented Table 10.1
    /// column (5..=7 reserved, ≥ 8 out of the 3-bit range).
    UnknownUwbMode(u8),
    /// The requested second-layer mode is an excitation-VQ column whose
    /// sub-frame geometry at the 16 kHz half-band is not pinned
    /// (modes 2..=4) — the encode-side face of
    /// [`crate::UwbDecodeError::UwbLayerUndocumented`].
    UndocumentedUwbMode(u8),
    /// Frame packing failed (should not happen for documented modes).
    Pack(FrameError),
}

impl core::fmt::Display for UwbEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            UwbEncodeError::Wideband(e) => write!(f, "ultra-wideband encode: {e}"),
            UwbEncodeError::UnknownUwbMode(m) => {
                write!(f, "ultra-wideband encode: second-layer mode {m} unknown")
            }
            UwbEncodeError::UndocumentedUwbMode(m) => write!(
                f,
                "ultra-wideband encode: second-layer mode {m} geometry is a docs gap"
            ),
            UwbEncodeError::Pack(e) => write!(f, "ultra-wideband encode: packing failed: {e}"),
        }
    }
}

impl std::error::Error for UwbEncodeError {}

impl From<WbEncodeError> for UwbEncodeError {
    fn from(e: WbEncodeError) -> Self {
        UwbEncodeError::Wideband(e)
    }
}

impl From<FrameError> for UwbEncodeError {
    fn from(e: FrameError) -> Self {
        UwbEncodeError::Pack(e)
    }
}

/// The encoded layers of one ultra-wideband frame: the embedded
/// wideband bodies plus the second high-band body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UwbFrameBodies {
    /// The embedded wideband layers (narrowband + first high band).
    pub wideband: WidebandFrameBodies,
    /// The second (8–16 kHz) high-band body.
    pub uwb_body: WidebandHighBandBody,
    /// The second layer's sub-mode ID (0 or 1).
    pub uwb_mode: u8,
}

/// Stateful ultra-wideband (32 kHz) encoder. One instance encodes a
/// continuous stream of 640-sample frames, carrying the outer QMF
/// history, the embedded wideband encoder state, and the second-layer
/// analysis state across frames.
#[derive(Debug, Clone)]
pub struct UltraWidebandEncoder {
    /// Outer QMF analysis bank (32 kHz full-rate FIR history).
    outer_qmf: QmfAnalysis,
    /// Embedded wideband encoder (low-half state).
    wideband: WidebandEncoder,
    /// Second-layer analysis (prediction-error) filter input history.
    uwb_analysis_hist: [f64; HB_LPC_ORDER],
}

impl Default for UltraWidebandEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl UltraWidebandEncoder {
    /// A fresh encoder with zero history (stream start).
    pub fn new() -> Self {
        Self {
            outer_qmf: QmfAnalysis::new(),
            wideband: WidebandEncoder::new(),
            uwb_analysis_hist: [0.0; HB_LPC_ORDER],
        }
    }

    /// Encode one 640-sample 32 kHz frame, returning the packed frame
    /// bytes (embedded wideband frame, then the second high-band frame)
    /// — the exact layout [`crate::UltraWidebandDecoder`] walks.
    ///
    /// `uwb_mode` selects the second layer's Table 10.1 column: `0`
    /// (silence) or `1` (gain-only, the RFC-pinned conformant choice).
    pub fn encode_frame(
        &mut self,
        pcm: &[i16; UWB_FRAME_SAMPLES],
        nb_mode: u8,
        hb_mode: u8,
        uwb_mode: u8,
    ) -> Result<Vec<u8>, UwbEncodeError> {
        let bodies = self.encode_frame_bodies(pcm, nb_mode, hb_mode, uwb_mode)?;
        let mut writer = BitWriter::new();
        write_uwb_frame(&mut writer, &bodies)?;
        Ok(writer.into_bytes())
    }

    /// Encode a whole **packet** of consecutive frames: each frame packs
    /// as the embedded layout, closed with the §5.5 mode-15 terminator
    /// and zero-padded to the byte boundary — directly consumable by
    /// [`crate::UltraWidebandDecoder::decode_packet`].
    pub fn encode_packet(
        &mut self,
        frames: &[[i16; UWB_FRAME_SAMPLES]],
        nb_mode: u8,
        hb_mode: u8,
        uwb_mode: u8,
    ) -> Result<Vec<u8>, UwbEncodeError> {
        let mut writer = BitWriter::new();
        for pcm in frames {
            let bodies = self.encode_frame_bodies(pcm, nb_mode, hb_mode, uwb_mode)?;
            write_uwb_frame(&mut writer, &bodies)?;
        }
        crate::nb_encode::write_packet_terminator(&mut writer).map_err(UwbEncodeError::Pack)?;
        Ok(writer.into_bytes())
    }

    /// Encode a packet at a §2.1 quality setting, using the
    /// [`crate::quality`] ladders for all three layers.
    ///
    /// The narrowband encode gaps constrain the usable range: qualities
    /// 0, 9 and 10 select narrowband modes 1 / 7 (undocumented
    /// innovation) or high-band mode 4 (docs-gapped codebook), which the
    /// embedded encoders reject; qualities 1..=8 encode.
    pub fn encode_packet_quality(
        &mut self,
        frames: &[[i16; UWB_FRAME_SAMPLES]],
        quality: u8,
    ) -> Result<Vec<u8>, UwbEncodeError> {
        let modes =
            uwb_modes_for_quality(quality).ok_or(UwbEncodeError::UnknownUwbMode(quality))?;
        self.encode_packet(frames, modes.nb_mode, modes.hb_mode, modes.uwb_hb_mode)
    }

    /// Encode one frame, returning the intermediate per-layer bodies
    /// (the quantised indices) instead of packed bytes.
    pub fn encode_frame_bodies(
        &mut self,
        pcm: &[i16; UWB_FRAME_SAMPLES],
        nb_mode: u8,
        hb_mode: u8,
        uwb_mode: u8,
    ) -> Result<UwbFrameBodies, UwbEncodeError> {
        // Validate the second-layer mode up-front (before any state
        // advances).
        match WidebandHighBandSubmode::for_id(uwb_mode) {
            Some(s) if s.excitation_vq_bits > 0 => {
                return Err(UwbEncodeError::UndocumentedUwbMode(uwb_mode))
            }
            Some(_) => {}
            None => return Err(UwbEncodeError::UnknownUwbMode(uwb_mode)),
        }

        // --- Outer QMF split: 32 kHz → two 16 kHz half-bands. ---
        let mut input = [0.0f64; UWB_FRAME_SAMPLES];
        for (slot, &s) in input.iter_mut().zip(pcm.iter()) {
            *slot = f64::from(s);
        }
        let mut low = [0.0f64; UWB_HALF_BAND_FRAME];
        let mut high = [0.0f64; UWB_HALF_BAND_FRAME];
        self.outer_qmf.split_slices(&input, &mut low, &mut high);

        // --- Low half: the embedded wideband encode. ---
        let mut low_i16 = [0i16; QMF_WIDEBAND_FRAME];
        for (slot, &s) in low_i16.iter_mut().zip(low.iter()) {
            *slot = saturate_i16(s);
        }
        let wideband = self
            .wideband
            .encode_frame_bodies(&low_i16, nb_mode, hb_mode)?;

        // --- High half: the second high-band layer. ---
        let uwb_body = self.encode_uwb_layer(&high, uwb_mode);

        Ok(UwbFrameBodies {
            wideband,
            uwb_body,
            uwb_mode,
        })
    }

    /// Encode the second high-band layer from the 8–16 kHz half-band.
    fn encode_uwb_layer(
        &mut self,
        high: &[f64; UWB_HALF_BAND_FRAME],
        uwb_mode: u8,
    ) -> WidebandHighBandBody {
        if uwb_mode == 0 {
            // Silence: zero-bit body.
            return WidebandHighBandBody {
                lsp_index: 0,
                subframes: [HighBandSubFrameIndices::default(); 4],
            };
        }

        // Envelope: order-8 LPC over the staged 200-sample window,
        // aligned to the frame end (module docs), quantised through the
        // 12-bit two-stage MSVQ.
        let (lsp_index, active_delta) = encode_uwb_envelope(high);
        let lpc = lpc_from_hb_lsp_delta_q10(&active_delta);

        // Gains: per 80-sample sub-frame, the order-8 prediction
        // residual RMS through the 32-level 5-bit folded-gain grid.
        let mut subframes = [HighBandSubFrameIndices::default(); 4];
        for (sf, slot) in subframes.iter_mut().enumerate() {
            let block = &high[sf * UWB_HB_SUBFRAME_SAMPLES..(sf + 1) * UWB_HB_SUBFRAME_SAMPLES];
            let mut energy = 0.0f64;
            for &x in block {
                let mut r = x;
                for (i, &c) in lpc.iter().enumerate() {
                    r -= c * self.uwb_analysis_hist[HB_LPC_ORDER - 1 - i];
                }
                self.uwb_analysis_hist.rotate_left(1);
                self.uwb_analysis_hist[HB_LPC_ORDER - 1] = x;
                energy += r * r;
            }
            let rms = (energy / UWB_HB_SUBFRAME_SAMPLES as f64).sqrt();
            let idx = quantise_hb_exc_gain(rms as f32, 5);
            slot.excitation_gain_index = idx.and_then(|i| i.raw_index()).unwrap_or(0);
            slot.excitation_vq_index = 0;
        }

        WidebandHighBandBody {
            lsp_index,
            subframes,
        }
    }
}

/// Second-layer envelope: order-8 LPC → LSP → Q10 delta → 2-stage MSVQ.
/// Returns `(packed 12-bit index, quantised delta-Q10)`.
fn encode_uwb_envelope(high: &[f64; UWB_HALF_BAND_FRAME]) -> (u16, [i32; HB_LPC_ORDER]) {
    let window = &high[UWB_HALF_BAND_FRAME - UWB_ANALYSIS_WINDOW..];
    let analysed = match analyse_hb(window) {
        Ok(c) => c,
        Err(_) => return (0, [0; HB_LPC_ORDER]),
    };
    let lsp_rad = match hb_lpc_to_lsp(&analysed.a) {
        Ok(r) => r,
        Err(_) => return (0, [0; HB_LPC_ORDER]),
    };
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

/// Write one complete ultra-wideband frame: the embedded wideband frame
/// (narrowband prefix with the wideband flag + Table 9.1 body + first
/// high-band frame), then the second high-band frame.
fn write_uwb_frame(writer: &mut BitWriter, bodies: &UwbFrameBodies) -> Result<(), UwbEncodeError> {
    let nb_submode = NarrowbandSubmode::for_id(bodies.wideband.nb_mode).ok_or(
        UwbEncodeError::Wideband(WbEncodeError::Narrowband(
            crate::encoder_nb::EncodeError::UnknownMode(bodies.wideband.nb_mode),
        )),
    )?;
    let hb_submode = WidebandHighBandSubmode::for_id(bodies.wideband.hb_mode).ok_or(
        UwbEncodeError::Wideband(WbEncodeError::UnknownHbMode(bodies.wideband.hb_mode)),
    )?;
    let uwb_submode = WidebandHighBandSubmode::for_id(bodies.uwb_mode)
        .ok_or(UwbEncodeError::UnknownUwbMode(bodies.uwb_mode))?;

    let header =
        NarrowbandFrameHeader::new(true, nb_submode.mode_id).map_err(UwbEncodeError::Pack)?;
    header.write(writer).map_err(UwbEncodeError::Pack)?;
    crate::nb_encode::write_narrowband_body(writer, &bodies.wideband.nb_body, &nb_submode)
        .map_err(|e| UwbEncodeError::Pack(FrameError::from(e)))?;
    write_high_band_frame(writer, &bodies.wideband.hb_body, &hb_submode)
        .map_err(|e| UwbEncodeError::Pack(FrameError::from(e)))?;
    write_high_band_frame(writer, &bodies.uwb_body, &uwb_submode)
        .map_err(|e| UwbEncodeError::Pack(FrameError::from(e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uwb_decoder::{UltraWidebandDecoder, UwbDecodedFrame};

    /// A 32 kHz test frame with voiced low-band content plus energy in
    /// both high bands (5 kHz for the 4–8 kHz layer at the inner split,
    /// 11 kHz for the 8–16 kHz second layer).
    fn uwb_frame(amp: f64) -> [i16; UWB_FRAME_SAMPLES] {
        let mut f = [0i16; UWB_FRAME_SAMPLES];
        for (n, s) in f.iter_mut().enumerate() {
            let t = n as f64 / 32_000.0;
            let low = (2.0 * std::f64::consts::PI * t * 300.0).sin();
            let mid = 0.5 * (2.0 * std::f64::consts::PI * t * 5_000.0).sin();
            let high = 0.4 * (2.0 * std::f64::consts::PI * t * 11_000.0).sin();
            let v = amp * (low + mid + high);
            *s = v.round().clamp(-32768.0, 32767.0) as i16;
        }
        f
    }

    #[test]
    fn encoded_frame_decodes_through_uwb_decoder() {
        let mut enc = UltraWidebandEncoder::new();
        let mut dec = UltraWidebandDecoder::new();
        let frame = uwb_frame(6000.0);
        let bytes = enc.encode_frame(&frame, 3, 2, 1).expect("encodes");
        let frames = dec.decode_packet(&bytes).expect("decodes");
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            UwbDecodedFrame::Audio(f) => {
                assert_eq!(f.uwb_hb_mode, 1);
                assert_eq!(f.uwb_pcm.len(), 640);
                assert!(f.uwb_pcm.iter().all(|s| s.is_finite()));
                assert!(f.uwb_pcm.iter().any(|&s| s != 0.0));
                // The 11 kHz component reaches the second layer's gain
                // track: at least one reconstructed gain is non-trivial.
                assert!(
                    f.uwb_gains.iter().any(|&g| g > 0.0),
                    "second-layer gains carry the 8–16 kHz envelope"
                );
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn encoded_bodies_reparse_exactly() {
        // The packed bytes must walk back to the bodies the encoder
        // chose — the wire-format contract with the decoder.
        let frame = uwb_frame(5000.0);
        let mut enc_a = UltraWidebandEncoder::new();
        let bodies = enc_a.encode_frame_bodies(&frame, 3, 2, 1).unwrap();
        let mut enc_b = UltraWidebandEncoder::new();
        let bytes = enc_b.encode_frame(&frame, 3, 2, 1).unwrap();

        let mut dec = UltraWidebandDecoder::new();
        let frames = dec.decode_packet(&bytes).unwrap();
        match &frames[0] {
            UwbDecodedFrame::Audio(f) => {
                assert_eq!(f.uwb_hb_body, bodies.uwb_body);
                assert_eq!(f.uwb_hb_mode, bodies.uwb_mode);
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn multi_frame_packet_round_trips() {
        let mut enc = UltraWidebandEncoder::new();
        let mut dec = UltraWidebandDecoder::new();
        let frames = [uwb_frame(4000.0), uwb_frame(6000.0), uwb_frame(2000.0)];
        let bytes = enc.encode_packet(&frames, 3, 2, 1).expect("encodes");
        let decoded = dec.decode_packet(&bytes).expect("decodes");
        assert_eq!(decoded.len(), 3);
        let mut dec2 = UltraWidebandDecoder::new();
        let pcm = dec2.decode_packet_pcm_i16(&bytes).unwrap();
        assert_eq!(pcm.len(), 3 * 640, "three 32 kHz frames");
    }

    #[test]
    fn quality_ladder_frame_sizes_match_staged_rates() {
        // For each encodable quality, one frame + terminator must pack
        // to exactly ceil((bits + 5) / 8) bytes where bits is the
        // quality's staged bits-per-frame total — the wire-budget tie
        // between the quality module and the packers.
        use crate::quality::{uwb_bitrate_bps, FRAMES_PER_SECOND};
        for q in 1..=8u8 {
            let mut enc = UltraWidebandEncoder::new();
            let frames = [uwb_frame(5000.0)];
            let bytes = enc
                .encode_packet_quality(&frames, q)
                .unwrap_or_else(|e| panic!("quality {q}: {e}"));
            let frame_bits = uwb_bitrate_bps(q).unwrap() / FRAMES_PER_SECOND;
            let expected_len = (frame_bits + 5).div_ceil(8) as usize;
            assert_eq!(bytes.len(), expected_len, "quality {q} packet size");

            // And it decodes.
            let mut dec = UltraWidebandDecoder::new();
            let decoded = dec.decode_packet(&bytes).expect("decodes");
            assert_eq!(decoded.len(), 1, "quality {q}");
        }
    }

    #[test]
    fn silence_second_layer_encodes_zero_bit_body() {
        let mut enc = UltraWidebandEncoder::new();
        let frame = uwb_frame(4000.0);
        let bodies = enc.encode_frame_bodies(&frame, 3, 2, 0).unwrap();
        assert_eq!(bodies.uwb_mode, 0);
        assert_eq!(bodies.uwb_body.lsp_index, 0);
        // Round-trips through the decoder.
        let mut enc2 = UltraWidebandEncoder::new();
        let bytes = enc2.encode_frame(&frame, 3, 2, 0).unwrap();
        let mut dec = UltraWidebandDecoder::new();
        let frames = dec.decode_packet(&bytes).unwrap();
        match &frames[0] {
            UwbDecodedFrame::Audio(f) => assert_eq!(f.uwb_hb_mode, 0),
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn silent_input_drives_gains_to_floor() {
        let mut enc = UltraWidebandEncoder::new();
        let frame = [0i16; UWB_FRAME_SAMPLES];
        let bodies = enc.encode_frame_bodies(&frame, 3, 1, 1).unwrap();
        for sf in &bodies.uwb_body.subframes {
            assert_eq!(sf.excitation_gain_index, 0);
        }
    }

    #[test]
    fn vq_and_reserved_second_layer_modes_rejected() {
        let mut enc = UltraWidebandEncoder::new();
        let frame = uwb_frame(4000.0);
        for m in [2u8, 3, 4] {
            assert_eq!(
                enc.encode_frame(&frame, 3, 2, m),
                Err(UwbEncodeError::UndocumentedUwbMode(m)),
                "mode {m}"
            );
        }
        for m in [5u8, 6, 7, 8, 255] {
            assert_eq!(
                enc.encode_frame(&frame, 3, 2, m),
                Err(UwbEncodeError::UnknownUwbMode(m)),
                "mode {m}"
            );
        }
    }

    #[test]
    fn embedded_wideband_error_propagates() {
        let mut enc = UltraWidebandEncoder::new();
        let frame = uwb_frame(4000.0);
        assert!(matches!(
            enc.encode_frame(&frame, 9, 2, 1),
            Err(UwbEncodeError::Wideband(_))
        ));
    }

    #[test]
    fn stationary_input_stabilises_second_layer_envelope() {
        // Same input frame after frame → the quantised second-layer LSP
        // index settles (steady-state analysis window).
        let frame = uwb_frame(5000.0);
        let mut enc = UltraWidebandEncoder::new();
        let b1 = enc.encode_frame_bodies(&frame, 3, 2, 1).unwrap();
        let b2 = enc.encode_frame_bodies(&frame, 3, 2, 1).unwrap();
        let b3 = enc.encode_frame_bodies(&frame, 3, 2, 1).unwrap();
        assert_eq!(b2.uwb_body.lsp_index, b3.uwb_body.lsp_index);
        let _ = b1;
    }
}
