//! Voice-activity detection + discontinuous transmission (VAD/DTX) —
//! the encoder-side silence handling of *The Speex Codec Manual* §2.1
//! (round r389 scope).
//!
//! ## What the staged material pins
//!
//! * **The DTX frame format**: §2.1 — *"Discontinuous transmission is
//!   an addition to VAD/VBR operation … In file-based operation, since
//!   we cannot just stop writing to the file, only 5 bits are used for
//!   such frames (corresponding to 250 bps)."* Five bits is exactly the
//!   Table 9.1 **mode-0** frame (1 wideband bit + 4 mode-ID bits, zero
//!   body), and Table 9.2 names mode 0 *"No transmission (DTX)"* at
//!   250 bps. For the higher rate classes the same degenerate frame is
//!   the mode-0 column of every layer: a wideband DTX frame is
//!   5 + 4 = 9 bits, an ultra-wideband one 5 + 4 + 4 = 13 bits.
//! * **The concept**: §2.1 — VAD *"detects whether the audio being
//!   encoded is speech or silence/background noise"*, and is what
//!   selects the DTX frames in non-VBR operation.
//!
//! ## What is encoder freedom (documented, not fished)
//!
//! The manual pins **no VAD decision algorithm** (the codec-internal
//! detector is unspecified; §2.3 only notes the preprocessor's VAD is
//! "more advanced"). [`EnergyVad`] is therefore a deliberately simple,
//! documented encoder-side policy: a frame is *active* when its RMS
//! crosses a caller-set threshold, with a configurable **hangover**
//! (trailing frames kept active after speech ends, so word endings are
//! not clipped — a standard VAD practice, not a Speex-specific claim).
//! Any conformant decoder plays the resulting stream; only *which*
//! frames get the 5-bit treatment is heuristic. Comfort-noise
//! generation on the decode side ("CNG") is likewise conceptual-only in
//! the staged manual and is not implemented.

use crate::bitreader::BitWriter;
use crate::encoder_nb::{EncodeError, NarrowbandEncoder, NB_FRAME_SAMPLES};
use crate::encoder_uwb::{write_uwb_frame, UltraWidebandEncoder, UwbEncodeError};
use crate::encoder_wb::{WbEncodeError, WidebandEncoder};
use crate::frame::{FrameError, NarrowbandFrameHeader};
use crate::qmf::QMF_WIDEBAND_FRAME;
use crate::submode::NarrowbandSubmode;
use crate::uwb_decoder::UWB_FRAME_SAMPLES;
use crate::wideband::WidebandHighBandSubmode;

/// The narrowband sub-mode DTX frames use — Table 9.2's mode 0
/// (*"No transmission (DTX)"*, 250 bps, 5 bits/frame).
pub const DTX_MODE: u8 = 0;

/// Energy-threshold voice-activity detector with hangover.
///
/// A frame is **active** (speech) when its RMS amplitude reaches
/// `threshold_rms`, or when a recent frame was active and the hangover
/// window has not yet elapsed. See the module docs for the clean-room
/// status of this policy (documented encoder freedom).
#[derive(Debug, Clone)]
pub struct EnergyVad {
    /// RMS activation threshold in `i16` sample units.
    threshold_rms: f64,
    /// Number of trailing frames kept active after the signal drops
    /// below the threshold.
    hangover_frames: u32,
    /// Remaining hangover budget.
    hangover_left: u32,
}

impl EnergyVad {
    /// A detector with the given RMS threshold (in `i16` sample units)
    /// and hangover length in frames.
    pub fn new(threshold_rms: f64, hangover_frames: u32) -> Self {
        Self {
            threshold_rms,
            hangover_frames,
            hangover_left: 0,
        }
    }

    /// Classify one frame, advancing the hangover state.
    pub fn frame_is_active(&mut self, pcm: &[i16]) -> bool {
        let energy: f64 = pcm.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        let rms = (energy / pcm.len().max(1) as f64).sqrt();
        if rms >= self.threshold_rms {
            self.hangover_left = self.hangover_frames;
            true
        } else if self.hangover_left > 0 {
            self.hangover_left -= 1;
            true
        } else {
            false
        }
    }
}

impl NarrowbandEncoder {
    /// Encode a packet with VAD/DTX: active frames encode at `mode`,
    /// inactive frames as the 5-bit mode-0 DTX frame (§2.1 / Table 9.2).
    ///
    /// The packet closes with the §5.5 terminator and is directly
    /// consumable by [`crate::SpeexDecoder::decode_packet`] — DTX
    /// frames decode as silence, so the frame count (and 20 ms timing)
    /// is preserved.
    pub fn encode_packet_dtx(
        &mut self,
        frames: &[[i16; NB_FRAME_SAMPLES]],
        mode: u8,
        vad: &mut EnergyVad,
    ) -> Result<Vec<u8>, EncodeError> {
        let mut writer = BitWriter::new();
        for pcm in frames {
            let frame_mode = if vad.frame_is_active(pcm) {
                mode
            } else {
                DTX_MODE
            };
            let submode = NarrowbandSubmode::for_id(frame_mode)
                .ok_or(EncodeError::UnknownMode(frame_mode))?;
            let body = self.encode_frame_body(pcm, frame_mode)?;
            let header =
                NarrowbandFrameHeader::new(false, submode.mode_id).map_err(EncodeError::Pack)?;
            header.write(&mut writer).map_err(EncodeError::Pack)?;
            crate::nb_encode::write_narrowband_body(&mut writer, &body, &submode)
                .map_err(|e| EncodeError::Pack(FrameError::from(e)))?;
        }
        crate::nb_encode::write_packet_terminator(&mut writer).map_err(EncodeError::Pack)?;
        Ok(writer.into_bytes())
    }
}

impl WidebandEncoder {
    /// Encode a packet with VAD/DTX: active frames encode at
    /// `(nb_mode, hb_mode)`, inactive frames as the 9-bit all-mode-0
    /// wideband DTX frame (narrowband mode 0 + high-band mode 0).
    pub fn encode_packet_dtx(
        &mut self,
        frames: &[[i16; QMF_WIDEBAND_FRAME]],
        nb_mode: u8,
        hb_mode: u8,
        vad: &mut EnergyVad,
    ) -> Result<Vec<u8>, WbEncodeError> {
        let mut writer = BitWriter::new();
        for pcm in frames {
            let (fnb, fhb) = if vad.frame_is_active(pcm) {
                (nb_mode, hb_mode)
            } else {
                (DTX_MODE, 0)
            };
            let bodies = self.encode_frame_bodies(pcm, fnb, fhb)?;
            let nb_submode = NarrowbandSubmode::for_id(fnb)
                .ok_or(WbEncodeError::Narrowband(EncodeError::UnknownMode(fnb)))?;
            let hb_submode =
                WidebandHighBandSubmode::for_id(fhb).ok_or(WbEncodeError::UnknownHbMode(fhb))?;
            let header = NarrowbandFrameHeader::new(true, nb_submode.mode_id)
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
}

impl UltraWidebandEncoder {
    /// Encode a packet with VAD/DTX: active frames encode at
    /// `(nb_mode, hb_mode, uwb_mode)`, inactive frames as the 13-bit
    /// all-mode-0 ultra-wideband DTX frame.
    pub fn encode_packet_dtx(
        &mut self,
        frames: &[[i16; UWB_FRAME_SAMPLES]],
        nb_mode: u8,
        hb_mode: u8,
        uwb_mode: u8,
        vad: &mut EnergyVad,
    ) -> Result<Vec<u8>, UwbEncodeError> {
        let mut writer = BitWriter::new();
        for pcm in frames {
            let (fnb, fhb, fuwb) = if vad.frame_is_active(pcm) {
                (nb_mode, hb_mode, uwb_mode)
            } else {
                (DTX_MODE, 0, 0)
            };
            let bodies = self.encode_frame_bodies(pcm, fnb, fhb, fuwb)?;
            write_uwb_frame(&mut writer, &bodies)?;
        }
        crate::nb_encode::write_packet_terminator(&mut writer).map_err(UwbEncodeError::Pack)?;
        Ok(writer.into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{DecodedFrame, SpeexDecoder};
    use crate::uwb_decoder::UltraWidebandDecoder;

    fn nb_tone(amp: f64) -> [i16; NB_FRAME_SAMPLES] {
        let mut f = [0i16; NB_FRAME_SAMPLES];
        for (n, s) in f.iter_mut().enumerate() {
            let v = amp * (2.0 * std::f64::consts::PI * n as f64 * 300.0 / 8_000.0).sin();
            *s = v.round() as i16;
        }
        f
    }

    #[test]
    fn vad_threshold_and_hangover() {
        let mut vad = EnergyVad::new(100.0, 2);
        let loud = nb_tone(5000.0);
        let quiet = [0i16; NB_FRAME_SAMPLES];
        assert!(!vad.frame_is_active(&quiet), "cold silence is inactive");
        assert!(vad.frame_is_active(&loud), "tone is active");
        // Hangover: the next two silent frames stay active, the third
        // goes inactive.
        assert!(vad.frame_is_active(&quiet), "hangover frame 1");
        assert!(vad.frame_is_active(&quiet), "hangover frame 2");
        assert!(!vad.frame_is_active(&quiet), "hangover exhausted");
    }

    #[test]
    fn nb_dtx_silent_packet_is_five_bits_per_frame() {
        // Four silent frames: 4 × 5 bits + 5-bit terminator = 25 bits →
        // 4 bytes. The §2.1 "only 5 bits … 250 bps" pin.
        let mut enc = NarrowbandEncoder::new();
        let mut vad = EnergyVad::new(100.0, 0);
        let frames = [[0i16; NB_FRAME_SAMPLES]; 4];
        let pkt = enc.encode_packet_dtx(&frames, 5, &mut vad).unwrap();
        assert_eq!(pkt.len(), 4, "4 DTX frames + terminator in 4 bytes");

        // The DTX frames still decode (as silent audio), preserving the
        // frame count.
        let mut dec = SpeexDecoder::new();
        let decoded = dec.decode_packet(&pkt).unwrap();
        assert_eq!(decoded.len(), 4);
        for f in &decoded {
            match f {
                DecodedFrame::Narrowband(pcm) => {
                    assert!(pcm.iter().all(|&s| s == 0.0), "DTX decodes silent")
                }
                other => panic!("expected Narrowband, got {other:?}"),
            }
        }
    }

    #[test]
    fn nb_dtx_mixed_packet_shrinks_but_keeps_frame_count() {
        let loud = nb_tone(6000.0);
        let quiet = [0i16; NB_FRAME_SAMPLES];
        let frames = [loud, quiet, quiet, loud];

        let mut enc_dtx = NarrowbandEncoder::new();
        let mut vad = EnergyVad::new(100.0, 0);
        let dtx_pkt = enc_dtx.encode_packet_dtx(&frames, 5, &mut vad).unwrap();

        let mut enc_all = NarrowbandEncoder::new();
        let full_pkt = enc_all.encode_packet(&frames, 5).unwrap();

        assert!(
            dtx_pkt.len() < full_pkt.len(),
            "DTX packet ({}) smaller than all-active ({})",
            dtx_pkt.len(),
            full_pkt.len()
        );

        let mut dec = SpeexDecoder::new();
        let decoded = dec.decode_packet(&dtx_pkt).unwrap();
        assert_eq!(decoded.len(), 4, "frame count (timing) preserved");
    }

    #[test]
    fn wb_dtx_silent_packet_uses_all_mode_0_layers() {
        // Two silent wideband frames: 2 × (5 + 4) + 5 = 23 bits → 3
        // bytes.
        let mut enc = WidebandEncoder::new();
        let mut vad = EnergyVad::new(100.0, 0);
        let frames = [[0i16; QMF_WIDEBAND_FRAME]; 2];
        let pkt = enc.encode_packet_dtx(&frames, 3, 2, &mut vad).unwrap();
        assert_eq!(pkt.len(), 3);

        let mut dec = SpeexDecoder::new();
        let decoded = dec.decode_packet(&pkt).unwrap();
        assert_eq!(decoded.len(), 2);
        assert!(decoded
            .iter()
            .all(|f| matches!(f, DecodedFrame::Wideband { .. })));
    }

    #[test]
    fn uwb_dtx_silent_packet_uses_all_mode_0_layers() {
        // Two silent UWB frames: 2 × (5 + 4 + 4) + 5 = 31 bits → 4
        // bytes.
        let mut enc = UltraWidebandEncoder::new();
        let mut vad = EnergyVad::new(100.0, 0);
        let frames = [[0i16; UWB_FRAME_SAMPLES]; 2];
        let pkt = enc.encode_packet_dtx(&frames, 3, 2, 1, &mut vad).unwrap();
        assert_eq!(pkt.len(), 4);

        let mut dec = UltraWidebandDecoder::new();
        let pcm = dec.decode_packet_pcm_i16(&pkt).unwrap();
        assert_eq!(pcm.len(), 2 * 640, "two 32 kHz frames of (silent) audio");
        assert!(pcm.iter().all(|&s| s == 0));
    }

    #[test]
    fn uwb_dtx_active_frames_encode_at_requested_modes() {
        let mut loud = [0i16; UWB_FRAME_SAMPLES];
        for (n, s) in loud.iter_mut().enumerate() {
            let t = n as f64 / 32_000.0;
            *s = (6000.0 * (2.0 * std::f64::consts::PI * t * 400.0).sin()).round() as i16;
        }
        let mut enc = UltraWidebandEncoder::new();
        let mut vad = EnergyVad::new(100.0, 0);
        let pkt = enc
            .encode_packet_dtx(&[loud, [0i16; UWB_FRAME_SAMPLES]], 3, 2, 1, &mut vad)
            .unwrap();

        let mut dec = UltraWidebandDecoder::new();
        let decoded = dec.decode_packet(&pkt).unwrap();
        assert_eq!(decoded.len(), 2);
        match &decoded[0] {
            crate::uwb_decoder::UwbDecodedFrame::Audio(f) => {
                assert_eq!(f.uwb_hb_mode, 1, "active frame keeps mode 1");
                assert!(f.uwb_pcm.iter().any(|&s| s != 0.0));
            }
            other => panic!("expected Audio, got {other:?}"),
        }
        match &decoded[1] {
            crate::uwb_decoder::UwbDecodedFrame::Audio(f) => {
                assert_eq!(f.uwb_hb_mode, 0, "silent frame drops to DTX");
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }
}
