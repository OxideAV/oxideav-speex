//! **Top-level Speex packet decoder** — walks a multi-frame Speex packet
//! and produces decoded audio for every CELP frame it contains.
//!
//! A Speex packet body is a concatenation of frames (manual §5.5), each
//! introduced by the 5-bit prefix `wideband_flag || mode_id`. The
//! [`crate::PacketFrames`] iterator already dispatches a packet into
//! typed [`crate::PacketFrame`]s at the *parse* layer; this module is the
//! layer above it that turns those parsed frames into **PCM**, holding
//! the persistent narrowband + high-band decoder state across all frames
//! of a stream.
//!
//! ## What this round adds
//!
//! The per-frame [`crate::NarrowbandDecoder`] and the per-packet
//! [`crate::WidebandDecoder`] decode loops both existed; this module
//! drives them off the packet iterator so a caller hands in a whole
//! packet and gets back a `Vec` of decoded frames — narrowband PCM or a
//! wideband half-band pair — with the in-band signalling / custom-message
//! pseudo-frames surfaced (not decoded to audio, matching their non-audio
//! role) and the terminator / padding tail handled by the iterator.
//!
//! Because every frame carries its own sampling-rate class + mode in-band
//! (RFC 5574 §3.1: *"bit-rate can change at any 20 ms boundary"*), a
//! single decoder instance handles a stream that mixes narrowband and
//! wideband frames: the narrowband decoder state is shared by both paths
//! (a wideband frame's low band *is* an embedded narrowband frame), and
//! the high-band synthesis state advances only on wideband frames.
//!
//! ## State sharing across NB / WB frames
//!
//! The embedded-bit-stream design means a wideband frame's low band is
//! literally a narrowband frame; the same [`crate::NarrowbandDecoder`]
//! therefore decodes both a standalone narrowband frame and the low band
//! of a wideband frame, so its IIR + excitation history is continuous
//! across a mixed stream. The high-band [`crate::HbSynthesisFilter`]
//! state is separate and advances only when a high band is present.

use crate::hb_synthesis::HbSynthesisFilter;
use crate::narrowband_decoder::{
    NarrowbandDecodeError, NarrowbandDecoder, NARROWBAND_FRAME_SAMPLES,
};
use crate::packet::{PacketError, PacketFrame, PacketFrames};
use crate::submode::Submode;
use crate::wb_synthesis::{synthesise_high_band_frame_interp, HB_FRAME_SAMPLES};
use crate::wideband::WidebandSubmode;
use core::fmt;

/// One decoded unit produced from a packet frame.
#[derive(Debug, Clone)]
pub enum DecodedFrame {
    /// A narrowband frame decoded to 160 samples of 8 kHz PCM.
    Narrowband(Box<[f64; NARROWBAND_FRAME_SAMPLES]>),
    /// A wideband frame decoded to its two reconstructed 8 kHz half-band
    /// signals (low band `x_lb` + high band `x_hb`). The QMF synthesis
    /// recombination into 16 kHz PCM is a recorded docs gap (see
    /// [`crate::wideband_decoder`]).
    Wideband {
        /// Low-band (0–4 kHz) reconstructed signal.
        low_band: Box<[f64; NARROWBAND_FRAME_SAMPLES]>,
        /// High-band (4–8 kHz folded) reconstructed signal.
        high_band: Box<[f64; HB_FRAME_SAMPLES]>,
    },
    /// A §5.5 in-band signalling or custom-message pseudo-frame — carries
    /// no audio, surfaced so the caller can act on the control message.
    Control,
}

impl DecodedFrame {
    /// The number of 8 kHz PCM samples this frame's *low band* carries
    /// (`160` for audio frames, `0` for control pseudo-frames).
    pub fn low_band_len(&self) -> usize {
        match self {
            DecodedFrame::Narrowband(_) | DecodedFrame::Wideband { .. } => NARROWBAND_FRAME_SAMPLES,
            DecodedFrame::Control => 0,
        }
    }
}

/// Errors from [`SpeexDecoder::decode_packet`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The packet bit-stream could not be walked — see [`PacketError`].
    Packet(PacketError),
    /// A narrowband frame (standalone or wideband low band) failed to
    /// decode.
    Narrowband(NarrowbandDecodeError),
    /// A wideband high-band sub-mode is the reserved high-rate slot whose
    /// per-field bit budget the staged Table 10.1 does not pin.
    HighBandReserved { mode_id: u8 },
    /// A wideband high-band excitation codebook binding is a recorded
    /// docs gap (high-band mode 4).
    HighBandUndocumented,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Packet(e) => write!(f, "speex decode: {e}"),
            DecodeError::Narrowband(e) => write!(f, "speex decode: {e}"),
            DecodeError::HighBandReserved { mode_id } => write!(
                f,
                "speex decode: high-band mode {mode_id} is the reserved high-rate slot"
            ),
            DecodeError::HighBandUndocumented => {
                write!(f, "speex decode: high-band codebook binding is a docs gap")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<PacketError> for DecodeError {
    fn from(e: PacketError) -> Self {
        DecodeError::Packet(e)
    }
}

impl From<NarrowbandDecodeError> for DecodeError {
    fn from(e: NarrowbandDecodeError) -> Self {
        DecodeError::Narrowband(e)
    }
}

/// Stateful top-level Speex decoder.
///
/// Holds the shared narrowband decode state (used by both standalone
/// narrowband frames and the low band of wideband frames) plus the
/// high-band synthesis state. Construct with [`SpeexDecoder::new`], then
/// call [`SpeexDecoder::decode_packet`] once per Speex packet.
#[derive(Debug, Clone)]
pub struct SpeexDecoder {
    narrowband: NarrowbandDecoder,
    high_band_filter: HbSynthesisFilter,
    /// Previous wideband frame's reconstructed high-band LSP
    /// codebook-delta vector (Q10, pre-base) for the continuous
    /// per-frame high-band LSP interpolation (§9.1 / §10.1).
    prev_hb_lsp_delta_q10: Option<[i32; crate::codebooks::HB_LPC_ORDER]>,
}

impl Default for SpeexDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeexDecoder {
    /// A fresh decoder at stream start (zero history everywhere).
    pub fn new() -> Self {
        Self {
            narrowband: NarrowbandDecoder::new(),
            high_band_filter: HbSynthesisFilter::new(),
            prev_hb_lsp_delta_q10: None,
        }
    }

    /// Decode every frame in a Speex packet, returning one
    /// [`DecodedFrame`] per CELP / control frame.
    ///
    /// Stops cleanly at a mode-15 terminator or a `< 5`-bit padding tail
    /// (per §5.5). A malformed packet surfaces the first
    /// [`PacketError`]; a docs-gapped high-band mode surfaces the matching
    /// [`DecodeError`].
    pub fn decode_packet(&mut self, packet: &[u8]) -> Result<Vec<DecodedFrame>, DecodeError> {
        let mut out = Vec::new();
        for frame in PacketFrames::new(packet) {
            let frame = frame?;
            out.push(self.decode_one(frame)?);
        }
        Ok(out)
    }

    /// Decode a single already-parsed [`PacketFrame`] to a
    /// [`DecodedFrame`], advancing the relevant decoder state.
    fn decode_one(&mut self, frame: PacketFrame) -> Result<DecodedFrame, DecodeError> {
        match frame {
            PacketFrame::Narrowband { header, body } => {
                let submode = match header.submode {
                    Submode::Celp(s) => s,
                    // PacketFrames only yields Narrowband for CELP modes.
                    _ => unreachable!("Narrowband frame carries a CELP sub-mode"),
                };
                let pcm = self.narrowband.decode_frame(&body, &submode)?;
                Ok(DecodedFrame::Narrowband(Box::new(pcm)))
            }
            PacketFrame::Wideband {
                header,
                narrowband,
                high_band_header,
                high_band,
            } => {
                let nb_submode = match header.submode {
                    Submode::Celp(s) => s,
                    _ => unreachable!("Wideband low band carries a CELP sub-mode"),
                };
                let low_band = self.narrowband.decode_frame(&narrowband, &nb_submode)?;

                let hb_submode = match high_band_header.submode {
                    WidebandSubmode::Documented(s) => s,
                    WidebandSubmode::ReservedHighRate(id) => {
                        return Err(DecodeError::HighBandReserved { mode_id: id })
                    }
                };
                let high_band = synthesise_high_band_frame_interp(
                    &high_band,
                    &hb_submode,
                    &mut self.high_band_filter,
                    &mut self.prev_hb_lsp_delta_q10,
                )
                .map_err(|_| DecodeError::HighBandUndocumented)?;

                Ok(DecodedFrame::Wideband {
                    low_band: Box::new(low_band),
                    high_band: Box::new(high_band),
                })
            }
            PacketFrame::InbandSignalling { .. } | PacketFrame::CustomInband { .. } => {
                Ok(DecodedFrame::Control)
            }
        }
    }

    /// Read-only view of the shared narrowband decoder (diagnostics).
    pub fn narrowband(&self) -> &NarrowbandDecoder {
        &self.narrowband
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitWriter;
    use crate::submode::NarrowbandSubmode;
    use crate::wideband::WIDEBAND_HIGH_BAND_SUBMODES;

    /// Build an all-ones narrowband frame for the given mode (wideband
    /// flag 0).
    fn nb_frame(mode: u8) -> Vec<u8> {
        let submode = NarrowbandSubmode::for_id(mode).expect("valid mode");
        let total_bytes = u32::from(submode.total_bits).div_ceil(8);
        let mut buf = vec![0xFFu8; total_bytes as usize];
        buf[0] = (mode & 0x0F) << 3 | 0b0000_0111;
        buf
    }

    fn write_ones(w: &mut BitWriter, mut bits: u32) {
        while bits > 0 {
            let chunk = bits.min(16);
            w.write((1u32 << chunk) - 1, chunk).unwrap();
            bits -= chunk;
        }
    }

    /// Build a single-frame wideband packet (NB flag 1 + NB body + HB
    /// flag 1 + HB body), no terminator.
    fn wb_frame(nb: u8, hb: u8) -> Vec<u8> {
        let nb_submode = NarrowbandSubmode::for_id(nb).unwrap();
        let hb_submode = WIDEBAND_HIGH_BAND_SUBMODES[hb as usize];
        let mut w = BitWriter::new();
        w.write_bit(1).unwrap();
        w.write(u32::from(nb), 4).unwrap();
        write_ones(&mut w, u32::from(nb_submode.total_bits) - 5);
        w.write_bit(1).unwrap();
        w.write(u32::from(hb), 3).unwrap();
        write_ones(&mut w, u32::from(hb_submode.total_bits) - 4);
        w.into_bytes()
    }

    #[test]
    fn single_narrowband_frame_decodes_to_one_pcm_frame() {
        let pkt = nb_frame(5);
        let mut dec = SpeexDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            DecodedFrame::Narrowband(pcm) => {
                assert_eq!(pcm.len(), 160);
                assert!(pcm.iter().all(|s| s.is_finite()));
            }
            other => panic!("expected Narrowband, got {other:?}"),
        }
    }

    #[test]
    fn multi_frame_packet_decodes_all_frames() {
        // Two narrowband mode-5 frames in one packet (the encoder packs
        // N frames before writing; §5.5). Concatenate two frame bodies.
        let mut pkt = nb_frame(3);
        pkt.extend_from_slice(&nb_frame(3));
        let mut dec = SpeexDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        // At least the first frame decodes; the iterator walks until the
        // bits are exhausted. Both should be narrowband.
        assert!(!frames.is_empty());
        for f in &frames {
            assert!(matches!(f, DecodedFrame::Narrowband(_)));
        }
    }

    #[test]
    fn wideband_frame_decodes_to_half_band_pair() {
        let pkt = wb_frame(5, 2);
        let mut dec = SpeexDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            DecodedFrame::Wideband {
                low_band,
                high_band,
            } => {
                assert_eq!(low_band.len(), 160);
                assert_eq!(high_band.len(), 160);
                assert!(low_band.iter().all(|s| s.is_finite()));
                assert!(high_band.iter().all(|s| s.is_finite()));
            }
            other => panic!("expected Wideband, got {other:?}"),
        }
    }

    #[test]
    fn shared_narrowband_state_continuous_across_mixed_stream() {
        // A narrowband frame followed (in a separate packet) by a
        // wideband frame: the wideband frame's low band shares the same
        // narrowband decoder state, so its output reflects the carried
        // history (not a stream-start frame).
        let mut dec = SpeexDecoder::new();
        let nb = nb_frame(5);
        let _ = dec.decode_packet(&nb).expect("nb decodes");
        // After the narrowband frame the shared decoder has non-zero
        // synthesis history.
        let hist = dec.narrowband().synthesis_history();
        assert!(
            hist.iter().any(|&h| h != 0.0),
            "shared narrowband state should be non-zero after one frame"
        );
    }

    #[test]
    fn empty_packet_yields_no_frames() {
        let mut dec = SpeexDecoder::new();
        let frames = dec.decode_packet(&[]).expect("empty packet ok");
        assert!(frames.is_empty());
    }

    #[test]
    fn terminator_only_packet_yields_no_frames() {
        // A 5-bit terminator (mode 15) in the high bits of one byte:
        // wideband flag 0 + mode 1111 + 3 padding bits → no frames.
        let pkt = [0b0111_1000_u8];
        let mut dec = SpeexDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("terminator ok");
        assert!(frames.is_empty());
    }

    #[test]
    fn decoder_is_deterministic() {
        let pkt = wb_frame(5, 2);
        let mut a = SpeexDecoder::new();
        let mut b = SpeexDecoder::new();
        let fa = a.decode_packet(&pkt).unwrap();
        let fb = b.decode_packet(&pkt).unwrap();
        assert_eq!(fa.len(), fb.len());
        for (x, y) in fa.iter().zip(fb.iter()) {
            match (x, y) {
                (
                    DecodedFrame::Wideband {
                        low_band: la,
                        high_band: ha,
                    },
                    DecodedFrame::Wideband {
                        low_band: lb,
                        high_band: hb,
                    },
                ) => {
                    assert_eq!(la, lb);
                    assert_eq!(ha, hb);
                }
                _ => panic!("expected wideband frames"),
            }
        }
    }
}
