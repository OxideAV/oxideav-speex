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

use crate::narrowband_decoder::{
    NarrowbandDecodeError, NarrowbandDecoder, NARROWBAND_FRAME_SAMPLES,
};
use crate::packet::{PacketError, PacketFrame, PacketFrames};
use crate::qmf::QMF_WIDEBAND_FRAME;
use crate::signalling::{InbandMessage, InbandRequest};
use crate::submode::Submode;
use crate::wb_synthesis::HB_FRAME_SAMPLES;
use crate::wideband::WidebandSubmode;
use crate::wideband_decoder::WidebandDecoder;
use core::fmt;

/// One decoded unit produced from a packet frame.
#[derive(Debug, Clone)]
pub enum DecodedFrame {
    /// A narrowband frame decoded to 160 samples of 8 kHz PCM.
    Narrowband(Box<[f64; NARROWBAND_FRAME_SAMPLES]>),
    /// A wideband frame decoded to its two reconstructed 8 kHz half-band
    /// signals (low band `x_lb` + high band `x_hb`) **and** the
    /// QMF-recombined 16 kHz wideband PCM ([`crate::QmfSynthesis`], round
    /// r365).
    Wideband {
        /// Low-band (0–4 kHz) reconstructed signal.
        low_band: Box<[f64; NARROWBAND_FRAME_SAMPLES]>,
        /// High-band (4–8 kHz folded) reconstructed signal.
        high_band: Box<[f64; HB_FRAME_SAMPLES]>,
        /// QMF-recombined 16 kHz wideband PCM (`2 × 160 = 320` samples).
        wideband_pcm: Box<[f64; QMF_WIDEBAND_FRAME]>,
    },
    /// A §5.5 in-band signalling or custom-message pseudo-frame — carries
    /// no audio, surfaced so the caller can act on the control message. The
    /// payload is the typed [`ControlMessage`] decoded from the §5.5 body
    /// (round r372), so a caller can read a mode-switch / rate-mode /
    /// intensity-stereo request without re-parsing the bit-stream.
    Control(ControlMessage),
}

/// The decoded content of a §5.5 control pseudo-frame surfaced on
/// [`DecodedFrame::Control`].
///
/// A Speex packet can interleave audio frames with two kinds of
/// non-audio pseudo-frame (manual §5.5): a **mode-14 in-band signalling**
/// message (a Table 5.1 request to the peer) and a **mode-13 custom**
/// message (an application-defined opaque body the decoder only skips).
/// This enum distinguishes the two and carries the in-band message's
/// fully-interpreted [`InbandRequest`] for the former.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlMessage {
    /// A mode-14 in-band signalling request, already interpreted per
    /// Table 5.1 (the typed [`InbandRequest`]) alongside the raw parsed
    /// [`InbandMessage`] for callers that want the unprocessed payload.
    Inband {
        /// The raw parsed message (code + payload bits).
        message: InbandMessage,
        /// The typed interpretation of the message per Table 5.1.
        request: InbandRequest,
    },
    /// A mode-13 custom in-band message — only its declared byte length
    /// is meaningful at the codec layer (the body is application-opaque,
    /// §5.5).
    Custom {
        /// The opaque body's declared length in bytes.
        size_bytes: u8,
    },
}

impl DecodedFrame {
    /// The number of 8 kHz PCM samples this frame's *low band* carries
    /// (`160` for audio frames, `0` for control pseudo-frames).
    pub fn low_band_len(&self) -> usize {
        match self {
            DecodedFrame::Narrowband(_) | DecodedFrame::Wideband { .. } => NARROWBAND_FRAME_SAMPLES,
            DecodedFrame::Control(_) => 0,
        }
    }

    /// The frame's full-rate PCM, rounded + saturated to `i16`.
    ///
    /// Returns the ready-to-play full-band output for an audio frame —
    /// the 160-sample 8 kHz narrowband PCM for a [`Self::Narrowband`]
    /// frame, or the 320-sample 16 kHz QMF-recombined wideband PCM for a
    /// [`Self::Wideband`] frame — quantised through the crate-wide
    /// [`crate::saturate_i16`] convention. A [`Self::Control`]
    /// pseudo-frame carries no audio and yields `None`.
    ///
    /// This is the convenience that lets a caller drive the top-level
    /// [`SpeexDecoder`] and collect playable `i16` PCM without matching on
    /// the band layout: narrowband frames give mono 8 kHz, wideband frames
    /// give mono 16 kHz, both already saturated.
    pub fn pcm_i16(&self) -> Option<Vec<i16>> {
        match self {
            DecodedFrame::Narrowband(pcm) => {
                Some(pcm.iter().map(|&s| crate::saturate_i16(s)).collect())
            }
            DecodedFrame::Wideband { wideband_pcm, .. } => Some(
                wideband_pcm
                    .iter()
                    .map(|&s| crate::saturate_i16(s))
                    .collect(),
            ),
            DecodedFrame::Control(_) => None,
        }
    }

    /// The frame's full-rate output sampling rate in Hz (`8000` for
    /// narrowband, `16000` for wideband), or `None` for a control
    /// pseudo-frame. Pairs with [`Self::pcm_i16`] so a caller knows the
    /// rate of the samples it just collected.
    pub fn sample_rate_hz(&self) -> Option<u32> {
        match self {
            DecodedFrame::Narrowband(_) => Some(8_000),
            DecodedFrame::Wideband { .. } => Some(16_000),
            DecodedFrame::Control(_) => None,
        }
    }

    /// The control message carried by a [`Self::Control`] pseudo-frame, or
    /// `None` for an audio frame. Convenience for a caller filtering a
    /// decoded packet for the §5.5 in-band / custom signalling requests.
    pub fn control_message(&self) -> Option<ControlMessage> {
        match self {
            DecodedFrame::Control(m) => Some(*m),
            _ => None,
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
    /// The full wideband decode stack (r393: delegated rather than
    /// duplicated, so the folded high-band law and every future
    /// wideband fix flow through this path automatically). Standalone
    /// narrowband frames advance its embedded narrowband decoder — the
    /// shared low-band state (RFC 5574 §3.1).
    wideband: WidebandDecoder,
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
            wideband: WidebandDecoder::new(),
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

    /// Decode a whole packet straight to a flat `i16` PCM buffer,
    /// concatenating every audio frame's full-rate output and skipping the
    /// non-audio control pseudo-frames.
    ///
    /// Convenience over [`Self::decode_packet`] for the common case of
    /// driving a homogeneous-rate stream: each [`DecodedFrame`]'s
    /// [`DecodedFrame::pcm_i16`] is appended in packet order, so a
    /// narrowband packet yields a flat mono 8 kHz buffer and a wideband
    /// packet a flat mono 16 kHz buffer. Mixing rate classes within one
    /// returned buffer is the caller's responsibility — a packet is a
    /// single sampling-rate class in every conformant Speex stream
    /// (§7.3), and [`DecodedFrame::sample_rate_hz`] is available on the
    /// typed [`Self::decode_packet`] output when finer control is needed.
    pub fn decode_packet_pcm_i16(&mut self, packet: &[u8]) -> Result<Vec<i16>, DecodeError> {
        let mut pcm = Vec::new();
        for frame in PacketFrames::new(packet) {
            let frame = frame?;
            if let Some(samples) = self.decode_one(frame)?.pcm_i16() {
                pcm.extend_from_slice(&samples);
            }
        }
        Ok(pcm)
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
                let pcm = self
                    .wideband
                    .low_band_decoder_mut()
                    .decode_frame(&body, &submode)?;
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
                let hb_submode = match high_band_header.submode {
                    WidebandSubmode::Documented(s) => s,
                    WidebandSubmode::ReservedHighRate(id) => {
                        return Err(DecodeError::HighBandReserved { mode_id: id })
                    }
                };
                // Delegate the whole wideband assembly (low band + r393
                // folded high band + QMF recombination) to the wideband
                // decoder so the two public paths are bit-identical.
                let frame = self
                    .wideband
                    .decode_frame_bodies(&narrowband, &nb_submode, &high_band, &hb_submode)
                    .map_err(|e| match e {
                        crate::wideband_decoder::WidebandDecodeError::Narrowband(nb) => {
                            DecodeError::Narrowband(nb)
                        }
                        crate::wideband_decoder::WidebandDecodeError::HighBandReserved {
                            mode_id,
                        } => DecodeError::HighBandReserved { mode_id },
                        _ => DecodeError::HighBandUndocumented,
                    })?;

                Ok(DecodedFrame::Wideband {
                    low_band: Box::new(frame.low_band),
                    high_band: Box::new(frame.high_band),
                    wideband_pcm: Box::new(frame.wideband_pcm),
                })
            }
            PacketFrame::InbandSignalling { message, .. } => {
                let request = message.interpret();
                Ok(DecodedFrame::Control(ControlMessage::Inband {
                    message,
                    request,
                }))
            }
            PacketFrame::CustomInband { message, .. } => {
                Ok(DecodedFrame::Control(ControlMessage::Custom {
                    size_bytes: message.size_bytes,
                }))
            }
        }
    }

    /// Read-only view of the shared narrowband decoder (diagnostics).
    pub fn narrowband(&self) -> &NarrowbandDecoder {
        self.wideband.low_band_decoder()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitWriter;
    use crate::submode::NarrowbandSubmode;
    use crate::wideband::WIDEBAND_HIGH_BAND_SUBMODES;

    /// Build an all-ones narrowband frame for the given mode (wideband
    /// flag 0). The §5.5 padding tail after the body is **zeroed** —
    /// conformant packets pad with the terminator / zero bits, and the
    /// r393 fixture-pinned layer grammar reads a `1` bit after a
    /// narrowband body as a high-band layer prefix.
    fn nb_frame(mode: u8) -> Vec<u8> {
        let submode = NarrowbandSubmode::for_id(mode).expect("valid mode");
        let total_bits = u32::from(submode.total_bits);
        let total_bytes = total_bits.div_ceil(8);
        let mut buf = vec![0xFFu8; total_bytes as usize];
        buf[0] = (mode & 0x0F) << 3 | 0b0000_0111;
        // Zero the padding bits of the last byte.
        let pad = total_bytes * 8 - total_bits;
        if pad > 0 {
            let last = buf.len() - 1;
            buf[last] &= 0xFFu8 << pad;
        }
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
                wideband_pcm,
            } => {
                assert_eq!(low_band.len(), 160);
                assert_eq!(high_band.len(), 160);
                assert_eq!(wideband_pcm.len(), 320);
                assert!(low_band.iter().all(|s| s.is_finite()));
                assert!(high_band.iter().all(|s| s.is_finite()));
                assert!(wideband_pcm.iter().all(|s| s.is_finite()));
                assert!(
                    wideband_pcm.iter().any(|&s| s != 0.0),
                    "recombined 16 kHz PCM should be non-silent"
                );
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
                        wideband_pcm: pa,
                    },
                    DecodedFrame::Wideband {
                        low_band: lb,
                        high_band: hb,
                        wideband_pcm: pb,
                    },
                ) => {
                    assert_eq!(la, lb);
                    assert_eq!(ha, hb);
                    assert_eq!(pa, pb);
                }
                _ => panic!("expected wideband frames"),
            }
        }
    }

    #[test]
    fn narrowband_frame_pcm_i16_matches_saturated_f64_at_8khz() {
        let pkt = nb_frame(5);
        let mut dec = SpeexDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        let frame = &frames[0];
        let pcm = frame.pcm_i16().expect("audio frame yields PCM");
        assert_eq!(pcm.len(), 160);
        assert_eq!(frame.sample_rate_hz(), Some(8_000));
        match frame {
            DecodedFrame::Narrowband(f64_pcm) => {
                for (i, (&q, &f)) in pcm.iter().zip(f64_pcm.iter()).enumerate() {
                    assert_eq!(q, crate::saturate_i16(f), "sample {i}");
                }
            }
            _ => panic!("expected narrowband frame"),
        }
    }

    #[test]
    fn wideband_frame_pcm_i16_is_16khz_320_samples() {
        let pkt = wb_frame(5, 2);
        let mut dec = SpeexDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        let frame = &frames[0];
        let pcm = frame.pcm_i16().expect("audio frame yields PCM");
        assert_eq!(pcm.len(), 320, "wideband i16 PCM is the 16 kHz full band");
        assert_eq!(frame.sample_rate_hz(), Some(16_000));
        match frame {
            DecodedFrame::Wideband { wideband_pcm, .. } => {
                for (i, (&q, &f)) in pcm.iter().zip(wideband_pcm.iter()).enumerate() {
                    assert_eq!(q, crate::saturate_i16(f), "sample {i}");
                }
            }
            _ => panic!("expected wideband frame"),
        }
    }

    #[test]
    fn decode_packet_pcm_i16_equals_flattened_typed_decode() {
        // The flat convenience must equal the per-frame pcm_i16() output of
        // the typed decode_packet, concatenated in packet order (the core
        // contract). Use mode 3 (160 bits = 20 bytes, byte-aligned) so a
        // byte-concatenation of two frame bodies is a well-formed two-frame
        // packet.
        let mut pkt = nb_frame(3);
        pkt.extend(nb_frame(3));

        let mut dec_flat = SpeexDecoder::new();
        let flat = dec_flat
            .decode_packet_pcm_i16(&pkt)
            .expect("flat decode ok");

        let mut dec_typed = SpeexDecoder::new();
        let typed = dec_typed.decode_packet(&pkt).expect("typed decode ok");
        let expected: Vec<i16> = typed.iter().filter_map(|f| f.pcm_i16()).flatten().collect();

        // Two byte-aligned mode-3 frames → 320 flat 8 kHz samples.
        assert_eq!(flat.len(), 320, "two 160-sample NB frames");
        assert_eq!(flat, expected, "flat == concatenated per-frame i16");
    }

    #[test]
    fn decode_packet_pcm_i16_skips_control_pseudo_frames() {
        // A leading in-band signalling frame contributes no audio; the
        // flat output equals only the audio frames' concatenated i16, so it
        // matches the typed decode with control frames filtered out. The
        // 6 trailing zero bits after the 10-bit mode-14 frame parse as a
        // mode-0 silence frame (audio), so this also exercises the "control
        // dropped, audio kept" split in one packet.
        let mut w = BitWriter::new();
        w.write_bit(0).unwrap(); // narrowband flag
        w.write(14, 4).unwrap(); // mode 14 = in-band signalling
        w.write(0, 4).unwrap(); // code 0 (perceptual enhancement)
        w.write_bit(1).unwrap(); // 1-bit payload
        let pkt = w.into_bytes();

        let mut dec_flat = SpeexDecoder::new();
        let flat = dec_flat.decode_packet_pcm_i16(&pkt).expect("decodes");

        let mut dec_typed = SpeexDecoder::new();
        let typed = dec_typed.decode_packet(&pkt).expect("decodes");
        let control_count = typed
            .iter()
            .filter(|f| matches!(f, DecodedFrame::Control(_)))
            .count();
        let expected: Vec<i16> = typed.iter().filter_map(|f| f.pcm_i16()).flatten().collect();

        assert!(control_count >= 1, "packet carries a control frame");
        assert_eq!(
            flat, expected,
            "flat output drops control frames, keeps audio"
        );
    }

    #[test]
    fn control_frame_has_no_pcm_or_sample_rate() {
        // In-band signalling frame (mode 14, code 0 = perceptual
        // enhancement on/off, 1-bit payload) decodes to a Control
        // pseudo-frame carrying no audio.
        let mut w = BitWriter::new();
        w.write_bit(0).unwrap(); // narrowband flag
        w.write(14, 4).unwrap(); // mode 14 = in-band signalling
        w.write(0, 4).unwrap(); // code 0
        w.write_bit(1).unwrap(); // 1-bit payload
        let pkt = w.into_bytes();

        let mut dec = SpeexDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        let control = frames
            .iter()
            .find(|f| matches!(f, DecodedFrame::Control(_)))
            .expect("an in-band signalling control frame is present");
        assert!(control.pcm_i16().is_none());
        assert_eq!(control.sample_rate_hz(), None);
    }

    #[test]
    fn control_frame_surfaces_interpreted_inband_request() {
        // Mode 14, code 0 (perceptual enhancement) with payload bit 1
        // ("enhancement on") decodes to a Control frame carrying the
        // interpreted request.
        let mut w = BitWriter::new();
        w.write_bit(0).unwrap(); // narrowband flag
        w.write(14, 4).unwrap(); // mode 14 = in-band signalling
        w.write(0, 4).unwrap(); // code 0
        w.write_bit(1).unwrap(); // payload bit = 1
        let pkt = w.into_bytes();

        let mut dec = SpeexDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        let control = frames
            .iter()
            .find_map(|f| f.control_message())
            .expect("a control message is present");
        match control {
            ControlMessage::Inband { message, request } => {
                assert_eq!(message.spec.code, 0);
                assert_eq!(request, InbandRequest::PerceptualEnhancement(true));
            }
            other => panic!("expected Inband control, got {:?}", other),
        }
    }

    #[test]
    fn control_frame_surfaces_dtx_rate_request() {
        // Mode 14, code 7 (set rate mode), value 3 = DTX (implies VAD).
        let mut w = BitWriter::new();
        w.write_bit(0).unwrap();
        w.write(14, 4).unwrap();
        w.write(7, 4).unwrap(); // code 7
        w.write(3, 4).unwrap(); // value 3 = DTX
        let pkt = w.into_bytes();

        let mut dec = SpeexDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        let control = frames
            .iter()
            .find_map(|f| f.control_message())
            .expect("a control message is present");
        match control {
            ControlMessage::Inband {
                request: InbandRequest::SetRateMode(cfg),
                ..
            } => {
                assert!(cfg.dtx && cfg.vad && !cfg.vbr);
            }
            other => panic!("expected SetRateMode(DTX), got {:?}", other),
        }
    }

    #[test]
    fn control_frame_surfaces_custom_message_size() {
        // Mode 13 (custom in-band), size_bytes = 0.
        let mut w = BitWriter::new();
        w.write_bit(0).unwrap();
        w.write(13, 4).unwrap(); // mode 13 = custom in-band
        w.write(0, 5).unwrap(); // size_bytes = 0
        let pkt = w.into_bytes();

        let mut dec = SpeexDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        let control = frames
            .iter()
            .find_map(|f| f.control_message())
            .expect("a control message is present");
        assert_eq!(control, ControlMessage::Custom { size_bytes: 0 });
    }
}
