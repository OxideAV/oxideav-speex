//! **Header-driven stream decoder** — binds a parsed Ogg/Speex stream
//! header ([`crate::SpeexHeader`]) to the decode path its `mode` field
//! selects, so a caller can decode a whole stream without hand-picking
//! among [`crate::SpeexDecoder`] and [`crate::UltraWidebandDecoder`]
//! (round r389 scope).
//!
//! The §5.5 frame concatenation is only walkable with the stream's
//! rate class known out-of-band (see [`crate::uwb_decoder`] — after a
//! high-band body, a second high-band prefix and a next frame's
//! narrowband prefix are bit-indistinguishable). The stream header's
//! `mode` field (§7.3) is that context: `0` narrowband, `1` wideband
//! (both served by the mixed-stream [`crate::SpeexDecoder`], since a
//! narrowband/wideband packet is self-describing frame-by-frame via
//! the wideband flag) and `2` ultra-wideband (served by
//! [`crate::UltraWidebandDecoder`], which claims the second high-band
//! layer per frame).

use crate::decoder::{DecodeError, SpeexDecoder};
use crate::header::{SpeexHeader, SPEEX_MODE_ULTRAWIDEBAND};
use crate::uwb_decoder::{UltraWidebandDecoder, UwbDecodeError, UWB_FRAME_SAMPLES};
use core::fmt;

/// Errors from [`SpeexStreamDecoder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamDecodeError {
    /// The header's `mode` field is not one of the three documented
    /// rate classes (§7.3).
    UnknownMode(u32),
    /// A narrowband / wideband packet failed to decode.
    Decode(DecodeError),
    /// An ultra-wideband packet failed to decode.
    Uwb(UwbDecodeError),
}

impl fmt::Display for StreamDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamDecodeError::UnknownMode(m) => {
                write!(f, "speex stream decode: unknown header mode {m}")
            }
            StreamDecodeError::Decode(e) => write!(f, "speex stream decode: {e}"),
            StreamDecodeError::Uwb(e) => write!(f, "speex stream decode: {e}"),
        }
    }
}

impl std::error::Error for StreamDecodeError {}

impl From<DecodeError> for StreamDecodeError {
    fn from(e: DecodeError) -> Self {
        StreamDecodeError::Decode(e)
    }
}

impl From<UwbDecodeError> for StreamDecodeError {
    fn from(e: UwbDecodeError) -> Self {
        StreamDecodeError::Uwb(e)
    }
}

/// One audio frame's decoded mono PCM paired with the code-9 stereo
/// payload seen ahead of it (`None` if the stream carries no intensity
/// message). Returned by
/// [`SpeexStreamDecoder::decode_packet_frames_stereo`].
pub type StereoFrame = (Vec<i16>, Option<u8>);

/// The rate-class-specific decoder behind the dispatch.
#[derive(Debug, Clone)]
enum Inner {
    /// Narrowband / wideband — the mixed-stream packet decoder (frames
    /// self-describe their band layout via the wideband flag).
    NarrowOrWide(Box<SpeexDecoder>),
    /// Ultra-wideband — the fixed three-layer walk.
    Ultra(Box<UltraWidebandDecoder>),
}

/// Stateful whole-stream decoder for one Ogg/Speex logical stream.
///
/// Construct once from the stream's parsed header with
/// [`SpeexStreamDecoder::for_header`], then feed every audio packet to
/// [`SpeexStreamDecoder::decode_packet_pcm_i16`]; the internal
/// per-band state carries across packets.
#[derive(Debug, Clone)]
pub struct SpeexStreamDecoder {
    inner: Inner,
    rate_hz: u32,
    frame_samples: usize,
}

impl SpeexStreamDecoder {
    /// Build the decoder the header's `mode` field calls for.
    ///
    /// Returns [`StreamDecodeError::UnknownMode`] for a mode outside
    /// the documented `0..=2` set — the same set
    /// [`SpeexHeader::is_known_mode`] accepts.
    pub fn for_header(header: &SpeexHeader) -> Result<Self, StreamDecodeError> {
        let rate_hz = header
            .mode_sampling_rate_hz()
            .ok_or(StreamDecodeError::UnknownMode(header.mode))?;
        let (inner, frame_samples) = if header.mode == SPEEX_MODE_ULTRAWIDEBAND {
            (Inner::Ultra(Box::default()), UWB_FRAME_SAMPLES)
        } else {
            (
                Inner::NarrowOrWide(Box::default()),
                // 160 at 8 kHz, 320 at 16 kHz — 20 ms either way.
                rate_hz as usize / 50,
            )
        };
        Ok(Self {
            inner,
            rate_hz,
            frame_samples,
        })
    }

    /// The stream's nominal output sampling rate in Hz (8000 / 16000 /
    /// 32000 per the header's rate class).
    pub fn output_rate_hz(&self) -> u32 {
        self.rate_hz
    }

    /// Samples per 20 ms audio frame at the stream's rate class.
    pub fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    /// Decode one packet to flat `i16` PCM at [`Self::output_rate_hz`],
    /// concatenating the packet's audio frames and skipping §5.5
    /// control pseudo-frames.
    pub fn decode_packet_pcm_i16(&mut self, packet: &[u8]) -> Result<Vec<i16>, StreamDecodeError> {
        match &mut self.inner {
            Inner::NarrowOrWide(d) => Ok(d.decode_packet_pcm_i16(packet)?),
            Inner::Ultra(d) => Ok(d.decode_packet_pcm_i16(packet)?),
        }
    }

    /// Decode a packet to per-audio-frame `(mono PCM, code-9 stereo
    /// payload)` pairs.
    ///
    /// The code-9 intensity-stereo message (Manual §5.5 / Table 5.1)
    /// precedes each audio frame in a stereo stream; this method pairs
    /// each audio frame's mono PCM with the most recent code-9 payload
    /// seen ahead of it (`None` if the stream carries no stereo message).
    /// The mono PCM is identical to what
    /// [`Self::decode_packet_pcm_i16`] concatenates — the stereo law
    /// (see [`crate::StereoDecoder`]) sits above this decode.
    pub fn decode_packet_frames_stereo(
        &mut self,
        packet: &[u8],
    ) -> Result<Vec<StereoFrame>, StreamDecodeError> {
        use crate::decoder::{ControlMessage, DecodedFrame};
        use crate::uwb_decoder::UwbDecodedFrame;

        const STEREO_CODE: u8 = 9;
        let mut out = Vec::new();
        let mut pending: Option<u8> = None;
        match &mut self.inner {
            Inner::NarrowOrWide(d) => {
                for f in d.decode_packet(packet)? {
                    match f {
                        DecodedFrame::Control(ControlMessage::Inband { message, .. })
                            if message.spec.code == STEREO_CODE =>
                        {
                            pending = Some(message.payload as u8);
                        }
                        DecodedFrame::Control(_) => {}
                        audio => {
                            if let Some(pcm) = audio.pcm_i16() {
                                out.push((pcm, pending.take()));
                            }
                        }
                    }
                }
            }
            Inner::Ultra(d) => {
                for f in d.decode_packet(packet)? {
                    match f {
                        UwbDecodedFrame::Control(ControlMessage::Inband { message, .. })
                            if message.spec.code == STEREO_CODE =>
                        {
                            pending = Some(message.payload as u8);
                        }
                        UwbDecodedFrame::Control(_) => {}
                        UwbDecodedFrame::Audio(a) => {
                            out.push((a.uwb_pcm_i16().to_vec(), pending.take()));
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder_nb::{NarrowbandEncoder, NB_FRAME_SAMPLES};
    use crate::encoder_uwb::UltraWidebandEncoder;
    use crate::encoder_wb::WidebandEncoder;
    use crate::header::{SPEEX_MODE_NARROWBAND, SPEEX_MODE_WIDEBAND};
    use crate::qmf::QMF_WIDEBAND_FRAME;

    /// A minimal header for a rate class (only the fields the stream
    /// decoder reads matter here).
    fn header(mode: u32, rate: u32, frame_size: u32) -> SpeexHeader {
        SpeexHeader {
            speex_string: *b"Speex   ",
            speex_version: [0; 20],
            speex_version_id: 1,
            header_size: 80,
            rate,
            mode,
            mode_bitstream_version: 4,
            nb_channels: 1,
            bitrate: u32::MAX,
            frame_size,
            vbr: 0,
            frames_per_packet: 1,
            extra_headers: 0,
            reserved1: 0,
            reserved2: 0,
        }
    }

    fn tone<const N: usize>(freq: f64, rate: f64) -> [i16; N] {
        let mut f = [0i16; N];
        for (n, s) in f.iter_mut().enumerate() {
            let v = 5000.0 * (2.0 * std::f64::consts::PI * n as f64 * freq / rate).sin();
            *s = v.round() as i16;
        }
        f
    }

    #[test]
    fn narrowband_header_drives_8khz_decode() {
        let h = header(SPEEX_MODE_NARROWBAND, 8_000, 160);
        let mut dec = SpeexStreamDecoder::for_header(&h).unwrap();
        assert_eq!(dec.output_rate_hz(), 8_000);
        assert_eq!(dec.frame_samples(), 160);

        let mut enc = NarrowbandEncoder::new();
        let frames = [tone::<NB_FRAME_SAMPLES>(300.0, 8_000.0); 2];
        let pkt = enc.encode_packet(&frames, 3).unwrap();
        let pcm = dec.decode_packet_pcm_i16(&pkt).unwrap();
        assert_eq!(pcm.len(), 2 * 160);
    }

    #[test]
    fn wideband_header_drives_16khz_decode() {
        let h = header(SPEEX_MODE_WIDEBAND, 16_000, 320);
        let mut dec = SpeexStreamDecoder::for_header(&h).unwrap();
        assert_eq!(dec.output_rate_hz(), 16_000);
        assert_eq!(dec.frame_samples(), 320);

        let mut enc = WidebandEncoder::new();
        let frames = [tone::<QMF_WIDEBAND_FRAME>(400.0, 16_000.0); 2];
        let pkt = enc.encode_packet(&frames, 3, 2).unwrap();
        let pcm = dec.decode_packet_pcm_i16(&pkt).unwrap();
        assert_eq!(pcm.len(), 2 * 320);
    }

    #[test]
    fn ultra_wideband_header_drives_32khz_decode() {
        let h = header(SPEEX_MODE_ULTRAWIDEBAND, 32_000, 640);
        let mut dec = SpeexStreamDecoder::for_header(&h).unwrap();
        assert_eq!(dec.output_rate_hz(), 32_000);
        assert_eq!(dec.frame_samples(), 640);

        let mut enc = UltraWidebandEncoder::new();
        let frames = [tone::<{ UWB_FRAME_SAMPLES }>(500.0, 32_000.0); 2];
        let pkt = enc.encode_packet(&frames, 3, 2, 1).unwrap();
        let pcm = dec.decode_packet_pcm_i16(&pkt).unwrap();
        assert_eq!(pcm.len(), 2 * 640);
    }

    #[test]
    fn unknown_header_mode_is_rejected() {
        let h = header(7, 32_000, 640);
        match SpeexStreamDecoder::for_header(&h) {
            Err(StreamDecodeError::UnknownMode(7)) => {}
            other => panic!("expected UnknownMode(7), got {other:?}"),
        }
    }

    #[test]
    fn state_carries_across_packets() {
        // Two consecutive single-frame packets of identical bytes decode
        // to different PCM (live decoder state) — the whole point of a
        // stream-lifetime decoder object.
        let h = header(SPEEX_MODE_ULTRAWIDEBAND, 32_000, 640);
        let mut dec = SpeexStreamDecoder::for_header(&h).unwrap();
        let mut enc = UltraWidebandEncoder::new();
        let frames = [tone::<{ UWB_FRAME_SAMPLES }>(500.0, 32_000.0)];
        let pkt = enc.encode_packet(&frames, 3, 2, 1).unwrap();
        let a = dec.decode_packet_pcm_i16(&pkt).unwrap();
        let b = dec.decode_packet_pcm_i16(&pkt).unwrap();
        assert_eq!(a.len(), b.len());
        assert_ne!(a, b, "decoder state should carry across packets");
    }
}
