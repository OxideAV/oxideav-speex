//! **Rate-agnostic stream encoder** — the encode-side counterpart of
//! [`crate::SpeexStreamDecoder`]. One type covers the three §2.2 rate
//! classes (8 / 16 / 32 kHz), the §2.1 quality knob (0..=10) and
//! intensity stereo, so a caller never hand-picks among
//! [`crate::NarrowbandEncoder`], [`crate::WidebandEncoder`] and
//! [`crate::UltraWidebandEncoder`] or assembles the stereo in-band
//! message itself.
//!
//! Each call to [`SpeexStreamEncoder::encode_frame`] turns one 20 ms
//! frame into one self-contained packet (mode-15 terminated, §5.5),
//! and [`SpeexStreamEncoder::header`] is the matching Table 7.1 stream
//! header (`frames_per_packet = 1`).
//!
//! A stereo encoder folds the two channels to the `(L+R)/2` mono signal
//! and prefixes each frame with the Table 5.1 code-9 intensity-stereo
//! message (17 bits: 1-bit wideband flag `0`, mode `14`, code `9`,
//! 8-bit payload), per `docs/audio/speex/intensity-stereo.md` §1 / §5.

use core::fmt;

use crate::bitreader::BitWriter;
use crate::encoder_nb::{EncodeError, NB_FRAME_SAMPLES, NarrowbandEncoder};
use crate::encoder_uwb::{UltraWidebandEncoder, UwbEncodeError};
use crate::encoder_wb::{WbEncodeError, WidebandEncoder};
use crate::header::SpeexHeader;
use crate::qmf::QMF_WIDEBAND_FRAME;
use crate::quality::{
    FRAMES_PER_SECOND, MAX_QUALITY, nb_mode_for_quality, uwb_bitrate_bps, wb_bitrate_bps,
};
use crate::stereo::{downmix_mean, encode_stereo_payload};
use crate::submode::NarrowbandSubmode;
use crate::uwb_decoder::UWB_FRAME_SAMPLES;

/// Errors from [`SpeexStreamEncoder`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamEncodeError {
    /// The sampling rate is not one of the three Speex rate classes
    /// (8000 / 16000 / 32000 Hz).
    UnsupportedSampleRate(u32),
    /// The quality is outside `0..=10`.
    QualityOutOfRange(u8),
    /// The input slice does not hold exactly one frame
    /// ([`SpeexStreamEncoder::frame_samples`] × channels samples).
    FrameLength {
        /// Samples expected (per-channel frame length × channels).
        expected: usize,
        /// Samples supplied.
        got: usize,
    },
    /// The narrowband encoder failed.
    Narrowband(EncodeError),
    /// The wideband encoder failed.
    Wideband(WbEncodeError),
    /// The ultra-wideband encoder failed.
    UltraWideband(UwbEncodeError),
}

impl fmt::Display for StreamEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamEncodeError::UnsupportedSampleRate(r) => write!(
                f,
                "speex stream encode: sample rate {r} is not a Speex rate class (8000/16000/32000)"
            ),
            StreamEncodeError::QualityOutOfRange(q) => {
                write!(f, "speex stream encode: quality {q} out of range 0..=10")
            }
            StreamEncodeError::FrameLength { expected, got } => write!(
                f,
                "speex stream encode: frame holds {got} samples, expected {expected}"
            ),
            StreamEncodeError::Narrowband(e) => write!(f, "speex stream encode: {e}"),
            StreamEncodeError::Wideband(e) => write!(f, "speex stream encode: {e}"),
            StreamEncodeError::UltraWideband(e) => write!(f, "speex stream encode: {e}"),
        }
    }
}

impl std::error::Error for StreamEncodeError {}

/// The rate-class-specific encoder behind the dispatch.
#[derive(Debug, Clone)]
enum Inner {
    Nb(Box<NarrowbandEncoder>),
    Wb(Box<WidebandEncoder>),
    Uwb(Box<UltraWidebandEncoder>),
}

/// Stateful whole-stream Speex encoder for 8, 16 or 32 kHz input,
/// mono or intensity stereo.
///
/// ```
/// use oxideav_speex::{SpeexStreamDecoder, SpeexStreamEncoder};
///
/// let mut enc = SpeexStreamEncoder::new(16_000, 8).unwrap();
/// let header = enc.header(); // 80-byte stream header via write_bytes()
/// let mut dec = SpeexStreamDecoder::for_header(&header).unwrap();
///
/// let frame = vec![0i16; enc.frame_samples()]; // 320 samples = 20 ms
/// let packet = enc.encode_frame(&frame).unwrap();
/// let pcm = dec.decode_packet_pcm_i16(&packet).unwrap();
/// assert_eq!(pcm.len(), 320);
/// ```
#[derive(Debug, Clone)]
pub struct SpeexStreamEncoder {
    inner: Inner,
    rate_hz: u32,
    quality: u8,
    channels: u16,
}

impl SpeexStreamEncoder {
    /// A mono encoder for `sample_rate` (8000 / 16000 / 32000) at the
    /// §2.1 `quality` (0..=10; 8 is the usual default).
    pub fn new(sample_rate: u32, quality: u8) -> Result<Self, StreamEncodeError> {
        Self::with_channels(sample_rate, quality, 1)
    }

    /// An intensity-stereo encoder: input is interleaved L/R, the stream
    /// carries the `(L+R)/2` mono signal plus a code-9 message per frame,
    /// and [`Self::header`] declares 2 channels.
    pub fn new_stereo(sample_rate: u32, quality: u8) -> Result<Self, StreamEncodeError> {
        Self::with_channels(sample_rate, quality, 2)
    }

    fn with_channels(
        sample_rate: u32,
        quality: u8,
        channels: u16,
    ) -> Result<Self, StreamEncodeError> {
        if quality > MAX_QUALITY {
            return Err(StreamEncodeError::QualityOutOfRange(quality));
        }
        let inner = match sample_rate {
            8_000 => Inner::Nb(Box::default()),
            16_000 => Inner::Wb(Box::default()),
            32_000 => Inner::Uwb(Box::default()),
            other => return Err(StreamEncodeError::UnsupportedSampleRate(other)),
        };
        Ok(Self {
            inner,
            rate_hz: sample_rate,
            quality,
            channels,
        })
    }

    /// The input sampling rate in Hz.
    pub fn sample_rate_hz(&self) -> u32 {
        self.rate_hz
    }

    /// Input channel count (1 or 2).
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// The §2.1 quality setting.
    pub fn quality(&self) -> u8 {
        self.quality
    }

    /// Samples **per channel** in one 20 ms frame: 160 / 320 / 640.
    pub fn frame_samples(&self) -> usize {
        match self.inner {
            Inner::Nb(_) => NB_FRAME_SAMPLES,
            Inner::Wb(_) => QMF_WIDEBAND_FRAME,
            Inner::Uwb(_) => UWB_FRAME_SAMPLES,
        }
    }

    /// Nominal bit-rate of the mono stream at this quality, in bit/s
    /// (Table 9.2 / Table 10.2 ladders). A stereo stream adds 17 bits
    /// per frame (850 bit/s) for the in-band message.
    pub fn nominal_bitrate_bps(&self) -> Option<u32> {
        match self.inner {
            Inner::Nb(_) => {
                let mode = nb_mode_for_quality(self.quality)?;
                let submode = NarrowbandSubmode::for_id(mode)?;
                Some(u32::from(submode.total_bits) * FRAMES_PER_SECOND)
            }
            Inner::Wb(_) => wb_bitrate_bps(self.quality),
            Inner::Uwb(_) => uwb_bitrate_bps(self.quality),
        }
    }

    /// The Table 7.1 stream header for this encoder's output (write it
    /// with [`SpeexHeader::write_bytes`]; it is the Ogg packet 0 /
    /// codec extradata).
    pub fn header(&self) -> SpeexHeader {
        SpeexHeader::for_sample_rate(self.rate_hz)
            .expect("constructor validated the rate")
            .with_channels(u32::from(self.channels))
            .with_bitrate(self.nominal_bitrate_bps().unwrap_or(0))
    }

    /// Encode exactly one 20 ms frame — [`Self::frame_samples`] samples
    /// per channel, interleaved for stereo — into one self-contained
    /// packet. Pad a short final frame with zeros (Speex has no
    /// partial-frame syntax).
    pub fn encode_frame(&mut self, pcm: &[i16]) -> Result<Vec<u8>, StreamEncodeError> {
        let n = self.frame_samples();
        let expected = n * usize::from(self.channels);
        if pcm.len() != expected {
            return Err(StreamEncodeError::FrameLength {
                expected,
                got: pcm.len(),
            });
        }
        if self.channels == 1 {
            return self.encode_mono(pcm);
        }
        // Intensity stereo (docs intensity-stereo.md §5): mean-absolute
        // per-channel magnitude drives the code-9 payload, the audio is
        // the (L+R)/2 fold.
        let (mut sl, mut sr) = (0.0f64, 0.0f64);
        let mut mono = Vec::with_capacity(n);
        for lr in pcm.as_chunks::<2>().0 {
            sl += f64::from(lr[0].unsigned_abs());
            sr += f64::from(lr[1].unsigned_abs());
            mono.push(downmix_mean(lr[0], lr[1]));
        }
        let payload = encode_stereo_payload(sl / n as f64, sr / n as f64);
        let frame_bytes = self.encode_mono(&mono)?;
        let mut w = BitWriter::new();
        // Each write fits its declared width, so none can fail.
        w.write(0, 1).ok(); // wideband flag of the in-band pseudo-frame
        w.write(14, 4).ok(); // mode 14 = in-band signalling
        w.write(9, 4).ok(); // code 9 = intensity stereo
        w.write(u32::from(payload), 8).ok();
        for &b in &frame_bytes {
            w.write(u32::from(b), 8).ok();
        }
        Ok(w.into_bytes())
    }

    fn encode_mono(&mut self, frame: &[i16]) -> Result<Vec<u8>, StreamEncodeError> {
        let q = self.quality;
        match &mut self.inner {
            Inner::Nb(enc) => {
                let mut pcm = [0i16; NB_FRAME_SAMPLES];
                pcm.copy_from_slice(frame);
                enc.encode_packet_quality(&[pcm], q)
                    .map_err(StreamEncodeError::Narrowband)
            }
            Inner::Wb(enc) => {
                let mut pcm = [0i16; QMF_WIDEBAND_FRAME];
                pcm.copy_from_slice(frame);
                enc.encode_packet_quality(&[pcm], q)
                    .map_err(StreamEncodeError::Wideband)
            }
            Inner::Uwb(enc) => {
                let mut pcm = [0i16; UWB_FRAME_SAMPLES];
                pcm.copy_from_slice(frame);
                enc.encode_packet_quality(&[pcm], q)
                    .map_err(StreamEncodeError::UltraWideband)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SpeexStreamDecoder;

    fn tone(n: usize, rate: u32, amp: f64) -> Vec<i16> {
        (0..n)
            .map(|i| {
                (amp * (2.0 * core::f64::consts::PI * 440.0 * i as f64 / rate as f64).sin()) as i16
            })
            .collect()
    }

    #[test]
    fn rejects_bad_configuration() {
        assert_eq!(
            SpeexStreamEncoder::new(44_100, 8).unwrap_err(),
            StreamEncodeError::UnsupportedSampleRate(44_100)
        );
        assert_eq!(
            SpeexStreamEncoder::new(8_000, 11).unwrap_err(),
            StreamEncodeError::QualityOutOfRange(11)
        );
        assert_eq!(
            SpeexStreamDecoder::for_sample_rate(44_100).unwrap_err(),
            crate::StreamDecodeError::UnsupportedSampleRate(44_100)
        );
        let mut enc = SpeexStreamEncoder::new(8_000, 8).unwrap();
        assert_eq!(
            enc.encode_frame(&[0; 100]).unwrap_err(),
            StreamEncodeError::FrameLength {
                expected: 160,
                got: 100
            }
        );
    }

    #[test]
    fn every_rate_and_quality_round_trips_through_the_stream_decoder() {
        for rate in [8_000, 16_000, 32_000] {
            for q in 0..=MAX_QUALITY {
                let mut enc = SpeexStreamEncoder::new(rate, q).unwrap();
                let h = enc.header();
                assert_eq!(h.rate, rate);
                assert_eq!(h.frame_size as usize, enc.frame_samples());
                let mut dec = SpeexStreamDecoder::for_header(&h).unwrap();
                let pcm = tone(enc.frame_samples() * 3, rate, 8000.0);
                for f in pcm.chunks_exact(enc.frame_samples()) {
                    let pkt = enc.encode_frame(f).unwrap();
                    let out = dec.decode_packet_pcm_i16(&pkt).unwrap();
                    assert_eq!(out.len(), enc.frame_samples(), "rate {rate} q {q}");
                }
            }
        }
    }

    #[test]
    fn stereo_frames_carry_the_pan() {
        let rate = 16_000;
        let mut enc = SpeexStreamEncoder::new_stereo(rate, 8).unwrap();
        assert_eq!(enc.header().nb_channels, 2);
        let mut dec = SpeexStreamDecoder::for_header(&enc.header()).unwrap();
        let n = enc.frame_samples();
        let mono = tone(n * 10, rate, 8000.0);
        // Hard-left-ish: right channel at 1/4 the amplitude.
        let mut inter = Vec::with_capacity(mono.len() * 2);
        for &s in &mono {
            inter.push(s);
            inter.push(s / 4);
        }
        let (mut el, mut er) = (0.0f64, 0.0f64);
        for f in inter.chunks_exact(n * 2) {
            let pkt = enc.encode_frame(f).unwrap();
            let out = dec.decode_packet_stereo_i16(&pkt).unwrap();
            assert_eq!(out.len(), n * 2);
            for lr in out.as_chunks::<2>().0 {
                el += f64::from(lr[0]).powi(2);
                er += f64::from(lr[1]).powi(2);
            }
        }
        assert!(el > 4.0 * er, "left energy {el} vs right {er}");
    }
}
