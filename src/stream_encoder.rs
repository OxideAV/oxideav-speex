//! **Rate-agnostic stream encoder** — the encode-side counterpart of
//! [`crate::SpeexStreamDecoder`]. One type covers the three §2.2 rate
//! classes (8 / 16 / 32 kHz), the §2.1 quality knob (0..=10) and
//! intensity stereo, so a caller never hand-picks among
//! [`crate::NarrowbandEncoder`], [`crate::WidebandEncoder`] and
//! [`crate::UltraWidebandEncoder`] or assembles the stereo in-band
//! message itself.
//!
//! Each call to [`SpeexStreamEncoder::encode_frame`] turns one 20 ms
//! frame into one self-contained packet (octet-padded, RFC 5574 §3.3),
//! and [`SpeexStreamEncoder::header`] is the matching Table 7.1 stream
//! header (`frames_per_packet = 1`).
//!
//! Beyond the constant-quality default the encoder carries the §2.1
//! rate modes — variable bit-rate ([`SpeexStreamEncoder::set_vbr`]),
//! average bit-rate ([`SpeexStreamEncoder::set_abr`]), voice-activity
//! detection with comfort noise ([`SpeexStreamEncoder::set_vad`]) and
//! discontinuous transmission ([`SpeexStreamEncoder::set_dtx`]) — the
//! search complexity ([`SpeexStreamEncoder::set_complexity`]), explicit
//! sub-mode selection ([`SpeexStreamEncoder::set_submodes`]), and the
//! §5.5 in-band channel ([`SpeexStreamEncoder::queue_inband`],
//! [`SpeexStreamEncoder::queue_user_data`]). Every frame names its own
//! sub-modes, so a decoder needs no notice of any of them.
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
    FRAMES_PER_SECOND, MAX_QUALITY, UWB_HIGH_BAND_MODE, nb_mode_for_quality, uwb_bitrate_bps,
    wb_bitrate_bps, wb_modes_for_quality,
};
use crate::rate_control::{FrameSpend, RateController};
use crate::signalling::{CUSTOM_INBAND_MAX_BYTES, CustomInbandMessage, InbandMessage};
use crate::stereo::{downmix_mean, encode_stereo_payload};
use crate::submode::NarrowbandSubmode;
use crate::uwb_decoder::UWB_FRAME_SAMPLES;
use crate::wideband::WidebandHighBandSubmode;

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
    /// A sub-mode pair this rate class cannot carry
    /// (`narrowband mode`, `high-band mode`).
    UnsupportedSubmodes(u8, u8),
    /// A user payload longer than the 5-bit size field allows (bytes).
    UserDataTooLong(usize),
    /// A bit-rate below the rate class's lowest coded mode (bit/s).
    BitrateTooLow(u32),
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
            StreamEncodeError::UnsupportedSubmodes(nb, hb) => write!(
                f,
                "speex stream encode: sub-modes (narrowband {nb}, high-band {hb}) are not codable in this rate class"
            ),
            StreamEncodeError::UserDataTooLong(n) => write!(
                f,
                "speex stream encode: user payload of {n} bytes exceeds the 31-byte in-band limit"
            ),
            StreamEncodeError::BitrateTooLow(b) => write!(
                f,
                "speex stream encode: no coded mode at or below {b} bit/s in this rate class"
            ),
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

/// One queued §5.5 side-channel item.
#[derive(Debug, Clone)]
enum SideMessage {
    Inband(InbandMessage),
    User(Vec<u8>),
}

/// The sub-modes one frame was coded with, per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct FrameSubmodes {
    /// Narrowband (Table 9.1) sub-mode; `0` is a null frame.
    pub nb_mode: u8,
    /// First high-band (Table 10.1) sub-mode — wideband and
    /// ultra-wideband streams only.
    pub hb_mode: Option<u8>,
    /// Second high-band sub-mode — ultra-wideband streams only.
    pub uwb_mode: Option<u8>,
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
    /// Explicit `(narrowband, high-band)` sub-modes for coded frames,
    /// instead of the quality ladder.
    submodes: Option<(u8, u8)>,
    rate: RateController,
    /// Side-channel items for the next packet.
    queued: Vec<SideMessage>,
    last_submodes: FrameSubmodes,
    last_bits: usize,
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
            submodes: None,
            rate: RateController::new(),
            queued: Vec::new(),
            last_submodes: FrameSubmodes::default(),
            last_bits: 0,
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

    /// Change the quality (0..=10) from the next frame on. Clears an
    /// explicit sub-mode selection.
    pub fn set_quality(&mut self, quality: u8) -> Result<(), StreamEncodeError> {
        if quality > MAX_QUALITY {
            return Err(StreamEncodeError::QualityOutOfRange(quality));
        }
        self.quality = quality;
        self.submodes = None;
        Ok(())
    }

    /// Pick the highest quality whose constant bit-rate does not exceed
    /// `bps` (the manual's `SET_BITRATE` semantics) and return it.
    pub fn set_bitrate(&mut self, bps: u32) -> Result<u8, StreamEncodeError> {
        let q = (0..=MAX_QUALITY)
            .rev()
            .find(|&q| self.ladder_bitrate(q) <= bps)
            .ok_or(StreamEncodeError::BitrateTooLow(bps))?;
        self.set_quality(q)?;
        Ok(q)
    }

    /// The search complexity, 1..=10 (see
    /// [`crate::NarrowbandEncoder::set_complexity`]).
    pub fn complexity(&self) -> u8 {
        match &self.inner {
            Inner::Nb(e) => e.complexity(),
            Inner::Wb(e) => e.complexity(),
            Inner::Uwb(e) => e.complexity(),
        }
    }

    /// Set the search complexity, 1..=10 (clamped): deeper searches for
    /// a lower coding error at a higher CPU cost. The stream format and
    /// bit-rate are unaffected. The default is
    /// [`crate::DEFAULT_COMPLEXITY`].
    pub fn set_complexity(&mut self, complexity: u8) {
        match &mut self.inner {
            Inner::Nb(e) => e.set_complexity(complexity),
            Inner::Wb(e) => e.set_complexity(complexity),
            Inner::Uwb(e) => e.set_complexity(complexity),
        }
    }

    /// Choose the search domain: `Some((γ1, γ2))` for the
    /// noise-weighted domain of manual §8.5 (the default, at the
    /// manual's 0.9 / 0.6), `None` for the plain waveform error (see
    /// [`crate::NarrowbandEncoder::set_noise_weighting`]).
    pub fn set_noise_weighting(&mut self, gammas: Option<(f64, f64)>) {
        match &mut self.inner {
            Inner::Nb(e) => e.set_noise_weighting(gammas),
            Inner::Wb(e) => e.set_noise_weighting(gammas),
            Inner::Uwb(e) => e.set_noise_weighting(gammas),
        }
    }

    /// Code every speech frame with explicit sub-modes instead of the
    /// quality ladder: `nb_mode` from Table 9.1 (1..=8) and, for
    /// wideband / ultra-wideband streams, `hb_mode` from Table 10.1
    /// (0..=4; ignored at 8 kHz; 1..=4 at 32 kHz). `None` returns to
    /// the ladder.
    ///
    /// The ultra-wideband second layer is always the gain-only mode 1
    /// (RFC 5574 Table 2).
    pub fn set_submodes(&mut self, modes: Option<(u8, u8)>) -> Result<(), StreamEncodeError> {
        if let Some((nb, hb)) = modes {
            let nb_ok = (1..=8).contains(&nb);
            let hb_ok = match self.inner {
                Inner::Nb(_) => true,
                Inner::Wb(_) => WidebandHighBandSubmode::for_id(hb).is_some(),
                // A silent first high band under a coded second layer
                // is not a combination decoders agree on.
                Inner::Uwb(_) => hb > 0 && WidebandHighBandSubmode::for_id(hb).is_some(),
            };
            if !nb_ok || !hb_ok {
                return Err(StreamEncodeError::UnsupportedSubmodes(nb, hb));
            }
        }
        self.submodes = modes;
        Ok(())
    }

    /// Variable bit-rate: `Some(q)` (0.0..=10.0, clamped) lets every
    /// frame pick its own sub-modes around quality `q` — more bits for
    /// vowels and onsets, fewer for unvoiced and quiet frames, comfort
    /// noise for the background (manual §2.1). `None` turns VBR off.
    pub fn set_vbr(&mut self, quality: Option<f32>) {
        self.rate.vbr_quality = quality.map(|q| q.clamp(0.0, f32::from(MAX_QUALITY)));
    }

    /// The VBR quality, when VBR is on.
    pub fn vbr(&self) -> Option<f32> {
        self.rate.vbr_quality
    }

    /// Average bit-rate: `Some(bps)` runs VBR with its quality steered
    /// so the stream averages `bps` (manual §2.1). `None` turns ABR off.
    pub fn set_abr(&mut self, bps: Option<u32>) {
        // VBR at quality q averages below the ladder's constant rate
        // (pauses, unvoiced frames), so start a little above the step
        // the target names.
        let base = bps
            .and_then(|b| {
                (0..=MAX_QUALITY)
                    .rev()
                    .find(|&q| u64::from(self.ladder_bitrate(q)) * 10 <= u64::from(b) * 13)
            })
            .unwrap_or(0);
        self.rate.set_abr(bps, base);
    }

    /// The ABR target, when ABR is on.
    pub fn abr(&self) -> Option<u32> {
        self.rate.abr_bps
    }

    /// Cap the rate of any single frame in VBR / ABR operation
    /// (bit/s, packet framing excluded); `None` removes the cap.
    pub fn set_vbr_max_bitrate(&mut self, bps: Option<u32>) {
        self.rate.max_bps = bps;
    }

    /// Voice-activity detection for constant-quality operation:
    /// non-speech frames are coded as comfort noise (the lowest ladder
    /// step). VBR and ABR always do this.
    pub fn set_vad(&mut self, on: bool) {
        self.rate.vad = on;
    }

    /// Whether VAD is on.
    pub fn vad(&self) -> bool {
        self.rate.vad
    }

    /// Discontinuous transmission: while the background holds still,
    /// frames shrink to the null frame (5 / 9 / 13 bits at 8 / 16 /
    /// 32 kHz; manual §2.1). Implies voice-activity detection.
    pub fn set_dtx(&mut self, on: bool) {
        self.rate.dtx = on;
    }

    /// Whether DTX is on.
    pub fn dtx(&self) -> bool {
        self.rate.dtx
    }

    /// Queue a Table 5.1 in-band message (mode 14) for the head of the
    /// next packet — a request to the far end, which it may honour or
    /// ignore (manual §5.5).
    pub fn queue_inband(&mut self, message: InbandMessage) {
        self.queued.push(SideMessage::Inband(message));
    }

    /// Queue an application-defined payload (mode 13, at most 31 bytes)
    /// for the head of the next packet, in the layout manual §5.5
    /// states: a 5-bit size in bytes, then the bytes. Decoders that do
    /// not know the payload skip it by its size field.
    ///
    /// Interoperability caveat: this crate's decoder reads that layout,
    /// but a widely deployed decoder was observed not to skip such a
    /// message correctly (the frames behind it are lost), so use it only
    /// between peers known to follow the manual. Table 5.1 messages
    /// ([`Self::queue_inband`]) do not have this problem.
    pub fn queue_user_data(&mut self, payload: &[u8]) -> Result<(), StreamEncodeError> {
        if payload.len() > CUSTOM_INBAND_MAX_BYTES as usize {
            return Err(StreamEncodeError::UserDataTooLong(payload.len()));
        }
        self.queued.push(SideMessage::User(payload.to_vec()));
        Ok(())
    }

    /// The sub-modes the most recent frame was coded with.
    pub fn last_submodes(&self) -> FrameSubmodes {
        self.last_submodes
    }

    /// The size in bits of the most recent frame's coded layers
    /// (side-channel messages, terminator and padding excluded).
    pub fn last_frame_bits(&self) -> usize {
        self.last_bits
    }

    /// Samples **per channel** in one 20 ms frame: 160 / 320 / 640.
    pub fn frame_samples(&self) -> usize {
        match self.inner {
            Inner::Nb(_) => NB_FRAME_SAMPLES,
            Inner::Wb(_) => QMF_WIDEBAND_FRAME,
            Inner::Uwb(_) => UWB_FRAME_SAMPLES,
        }
    }

    /// The quality ladder's constant bit-rate at `quality` in this rate
    /// class.
    fn ladder_bitrate(&self, quality: u8) -> u32 {
        match self.inner {
            Inner::Nb(_) => nb_mode_for_quality(quality)
                .and_then(NarrowbandSubmode::for_id)
                .map_or(0, |m| u32::from(m.total_bits) * FRAMES_PER_SECOND),
            Inner::Wb(_) => wb_bitrate_bps(quality).unwrap_or(0),
            Inner::Uwb(_) => uwb_bitrate_bps(quality).unwrap_or(0),
        }
    }

    /// Bits of a frame coded with the given layer sub-modes.
    fn frame_bits(&self, nb: u8, hb: u8) -> usize {
        let nb_bits = NarrowbandSubmode::for_id(nb).map_or(0, |m| usize::from(m.total_bits));
        let hb_bits =
            |m: u8| WidebandHighBandSubmode::for_id(m).map_or(0, |s| usize::from(s.total_bits));
        match self.inner {
            Inner::Nb(_) => nb_bits,
            Inner::Wb(_) => nb_bits + hb_bits(hb),
            Inner::Uwb(_) => {
                nb_bits + hb_bits(hb) + hb_bits(if nb == 0 { 0 } else { UWB_HIGH_BAND_MODE })
            }
        }
    }

    /// Nominal bit-rate of the mono stream in bit/s: the constant rate
    /// of the configured quality (Table 9.2 / Table 10.2 ladders) or
    /// sub-modes; with VBR / VAD / DTX on, the rate speech frames are
    /// coded around; with ABR, the target. A stereo stream adds 17 bits
    /// per frame (850 bit/s) for the in-band message.
    pub fn nominal_bitrate_bps(&self) -> Option<u32> {
        if let Some(target) = self.rate.abr_bps {
            return Some(target);
        }
        if let Some((nb, hb)) = self.submodes {
            return Some(self.frame_bits(nb, hb) as u32 * FRAMES_PER_SECOND);
        }
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
    /// codec extradata). Take it after configuring the rate mode: it
    /// records the nominal bit-rate and the VBR flag.
    pub fn header(&self) -> SpeexHeader {
        let mut h = SpeexHeader::for_sample_rate(self.rate_hz)
            .expect("constructor validated the rate")
            .with_channels(u32::from(self.channels))
            .with_bitrate(self.nominal_bitrate_bps().unwrap_or(0));
        h.vbr = u32::from(self.rate.vbr_quality.is_some() || self.rate.abr_bps.is_some());
        h
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
        let mut w = BitWriter::new();
        // Each write below fits its declared width, so none can fail.
        for item in core::mem::take(&mut self.queued) {
            w.write(0, 1).ok(); // wideband flag of the pseudo-frame
            match item {
                SideMessage::Inband(m) => {
                    w.write(14, 4).ok(); // mode 14 = in-band signalling
                    m.write(&mut w).ok();
                }
                SideMessage::User(bytes) => {
                    w.write(13, 4).ok(); // mode 13 = application-defined
                    CustomInbandMessage {
                        size_bytes: bytes.len() as u8,
                    }
                    .write(&mut w, &bytes)
                    .ok();
                }
            }
        }
        if self.channels == 1 {
            self.encode_mono(&mut w, pcm)?;
        } else {
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
            w.write(0, 1).ok(); // wideband flag of the in-band pseudo-frame
            w.write(14, 4).ok(); // mode 14 = in-band signalling
            w.write(9, 4).ok(); // code 9 = intensity stereo
            w.write(u32::from(payload), 8).ok();
            self.encode_mono(&mut w, &mono)?;
        }
        crate::nb_encode::write_packet_terminator(&mut w).ok();
        Ok(w.into_bytes())
    }

    /// The layer sub-modes for one frame's spend.
    fn submodes_for(&self, spend: FrameSpend, configured: u8) -> (u8, u8) {
        let ladder = |q: u8| -> (u8, u8) {
            match self.inner {
                Inner::Nb(_) => (nb_mode_for_quality(q).unwrap_or(1), 0),
                _ => wb_modes_for_quality(q).map_or((1, 1), |m| (m.nb_mode, m.hb_mode)),
            }
        };
        match spend {
            FrameSpend::Null => (0, 0),
            FrameSpend::ComfortNoise => ladder(0),
            FrameSpend::Quality(q) => match self.submodes {
                Some(m) if q == configured && self.rate.steering_quality().is_none() => m,
                _ => ladder(q),
            },
        }
    }

    fn encode_mono(&mut self, w: &mut BitWriter, frame: &[i16]) -> Result<(), StreamEncodeError> {
        let q = self.quality;
        let spend = {
            let this = &*self;
            let rate_of = |q: u8| this.ladder_bitrate(q);
            // (The controller is cloned out so the ladder closure can
            // borrow the encoder.)
            let mut rc = this.rate.clone();
            let s = rc.decide(frame, q, rate_of);
            self.rate = rc;
            s
        };
        let (nb, hb) = self.submodes_for(spend, q);
        let bits = self.frame_bits(nb, hb);
        self.rate.account(bits);
        self.last_bits = bits;
        match &mut self.inner {
            Inner::Nb(enc) => {
                self.last_submodes = FrameSubmodes {
                    nb_mode: nb,
                    hb_mode: None,
                    uwb_mode: None,
                };
                let mut pcm = [0i16; NB_FRAME_SAMPLES];
                pcm.copy_from_slice(frame);
                enc.encode_frame_into(w, &pcm, nb)
                    .map_err(StreamEncodeError::Narrowband)
            }
            Inner::Wb(enc) => {
                self.last_submodes = FrameSubmodes {
                    nb_mode: nb,
                    hb_mode: Some(hb),
                    uwb_mode: None,
                };
                let mut pcm = [0i16; QMF_WIDEBAND_FRAME];
                pcm.copy_from_slice(frame);
                enc.encode_frame_into(w, &pcm, nb, hb)
                    .map_err(StreamEncodeError::Wideband)
            }
            Inner::Uwb(enc) => {
                let uwb = if nb == 0 { 0 } else { UWB_HIGH_BAND_MODE };
                self.last_submodes = FrameSubmodes {
                    nb_mode: nb,
                    hb_mode: Some(hb),
                    uwb_mode: Some(uwb),
                };
                let mut pcm = [0i16; UWB_FRAME_SAMPLES];
                pcm.copy_from_slice(frame);
                enc.encode_frame_into(w, &pcm, nb, hb, uwb)
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
