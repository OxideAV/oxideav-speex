//! **oxideav-core framework integration** (round r438) — the generic
//! [`Decoder`] / [`Encoder`] trait surface over the crate's working
//! codec types, the [`register`] entry point that installs them into a
//! [`RuntimeContext`], and the dual-API [`make_decoder`] /
//! [`make_encoder`] factories.
//!
//! ## What this wires (and what it does not change)
//!
//! Everything here is *plumbing over the existing decode/encode paths*:
//! the packet→PCM work is done by [`crate::SpeexStreamDecoder`] /
//! [`crate::SpeexDecoder`] and the three direct encoders
//! ([`crate::NarrowbandEncoder`] / [`crate::WidebandEncoder`] /
//! [`crate::UltraWidebandEncoder`]); no bit-stream behaviour is added.
//! The direct types remain the recommended API for callers that don't
//! need the framework registry.
//!
//! ## Decoder packet model
//!
//! An Ogg/Speex logical stream (manual §7, Table 7.1) opens with the
//! `Speex   ` header packet, then one comment packet plus
//! `extra_headers` further metadata packets, then audio packets. The
//! framework decoder accepts either arrangement:
//!
//! * **In-band header** — a packet starting with the 8-byte
//!   [`SPEEX_MAGIC`] (re)initialises the stream state and schedules the
//!   `1 + extra_headers` metadata packets that follow to be consumed
//!   without producing audio.
//! * **Out-of-band header** — [`CodecParameters::extradata`] holding the
//!   80-byte header configures the decoder at construction; failing
//!   that, `CodecParameters::sample_rate` (8000 / 16000 / 32000) selects
//!   the rate class directly.
//!
//! Output is interleaved signed-16-bit PCM ([`SampleFormat::S16`]), one
//! [`AudioFrame`] per packet (a packet's audio frames concatenated, §5.5
//! control pseudo-frames skipped).
//!
//! ## Stereo posture (documented limitation)
//!
//! Speex stereo is the in-band *intensity stereo* extension: the frames
//! stay mono and a Table 5.1 code-9 in-band message ("Intensity stereo
//! information", 8 bits) rides alongside them. The staged
//! `docs/audio/speex/intensity-stereo.md` clean-room note pins the
//! payload layout, the L/R reconstruction law and the encoder fold, so
//! this framework path implements true intensity stereo
//! ([`crate::stereo`]):
//!
//! * **Decode** — for a stream whose header/parameters declare 2
//!   channels, each frame's code-9 payload reconstructs an `(gL, gR)`
//!   gain pair (with the §4 intra-frame interpolation) applied to the
//!   decoded mono signal, producing interleaved L/R. Absent a stereo
//!   message the neutral unit gains reproduce the previous
//!   duplicate-mono behaviour. The block-phase offset the reference file
//!   carries (`intensity-stereo.md` §4.1) is not reproduced, so
//!   byte-exactness against a reference decode is bounded by that
//!   sub-frame phase while the per-sample gains are reference-correct.
//! * **Encode** — a 2-channel input stream emits the `(L+R)/2` downmix
//!   with the per-frame code-9 message prefixed (balance from the L/R
//!   amplitude ratio, `e_ratio` from the total power); the output stream
//!   declares 2 channels. A 1-channel input encodes plain mono.
//!
//! ## Encoder
//!
//! `sample_rate` selects the rate class (8000 → narrowband, 16000 →
//! wideband, 32000 → ultra-wideband; §2.2), the `quality` option (0..=10,
//! default 8) selects the sub-mode ladder ([`crate::quality`]), and each
//! 20 ms frame is emitted as one self-contained packet
//! (`frames_per_packet = 1`, mode-15 terminated). Wideband /
//! ultra-wideband quality 10 needs high-band mode 4, whose
//! innovation-codebook binding is the remaining recorded docs gap — the
//! factory rejects it up front. [`Encoder::output_params`] carries the
//! 80-byte stream header as `extradata` (every numeric field mirrored
//! from the staged real-capture headers, e.g. `mode_bitstream_version =
//! 4`).

use std::collections::VecDeque;

use oxideav_core::{
    parse_options, AudioFrame, CodecCapabilities, CodecId, CodecInfo, CodecOptionsStruct,
    CodecParameters, Decoder, Encoder, Error, Frame, OptionField, OptionKind, OptionValue, Packet,
    Result, RuntimeContext, SampleFormat, TimeBase,
};

use crate::encoder_nb::{NarrowbandEncoder, NB_FRAME_SAMPLES};
use crate::encoder_uwb::UltraWidebandEncoder;
use crate::encoder_wb::WidebandEncoder;
use crate::header::{
    SpeexHeader, SPEEX_HEADER_LEN, SPEEX_MAGIC, SPEEX_MODE_NARROWBAND, SPEEX_MODE_ULTRAWIDEBAND,
    SPEEX_MODE_WIDEBAND, SPEEX_STRING_LEN, SPEEX_VERSION_LEN,
};
use crate::qmf::QMF_WIDEBAND_FRAME;
use crate::quality::{nb_mode_for_quality, uwb_bitrate_bps, wb_bitrate_bps, MAX_QUALITY};
use crate::stereo::{downmix_mean, encode_stereo_payload, StereoDecoder};
use crate::stream_decoder::SpeexStreamDecoder;
use crate::submode::NarrowbandSubmode;
use crate::uwb_decoder::UWB_FRAME_SAMPLES;

/// Registry identifier this crate registers under.
const CODEC_NAME: &str = "speex";

/// The three §2.2 rate classes, keyed by the sampling rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RateClass {
    /// 8 kHz narrowband.
    Narrowband,
    /// 16 kHz wideband.
    Wideband,
    /// 32 kHz ultra-wideband.
    UltraWideband,
}

impl RateClass {
    fn for_sample_rate(rate: u32) -> Option<Self> {
        match rate {
            8_000 => Some(RateClass::Narrowband),
            16_000 => Some(RateClass::Wideband),
            32_000 => Some(RateClass::UltraWideband),
            _ => None,
        }
    }

    fn sample_rate(self) -> u32 {
        match self {
            RateClass::Narrowband => 8_000,
            RateClass::Wideband => 16_000,
            RateClass::UltraWideband => 32_000,
        }
    }

    /// Samples per 20 ms frame (§2.1).
    fn frame_samples(self) -> usize {
        match self {
            RateClass::Narrowband => NB_FRAME_SAMPLES,
            RateClass::Wideband => QMF_WIDEBAND_FRAME,
            RateClass::UltraWideband => UWB_FRAME_SAMPLES,
        }
    }

    /// The Table 7.1 `mode` field value.
    fn header_mode(self) -> u32 {
        match self {
            RateClass::Narrowband => SPEEX_MODE_NARROWBAND,
            RateClass::Wideband => SPEEX_MODE_WIDEBAND,
            RateClass::UltraWideband => SPEEX_MODE_ULTRAWIDEBAND,
        }
    }

    /// Nominal bit-rate of the class's quality ladder at `quality`.
    fn nominal_bitrate(self, quality: u8) -> Option<u32> {
        match self {
            RateClass::Narrowband => {
                let mode = nb_mode_for_quality(quality)?;
                let submode = NarrowbandSubmode::for_id(mode)?;
                Some(u32::from(submode.total_bits) * crate::quality::FRAMES_PER_SECOND)
            }
            RateClass::Wideband => wb_bitrate_bps(quality),
            RateClass::UltraWideband => uwb_bitrate_bps(quality),
        }
    }

    /// A synthetic minimal stream header for this class (all numeric
    /// fields mirroring the staged real-capture headers).
    fn stream_header(self, quality: u8, vbr: bool) -> SpeexHeader {
        let mut speex_version = [0u8; SPEEX_VERSION_LEN];
        let tag = env!("CARGO_PKG_VERSION").as_bytes();
        let n = tag.len().min(SPEEX_VERSION_LEN);
        speex_version[..n].copy_from_slice(&tag[..n]);
        SpeexHeader {
            speex_string: *SPEEX_MAGIC,
            speex_version,
            speex_version_id: 1,
            header_size: SPEEX_HEADER_LEN as u32,
            rate: self.sample_rate(),
            mode: self.header_mode(),
            // Every staged real capture across the three rate classes
            // declares bitstream version 4.
            mode_bitstream_version: 4,
            nb_channels: 1,
            bitrate: self.nominal_bitrate(quality).unwrap_or(0),
            frame_size: self.frame_samples() as u32,
            vbr: u32::from(vbr),
            frames_per_packet: 1,
            extra_headers: 0,
            reserved1: 0,
            reserved2: 0,
        }
    }
}

// ───────────────────────── decoder ─────────────────────────

/// [`Decoder`] implementation over [`SpeexStreamDecoder`]. Built by
/// [`make_decoder`]; see the module docs for the packet model.
#[derive(Debug)]
pub struct SpeexFrameworkDecoder {
    codec_id: CodecId,
    /// Rate-class decode state; `None` until a header (in-band or
    /// out-of-band) or a usable `sample_rate` fixes the class.
    stream: Option<SpeexStreamDecoder>,
    /// Header the stream was configured from, kept for `reset`.
    header: Option<SpeexHeader>,
    /// Rate-class fallback from `CodecParameters::sample_rate`.
    param_rate: Option<u32>,
    /// Metadata packets still to swallow after an in-band header
    /// (`1 + extra_headers`: the comment packet plus any extras, §7).
    meta_skip: u32,
    /// Output channel count (1, or 2 for the duplicated fallback).
    channels: u16,
    /// Decoded frames not yet pulled.
    pending: VecDeque<AudioFrame>,
    /// `flush` seen — drain then Eof.
    flushed: bool,
    /// Intensity-stereo reconstruction state (used only when
    /// `channels == 2`); carries the per-frame gain interpolation.
    stereo: StereoDecoder,
}

impl SpeexFrameworkDecoder {
    fn from_params(params: &CodecParameters) -> Result<Self> {
        let mut dec = Self {
            codec_id: params.codec_id.clone(),
            stream: None,
            header: None,
            param_rate: params.sample_rate,
            meta_skip: 0,
            channels: params.channels.unwrap_or(1).clamp(1, 2),
            pending: VecDeque::new(),
            flushed: false,
            stereo: StereoDecoder::new(),
        };
        if params.channels.is_some_and(|c| c > 2) {
            return Err(Error::unsupported(
                "speex: streams carry at most 2 channels (mono + in-band intensity stereo)",
            ));
        }
        if !params.extradata.is_empty() {
            let header = SpeexHeader::parse(&params.extradata)
                .map_err(|e| Error::invalid(format!("speex: bad extradata header: {e}")))?;
            dec.install_header(header, false)?;
        } else if let Some(rate) = params.sample_rate {
            dec.install_rate(rate)?;
        }
        Ok(dec)
    }

    /// Configure from a parsed stream header. `in_band` schedules the
    /// following `1 + extra_headers` metadata packets to be swallowed.
    fn install_header(&mut self, header: SpeexHeader, in_band: bool) -> Result<()> {
        let stream = SpeexStreamDecoder::for_header(&header)
            .map_err(|e| Error::invalid(format!("speex: {e}")))?;
        self.meta_skip = if in_band {
            1 + header.extra_headers.min(15)
        } else {
            0
        };
        // The header's channel declaration wins over the construction
        // parameters (it is the stream's own metadata).
        self.channels = header.nb_channels.clamp(1, 2) as u16;
        self.header = Some(header);
        self.stream = Some(stream);
        Ok(())
    }

    /// Configure from a bare sampling rate (no header available).
    fn install_rate(&mut self, rate: u32) -> Result<()> {
        let class = RateClass::for_sample_rate(rate).ok_or_else(|| {
            Error::unsupported(format!(
                "speex: sample rate {rate} is not a Speex rate class (8000/16000/32000)"
            ))
        })?;
        // A synthetic header carries exactly the mode field the stream
        // decoder dispatches on.
        let header = class.stream_header(MAX_QUALITY, false);
        let stream = SpeexStreamDecoder::for_header(&header)
            .map_err(|e| Error::invalid(format!("speex: {e}")))?;
        self.header = Some(header);
        self.stream = Some(stream);
        Ok(())
    }

    /// Interleave the decoded mono signal into the declared channel
    /// shape (see the module docs' stereo posture).
    fn shape_pcm(&self, mono: &[i16]) -> Vec<u8> {
        let ch = usize::from(self.channels.max(1));
        let mut out = Vec::with_capacity(mono.len() * ch * 2);
        for &s in mono {
            for _ in 0..ch {
                out.extend_from_slice(&s.to_le_bytes());
            }
        }
        out
    }
}

impl Decoder for SpeexFrameworkDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let data = &packet.data;
        // In-band stream header (Ogg packet 0): (re)configure and
        // schedule the metadata packets that follow.
        if data.len() >= SPEEX_STRING_LEN && &data[..SPEEX_STRING_LEN] == SPEEX_MAGIC {
            if let Ok(header) = SpeexHeader::parse(data) {
                self.install_header(header, true)?;
                return Ok(());
            }
        }
        // Comment / extra-header packets after an in-band header carry
        // no audio.
        if self.meta_skip > 0 {
            self.meta_skip -= 1;
            return Ok(());
        }
        let Some(stream) = self.stream.as_mut() else {
            return Err(Error::invalid(
                "speex: no stream header seen and no sample_rate in parameters",
            ));
        };
        // Two-channel output applies the in-band intensity-stereo law
        // (`crate::stereo`); mono output takes the flat concatenation.
        if self.channels >= 2 {
            let frames = stream
                .decode_packet_frames_stereo(data)
                .map_err(|e| Error::invalid(format!("speex: {e}")))?;
            let mut interleaved: Vec<i16> = Vec::new();
            let mut total = 0u32;
            for (mono, payload) in frames {
                if mono.is_empty() {
                    continue;
                }
                total += mono.len() as u32;
                // Absent payload (non-stereo message stream): neutral
                // duplicate, the un-panned identity.
                let p = payload.unwrap_or(0b0000_0011); // bal 0, e 3 = unit gains
                interleaved.extend_from_slice(&self.stereo.interleave_frame(&mono, p));
            }
            if total == 0 {
                return Ok(());
            }
            let bytes: Vec<u8> = interleaved.iter().flat_map(|s| s.to_le_bytes()).collect();
            self.pending.push_back(AudioFrame {
                samples: total,
                pts: packet.pts,
                data: vec![bytes],
            });
            return Ok(());
        }
        let mono = stream
            .decode_packet_pcm_i16(data)
            .map_err(|e| Error::invalid(format!("speex: {e}")))?;
        if mono.is_empty() {
            // Control-only packet (§5.5 in-band signalling) — no audio.
            return Ok(());
        }
        let frame = AudioFrame {
            samples: mono.len() as u32,
            pts: packet.pts,
            data: vec![self.shape_pcm(&mono)],
        };
        self.pending.push_back(frame);
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match self.pending.pop_front() {
            Some(f) => Ok(Frame::Audio(f)),
            None if self.flushed => Err(Error::Eof),
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        // Speex packets are self-contained (every frame decodes on
        // arrival) — flushing only marks the drain-then-Eof state.
        self.flushed = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.pending.clear();
        self.flushed = false;
        self.meta_skip = 0;
        self.stereo = StereoDecoder::new();
        // Rebuild the rate-class state fresh (zero IIR / excitation /
        // QMF history) so post-seek decode starts clean.
        if let Some(header) = &self.header {
            self.stream = Some(
                SpeexStreamDecoder::for_header(header)
                    .map_err(|e| Error::invalid(format!("speex: {e}")))?,
            );
        } else if let Some(rate) = self.param_rate {
            self.install_rate(rate)?;
        }
        Ok(())
    }
}

// ───────────────────────── encoder ─────────────────────────

/// Typed options accepted by the Speex encoder factory.
#[derive(Debug, Clone, Copy)]
pub struct SpeexEncoderOptions {
    /// §2.1 quality knob, 0..=10 (Table 9.2 / Table 10.2 / RFC 5574
    /// Table 2 ladders). Default 8.
    pub quality: u8,
}

impl Default for SpeexEncoderOptions {
    fn default() -> Self {
        Self { quality: 8 }
    }
}

impl CodecOptionsStruct for SpeexEncoderOptions {
    const SCHEMA: &'static [OptionField] = &[OptionField {
        name: "quality",
        kind: OptionKind::U32,
        default: OptionValue::U32(8),
        help: "Speex quality 0-10 (sub-mode ladder per rate class)",
    }];

    fn apply(&mut self, key: &str, value: &OptionValue) -> Result<()> {
        match key {
            "quality" => {
                let q = value.as_u32()?;
                if q > u32::from(MAX_QUALITY) {
                    return Err(Error::invalid(format!(
                        "speex: quality {q} out of range 0..=10"
                    )));
                }
                self.quality = q as u8;
            }
            other => return Err(Error::invalid(format!("speex: unknown option '{other}'"))),
        }
        Ok(())
    }
}

/// Rate-class-specific encoder state behind [`SpeexFrameworkEncoder`].
#[derive(Debug)]
enum EncoderInner {
    /// 8 kHz narrowband.
    Nb(Box<NarrowbandEncoder>),
    /// 16 kHz wideband.
    Wb(Box<WidebandEncoder>),
    /// 32 kHz ultra-wideband.
    Uwb(Box<UltraWidebandEncoder>),
}

/// [`Encoder`] implementation over the three direct encoders. Built by
/// [`make_encoder`]; consumes interleaved S16 audio frames of any
/// length, re-blocks them into 20 ms Speex frames, and emits one
/// self-contained packet per frame.
#[derive(Debug)]
pub struct SpeexFrameworkEncoder {
    codec_id: CodecId,
    inner: EncoderInner,
    class: RateClass,
    quality: u8,
    /// Input channel count (2 is downmixed to mono before encoding).
    in_channels: u16,
    /// Mono sample accumulator awaiting a full 20 ms frame.
    buf: Vec<i16>,
    /// Stereo (L, R) accumulator when `in_channels == 2`: true intensity
    /// stereo emits the code-9 message + the `(L+R)/2` downmix per frame.
    stereo_buf: Vec<(i16, i16)>,
    /// Encoded packets not yet pulled.
    pending: VecDeque<Packet>,
    /// Running output position in samples (packet pts).
    sample_pos: i64,
    output_params: CodecParameters,
}

impl SpeexFrameworkEncoder {
    fn from_params(params: &CodecParameters) -> Result<Self> {
        let rate = params
            .sample_rate
            .ok_or_else(|| Error::invalid("speex: encoder needs sample_rate"))?;
        let class = RateClass::for_sample_rate(rate).ok_or_else(|| {
            Error::unsupported(format!(
                "speex: sample rate {rate} is not a Speex rate class (8000/16000/32000)"
            ))
        })?;
        let in_channels = params.channels.unwrap_or(1);
        if !(1..=2).contains(&in_channels) {
            return Err(Error::unsupported(format!(
                "speex: encoder accepts 1 or 2 input channels (2 is downmixed to mono), got {in_channels}"
            )));
        }
        if let Some(fmt) = params.sample_format {
            if fmt != SampleFormat::S16 {
                return Err(Error::unsupported(format!(
                    "speex: encoder consumes interleaved S16 input, got {fmt:?}"
                )));
            }
        }
        let opts: SpeexEncoderOptions = parse_options(&params.options)?;
        if opts.quality == 10 && class != RateClass::Narrowband {
            // WB/UWB quality 10 selects high-band mode 4. Its codebook
            // binding now *decodes* (crate::hb_innovation two-stage law),
            // but the encoder mode-4 codebook search + the absolute
            // per-frame HB-innovation gain law are not yet pinned, so the
            // factory still declines to *emit* quality 10.
            return Err(Error::unsupported(
                "speex: wideband/ultra-wideband quality 10 (high-band mode 4) decodes but \
                 encoding it is not yet supported; use quality <= 9",
            ));
        }
        let inner = match class {
            RateClass::Narrowband => EncoderInner::Nb(Box::default()),
            RateClass::Wideband => EncoderInner::Wb(Box::default()),
            RateClass::UltraWideband => EncoderInner::Uwb(Box::default()),
        };

        let mut header = class.stream_header(opts.quality, false);
        // True intensity-stereo output declares 2 channels so a decoder
        // installs the stereo callback (docs intensity-stereo.md §1).
        if in_channels == 2 {
            header.nb_channels = 2;
        }
        let mut output_params = CodecParameters::audio(params.codec_id.clone());
        output_params.sample_rate = Some(rate);
        output_params.channels = Some(u16::from(in_channels == 2) + 1);
        output_params.sample_format = Some(SampleFormat::S16);
        output_params.bit_rate = class.nominal_bitrate(opts.quality).map(u64::from);
        output_params.extradata = header.write_bytes().to_vec();

        Ok(Self {
            codec_id: params.codec_id.clone(),
            inner,
            class,
            quality: opts.quality,
            in_channels,
            buf: Vec::new(),
            stereo_buf: Vec::new(),
            pending: VecDeque::new(),
            sample_pos: 0,
            output_params,
        })
    }

    /// Encode every complete 20 ms frame buffered so far.
    fn drain_full_frames(&mut self) -> Result<()> {
        let frame_len = self.class.frame_samples();
        while self.buf.len() >= frame_len {
            let frame: Vec<i16> = self.buf.drain(..frame_len).collect();
            self.encode_one(&frame)?;
        }
        Ok(())
    }

    /// Encode one exactly-frame-length block of mono samples as a
    /// self-contained packet.
    fn encode_one(&mut self, frame: &[i16]) -> Result<()> {
        let frame_len = self.class.frame_samples();
        debug_assert_eq!(frame.len(), frame_len);
        let bytes = self.encode_mono_frame_bytes(frame)?;
        let rate = self.class.sample_rate();
        let packet = Packet::new(0, TimeBase::from_rate(rate), bytes)
            .with_pts(self.sample_pos)
            .with_dts(self.sample_pos)
            .with_duration(frame_len as i64)
            .with_keyframe(true);
        self.sample_pos += frame_len as i64;
        self.pending.push_back(packet);
        Ok(())
    }

    /// Encode every complete stereo frame buffered so far: derive the
    /// code-9 payload from the L/R magnitudes, encode the `(L+R)/2`
    /// downmix, and emit the 17-bit in-band message prefixed to the frame.
    fn drain_stereo_frames(&mut self) -> Result<()> {
        let frame_len = self.class.frame_samples();
        while self.stereo_buf.len() >= frame_len {
            let pairs: Vec<(i16, i16)> = self.stereo_buf.drain(..frame_len).collect();
            self.encode_one_stereo(&pairs)?;
        }
        Ok(())
    }

    /// Encode one stereo frame: mono downmix packet + code-9 message.
    fn encode_one_stereo(&mut self, pairs: &[(i16, i16)]) -> Result<()> {
        let frame_len = self.class.frame_samples();
        debug_assert_eq!(pairs.len(), frame_len);
        // Mean-absolute per-channel magnitude (any consistent measure —
        // only the quantiser grid + clamp are pinned, docs §5).
        let (mut sl, mut sr) = (0.0f64, 0.0f64);
        let mut mono = Vec::with_capacity(frame_len);
        for &(l, r) in pairs {
            sl += f64::from(l.unsigned_abs());
            sr += f64::from(r.unsigned_abs());
            mono.push(downmix_mean(l, r));
        }
        let n = frame_len as f64;
        let payload = encode_stereo_payload(sl / n, sr / n);
        // Encode the mono downmix as a normal frame, then bit-prefix the
        // 17-bit code-9 message (docs intensity-stereo.md §1).
        let frame_bytes = self.encode_mono_frame_bytes(&mono)?;
        let mut w = crate::bitreader::BitWriter::new();
        w.write(0, 1).ok(); // wideband flag of the in-band pseudo-frame
        w.write(14, 4).ok(); // mode 14 = in-band signalling
        w.write(9, 4).ok(); // code 9 = intensity stereo
        w.write(u32::from(payload), 8).ok();
        for &b in &frame_bytes {
            w.write(u32::from(b), 8).ok();
        }
        let bytes = w.into_bytes();
        let rate = self.class.sample_rate();
        let packet = Packet::new(0, TimeBase::from_rate(rate), bytes)
            .with_pts(self.sample_pos)
            .with_dts(self.sample_pos)
            .with_duration(frame_len as i64)
            .with_keyframe(true);
        self.sample_pos += frame_len as i64;
        self.pending.push_back(packet);
        Ok(())
    }

    /// Encode one exactly-frame-length mono block to its packet bytes.
    fn encode_mono_frame_bytes(&mut self, frame: &[i16]) -> Result<Vec<u8>> {
        let frame_len = self.class.frame_samples();
        debug_assert_eq!(frame.len(), frame_len);
        match &mut self.inner {
            EncoderInner::Nb(enc) => {
                let mut pcm = [0i16; NB_FRAME_SAMPLES];
                pcm.copy_from_slice(frame);
                enc.encode_packet_quality(&[pcm], self.quality)
                    .map_err(|e| Error::invalid(format!("speex: {e}")))
            }
            EncoderInner::Wb(enc) => {
                let mut pcm = [0i16; QMF_WIDEBAND_FRAME];
                pcm.copy_from_slice(frame);
                enc.encode_packet_quality(&[pcm], self.quality)
                    .map_err(|e| Error::invalid(format!("speex: {e}")))
            }
            EncoderInner::Uwb(enc) => {
                let mut pcm = [0i16; UWB_FRAME_SAMPLES];
                pcm.copy_from_slice(frame);
                enc.encode_packet_quality(&[pcm], self.quality)
                    .map_err(|e| Error::invalid(format!("speex: {e}")))
            }
        }
    }
}

impl Encoder for SpeexFrameworkEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let audio = match frame {
            Frame::Audio(a) => a,
            _ => return Err(Error::unsupported("speex: encoder consumes audio frames")),
        };
        let plane = audio
            .data
            .first()
            .ok_or_else(|| Error::invalid("speex: audio frame has no data plane"))?;
        if audio.data.len() != 1 {
            return Err(Error::unsupported(
                "speex: encoder consumes interleaved S16 input (one plane)",
            ));
        }
        let ch = usize::from(self.in_channels);
        let expect = audio.samples as usize * ch * 2;
        if plane.len() < expect {
            return Err(Error::invalid(format!(
                "speex: audio frame plane holds {} bytes, need {expect}",
                plane.len()
            )));
        }
        // Two-channel input: keep the L/R pair for the per-frame code-9
        // message; the audio itself is the `(L+R)/2` downmix (§5).
        if ch == 2 {
            self.stereo_buf.reserve(audio.samples as usize);
            for s in 0..audio.samples as usize {
                let base = s * 4;
                let l = i16::from_le_bytes([plane[base], plane[base + 1]]);
                let r = i16::from_le_bytes([plane[base + 2], plane[base + 3]]);
                self.stereo_buf.push((l, r));
            }
            return self.drain_stereo_frames();
        }
        // Mono input.
        self.buf.reserve(audio.samples as usize);
        for s in 0..audio.samples as usize {
            let off = s * 2;
            self.buf
                .push(i16::from_le_bytes([plane[off], plane[off + 1]]));
        }
        self.drain_full_frames()
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.drain_full_frames()?;
        self.drain_stereo_frames()?;
        let frame_len = self.class.frame_samples();
        if !self.buf.is_empty() {
            // Zero-pad the trailing partial frame to a whole 20 ms frame
            // (Speex has no partial-frame syntax).
            let mut tail = std::mem::take(&mut self.buf);
            tail.resize(frame_len, 0);
            self.encode_one(&tail)?;
        }
        if !self.stereo_buf.is_empty() {
            let mut tail = std::mem::take(&mut self.stereo_buf);
            tail.resize(frame_len, (0, 0));
            self.encode_one_stereo(&tail)?;
        }
        Ok(())
    }
}

// ───────────────────────── factories + registration ─────────────────────────

/// Dual-API factory: build a framework [`Decoder`] for `params`. This is
/// the same factory [`register`] installs in the codec registry.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(SpeexFrameworkDecoder::from_params(params)?))
}

/// Dual-API factory: build a framework [`Encoder`] for `params`. This is
/// the same factory [`register`] installs in the codec registry.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Ok(Box::new(SpeexFrameworkEncoder::from_params(params)?))
}

/// Install the Speex codec into a [`RuntimeContext`]: decoder + encoder
/// factories under the id `"speex"`, claiming the Ogg logical-stream
/// payload magic `Speex   ` (Table 7.1's `speex_string`).
pub fn register(ctx: &mut RuntimeContext) {
    let mut caps = CodecCapabilities::audio("speex_sw");
    caps.lossy = true;
    caps.max_sample_rate = Some(32_000);
    caps.max_channels = Some(2);
    ctx.codecs.register(
        CodecInfo::new(CodecId::new(CODEC_NAME))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder)
            .encoder_options::<SpeexEncoderOptions>()
            .payload_magic(SPEEX_MAGIC.as_slice()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio_params(rate: u32, channels: u16) -> CodecParameters {
        let mut p = CodecParameters::audio(CodecId::new(CODEC_NAME));
        p.sample_rate = Some(rate);
        p.channels = Some(channels);
        p.sample_format = Some(SampleFormat::S16);
        p
    }

    fn tone_frame(samples: usize, rate: f64, channels: u16, pts: i64) -> Frame {
        let mut data = Vec::with_capacity(samples * usize::from(channels) * 2);
        for n in 0..samples {
            let v = (6000.0 * (2.0 * std::f64::consts::PI * 320.0 * n as f64 / rate).sin()) as i16;
            for _ in 0..channels {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        Frame::Audio(AudioFrame {
            samples: samples as u32,
            pts: Some(pts),
            data: vec![data],
        })
    }

    fn drain_packets(enc: &mut dyn Encoder) -> Vec<Packet> {
        let mut out = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(p) => out.push(p),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_packet: {e}"),
            }
        }
        out
    }

    #[test]
    fn registry_installs_decoder_encoder_and_payload_magic() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let id = CodecId::new(CODEC_NAME);
        assert!(ctx.codecs.has_decoder(&id));
        assert!(ctx.codecs.has_encoder(&id));
        // An Ogg BOS packet's leading bytes resolve to this codec.
        assert_eq!(
            ctx.codecs.resolve_payload_magic_ref(b"Speex   1.2beta3"),
            Some(&id),
        );
        // The quality option schema is discoverable.
        let schema = ctx.codecs.encoder_options_schema(&id).unwrap();
        assert!(schema.iter().any(|f| f.name == "quality"));
    }

    #[test]
    fn nb_encode_decode_round_trip_through_traits() {
        let params = audio_params(8_000, 1);
        let mut enc = make_encoder(&params).unwrap();
        // 2.5 frames of input across two send_frame calls + flush.
        enc.send_frame(&tone_frame(240, 8_000.0, 1, 0)).unwrap();
        enc.send_frame(&tone_frame(160, 8_000.0, 1, 240)).unwrap();
        enc.flush().unwrap();
        let packets = drain_packets(enc.as_mut());
        assert_eq!(packets.len(), 3, "400 samples pad to three 20 ms frames");
        assert_eq!(packets[0].pts, Some(0));
        assert_eq!(packets[1].pts, Some(160));
        assert_eq!(packets[0].duration, Some(160));

        // Decode through a decoder constructed from the encoder's
        // output parameters (extradata header path).
        let mut dec = make_decoder(enc.output_params()).unwrap();
        let mut decoded = 0usize;
        for p in &packets {
            dec.send_packet(p).unwrap();
        }
        dec.flush().unwrap();
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    assert_eq!(a.samples, 160);
                    assert_eq!(a.data[0].len(), 160 * 2);
                    decoded += a.samples as usize;
                }
                Ok(_) => panic!("expected audio frames"),
                Err(Error::Eof) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
        assert_eq!(decoded, 3 * 160);
    }

    #[test]
    fn wb_and_uwb_classes_round_trip() {
        for (rate, frame_samples) in [(16_000u32, 320usize), (32_000, 640)] {
            let params = audio_params(rate, 1);
            let mut enc = make_encoder(&params).unwrap();
            enc.send_frame(&tone_frame(frame_samples, f64::from(rate), 1, 0))
                .unwrap();
            enc.flush().unwrap();
            let packets = drain_packets(enc.as_mut());
            assert_eq!(packets.len(), 1, "rate {rate}");

            let mut dec = make_decoder(enc.output_params()).unwrap();
            dec.send_packet(&packets[0]).unwrap();
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    assert_eq!(a.samples as usize, frame_samples, "rate {rate}");
                }
                other => panic!("rate {rate}: {other:?}"),
            }
        }
    }

    #[test]
    fn decoder_handles_in_band_header_and_metadata_packets() {
        // Simulate the Ogg packet sequence: header, comment, audio.
        let params = audio_params(8_000, 1);
        let mut enc = make_encoder(&params).unwrap();
        enc.send_frame(&tone_frame(160, 8_000.0, 1, 0)).unwrap();
        let packets = drain_packets(enc.as_mut());

        let header = RateClass::Narrowband.stream_header(8, false);
        let tb = TimeBase::from_rate(8_000);
        let header_pkt = Packet::new(0, tb, header.write_bytes().to_vec());
        let comment_pkt = Packet::new(0, tb, b"not-a-speex-frame comment body".to_vec());

        // No extradata, no sample_rate: the in-band header must carry
        // all configuration.
        let bare = CodecParameters::audio(CodecId::new(CODEC_NAME));
        let mut dec = make_decoder(&bare).unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
        dec.send_packet(&header_pkt).unwrap();
        dec.send_packet(&comment_pkt).unwrap();
        assert!(
            matches!(dec.receive_frame(), Err(Error::NeedMore)),
            "header/comment packets must not produce audio"
        );
        dec.send_packet(&packets[0]).unwrap();
        match dec.receive_frame() {
            Ok(Frame::Audio(a)) => assert_eq!(a.samples, 160),
            other => panic!("expected audio, got {other:?}"),
        }
    }

    #[test]
    fn two_channel_stream_duplicates_the_transmitted_signal() {
        // Encode mono, decode with a 2-channel declaration: the decoder
        // emits shape-correct interleaved stereo with L == R (the
        // documented un-panned fallback while the intensity-stereo
        // reconstruction law is a docs gap).
        let params = audio_params(8_000, 1);
        let mut enc = make_encoder(&params).unwrap();
        enc.send_frame(&tone_frame(160, 8_000.0, 1, 0)).unwrap();
        let packets = drain_packets(enc.as_mut());

        let dec_params = audio_params(8_000, 2);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&packets[0]).unwrap();
        match dec.receive_frame() {
            Ok(Frame::Audio(a)) => {
                assert_eq!(a.samples, 160, "samples stay per-channel");
                assert_eq!(a.data[0].len(), 160 * 2 * 2, "interleaved stereo bytes");
                for s in 0..160 {
                    let l = &a.data[0][s * 4..s * 4 + 2];
                    let r = &a.data[0][s * 4 + 2..s * 4 + 4];
                    assert_eq!(l, r, "sample {s}: both channels carry the signal");
                }
            }
            other => panic!("expected audio, got {other:?}"),
        }
    }

    /// A panned 2-channel frame: left carries `ratio`× the right.
    fn panned_frame(samples: usize, rate: f64, ratio: f64, pts: i64) -> Frame {
        let mut data = Vec::with_capacity(samples * 4);
        for n in 0..samples {
            let base = 8000.0 * (2.0 * std::f64::consts::PI * 300.0 * n as f64 / rate).sin();
            let l = (base * ratio / (1.0 + ratio) * 2.0) as i16;
            let r = (base / (1.0 + ratio) * 2.0) as i16;
            data.extend_from_slice(&l.to_le_bytes());
            data.extend_from_slice(&r.to_le_bytes());
        }
        Frame::Audio(AudioFrame {
            samples: samples as u32,
            pts: Some(pts),
            data: vec![data],
        })
    }

    #[test]
    fn stereo_input_emits_intensity_messages_and_declares_two_channels() {
        // A 2-channel input encodes the (L+R)/2 downmix with a code-9
        // message per frame; the output stream declares 2 channels.
        let params = audio_params(8_000, 2);
        let mut enc = make_encoder(&params).unwrap();
        assert_eq!(enc.output_params().channels, Some(2), "output is stereo");
        // Left ~4× right → a clear balance.
        for i in 0..3 {
            enc.send_frame(&panned_frame(160, 8_000.0, 4.0, i * 160))
                .unwrap();
        }
        enc.flush().unwrap();
        let packets = drain_packets(enc.as_mut());
        assert!(packets.len() >= 3);
        // Every audio packet is 17 bits longer than the mono frame: the
        // in-band code-9 message. Decode it back and confirm the panning
        // (left louder) is reconstructed.
        let mut dec = make_decoder(enc.output_params()).unwrap();
        for p in &packets {
            dec.send_packet(p).unwrap();
        }
        dec.flush().unwrap();
        let mut suml = 0.0f64;
        let mut sumr = 0.0f64;
        while let Ok(Frame::Audio(a)) = dec.receive_frame() {
            assert_eq!(
                a.data[0].len(),
                a.samples as usize * 4,
                "interleaved stereo"
            );
            for s in 0..a.samples as usize {
                let l = i16::from_le_bytes([a.data[0][s * 4], a.data[0][s * 4 + 1]]);
                let r = i16::from_le_bytes([a.data[0][s * 4 + 2], a.data[0][s * 4 + 3]]);
                suml += f64::from(l.unsigned_abs());
                sumr += f64::from(r.unsigned_abs());
            }
        }
        assert!(
            suml > sumr * 1.5,
            "left should be reconstructed louder: L={suml:.0} R={sumr:.0}"
        );
    }

    #[test]
    fn output_params_extradata_is_a_valid_stream_header() {
        for (rate, mode) in [
            (8_000u32, SPEEX_MODE_NARROWBAND),
            (16_000, SPEEX_MODE_WIDEBAND),
            (32_000, SPEEX_MODE_ULTRAWIDEBAND),
        ] {
            let enc = SpeexFrameworkEncoder::from_params(&audio_params(rate, 1)).unwrap();
            let h = SpeexHeader::parse(&enc.output_params.extradata).unwrap();
            assert_eq!(h.rate, rate);
            assert_eq!(h.mode, mode);
            assert_eq!(h.mode_bitstream_version, 4);
            assert_eq!(h.nb_channels, 1);
            assert_eq!(h.frames_per_packet, 1);
            assert!(h.rate_matches_mode());
        }
    }

    #[test]
    fn quality_option_is_parsed_and_bounded() {
        let mut params = audio_params(8_000, 1);
        params.options.insert("quality", "3");
        let enc = SpeexFrameworkEncoder::from_params(&params).unwrap();
        assert_eq!(enc.quality, 3);
        assert_eq!(
            enc.output_params.bit_rate,
            Some(8_000),
            "NB quality 3 is the 8 kbps mode"
        );

        let mut bad = audio_params(8_000, 1);
        bad.options.insert("quality", "11");
        assert!(SpeexFrameworkEncoder::from_params(&bad).is_err());
    }

    #[test]
    fn nb_quality_10_encodes_and_wb_uwb_quality_10_reports_docs_gap() {
        // NB quality 10 = the r438-bound two-stage mode 7.
        let mut nb = audio_params(8_000, 1);
        nb.options.insert("quality", "10");
        let mut enc = make_encoder(&nb).unwrap();
        enc.send_frame(&tone_frame(160, 8_000.0, 1, 0)).unwrap();
        assert_eq!(drain_packets(enc.as_mut()).len(), 1);

        // WB/UWB quality 10 needs the docs-gapped high-band mode 4.
        for rate in [16_000u32, 32_000] {
            let mut p = audio_params(rate, 1);
            p.options.insert("quality", "10");
            let err = make_encoder(&p).err().expect("factory must reject");
            assert!(
                err.to_string().contains("not yet supported"),
                "rate {rate}: {err}"
            );
        }
    }

    #[test]
    fn reset_clears_state_for_seek() {
        let params = audio_params(8_000, 1);
        let mut enc = make_encoder(&params).unwrap();
        for k in 0..3 {
            enc.send_frame(&tone_frame(160, 8_000.0, 1, k * 160))
                .unwrap();
        }
        let packets = drain_packets(enc.as_mut());

        let mut dec = make_decoder(enc.output_params()).unwrap();
        dec.send_packet(&packets[0]).unwrap();
        dec.send_packet(&packets[1]).unwrap();
        let _ = dec.receive_frame().unwrap();
        // Seek: reset, then decode packet 0 again — the output must
        // equal a fresh decoder's (state fully wiped).
        dec.reset().unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
        dec.send_packet(&packets[0]).unwrap();
        let after_reset = match dec.receive_frame().unwrap() {
            Frame::Audio(a) => a.data[0].clone(),
            _ => panic!("expected audio"),
        };
        let mut fresh = make_decoder(enc.output_params()).unwrap();
        fresh.send_packet(&packets[0]).unwrap();
        let fresh_out = match fresh.receive_frame().unwrap() {
            Frame::Audio(a) => a.data[0].clone(),
            _ => panic!("expected audio"),
        };
        assert_eq!(after_reset, fresh_out);
    }

    #[test]
    fn unsupported_shapes_are_rejected_up_front() {
        // Non-Speex sample rate.
        assert!(make_encoder(&audio_params(44_100, 1)).is_err());
        // Too many channels.
        assert!(make_encoder(&audio_params(8_000, 3)).is_err());
        let mut p = audio_params(8_000, 1);
        p.sample_format = Some(SampleFormat::F32);
        assert!(make_encoder(&p).is_err());
        // Decoder with no header, no rate: construction succeeds (the
        // in-band header may still arrive) but audio packets are
        // rejected until one does.
        let bare = CodecParameters::audio(CodecId::new(CODEC_NAME));
        let mut dec = make_decoder(&bare).unwrap();
        let pkt = Packet::new(0, TimeBase::from_rate(8_000), vec![0x12, 0x34]);
        assert!(dec.send_packet(&pkt).is_err());
    }
}
