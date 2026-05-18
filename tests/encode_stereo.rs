//! End-to-end intensity-stereo encode → decode round trip.
//!
//! The Speex manual §5.5 Table 5.1 code 9 packet carries a per-frame
//! `(sign, dexp, e_ratio_idx)` triple ahead of the mono CELP frame.
//! Our encoder factory now accepts `channels=2` S16 input at the NB,
//! WB, and UWB rates, emits the side channel for every frame, mixes
//! L+R to mono, and runs the standard CELP analysis on the mono
//! signal. The companion top-level decoder reads the side channel and
//! expands the mono CELP synthesis back to interleaved L/R via
//! `StereoState`.
//!
//! These tests feed a known L/R balance into the encoder, decode the
//! packets it produces, and confirm the recovered L/R RMS ratio is
//! within the smoothing-filter tolerance the decoder uses.

#![allow(clippy::needless_range_loop)]

use oxideav_core::{AudioFrame, CodecId, CodecParameters, Error, Frame, SampleFormat};
use oxideav_speex::decoder::make_decoder;
use oxideav_speex::encoder::make_encoder;
use oxideav_speex::nb_decoder::NB_FRAME_SIZE;
use oxideav_speex::wb_decoder::WB_FULL_FRAME_SIZE;

/// Generate `n_frames * frame_size` L/R sample pairs of a sine at
/// `freq` Hz. L amplitude is `amp_l`, R amplitude is `amp_r`. Returns
/// interleaved i16-LE bytes ready to feed into `AudioFrame::data[0]`.
fn lr_sine_bytes(
    n_frames: usize,
    frame_size: usize,
    sample_rate: f32,
    freq: f32,
    amp_l: f32,
    amp_r: f32,
) -> Vec<u8> {
    let total = n_frames * frame_size;
    let mut out = Vec::with_capacity(total * 4);
    for i in 0..total {
        let t = i as f32 / sample_rate;
        let s = (2.0 * std::f32::consts::PI * freq * t).sin();
        let l = (amp_l * s).round().clamp(-32768.0, 32767.0) as i16;
        let r = (amp_r * s).round().clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&l.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
    }
    out
}

fn rms_i16(x: &[i16]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let s: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    ((s / x.len() as f64).sqrt()) as f32
}

fn decode_all(
    extradata: Vec<u8>,
    sample_rate: u32,
    packets: Vec<oxideav_core::Packet>,
) -> (Vec<i16>, Vec<i16>) {
    let mut dec_params = CodecParameters::audio(CodecId::new("speex"));
    dec_params.sample_rate = Some(sample_rate);
    dec_params.channels = Some(2);
    dec_params.sample_format = Some(SampleFormat::S16);
    dec_params.extradata = extradata;
    let mut dec = make_decoder(&dec_params).expect("stereo decoder factory");

    let mut left = Vec::new();
    let mut right = Vec::new();
    for p in &packets {
        dec.send_packet(p).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    for chunk in af.data[0].chunks_exact(4) {
                        let l = i16::from_le_bytes([chunk[0], chunk[1]]);
                        let r = i16::from_le_bytes([chunk[2], chunk[3]]);
                        left.push(l);
                        right.push(r);
                    }
                }
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("decode: {e}"),
            }
        }
    }
    let _ = dec.flush();
    (left, right)
}

#[test]
fn encoder_factory_accepts_nb_stereo_s16() {
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(8_000);
    params.channels = Some(2);
    params.sample_format = Some(SampleFormat::S16);
    let enc = make_encoder(&params).expect("factory accepts NB stereo S16");
    assert_eq!(enc.output_params().channels, Some(2));
    // Header byte 48..52 should reflect nb_channels=2.
    let extradata = &enc.output_params().extradata;
    assert!(
        extradata.len() >= 52,
        "extradata too short: {}",
        extradata.len()
    );
    let nb_channels =
        u32::from_le_bytes([extradata[48], extradata[49], extradata[50], extradata[51]]);
    assert_eq!(nb_channels, 2, "speex header nb_channels field");
}

#[test]
fn encoder_factory_accepts_wb_stereo_s16() {
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(16_000);
    params.channels = Some(2);
    params.sample_format = Some(SampleFormat::S16);
    let enc = make_encoder(&params).expect("factory accepts WB stereo S16");
    assert_eq!(enc.output_params().channels, Some(2));
}

#[test]
fn encoder_factory_accepts_uwb_stereo_s16() {
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(32_000);
    params.channels = Some(2);
    params.sample_format = Some(SampleFormat::S16);
    let enc = make_encoder(&params).expect("factory accepts UWB stereo S16");
    assert_eq!(enc.output_params().channels, Some(2));
}

#[test]
fn encoder_factory_rejects_four_channels() {
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(8_000);
    params.channels = Some(4);
    params.sample_format = Some(SampleFormat::S16);
    let err = match make_encoder(&params) {
        Ok(_) => panic!("factory must reject quad-channel input"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("4 channels") || msg.contains("got 4"),
        "error must mention channel count: {msg}"
    );
}

#[test]
fn nb_stereo_packets_are_larger_than_mono_by_at_least_one_byte() {
    // 300 bits NB mode 5 ⇒ 38 bytes (with 4-bit pad).
    // 17 bits stereo prefix + 300 bits CELP = 317 bits ⇒ 40 bytes
    // (rounded up to byte boundary, with up to 7 bits of pad).
    let mut mono_params = CodecParameters::audio(CodecId::new("speex"));
    mono_params.sample_rate = Some(8_000);
    mono_params.channels = Some(1);
    mono_params.sample_format = Some(SampleFormat::S16);
    let mut mono_enc = make_encoder(&mono_params).unwrap();

    let mut stereo_params = CodecParameters::audio(CodecId::new("speex"));
    stereo_params.sample_rate = Some(8_000);
    stereo_params.channels = Some(2);
    stereo_params.sample_format = Some(SampleFormat::S16);
    let mut stereo_enc = make_encoder(&stereo_params).unwrap();

    // 1 frame of NB audio — mono and stereo with identical L=R.
    let mut mono_bytes = Vec::with_capacity(NB_FRAME_SIZE * 2);
    let mut stereo_bytes = Vec::with_capacity(NB_FRAME_SIZE * 4);
    for i in 0..NB_FRAME_SIZE {
        let t = i as f32;
        let v = (4000.0 * (2.0 * std::f32::consts::PI * 400.0 * t / 8_000.0).sin())
            .round()
            .clamp(-32768.0, 32767.0) as i16;
        mono_bytes.extend_from_slice(&v.to_le_bytes());
        stereo_bytes.extend_from_slice(&v.to_le_bytes()); // L
        stereo_bytes.extend_from_slice(&v.to_le_bytes()); // R
    }

    mono_enc
        .send_frame(&Frame::Audio(AudioFrame {
            samples: NB_FRAME_SIZE as u32,
            pts: None,
            data: vec![mono_bytes],
        }))
        .unwrap();
    mono_enc.flush().unwrap();

    stereo_enc
        .send_frame(&Frame::Audio(AudioFrame {
            samples: NB_FRAME_SIZE as u32,
            pts: None,
            data: vec![stereo_bytes],
        }))
        .unwrap();
    stereo_enc.flush().unwrap();

    let mono_pkt = mono_enc.receive_packet().unwrap();
    let stereo_pkt = stereo_enc.receive_packet().unwrap();
    assert_eq!(mono_pkt.data.len(), 38, "mono NB-5 should be 38 bytes");
    // 317 bits → ceil(317 / 8) = 40 bytes.
    assert_eq!(
        stereo_pkt.data.len(),
        40,
        "stereo NB-5 should be 40 bytes (17 prefix + 300 CELP = 317 bits)"
    );
}

#[test]
fn nb_stereo_roundtrip_recovers_left_loud_balance() {
    // Encode 40 frames of L=4× R. Decoded L/R RMS ratio should be
    // ≥ 2.5 after the smoothing filter has converged.
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(8_000);
    params.channels = Some(2);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = make_encoder(&params).unwrap();

    let n_frames = 40;
    let lr = lr_sine_bytes(n_frames, NB_FRAME_SIZE, 8_000.0, 500.0, 8000.0, 2000.0);
    enc.send_frame(&Frame::Audio(AudioFrame {
        samples: (n_frames * NB_FRAME_SIZE) as u32,
        pts: None,
        data: vec![lr],
    }))
    .unwrap();
    enc.flush().unwrap();

    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e}"),
        }
    }
    assert_eq!(packets.len(), n_frames, "expected one packet per frame");

    let extradata = enc.output_params().extradata.clone();
    let (left, right) = decode_all(extradata, 8_000, packets);
    assert!(!left.is_empty());
    assert_eq!(left.len(), right.len());

    // Drop the first ~1024 samples so the 0.98/0.02 smoothing filter
    // has settled.
    let warm = 1024.min(left.len() / 2);
    let l_rms = rms_i16(&left[warm..]);
    let r_rms = rms_i16(&right[warm..]);
    let ratio = l_rms / r_rms.max(1.0);
    eprintln!("NB stereo L/R RMS ratio (L=4·R input): {ratio:.2}");
    assert!(
        ratio > 2.5,
        "L should be substantially louder than R, got L/R={ratio:.2}"
    );
    assert!(
        ratio < 5.5,
        "L/R ratio too large (clipping or quantiser bug?): {ratio:.2}"
    );
}

#[test]
fn nb_stereo_roundtrip_recovers_right_loud_balance() {
    // Same setup with R = 4·L.
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(8_000);
    params.channels = Some(2);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = make_encoder(&params).unwrap();

    let n_frames = 40;
    let lr = lr_sine_bytes(n_frames, NB_FRAME_SIZE, 8_000.0, 500.0, 2000.0, 8000.0);
    enc.send_frame(&Frame::Audio(AudioFrame {
        samples: (n_frames * NB_FRAME_SIZE) as u32,
        pts: None,
        data: vec![lr],
    }))
    .unwrap();
    enc.flush().unwrap();

    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e}"),
        }
    }

    let extradata = enc.output_params().extradata.clone();
    let (left, right) = decode_all(extradata, 8_000, packets);
    let warm = 1024.min(left.len() / 2);
    let l_rms = rms_i16(&left[warm..]);
    let r_rms = rms_i16(&right[warm..]);
    let ratio = r_rms / l_rms.max(1.0);
    eprintln!("NB stereo R/L RMS ratio (R=4·L input): {ratio:.2}");
    assert!(
        ratio > 2.5,
        "R should be substantially louder than L, got R/L={ratio:.2}"
    );
}

#[test]
fn nb_stereo_roundtrip_balanced_lr_stays_balanced() {
    // L = R input. After decode, ratio should be very close to 1.
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(8_000);
    params.channels = Some(2);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = make_encoder(&params).unwrap();

    let n_frames = 30;
    let lr = lr_sine_bytes(n_frames, NB_FRAME_SIZE, 8_000.0, 400.0, 5000.0, 5000.0);
    enc.send_frame(&Frame::Audio(AudioFrame {
        samples: (n_frames * NB_FRAME_SIZE) as u32,
        pts: None,
        data: vec![lr],
    }))
    .unwrap();
    enc.flush().unwrap();

    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e}"),
        }
    }
    let extradata = enc.output_params().extradata.clone();
    let (left, right) = decode_all(extradata, 8_000, packets);
    let warm = 1024.min(left.len() / 2);
    let l_rms = rms_i16(&left[warm..]);
    let r_rms = rms_i16(&right[warm..]);
    let ratio = (l_rms / r_rms.max(1.0)).max(r_rms / l_rms.max(1.0));
    eprintln!("NB stereo balanced L=R: ratio={ratio:.3}");
    assert!(
        ratio < 1.15,
        "balanced L=R should round-trip with ratio≈1, got {ratio:.3}"
    );
}

#[test]
fn wb_stereo_roundtrip_recovers_left_loud_balance() {
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(16_000);
    params.channels = Some(2);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = make_encoder(&params).unwrap();

    let n_frames = 20;
    let lr = lr_sine_bytes(
        n_frames,
        WB_FULL_FRAME_SIZE,
        16_000.0,
        800.0,
        8000.0,
        2000.0,
    );
    enc.send_frame(&Frame::Audio(AudioFrame {
        samples: (n_frames * WB_FULL_FRAME_SIZE) as u32,
        pts: None,
        data: vec![lr],
    }))
    .unwrap();
    enc.flush().unwrap();

    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e}"),
        }
    }
    let extradata = enc.output_params().extradata.clone();
    let (left, right) = decode_all(extradata, 16_000, packets);
    let warm = 2048.min(left.len() / 2);
    let l_rms = rms_i16(&left[warm..]);
    let r_rms = rms_i16(&right[warm..]);
    let ratio = l_rms / r_rms.max(1.0);
    eprintln!("WB stereo L/R ratio (L=4·R input): {ratio:.2}");
    assert!(
        ratio > 2.5,
        "WB stereo: L should be substantially louder, got {ratio:.2}"
    );
}
