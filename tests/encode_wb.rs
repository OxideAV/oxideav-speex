//! Speex wideband (16 kHz) encoder ↔ decoder roundtrip.
//!
//! Encode a synthetic wideband chirp with the `speex` encoder running
//! in WB mode, decode with the in-tree WB decoder, and assert the
//! output has finite energy and tracks the input spectrum.
//!
//! Two WB sub-modes are exercised:
//! * **Sub-mode 1** — spectral folding, 42-byte packets (~16.8 kbps).
//!   Selected by passing `bit_rate = Some(16_800)`.
//! * **Sub-mode 3** — stochastic split-VQ, 62-byte packets (~24.6
//!   kbps). Default (no `bit_rate`).
//!
//! ### Quality expectations
//!
//! WB sub-mode 1 is the lowest-rate extension layer (36 bits/frame).
//! The high band (4–8 kHz) is reconstructed by spectrally folding the
//! NB innovation — effectively duplicating the NB band's noise-like
//! component into the high band and relying on LPC shaping to carve
//! out the formants.
//!
//! WB sub-mode 3 transmits a proper stochastic innovation (192
//! bits/frame) so the high band has an independent shape search with
//! sign bits and a 4-bit gain index; the reconstructed 4–8 kHz signal
//! is close to the input's high-band envelope rather than a folded
//! copy of the NB innovation.
//!
//! The roundtrip PSNR we measure here is computed on a gain-corrected
//! residual (best linear gain removed before noise measurement). We
//! set the floor at **8 dB** for each sub-mode — well above the
//! "total garbage" line (< 3 dB) but below what a bit-exact encoder
//! would produce, since we still lack perceptual weighting on the
//! analysis side.

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, SampleFormat, TimeBase,
};
use oxideav_speex::decoder::make_decoder;
use oxideav_speex::encoder::make_encoder;
use oxideav_speex::wb_decoder::WB_FULL_FRAME_SIZE;

fn build_input(total_frames: usize) -> Vec<i16> {
    // Speech-like wideband signal: 3 Hz syllable envelope applied to a
    // multi-harmonic carrier (200 / 600 / 1600 / 3200 Hz) plus a
    // little high-band energy (4500 Hz) that only the WB extension
    // can carry. Amplitude kept moderate so the NB synthesis filter
    // stays comfortably under saturation.
    let sr = 16_000.0f32;
    let n = total_frames * WB_FULL_FRAME_SIZE;
    let mut out = Vec::with_capacity(n);
    let mut rng = 0x12345u32;
    for i in 0..n {
        let t = i as f32 / sr;
        let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin().abs();
        let carrier = (2.0 * std::f32::consts::PI * 200.0 * t).sin()
            + 0.5 * (2.0 * std::f32::consts::PI * 600.0 * t).sin()
            + 0.25 * (2.0 * std::f32::consts::PI * 1600.0 * t).sin()
            + 0.15 * (2.0 * std::f32::consts::PI * 3200.0 * t).sin()
            + 0.1 * (2.0 * std::f32::consts::PI * 4500.0 * t).sin();
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = (((rng >> 16) & 0x7FFF) as f32 / 32768.0) - 0.5;
        let s = 3000.0 * env * carrier + 150.0 * noise;
        out.push(s.round().clamp(-32768.0, 32767.0) as i16);
    }
    out
}

fn audio_frame_s16(samples: &[i16]) -> AudioFrame {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    AudioFrame {
        format: SampleFormat::S16,
        channels: 1,
        sample_rate: 16_000,
        samples: samples.len() as u32,
        pts: None,
        time_base: TimeBase::new(1, 16_000),
        data: vec![bytes],
    }
}

fn decode_all(decoder: &mut Box<dyn Decoder>, packets: &[Packet]) -> Vec<i16> {
    let mut pcm = Vec::new();
    for p in packets {
        decoder.send_packet(p).expect("send_packet");
        loop {
            match decoder.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    let bytes = &af.data[0];
                    for chunk in bytes.chunks_exact(2) {
                        pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("decode error: {e}"),
            }
        }
    }
    let _ = decoder.flush();
    pcm
}

fn rms_i16(x: &[i16]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let sum: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    ((sum / x.len() as f64).sqrt()) as f32
}

/// Gain-corrected segmental SNR. Scans a short alignment window to pick
/// the best time offset (decoder has a QMF + sub-frame group delay of
/// up to ~96 samples at 16 kHz), then computes the best linear gain
/// that minimises ||ref - g·tst||^2 and reports the remaining noise
/// energy under that correction.
fn snr_db(reference: &[i16], test: &[i16]) -> f32 {
    let n = reference.len().min(test.len());
    // Warm-up: skip the first ~200 samples so the QMF synth memory
    // settles and the NB decoder's first sub-frame (which emits ~zero
    // from a cold-start) doesn't dominate the SNR.
    let warm = 400;
    let max_shift = 240usize; // ~15 ms of search — generous for WB
    if n <= warm + max_shift + 64 {
        return 0.0;
    }
    let mut best = f32::NEG_INFINITY;
    for shift in 0..max_shift {
        let end = n.saturating_sub(max_shift);
        if end <= warm {
            break;
        }
        let ref_slice = &reference[warm..end];
        let tst_slice = &test[warm + shift..end + shift];
        let sig_pow: f64 = ref_slice
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            / ref_slice.len() as f64;
        let rt: f64 = ref_slice
            .iter()
            .zip(tst_slice.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum::<f64>();
        let tt: f64 = tst_slice.iter().map(|&b| (b as f64) * (b as f64)).sum();
        let g = if tt > 1e-9 { rt / tt } else { 1.0 };
        let noise_pow: f64 = ref_slice
            .iter()
            .zip(tst_slice.iter())
            .map(|(&a, &b)| {
                let d = (a as f64) - g * (b as f64);
                d * d
            })
            .sum::<f64>()
            / ref_slice.len() as f64;
        if noise_pow < 1e-9 {
            return 200.0;
        }
        let snr = 10.0 * (sig_pow / noise_pow).log10();
        if snr as f32 > best {
            best = snr as f32;
        }
    }
    best
}

#[test]
fn encode_decode_wb_zero_input_stays_quiet() {
    // Sanity: 16 kHz zeros should decode back to (near-)silence.
    let input = vec![0i16; 10 * WB_FULL_FRAME_SIZE];
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(16_000);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = make_encoder(&params).expect("speex wb encoder");
    enc.send_frame(&Frame::Audio(audio_frame_s16(&input)))
        .unwrap();
    enc.flush().unwrap();
    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("{e}"),
        }
    }
    let mut dec_params = enc.output_params().clone();
    dec_params.codec_id = CodecId::new("speex");
    let mut dec = make_decoder(&dec_params).expect("speex wb decoder");
    let decoded = decode_all(&mut dec, &packets);
    let out_rms = rms_i16(&decoded);
    assert!(
        out_rms < 500.0,
        "zero-input WB encode should decode to (near-)silence, got RMS {out_rms}"
    );
}

/// Shared helper: encode → decode a WB stream and return `(packets, decoded_pcm)`.
fn wb_encode_decode(bit_rate: Option<u64>, input: &[i16]) -> (Vec<Packet>, Vec<i16>) {
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(16_000);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    params.bit_rate = bit_rate;
    let mut enc = make_encoder(&params).expect("speex wb encoder");
    enc.send_frame(&Frame::Audio(audio_frame_s16(input)))
        .expect("send_frame");
    enc.flush().expect("flush encoder");

    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("encoder receive_packet: {e}"),
        }
    }

    let mut dec_params = enc.output_params().clone();
    dec_params.codec_id = CodecId::new("speex");
    let mut dec = make_decoder(&dec_params).expect("speex wb decoder");
    let decoded = decode_all(&mut dec, &packets);
    (packets, decoded)
}

#[test]
fn encode_decode_wb_submode_1_roundtrip_is_coherent() {
    let input = build_input(25);
    let input_rms = rms_i16(&input);
    assert!(input_rms > 100.0);

    // bit_rate ≤ 20_000 picks WB sub-mode 1 (folding, 336 bits).
    let (packets, decoded) = wb_encode_decode(Some(16_800), &input);
    let n_frames = input.len() / WB_FULL_FRAME_SIZE;
    assert_eq!(packets.len(), n_frames);
    for p in &packets {
        assert_eq!(
            p.data.len(),
            42,
            "sub-mode 1 packets are 336 bits = 42 bytes"
        );
    }

    let out_rms = rms_i16(&decoded);
    eprintln!("WB-1 input RMS = {input_rms}, decoded RMS = {out_rms}");
    assert!(out_rms > 10.0);

    let snr = snr_db(&input, &decoded);
    eprintln!("WB sub-mode 1 gain-corrected SNR ≈ {snr:.1} dB (floor 8 dB)");
    assert!(snr > 8.0, "WB-1 SNR should clear 8 dB, got {snr:.1} dB");
}

#[test]
fn encode_decode_wb_submode_3_roundtrip_is_coherent() {
    let input = build_input(25);
    let input_rms = rms_i16(&input);
    assert!(input_rms > 100.0);

    // No bit_rate → default WB sub-mode 3 (stochastic split-VQ, 492 bits).
    let (packets, decoded) = wb_encode_decode(None, &input);
    let n_frames = input.len() / WB_FULL_FRAME_SIZE;
    assert_eq!(packets.len(), n_frames);
    for p in &packets {
        assert_eq!(
            p.data.len(),
            62,
            "sub-mode 3 packets are 492 bits = 62 bytes"
        );
    }

    let out_rms = rms_i16(&decoded);
    eprintln!("WB-3 input RMS = {input_rms}, decoded RMS = {out_rms}");
    assert!(out_rms > 10.0);

    let snr = snr_db(&input, &decoded);
    eprintln!("WB sub-mode 3 gain-corrected SNR ≈ {snr:.1} dB (floor 8 dB)");
    assert!(snr > 8.0, "WB-3 SNR should clear 8 dB, got {snr:.1} dB");
}

/// 200 ms at 16 kHz of a modulated sine + harmonic sweep. Speech-like
/// spectral envelope (syllable rate AM, multi-harmonic carrier) chosen
/// so the NB residual is well-behaved — a pure sweeping chirp would
/// hit the first-cut NB encoder's known weakness for unvoiced transient
/// content, independent of the WB layer.
fn sine_chirp_200ms() -> Vec<i16> {
    let sr = 16_000.0f32;
    let n_ms = 200.0f32;
    let n = (sr * n_ms / 1000.0) as usize;
    let n = (n / WB_FULL_FRAME_SIZE) * WB_FULL_FRAME_SIZE;
    assert!(n >= 2 * WB_FULL_FRAME_SIZE, "need ≥2 WB frames");
    let mut input = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr;
        // Syllable-rate amplitude envelope.
        let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin().abs();
        // Multi-harmonic carrier (speech-like formant structure).
        let carrier = (2.0 * std::f32::consts::PI * 200.0 * t).sin()
            + 0.5 * (2.0 * std::f32::consts::PI * 600.0 * t).sin()
            + 0.25 * (2.0 * std::f32::consts::PI * 1600.0 * t).sin()
            + 0.15 * (2.0 * std::f32::consts::PI * 3200.0 * t).sin();
        // Gentle "chirp-like" cross-band energy — a slow modulation
        // of a mid-band tone rather than a true sweeping chirp.
        let chirp = 0.1
            * (2.0
                * std::f32::consts::PI
                * (2200.0 + 300.0 * (2.0 * std::f32::consts::PI * 5.0 * t).sin())
                * t)
                .sin();
        let s = 3000.0 * env * (carrier + chirp);
        input.push(s.round().clamp(-32768.0, 32767.0) as i16);
    }
    input
}

#[test]
fn encode_decode_wb_submode_1_sine_chirp_snr_above_floor() {
    let input = sine_chirp_200ms();
    let (_, decoded) = wb_encode_decode(Some(16_800), &input);
    let snr = snr_db(&input, &decoded);
    eprintln!("WB-1 sine+chirp SNR ≈ {snr:.1} dB (floor 8 dB)");
    assert!(
        snr > 8.0,
        "WB-1 sine+chirp SNR should clear 8 dB, got {snr:.1} dB"
    );
}

#[test]
fn encode_decode_wb_submode_3_sine_chirp_snr_above_floor() {
    let input = sine_chirp_200ms();
    let (_, decoded) = wb_encode_decode(None, &input);
    let snr = snr_db(&input, &decoded);
    eprintln!("WB-3 sine+chirp SNR ≈ {snr:.1} dB (floor 8 dB)");
    assert!(
        snr > 8.0,
        "WB-3 sine+chirp SNR should clear 8 dB, got {snr:.1} dB"
    );
}

#[test]
fn encode_decode_wb_submode_2_roundtrip_is_coherent() {
    // bit_rate 18_001..=22_000 picks WB sub-mode 2 (LBR split-VQ, 412
    // bits → 52 bytes / 20 ms frame after 4-bit pad).
    let input = build_input(25);
    let input_rms = rms_i16(&input);
    assert!(input_rms > 100.0);

    let (packets, decoded) = wb_encode_decode(Some(20_600), &input);
    let n_frames = input.len() / WB_FULL_FRAME_SIZE;
    assert_eq!(packets.len(), n_frames);
    for p in &packets {
        assert_eq!(
            p.data.len(),
            52,
            "sub-mode 2 packets are 412 bits → 52 bytes"
        );
    }
    let out_rms = rms_i16(&decoded);
    eprintln!("WB-2 input RMS = {input_rms}, decoded RMS = {out_rms}");
    assert!(out_rms > 10.0);

    let snr = snr_db(&input, &decoded);
    eprintln!("WB sub-mode 2 gain-corrected SNR ≈ {snr:.1} dB (floor 8 dB)");
    assert!(snr > 8.0, "WB-2 SNR should clear 8 dB, got {snr:.1} dB");
}

#[test]
fn encode_decode_wb_submode_4_roundtrip_is_coherent() {
    // bit_rate > 28_000 picks WB sub-mode 4 (double-codebook split-VQ,
    // 652 bits → 82 bytes / 20 ms frame after 4-bit pad).
    let input = build_input(25);
    let input_rms = rms_i16(&input);
    assert!(input_rms > 100.0);

    let (packets, decoded) = wb_encode_decode(Some(32_600), &input);
    let n_frames = input.len() / WB_FULL_FRAME_SIZE;
    assert_eq!(packets.len(), n_frames);
    for p in &packets {
        assert_eq!(
            p.data.len(),
            82,
            "sub-mode 4 packets are 652 bits → 82 bytes"
        );
    }
    let out_rms = rms_i16(&decoded);
    eprintln!("WB-4 input RMS = {input_rms}, decoded RMS = {out_rms}");
    assert!(out_rms > 10.0);

    let snr = snr_db(&input, &decoded);
    eprintln!("WB sub-mode 4 gain-corrected SNR ≈ {snr:.1} dB (floor 8 dB)");
    assert!(snr > 8.0, "WB-4 SNR should clear 8 dB, got {snr:.1} dB");
}

#[test]
fn encode_decode_wb_submode_2_sine_chirp_snr_above_floor() {
    let input = sine_chirp_200ms();
    let (_, decoded) = wb_encode_decode(Some(20_600), &input);
    let snr = snr_db(&input, &decoded);
    eprintln!("WB-2 sine+chirp SNR ≈ {snr:.1} dB (floor 8 dB)");
    assert!(
        snr > 8.0,
        "WB-2 sine+chirp SNR should clear 8 dB, got {snr:.1} dB"
    );
}

#[test]
fn encode_decode_wb_submode_4_sine_chirp_snr_above_floor() {
    let input = sine_chirp_200ms();
    let (_, decoded) = wb_encode_decode(Some(32_600), &input);
    let snr = snr_db(&input, &decoded);
    eprintln!("WB-4 sine+chirp SNR ≈ {snr:.1} dB (floor 8 dB)");
    assert!(
        snr > 8.0,
        "WB-4 sine+chirp SNR should clear 8 dB, got {snr:.1} dB"
    );
}
