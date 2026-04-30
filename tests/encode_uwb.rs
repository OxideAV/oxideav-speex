//! Speex ultra-wideband (32 kHz) encoder ↔ decoder roundtrip.
//!
//! Encodes a synthetic UWB waveform through the stacked NB + WB + UWB
//! pipeline and decodes the resulting packets with the in-tree UWB
//! decoder. Both UWB layer flavours are exercised:
//!
//! * **Null layer (uwb-bit = 0)** — the UWB encoder emits WB mode-3
//!   followed by a single 0 bit. Decoder zero-pads the 8–16 kHz half
//!   through its QMF synthesis and we recover the 0–8 kHz spectrum
//!   that WB-3 produces, just upsampled to 32 kHz.
//! * **Folding layer (uwb-bit = 1, submode = 1)** — 36 bits of UWB
//!   extension on top of WB mode-3: 12 LSP bits + 4 × 5-bit
//!   per-sub-frame folding gain. The 8–16 kHz half is a spectrally
//!   folded copy of the WB high-band excitation, gain-balanced by
//!   the LPC synthesis filter.
//!
//! The gain-corrected SNR floor here is lower (6 dB) than the WB tests
//! because UWB's folding extension is an approximation of the 8–16 kHz
//! band rather than a direct encode — the test guards against total
//! garbage output, not bit-exact reproduction.

use oxideav_core::Decoder;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, SampleFormat};
use oxideav_speex::decoder::make_decoder;
use oxideav_speex::encoder::make_encoder;
use oxideav_speex::uwb_decoder::UWB_FULL_FRAME_SIZE;

fn build_input(total_frames: usize) -> Vec<i16> {
    let sr = 32_000.0f32;
    let n = total_frames * UWB_FULL_FRAME_SIZE;
    let mut out = Vec::with_capacity(n);
    let mut rng = 0x12345u32;
    for i in 0..n {
        let t = i as f32 / sr;
        // Syllable-rate envelope × mixed-harmonic speech-like carrier.
        let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin().abs();
        let carrier = (2.0 * std::f32::consts::PI * 200.0 * t).sin()
            + 0.5 * (2.0 * std::f32::consts::PI * 600.0 * t).sin()
            + 0.25 * (2.0 * std::f32::consts::PI * 1600.0 * t).sin()
            + 0.15 * (2.0 * std::f32::consts::PI * 3200.0 * t).sin()
            + 0.08 * (2.0 * std::f32::consts::PI * 6500.0 * t).sin();
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
        samples: samples.len() as u32,
        pts: None,
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

/// Gain-corrected segmental SNR with a short cross-correlation search
/// — mirrors the WB roundtrip test. The UWB pipeline has a slightly
/// larger group delay (two nested QMFs plus sub-frame delays) so we
/// widen the alignment window.
fn snr_db(reference: &[i16], test: &[i16]) -> f32 {
    let n = reference.len().min(test.len());
    let warm = 1000;
    let max_shift = 480usize; // ~15 ms at 32 kHz
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

fn uwb_encode_decode(bit_rate: Option<u64>, input: &[i16]) -> (Vec<Packet>, Vec<i16>) {
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(32_000);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    params.bit_rate = bit_rate;
    let mut enc = make_encoder(&params).expect("speex uwb encoder");
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
    let mut dec = make_decoder(&dec_params).expect("speex uwb decoder");
    let decoded = decode_all(&mut dec, &packets);
    (packets, decoded)
}

#[test]
fn encode_decode_uwb_zero_input_stays_quiet() {
    let input = vec![0i16; 6 * UWB_FULL_FRAME_SIZE];
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(32_000);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = make_encoder(&params).expect("speex uwb encoder");
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
    let mut dec = make_decoder(&dec_params).expect("speex uwb decoder");
    let decoded = decode_all(&mut dec, &packets);
    let out_rms = rms_i16(&decoded);
    assert!(
        out_rms < 500.0,
        "zero-input UWB encode should decode to (near-)silence, got RMS {out_rms}"
    );
}

#[test]
fn encode_decode_uwb_null_layer_roundtrip_is_coherent() {
    // bit_rate ≤ 25_000 selects null UWB layer.
    let input = build_input(20);
    let (packets, decoded) = uwb_encode_decode(Some(24_000), &input);
    let n_frames = input.len() / UWB_FULL_FRAME_SIZE;
    assert_eq!(packets.len(), n_frames);
    // Null UWB layer: 492 (WB-3) + 1 (uwb-bit=0) = 493 bits → 62 bytes
    // (4-bit zero pad).
    for p in &packets {
        assert_eq!(
            p.data.len(),
            62,
            "UWB null layer packets are 493 bits = 62 bytes"
        );
    }

    let out_rms = rms_i16(&decoded);
    assert!(out_rms > 10.0, "decoded RMS should be non-trivial");

    let snr = snr_db(&input, &decoded);
    eprintln!("UWB null-layer gain-corrected SNR ≈ {snr:.1} dB (floor 6 dB)");
    assert!(snr > 6.0, "UWB null SNR should clear 6 dB, got {snr:.1} dB");
}

#[test]
fn encode_decode_uwb_folding_layer_roundtrip_is_coherent() {
    // Default bit_rate (None) selects folding UWB layer.
    let input = build_input(20);
    let (packets, decoded) = uwb_encode_decode(None, &input);
    let n_frames = input.len() / UWB_FULL_FRAME_SIZE;
    assert_eq!(packets.len(), n_frames);
    // Folding UWB: 492 (WB-3) + 36 = 528 bits → 66 bytes.
    for p in &packets {
        assert_eq!(
            p.data.len(),
            66,
            "UWB folding layer packets are 528 bits = 66 bytes"
        );
    }

    let out_rms = rms_i16(&decoded);
    assert!(out_rms > 10.0, "decoded RMS should be non-trivial");

    let snr = snr_db(&input, &decoded);
    eprintln!("UWB folding gain-corrected SNR ≈ {snr:.1} dB (floor 6 dB)");
    assert!(
        snr > 6.0,
        "UWB folding SNR should clear 6 dB, got {snr:.1} dB"
    );
}
