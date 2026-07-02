//! End-to-end narrowband encode → decode round-trip (round r382).
//!
//! Drives the full [`NarrowbandEncoder`] → wire bytes → parse →
//! [`NarrowbandDecoder`] path and confirms the encoded frames are valid,
//! decodable, and reconstruct a signal that tracks the input. This is a
//! *functional* round-trip (the encoder is not bit-exact against the
//! reference — see the crate's gain-Q-format gap), so the assertions
//! check structural validity + signal tracking, not sample equality.

use oxideav_speex::{
    BitReader, NarrowbandDecoder, NarrowbandEncoder, NarrowbandFrameBody, NarrowbandFrameHeader,
    NarrowbandSubmode, NB_FRAME_SAMPLES,
};

/// A deterministic voiced test signal: a pitch-periodic pulse train run
/// through a two-pole "vocal-tract" resonator, so it has both a clear
/// pitch and a spectral envelope for the LPC analysis to capture.
fn voiced_signal(frames: usize, pitch: usize) -> Vec<i16> {
    let n = frames * NB_FRAME_SAMPLES;
    let mut exc = vec![0.0f64; n];
    for (i, e) in exc.iter_mut().enumerate() {
        if i % pitch == 0 {
            *e = 1.0;
        }
    }
    // Resonator y[n] = e[n] + 1.3 y[n-1] - 0.6 y[n-2].
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut v = exc[i];
        if i >= 1 {
            v += 1.3 * y[i - 1];
        }
        if i >= 2 {
            v -= 0.6 * y[i - 2];
        }
        y[i] = v;
    }
    // Normalise to a comfortable amplitude.
    let peak = y.iter().fold(0.0f64, |m, &v| m.max(v.abs())).max(1e-9);
    y.iter()
        .map(|&v| (v / peak * 8000.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect()
}

fn decode_bytes(dec: &mut NarrowbandDecoder, bytes: &[u8]) -> [f64; NB_FRAME_SAMPLES] {
    let mut reader = BitReader::new(bytes);
    let header = NarrowbandFrameHeader::parse(&mut reader).expect("header parses");
    let submode = NarrowbandSubmode::for_id(header.mode_id).expect("valid mode");
    let body = NarrowbandFrameBody::parse(&mut reader, &submode).expect("body parses");
    dec.decode_frame(&body, &submode).expect("decodes")
}

#[test]
fn documented_modes_encode_and_decode_to_finite_pcm() {
    for mode in [2u8, 3, 4, 5, 6, 8] {
        let signal = voiced_signal(4, 40);
        let mut enc = NarrowbandEncoder::new();
        let mut dec = NarrowbandDecoder::new();
        for chunk in signal.chunks(NB_FRAME_SAMPLES) {
            let mut frame = [0i16; NB_FRAME_SAMPLES];
            frame.copy_from_slice(chunk);
            let bytes = enc.encode_frame(&frame, mode).expect("encode");
            let pcm = decode_bytes(&mut dec, &bytes);
            assert!(
                pcm.iter().all(|s| s.is_finite()),
                "mode {mode} produced non-finite PCM"
            );
        }
    }
}

#[test]
fn reconstruction_tracks_input_energy() {
    // A high-rate mode (mode 6, 18.2 kbps) should reconstruct a signal
    // whose energy is in the same ballpark as the input — the encoder is
    // not silencing the frame.
    let mode = 6u8;
    let signal = voiced_signal(6, 40);
    let mut enc = NarrowbandEncoder::new();
    let mut dec = NarrowbandDecoder::new();

    let mut in_energy = 0.0f64;
    let mut out_energy = 0.0f64;
    // Skip the first two frames (filter warm-up) when accumulating energy.
    for (fi, chunk) in signal.chunks(NB_FRAME_SAMPLES).enumerate() {
        let mut frame = [0i16; NB_FRAME_SAMPLES];
        frame.copy_from_slice(chunk);
        let bytes = enc.encode_frame(&frame, mode).expect("encode");
        let pcm = decode_bytes(&mut dec, &bytes);
        if fi >= 2 {
            for (&s, &o) in chunk.iter().zip(pcm.iter()) {
                in_energy += f64::from(s) * f64::from(s);
                out_energy += o * o;
            }
        }
    }
    assert!(in_energy > 0.0, "test signal has no energy");
    assert!(out_energy > 0.0, "reconstruction is silent");
    // The output energy should be within a couple of orders of magnitude
    // of the input — a loose but meaningful "not garbage" bound.
    let ratio = out_energy / in_energy;
    assert!(
        (0.01..100.0).contains(&ratio),
        "reconstruction energy ratio {ratio} is implausible (in={in_energy}, out={out_energy})"
    );
}

#[test]
fn encoded_bytes_reparse_to_encoder_body() {
    // The wire bytes must parse back to exactly the body the encoder
    // built — the pack/parse round-trip is exact even though the audio
    // reconstruction is approximate.
    let mode = 5u8;
    let signal = voiced_signal(1, 45);
    let mut enc = NarrowbandEncoder::new();
    let mut frame = [0i16; NB_FRAME_SAMPLES];
    frame.copy_from_slice(&signal);

    // Encode twice with identical fresh encoders → identical bytes
    // (determinism), and the body path matches the bytes path.
    let mut enc2 = NarrowbandEncoder::new();
    let body = enc2.encode_frame_body(&frame, mode).expect("body");
    let bytes = enc.encode_frame(&frame, mode).expect("bytes");

    let mut reader = BitReader::new(&bytes);
    let header = NarrowbandFrameHeader::parse(&mut reader).unwrap();
    assert_eq!(header.mode_id, mode);
    let submode = NarrowbandSubmode::for_id(mode).unwrap();
    let parsed = NarrowbandFrameBody::parse(&mut reader, &submode).unwrap();
    assert_eq!(parsed, body, "bytes body != direct body");
}
