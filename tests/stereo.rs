//! End-to-end stereo (intensity side-channel) decode.
//!
//! The crate ships a mono-only encoder, so we can't produce a real
//! stereo encoded clip out of our own encoder. Instead we build the
//! bitstream by hand: encode a mono NB frame through `NbEncoder`, then
//! prepend an in-band intensity-stereo request packet (`m=14, id=9`)
//! carrying a known (sign, dexp, e_ratio_idx) triple. Wrap the whole
//! thing in a Speex header with `nb_channels=2` and confirm the top-
//! level decoder returns interleaved L/R samples whose L/R ratio
//! matches the encoded balance within the smoothing tolerance.
//!
//! Reference for the bitstream layout:
//!   * `libspeex/stereo.c`       — state + decode + encode.
//!   * `libspeex/speex_callbacks.c` — `speex_inband_handler`, 4-bit id.

#![allow(clippy::needless_range_loop)]

use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, TimeBase};
use oxideav_speex::bitwriter::BitWriter;
use oxideav_speex::decoder::make_decoder;
use oxideav_speex::header::{SPEEX_HEADER_SIZE, SPEEX_SIGNATURE};
use oxideav_speex::nb_decoder::NB_FRAME_SIZE;
use oxideav_speex::nb_encoder::NbEncoder;

/// Build the 80-byte Speex-in-Ogg header for an NB stereo stream.
fn stereo_nb_header() -> Vec<u8> {
    let mut h = vec![0u8; SPEEX_HEADER_SIZE];
    h[0..8].copy_from_slice(SPEEX_SIGNATURE);
    let ver = b"oxideav-stereo-test";
    h[8..8 + ver.len()].copy_from_slice(ver);
    h[28..32].copy_from_slice(&1u32.to_le_bytes()); // version_id
    h[32..36].copy_from_slice(&(SPEEX_HEADER_SIZE as u32).to_le_bytes());
    h[36..40].copy_from_slice(&8_000u32.to_le_bytes()); // rate
    h[40..44].copy_from_slice(&0u32.to_le_bytes()); // mode = NB
    h[44..48].copy_from_slice(&4u32.to_le_bytes()); // mode_bitstream_version
    h[48..52].copy_from_slice(&2u32.to_le_bytes()); // **stereo**
    h[52..56].copy_from_slice(&15_000i32.to_le_bytes()); // nominal bitrate
    h[56..60].copy_from_slice(&(NB_FRAME_SIZE as u32).to_le_bytes());
    h[60..64].copy_from_slice(&0u32.to_le_bytes()); // no VBR
    h[64..68].copy_from_slice(&1u32.to_le_bytes()); // frames_per_packet = 1
    h[68..72].copy_from_slice(&0u32.to_le_bytes()); // no extra headers
    h
}

/// Generate one packet: in-band stereo side channel + NB mode-5 frame.
/// Bit layout mirrors the reference's `speex_encode_stereo_int`:
///   * 5-bit in-band marker = 14 (wideband=0 || m=14 in 4 bits),
///   * 4-bit request id = 9 (SPEEX_INBAND_STEREO),
///   * 1-bit sign + 5-bit dexp + 2-bit e_ratio_idx,
///   * encoded NB frame (wideband=0 + m=5 + ...) from `NbEncoder`.
fn build_stereo_packet(
    enc: &mut NbEncoder,
    mono: &[f32],
    sign: u32,
    dexp: u32,
    eidx: u32,
) -> Vec<u8> {
    let mut bw = BitWriter::new();
    // 5 bits: the reference packs value 14 into 5 bits, which on the
    // wire is `0 || 1110` = wideband-bit=0 followed by m=14.
    bw.write_bits(14, 5);
    bw.write_bits(9, 4);
    bw.write_bits(sign, 1);
    bw.write_bits(dexp, 5);
    bw.write_bits(eidx, 2);
    // Then the actual CELP frame — wb=0 + sub=5 + payload, appended by
    // the NB encoder.
    enc.encode_frame(mono, &mut bw).unwrap();
    bw.finish()
}

fn audio_samples_interleaved(bytes: &[u8]) -> Vec<(i16, i16)> {
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let l = i16::from_le_bytes([chunk[0], chunk[1]]);
        let r = i16::from_le_bytes([chunk[2], chunk[3]]);
        out.push((l, r));
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

#[test]
fn stereo_decode_balances_left_and_right_per_payload() {
    // Synthetic mono input: a 500 Hz sine with moderate amplitude.
    let mut mono = [0.0f32; NB_FRAME_SIZE];
    for i in 0..NB_FRAME_SIZE {
        let t = i as f32;
        mono[i] = 6000.0 * (2.0 * std::f32::consts::PI * 500.0 * t / 8_000.0).sin();
    }

    let mut enc = NbEncoder::with_submode(5).expect("NB mode 5");
    // 30 frames at constant balance so the smoothing filter has time to
    // converge. sign=0 (positive), dexp=8 ⇒ balance = exp(0.25·8) ≈ 7.389,
    // eidx=3 ⇒ e_ratio = 0.5. That gives:
    //   e_right = 1 / sqrt(0.5 · (1 + 7.389)) = 1 / sqrt(4.195) ≈ 0.488
    //   e_left  = sqrt(7.389) · e_right ≈ 2.718 · 0.488 ≈ 1.327
    // So the left channel should be louder than the right by a factor
    // of sqrt(balance) ≈ 2.72.
    let frames = 30usize;
    let mut packets = Vec::with_capacity(frames);
    for _ in 0..frames {
        let data = build_stereo_packet(&mut enc, &mono, 0, 8, 3);
        packets.push(Packet {
            stream_index: 0,
            time_base: TimeBase::new(1, 8_000),
            pts: None,
            dts: None,
            duration: Some(NB_FRAME_SIZE as i64),
            flags: Default::default(),
            data,
        });
    }

    // Build decoder from the stereo header.
    let mut dec_params = CodecParameters::audio(CodecId::new("speex"));
    dec_params.sample_rate = Some(8_000);
    dec_params.channels = Some(2);
    dec_params.sample_format = Some(oxideav_core::SampleFormat::S16);
    dec_params.extradata = stereo_nb_header();
    let mut dec = make_decoder(&dec_params).expect("stereo decoder factory");

    let mut left = Vec::new();
    let mut right = Vec::new();
    for p in &packets {
        dec.send_packet(p).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    assert_eq!(af.channels, 2, "frame must be stereo");
                    for (l, r) in audio_samples_interleaved(&af.data[0]) {
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

    assert!(!left.is_empty(), "decoder produced no stereo samples");
    assert_eq!(left.len(), right.len(), "L and R must have same length");

    // Discard the first ~1000 samples so the 0.98/0.02 smoothing filter
    // has fully converged to the per-frame L/R gains.
    let warm = 1024usize.min(left.len() / 2);
    let l_rms = rms_i16(&left[warm..]);
    let r_rms = rms_i16(&right[warm..]);
    eprintln!(
        "stereo: L rms={l_rms:.1}, R rms={r_rms:.1}, ratio={:.2}",
        l_rms / r_rms.max(1.0)
    );
    // Expected ratio ≈ sqrt(balance) = sqrt(exp(2)) ≈ 2.72. The
    // smoothing filter still has a small asymptotic lag so the measured
    // ratio will be slightly below the target — test for 2.2× to 3.2×.
    assert!(
        l_rms / r_rms.max(1.0) > 2.0,
        "L should be substantially louder than R (balance > 1), got ratio {:.2}",
        l_rms / r_rms.max(1.0)
    );
    assert!(
        l_rms / r_rms.max(1.0) < 3.5,
        "L/R ratio too large, got {:.2}",
        l_rms / r_rms.max(1.0)
    );
}

#[test]
fn stereo_decode_negative_sign_inverts_balance() {
    // Same setup but with sign=1 ⇒ balance = exp(-0.25·8) = 1/7.389 ≈ 0.135.
    // Now e_right ≈ 1.327 and e_left ≈ 0.488 — R louder than L.
    let mut mono = [0.0f32; NB_FRAME_SIZE];
    for i in 0..NB_FRAME_SIZE {
        let t = i as f32;
        mono[i] = 6000.0 * (2.0 * std::f32::consts::PI * 500.0 * t / 8_000.0).sin();
    }
    let mut enc = NbEncoder::with_submode(5).unwrap();
    let frames = 30usize;
    let mut packets = Vec::with_capacity(frames);
    for _ in 0..frames {
        let data = build_stereo_packet(&mut enc, &mono, 1, 8, 3);
        packets.push(Packet {
            stream_index: 0,
            time_base: TimeBase::new(1, 8_000),
            pts: None,
            dts: None,
            duration: Some(NB_FRAME_SIZE as i64),
            flags: Default::default(),
            data,
        });
    }
    let mut dec_params = CodecParameters::audio(CodecId::new("speex"));
    dec_params.sample_rate = Some(8_000);
    dec_params.channels = Some(2);
    dec_params.sample_format = Some(oxideav_core::SampleFormat::S16);
    dec_params.extradata = stereo_nb_header();
    let mut dec = make_decoder(&dec_params).unwrap();
    let mut left = Vec::new();
    let mut right = Vec::new();
    for p in &packets {
        dec.send_packet(p).unwrap();
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    for (l, r) in audio_samples_interleaved(&af.data[0]) {
                        left.push(l);
                        right.push(r);
                    }
                }
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("{e}"),
            }
        }
    }
    let _ = dec.flush();
    let warm = 1024.min(left.len() / 2);
    let l_rms = rms_i16(&left[warm..]);
    let r_rms = rms_i16(&right[warm..]);
    eprintln!("stereo (inverted): L={l_rms:.1}, R={r_rms:.1}");
    assert!(
        r_rms / l_rms.max(1.0) > 2.0,
        "R should be louder than L (balance < 1), got {:.2}",
        r_rms / l_rms.max(1.0)
    );
}

#[test]
fn stereo_decode_zero_balance_equal_channels() {
    // sign=0, dexp=0 ⇒ balance = exp(0) = 1 ⇒ e_left = e_right.
    // eidx=3 ⇒ e_ratio = 0.5 ⇒ both gains = 1 / sqrt(0.5 · 2) = 1.
    // So L and R should be approximately equal.
    let mut mono = [0.0f32; NB_FRAME_SIZE];
    for i in 0..NB_FRAME_SIZE {
        let t = i as f32;
        mono[i] = 4000.0 * (2.0 * std::f32::consts::PI * 300.0 * t / 8_000.0).sin();
    }
    let mut enc = NbEncoder::with_submode(5).unwrap();
    let frames = 20usize;
    let mut packets = Vec::with_capacity(frames);
    for _ in 0..frames {
        let data = build_stereo_packet(&mut enc, &mono, 0, 0, 3);
        packets.push(Packet {
            stream_index: 0,
            time_base: TimeBase::new(1, 8_000),
            pts: None,
            dts: None,
            duration: Some(NB_FRAME_SIZE as i64),
            flags: Default::default(),
            data,
        });
    }
    let mut dec_params = CodecParameters::audio(CodecId::new("speex"));
    dec_params.sample_rate = Some(8_000);
    dec_params.channels = Some(2);
    dec_params.sample_format = Some(oxideav_core::SampleFormat::S16);
    dec_params.extradata = stereo_nb_header();
    let mut dec = make_decoder(&dec_params).unwrap();
    let mut left = Vec::new();
    let mut right = Vec::new();
    for p in &packets {
        dec.send_packet(p).unwrap();
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    for (l, r) in audio_samples_interleaved(&af.data[0]) {
                        left.push(l);
                        right.push(r);
                    }
                }
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("{e}"),
            }
        }
    }
    let _ = dec.flush();
    let warm = 512.min(left.len() / 2);
    let l_rms = rms_i16(&left[warm..]);
    let r_rms = rms_i16(&right[warm..]);
    eprintln!("stereo (balanced): L={l_rms:.1}, R={r_rms:.1}");
    let ratio = (l_rms / r_rms.max(1.0)).max(r_rms / l_rms.max(1.0));
    assert!(
        ratio < 1.2,
        "balance=1 should yield L≈R, got ratio {:.2}",
        ratio
    );
}
