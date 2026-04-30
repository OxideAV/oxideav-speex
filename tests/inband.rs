//! End-to-end Speex in-band signalling — RFC 5574 §3 + Speex manual §5.5.
//!
//! Two scenarios are exercised:
//!
//! 1. **Pre-pended in-band requests are silently consumed by the NB
//!    decoder** — encode a real CELP frame, manually prepend a chain
//!    of `m=14` requests (mode-switch, max-bitrate, ack-packet, char
//!    transmit, perceptual-enhance) using
//!    [`oxideav_speex::inband::InbandMessage::encode`], then verify
//!    the top-level decoder still decodes the frame to coherent audio
//!    (the in-band requests advance the bit cursor exactly as
//!    `inband_skip_bits` says, so the CELP reader stays aligned).
//!
//! 2. **Encode → decode round-trip of the typed enum** — pack one of
//!    every supported message kind into a `BitWriter`, then read them
//!    back through [`oxideav_speex::inband::decode_inband`]. Confirms
//!    every wire form parses to the identical `InbandMessage`. This
//!    is the integration mirror of the unit-level round-trips in
//!    `src/inband.rs`.
//!
//! 3. **RFC 5574 §3.3 padding** — emit a terminator + the LSB-aligned
//!    `0`/all-ones pad and verify the produced byte string matches
//!    the figure in §3.3 (single Speex frame, 5-bit padding ⇒
//!    `..speex data.. | 0 1 1 1 1`).

#![allow(clippy::needless_range_loop)]

use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{CodecId, CodecParameters, Decoder, Error, Frame, Packet, TimeBase};
use oxideav_speex::decoder::make_decoder;
use oxideav_speex::header::{SPEEX_HEADER_SIZE, SPEEX_SIGNATURE};
use oxideav_speex::inband::{
    decode_inband, pad_to_octet_boundary, AckPolicy, InbandMessage, RateControl,
};
use oxideav_speex::nb_decoder::NB_FRAME_SIZE;
use oxideav_speex::nb_encoder::NbEncoder;

/// Build the 80-byte Speex-in-Ogg header for an NB **mono** stream.
fn nb_mono_header() -> Vec<u8> {
    let mut h = vec![0u8; SPEEX_HEADER_SIZE];
    h[0..8].copy_from_slice(SPEEX_SIGNATURE);
    h[28..32].copy_from_slice(&1u32.to_le_bytes());
    h[32..36].copy_from_slice(&(SPEEX_HEADER_SIZE as u32).to_le_bytes());
    h[36..40].copy_from_slice(&8_000u32.to_le_bytes()); // rate
    h[40..44].copy_from_slice(&0u32.to_le_bytes()); // mode = NB
    h[44..48].copy_from_slice(&4u32.to_le_bytes());
    h[48..52].copy_from_slice(&1u32.to_le_bytes()); // mono
    h[52..56].copy_from_slice(&15_000i32.to_le_bytes());
    h[56..60].copy_from_slice(&(NB_FRAME_SIZE as u32).to_le_bytes());
    h[60..64].copy_from_slice(&0u32.to_le_bytes());
    h[64..68].copy_from_slice(&1u32.to_le_bytes());
    h[68..72].copy_from_slice(&0u32.to_le_bytes());
    h
}

fn rms_i16(x: &[i16]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let s: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    ((s / x.len() as f64).sqrt()) as f32
}

fn collect_pcm(dec: &mut Box<dyn Decoder>) -> Vec<i16> {
    let mut pcm = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Audio(af)) => {
                for chunk in af.data[0].chunks_exact(2) {
                    pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
            Ok(_) => {}
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("decode: {e}"),
        }
    }
    pcm
}

#[test]
fn nb_decoder_skips_chain_of_inband_requests_before_celp_frame() {
    // Encode one mono NB-mode-5 CELP frame, then build a packet that
    // pre-pends five different in-band requests in front of it. The
    // bit count of the prepended payload is irregular (5+5+9+9+13+13=
    // 1 + 9 + 12 + 36 + 8 + 9 = etc.) so any off-by-one in the skip
    // ladder will desync the NB reader and crash before producing a
    // frame.
    //
    // We construct each prefix message directly via `InbandMessage::encode`;
    // the decoder then opaquely advances past them via the skip ladder
    // in `nb_decoder.rs`. The CELP frame still decodes to non-silence.
    let mut mono = [0.0f32; NB_FRAME_SIZE];
    for i in 0..NB_FRAME_SIZE {
        let t = i as f32;
        mono[i] = 6000.0 * (2.0 * std::f32::consts::PI * 500.0 * t / 8_000.0).sin();
    }
    let mut enc = NbEncoder::with_submode(5).unwrap();

    // Five different in-band messages — pick representatives across the
    // bit-width ladder:
    //   * PerceptualEnhance — 1-bit payload
    //   * ModeSwitch        — 4-bit payload
    //   * Character         — 8-bit payload
    //   * MaxBitrate        — 16-bit payload
    //   * AckPacket         — 32-bit payload
    let prefix_msgs = [
        InbandMessage::PerceptualEnhance(true),
        InbandMessage::ModeSwitch(5),
        InbandMessage::Character(b'!'),
        InbandMessage::MaxBitrate(8_000),
        InbandMessage::AckPacket(1234),
    ];
    let frames = 25usize;
    let mut packets = Vec::with_capacity(frames);
    for _ in 0..frames {
        let mut bw = BitWriter::new();
        // Wideband-bit = 0 (NB stream) — written by the NB encoder
        // *inside* its `encode_frame`, but the in-band marker `m=14`
        // shares the same 4-bit `m` slot. Reference encoder packs the
        // marker as a 5-bit field where the leading bit is the
        // wideband flag. Match that here: write a 5-bit (`0||14`)
        // header for each in-band request, then the 4-bit id, then
        // the payload.
        for msg in &prefix_msgs {
            // Manually replicate `InbandMessage::encode`'s bit layout
            // but with the leading wideband=0 prefix that the NB
            // decoder's main loop expects.
            bw.write_bits(0, 1); // wideband-bit = 0 (NB stream marker)
                                 // The remaining 4 bits + payload come from the typed encoder.
            msg.encode(&mut bw).unwrap();
        }
        // Then the actual NB frame (which writes its own wideband-bit).
        enc.encode_frame(&mono, &mut bw).unwrap();
        let data = bw.finish();
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
    dec_params.channels = Some(1);
    dec_params.sample_format = Some(oxideav_core::SampleFormat::S16);
    dec_params.extradata = nb_mono_header();
    let mut dec = make_decoder(&dec_params).expect("NB decoder");

    let mut pcm = Vec::new();
    for p in &packets {
        dec.send_packet(p).expect("send");
        pcm.extend(collect_pcm(&mut dec));
    }
    let _ = dec.flush();
    pcm.extend(collect_pcm(&mut dec));

    assert!(
        !pcm.is_empty(),
        "decoder produced no samples — in-band skip desynced"
    );
    let warm = 200usize.min(pcm.len() / 2);
    let out_rms = rms_i16(&pcm[warm..]);
    eprintln!("in-band-prefixed packets: out RMS = {out_rms:.1}");
    assert!(
        out_rms > 200.0,
        "decoder should still produce coherent NB output, got RMS {out_rms:.1}"
    );
}

#[test]
fn typed_inband_roundtrip_through_writer_reader() {
    // Pack one of each variant in a row, decode them back, assert
    // bit-for-bit equality.
    let messages = [
        InbandMessage::PerceptualEnhance(false),
        InbandMessage::PacketLossLessAggressive(true),
        InbandMessage::ModeSwitch(7),
        InbandMessage::LowBandMode(3),
        InbandMessage::HighBandMode(2),
        InbandMessage::VbrQuality(8),
        InbandMessage::RequestAck(AckPolicy::All),
        InbandMessage::RateControl(RateControl::Vbr),
        InbandMessage::Character(b'A'),
        InbandMessage::StereoSideChannel(0xA5),
        InbandMessage::MaxBitrate(20_000),
        InbandMessage::Reserved11(0xC0DE),
        InbandMessage::AckPacket(0xDEAD_BEEF),
        InbandMessage::Reserved13(42),
        InbandMessage::Reserved14(0x1122_3344_5566_7788),
        InbandMessage::Reserved15(0xFFEE_DDCC_BBAA_9988),
        InbandMessage::UserPayload(b"speex".to_vec()),
        InbandMessage::Terminator,
    ];

    let mut bw = BitWriter::new();
    for m in &messages {
        m.encode(&mut bw).unwrap();
    }
    let bytes = bw.finish();

    let mut br = BitReader::new(&bytes);
    for expected in &messages {
        let got = decode_inband(&mut br).expect("decode").expect("some");
        assert_eq!(&got, expected, "round-trip mismatch on {expected:?}");
    }
}

#[test]
fn rfc5574_padding_terminator_pattern() {
    // RFC 5574 §3.3: padding is "0 followed by all ones (until the end
    // of the octet)". The figure on page 5 shows a single Speex frame
    // with 5 bits of padding at the end (= 0,1,1,1,1).
    //
    // We emulate that exact pattern: pretend our "frame" is a single
    // m=15 terminator (4 bits), then ask for padding. After 4 bits, we
    // need 4 padding bits — the leading `0` followed by 3 ones,
    // packed MSB-first as `0111`. So the final byte is
    // `1111_0111` = 0xF7.
    let mut bw = BitWriter::new();
    InbandMessage::Terminator.encode(&mut bw).unwrap();
    let pad_bits = pad_to_octet_boundary(&mut bw);
    assert_eq!(pad_bits, 4, "4 padding bits after a single m=15 marker");
    let bytes = bw.finish();
    assert_eq!(bytes, vec![0xF7]);

    // A 3-bit "frame" → 5-bit padding `01111` → byte `???_01111`.
    // Emit 0b101 as the "frame" payload and check the byte.
    let mut bw2 = BitWriter::new();
    bw2.write_bits(0b101, 3);
    let pad2 = pad_to_octet_boundary(&mut bw2);
    assert_eq!(pad2, 5);
    let bytes2 = bw2.finish();
    // 3-bit 0b101 then 5-bit 0b01111 → 1010_1111 = 0xAF.
    assert_eq!(bytes2, vec![0xAF]);
}
