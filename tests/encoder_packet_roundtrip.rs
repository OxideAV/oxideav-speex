//! Packet-level encode → top-level decode round-trip (round r385).
//!
//! Drives the two encoders' `encode_packet` output (frames packed
//! back-to-back + the §5.5 mode-15 terminator + byte padding) through
//! the top-level [`SpeexDecoder`] — the same entry point a container
//! demuxer feeds — asserting the packet walker recovers exactly the
//! encoded frames (no misparse of the padding tail) and the PCM
//! convenience surface produces the right shape at the right rate.

use oxideav_speex::{
    DecodedFrame, NarrowbandEncoder, SpeexDecoder, WidebandEncoder, NB_FRAME_SAMPLES,
    QMF_WIDEBAND_FRAME,
};

fn nb_frame(amp: f64, frame_idx: usize) -> [i16; NB_FRAME_SAMPLES] {
    let mut f = [0i16; NB_FRAME_SAMPLES];
    for (n, s) in f.iter_mut().enumerate() {
        let t = (frame_idx * NB_FRAME_SAMPLES + n) as f64;
        let v = amp * (2.0 * std::f64::consts::PI * t * 300.0 / 8_000.0).sin();
        *s = v.round().clamp(-32768.0, 32767.0) as i16;
    }
    f
}

fn wb_frame(amp: f64, frame_idx: usize) -> [i16; QMF_WIDEBAND_FRAME] {
    let mut f = [0i16; QMF_WIDEBAND_FRAME];
    for (n, s) in f.iter_mut().enumerate() {
        let t = (frame_idx * QMF_WIDEBAND_FRAME + n) as f64;
        let low = (2.0 * std::f64::consts::PI * t * 300.0 / 16_000.0).sin();
        let high = 0.5 * (2.0 * std::f64::consts::PI * t * 6_000.0 / 16_000.0).sin();
        let v = amp * (low + high);
        *s = v.round().clamp(-32768.0, 32767.0) as i16;
    }
    f
}

#[test]
fn narrowband_multi_frame_packet_decodes_through_speex_decoder() {
    // Three NB frames in one packet: the top-level decoder must yield
    // exactly three narrowband audio frames (terminator consumed, no
    // padding misparse).
    let mut enc = NarrowbandEncoder::new();
    let frames = [
        nb_frame(5000.0, 0),
        nb_frame(5000.0, 1),
        nb_frame(5000.0, 2),
    ];
    let packet = enc.encode_packet(&frames, 5).expect("encode packet");

    let mut dec = SpeexDecoder::new();
    let out = dec.decode_packet(&packet).expect("decode packet");
    assert_eq!(out.len(), 3, "expected 3 frames, got {}", out.len());
    for (i, f) in out.iter().enumerate() {
        assert!(
            matches!(f, DecodedFrame::Narrowband { .. }),
            "frame {i} should be narrowband"
        );
        assert_eq!(f.sample_rate_hz(), Some(8_000), "frame {i} rate");
    }

    // The flat i16 convenience concatenates 3 × 160 samples.
    let mut dec2 = SpeexDecoder::new();
    let pcm = dec2.decode_packet_pcm_i16(&packet).expect("flat decode");
    assert_eq!(pcm.len(), 3 * NB_FRAME_SAMPLES);
}

#[test]
fn wideband_multi_frame_packet_decodes_through_speex_decoder() {
    // Two WB frames in one packet: the walker must skip the high-band
    // parts correctly to stay in sync (§10.4) and yield two wideband
    // frames at 16 kHz.
    let mut enc = WidebandEncoder::new();
    let frames = [wb_frame(6000.0, 0), wb_frame(6000.0, 1)];
    let packet = enc.encode_packet(&frames, 3, 2).expect("encode packet");

    let mut dec = SpeexDecoder::new();
    let out = dec.decode_packet(&packet).expect("decode packet");
    assert_eq!(out.len(), 2, "expected 2 frames, got {}", out.len());
    for (i, f) in out.iter().enumerate() {
        assert!(
            matches!(f, DecodedFrame::Wideband { .. }),
            "frame {i} should be wideband"
        );
        assert_eq!(f.sample_rate_hz(), Some(16_000), "frame {i} rate");
    }

    let mut dec2 = SpeexDecoder::new();
    let pcm = dec2.decode_packet_pcm_i16(&packet).expect("flat decode");
    assert_eq!(pcm.len(), 2 * QMF_WIDEBAND_FRAME);
    assert!(pcm.iter().any(|&s| s != 0), "PCM should be non-silent");
}

#[test]
fn single_frame_wideband_packet_decodes() {
    let mut enc = WidebandEncoder::new();
    let packet = enc
        .encode_packet(&[wb_frame(4000.0, 0)], 3, 3)
        .expect("encode");
    let mut dec = SpeexDecoder::new();
    let out = dec.decode_packet(&packet).expect("decode");
    assert_eq!(out.len(), 1);
    assert!(matches!(out[0], DecodedFrame::Wideband { .. }));
}

#[test]
fn empty_frame_list_encodes_terminator_only_packet() {
    // An empty packet is just the terminator + padding; the decoder
    // yields zero frames.
    let mut enc = NarrowbandEncoder::new();
    let packet = enc.encode_packet(&[], 5).expect("encode");
    assert!(!packet.is_empty(), "terminator still occupies one byte");
    let mut dec = SpeexDecoder::new();
    let out = dec.decode_packet(&packet).expect("decode");
    assert!(out.is_empty(), "terminator-only packet has no frames");
}

#[test]
fn packet_stream_is_continuous_across_packets() {
    // Decoding N frames as one packet vs N single-frame packets through
    // one decoder must produce identical PCM (the packetisation is
    // transparent to the decode state).
    let frames = [
        wb_frame(5000.0, 0),
        wb_frame(5000.0, 1),
        wb_frame(5000.0, 2),
    ];

    let mut enc_a = WidebandEncoder::new();
    let one_packet = enc_a.encode_packet(&frames, 3, 2).expect("encode");
    let mut dec_a = SpeexDecoder::new();
    let pcm_a = dec_a.decode_packet_pcm_i16(&one_packet).expect("decode");

    let mut enc_b = WidebandEncoder::new();
    let mut dec_b = SpeexDecoder::new();
    let mut pcm_b = Vec::new();
    for f in &frames {
        let p = enc_b.encode_packet(&[*f], 3, 2).expect("encode");
        pcm_b.extend(dec_b.decode_packet_pcm_i16(&p).expect("decode"));
    }

    assert_eq!(pcm_a, pcm_b, "packetisation must be decode-transparent");
}
