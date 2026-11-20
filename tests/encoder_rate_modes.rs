//! The stream encoder's rate modes and side channel, end to end through
//! the crate's own decoder: variable / average bit-rate, voice-activity
//! detection with comfort noise, discontinuous transmission, explicit
//! sub-mode selection, the complexity setting, in-band messages and
//! user payloads (manual §2.1, §5.5).

mod common;

use common::{fidelity, speech_like};
use oxideav_speex::{
    ControlMessage, DecodedFrame, InbandRequest, SpeexDecoder, SpeexStreamDecoder,
    SpeexStreamEncoder, inband_code_spec,
};

/// What one configured encode of `x` produced.
struct Run {
    packets: Vec<Vec<u8>>,
    pcm: Vec<i16>,
    /// Mean coded bit-rate (frame bits only), bit/s.
    bps: f64,
    /// Frames per narrowband sub-mode.
    modes: [usize; 9],
    /// Largest single frame, bits.
    peak_bits: usize,
}

fn run(enc: &mut SpeexStreamEncoder, x: &[i16]) -> Run {
    let n = enc.frame_samples();
    let mut dec = SpeexStreamDecoder::for_header(&enc.header()).unwrap();
    dec.set_enhancer(false);
    let mut r = Run {
        packets: Vec::new(),
        pcm: Vec::new(),
        bps: 0.0,
        modes: [0; 9],
        peak_bits: 0,
    };
    let mut bits = 0usize;
    for f in x.chunks_exact(n) {
        let p = enc.encode_frame(f).unwrap();
        bits += enc.last_frame_bits();
        r.peak_bits = r.peak_bits.max(enc.last_frame_bits());
        r.modes[usize::from(enc.last_submodes().nb_mode)] += 1;
        let out = dec.decode_packet_pcm_i16(&p).unwrap();
        assert_eq!(out.len(), n);
        r.pcm.extend(out);
        r.packets.push(p);
    }
    r.bps = bits as f64 * 50.0 / r.packets.len() as f64;
    r
}

#[test]
fn vbr_spends_fewer_bits_than_constant_quality_and_varies_the_mode() {
    let x = speech_like(8_000, 4.0, 4_917);
    let mut cbr = SpeexStreamEncoder::new(8_000, 6).unwrap();
    let c = run(&mut cbr, &x);
    let mut vbr = SpeexStreamEncoder::new(8_000, 6).unwrap();
    vbr.set_vbr(Some(6.0));
    assert_eq!(vbr.header().vbr, 1);
    let v = run(&mut vbr, &x);
    assert!(
        v.bps < 0.95 * c.bps,
        "VBR {:.0} bit/s vs constant {:.0}",
        v.bps,
        c.bps
    );
    assert!(
        v.modes.iter().filter(|&&m| m > 0).count() >= 3,
        "VBR used modes {:?}",
        v.modes
    );
    assert!(v.modes[1] > 0, "pauses are comfort noise: {:?}", v.modes);
    // The speech itself is still coded: within 2 dB of constant quality.
    let (fc, fv) = (
        fidelity(&x, &c.pcm, 8_000, 400),
        fidelity(&x, &v.pcm, 8_000, 400),
    );
    assert!(
        fv.seg_snr > fc.seg_snr - 2.0,
        "VBR segSNR {:.2} vs {:.2}",
        fv.seg_snr,
        fc.seg_snr
    );
    // A higher VBR quality spends more.
    let mut hi = SpeexStreamEncoder::new(8_000, 6).unwrap();
    hi.set_vbr(Some(9.0));
    assert!(run(&mut hi, &x).bps > v.bps);
}

#[test]
fn abr_lands_on_its_target() {
    for (rate, target) in [(8_000u32, 9_000u32), (16_000, 16_000)] {
        let x = speech_like(rate, 5.0, 77);
        let mut enc = SpeexStreamEncoder::new(rate, 8).unwrap();
        enc.set_abr(Some(target));
        assert_eq!(enc.nominal_bitrate_bps(), Some(target));
        let r = run(&mut enc, &x);
        let off = (r.bps - f64::from(target)).abs() / f64::from(target);
        assert!(off < 0.06, "{rate} Hz ABR {target}: got {:.0} bit/s", r.bps);
    }
}

#[test]
fn vbr_ceiling_bounds_every_frame() {
    let x = speech_like(8_000, 3.0, 9);
    let mut enc = SpeexStreamEncoder::new(8_000, 8).unwrap();
    enc.set_vbr(Some(10.0));
    enc.set_vbr_max_bitrate(Some(11_000));
    let r = run(&mut enc, &x);
    assert_eq!(r.peak_bits, 220, "11 kbit/s = the 220-bit mode 4");
}

#[test]
fn vad_codes_pauses_as_comfort_noise_and_dtx_as_null_frames() {
    let x = speech_like(8_000, 4.0, 4_917);
    let mut plain = SpeexStreamEncoder::new(8_000, 8).unwrap();
    let p = run(&mut plain, &x);
    assert_eq!(p.modes[5], p.packets.len(), "constant quality 8 = mode 5");

    let mut vad = SpeexStreamEncoder::new(8_000, 8).unwrap();
    vad.set_vad(true);
    let v = run(&mut vad, &x);
    assert!(v.modes[1] > 5, "comfort-noise frames: {:?}", v.modes);
    assert_eq!(v.modes[0], 0, "no null frames without DTX");
    assert_eq!(v.modes[1] + v.modes[5], v.packets.len());

    let mut dtx = SpeexStreamEncoder::new(8_000, 8).unwrap();
    dtx.set_dtx(true);
    let d = run(&mut dtx, &x);
    assert!(d.modes[0] > 5, "null frames: {:?}", d.modes);
    assert!(d.bps < v.bps && v.bps < p.bps);
    // A null frame is the 5-bit frame of manual §2.1: one octet.
    assert!(d.packets.iter().any(|p| p.len() == 1));
    // Speech is untouched: the decode stays within 1 dB of the plain one.
    let (fp, fd) = (
        fidelity(&x, &p.pcm, 8_000, 400),
        fidelity(&x, &d.pcm, 8_000, 400),
    );
    assert!(
        fd.snr > fp.snr - 1.0,
        "DTX SNR {:.2} vs {:.2}",
        fd.snr,
        fp.snr
    );
}

#[test]
fn dtx_null_frames_are_9_and_13_bits_in_the_wider_classes() {
    for (rate, bits) in [(16_000u32, 9usize), (32_000, 13)] {
        let n = (rate / 50) as usize;
        let mut enc = SpeexStreamEncoder::new(rate, 6).unwrap();
        enc.set_dtx(true);
        let mut dec = SpeexStreamDecoder::for_sample_rate(rate).unwrap();
        let mut null_frames = 0;
        for k in 0..12i16 {
            // A faint steady hiss.
            let f: Vec<i16> = (0..n as i16).map(|i| (i * 7 + k * 3) % 5 - 2).collect();
            let p = enc.encode_frame(&f).unwrap();
            if enc.last_submodes().nb_mode == 0 {
                null_frames += 1;
                assert_eq!(enc.last_frame_bits(), bits);
                assert_eq!(p.len(), 2);
            }
            assert_eq!(dec.decode_packet_pcm_i16(&p).unwrap().len(), n);
        }
        assert!(null_frames >= 8, "{rate} Hz: {null_frames} null frames");
    }
}

#[test]
fn every_sub_mode_pair_encodes_at_its_table_size_and_decodes() {
    // Table 9.1 / Table 10.1 frame totals.
    const NB_BITS: [usize; 9] = [5, 43, 119, 160, 220, 300, 364, 492, 79];
    const HB_BITS: [usize; 5] = [4, 36, 112, 192, 352];
    let x8 = speech_like(8_000, 0.4, 3);
    for nb in 1..=8u8 {
        let mut enc = SpeexStreamEncoder::new(8_000, 8).unwrap();
        enc.set_submodes(Some((nb, 0))).unwrap();
        let r = run(&mut enc, &x8);
        assert_eq!(r.peak_bits, NB_BITS[usize::from(nb)]);
        assert_eq!(r.modes[usize::from(nb)], r.packets.len());
        assert_eq!(
            enc.nominal_bitrate_bps(),
            Some(NB_BITS[usize::from(nb)] as u32 * 50)
        );
    }
    let x16 = speech_like(16_000, 0.2, 3);
    for nb in [1u8, 2, 5, 8] {
        for hb in 0..=4u8 {
            let mut enc = SpeexStreamEncoder::new(16_000, 8).unwrap();
            enc.set_submodes(Some((nb, hb))).unwrap();
            let r = run(&mut enc, &x16);
            assert_eq!(
                r.peak_bits,
                NB_BITS[usize::from(nb)] + HB_BITS[usize::from(hb)]
            );
            assert_eq!(enc.last_submodes().hb_mode, Some(hb));
        }
    }
    let x32 = speech_like(32_000, 0.2, 3);
    let mut enc = SpeexStreamEncoder::new(32_000, 8).unwrap();
    enc.set_submodes(Some((4, 3))).unwrap();
    let r = run(&mut enc, &x32);
    assert_eq!(r.peak_bits, 220 + 192 + 36);
    assert_eq!(enc.last_submodes().uwb_mode, Some(1));

    // Out-of-table requests are refused.
    let mut enc = SpeexStreamEncoder::new(16_000, 8).unwrap();
    assert!(enc.set_submodes(Some((0, 1))).is_err());
    assert!(enc.set_submodes(Some((9, 1))).is_err());
    assert!(enc.set_submodes(Some((5, 5))).is_err());
    let mut enc = SpeexStreamEncoder::new(32_000, 8).unwrap();
    assert!(enc.set_submodes(Some((5, 0))).is_err());
}

#[test]
fn bitrate_picks_the_highest_quality_at_or_below_it() {
    let mut enc = SpeexStreamEncoder::new(8_000, 8).unwrap();
    assert_eq!(enc.set_bitrate(11_000).unwrap(), 6);
    assert_eq!(enc.set_bitrate(10_999).unwrap(), 4);
    assert_eq!(enc.set_bitrate(1_000_000).unwrap(), 10);
    assert!(enc.set_bitrate(2_000).is_err());
    let mut enc = SpeexStreamEncoder::new(32_000, 8).unwrap();
    assert_eq!(enc.set_bitrate(30_000).unwrap(), 8);
}

#[test]
fn complexity_trades_time_for_error_without_touching_the_rate() {
    let x = speech_like(8_000, 1.5, 21);
    let mut err = Vec::new();
    for c in [1u8, 3, 9] {
        let mut enc = SpeexStreamEncoder::new(8_000, 6).unwrap();
        enc.set_complexity(c);
        assert_eq!(enc.complexity(), c);
        let r = run(&mut enc, &x);
        assert_eq!(r.peak_bits, 220);
        err.push(fidelity(&x, &r.pcm, 8_000, 400).seg_snr);
    }
    assert!(
        err[0] < err[1] && err[1] < err[2],
        "segSNR by complexity 1 / 3 / 9: {err:?}"
    );
    // Out-of-range settings clamp.
    let mut enc = SpeexStreamEncoder::new(8_000, 6).unwrap();
    enc.set_complexity(0);
    assert_eq!(enc.complexity(), 1);
    enc.set_complexity(200);
    assert_eq!(enc.complexity(), 10);
}

#[test]
fn inband_messages_and_user_payloads_ride_ahead_of_the_frame() {
    let x = speech_like(8_000, 0.2, 5);
    let frame = &x[..160];
    let mut plain = SpeexStreamEncoder::new(8_000, 4).unwrap();
    let bare = plain.encode_frame(frame).unwrap();

    let mut enc = SpeexStreamEncoder::new(8_000, 4).unwrap();
    // Table 5.1 code 10: announce a 2 000 byte/s ceiling; code 8: a
    // character; then an application payload.
    enc.queue_inband(oxideav_speex::InbandMessage {
        spec: inband_code_spec(10),
        payload: 2_000,
    });
    enc.queue_inband(oxideav_speex::InbandMessage {
        spec: inband_code_spec(8),
        payload: u64::from(b'k'),
    });
    enc.queue_user_data(b"oxide").unwrap();
    assert!(enc.queue_user_data(&[0; 32]).is_err());
    let packet = enc.encode_frame(frame).unwrap();
    // 5+4+16, 5+4+8, 5+5+40 bits ahead of the 160-bit frame.
    assert_eq!(packet.len(), (25 + 17 + 50 + 160usize).div_ceil(8));

    let mut dec = SpeexDecoder::new();
    let frames = dec.decode_packet(&packet).unwrap();
    assert_eq!(frames.len(), 4);
    match &frames[0] {
        DecodedFrame::Control(ControlMessage::Inband { request, .. }) => {
            assert_eq!(*request, InbandRequest::AnnounceMaxBitrate(2_000));
        }
        other => panic!("expected the max-bit-rate message, got {other:?}"),
    }
    match &frames[1] {
        DecodedFrame::Control(ControlMessage::Inband { request, .. }) => {
            assert_eq!(*request, InbandRequest::TransmitCharacter(b'k'));
        }
        other => panic!("expected the character message, got {other:?}"),
    }
    match &frames[2] {
        DecodedFrame::Control(ControlMessage::Custom { size_bytes }) => {
            assert_eq!(*size_bytes, 5);
        }
        other => panic!("expected the user payload, got {other:?}"),
    }
    // The audio frame is the one a bare packet carries.
    let mut bare_dec = SpeexDecoder::new();
    let bare_frames = bare_dec.decode_packet(&bare).unwrap();
    assert_eq!(frames[3].pcm_i16(), bare_frames[0].pcm_i16());
    // The queue is spent: the next packet is bare again.
    assert_eq!(enc.encode_frame(frame).unwrap().len(), 20);
}

#[test]
fn stereo_streams_keep_their_message_in_every_rate_mode() {
    let mono = speech_like(16_000, 1.0, 13);
    let inter: Vec<i16> = mono.iter().flat_map(|&s| [s, s / 3]).collect();
    let mut enc = SpeexStreamEncoder::new_stereo(16_000, 6).unwrap();
    enc.set_vbr(Some(6.0));
    enc.set_dtx(true);
    let mut dec = SpeexStreamDecoder::for_header(&enc.header()).unwrap();
    let (mut el, mut er) = (0.0f64, 0.0f64);
    for f in inter.as_chunks::<640>().0 {
        let p = enc.encode_frame(f).unwrap();
        let out = dec.decode_packet_stereo_i16(&p).unwrap();
        assert_eq!(out.len(), 640);
        for lr in out.as_chunks::<2>().0 {
            el += f64::from(lr[0]).powi(2);
            er += f64::from(lr[1]).powi(2);
        }
    }
    assert!(el > 4.0 * er, "left {el:.0} vs right {er:.0}");
}

/// Encode `x` through the framework factory with the given options;
/// returns the packet sizes and the encoder's output parameters.
fn framework_encode(
    x: &[i16],
    rate: u32,
    options: &[(&str, &str)],
) -> oxideav_core::Result<(Vec<usize>, oxideav_core::CodecParameters)> {
    use oxideav_core::{AudioFrame, CodecId, CodecParameters, Error, Frame, SampleFormat};
    let mut p = CodecParameters::audio(CodecId::new("speex"));
    p.sample_rate = Some(rate);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::S16);
    for (k, v) in options {
        p.options.insert(*k, *v);
    }
    let mut enc = oxideav_speex::make_encoder(&p)?;
    enc.send_frame(&Frame::Audio(AudioFrame {
        samples: x.len() as u32,
        pts: Some(0),
        data: vec![x.iter().flat_map(|s| s.to_le_bytes()).collect()],
    }))?;
    let mut sizes = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(pk) => sizes.push(pk.data.len()),
            Err(Error::NeedMore) => break,
            Err(e) => return Err(e),
        }
    }
    Ok((sizes, enc.output_params().clone()))
}

#[test]
fn framework_options_reach_the_rate_modes() {
    let x = speech_like(8_000, 3.0, 4_917);
    // Constant quality: every packet the same size.
    let (plain, params) = framework_encode(&x, 8_000, &[("quality", "6")]).unwrap();
    assert!(
        plain.iter().all(|&n| n == 28),
        "mode 4 = 220 bits = 28 octets"
    );
    assert_eq!(params.bit_rate, Some(11_000));
    let header = oxideav_speex::SpeexHeader::parse(&params.extradata).unwrap();
    assert_eq!(header.vbr, 0);

    // `bitrate` picks the quality.
    let (by_rate, params) = framework_encode(&x, 8_000, &[("bitrate", "15000")]).unwrap();
    assert!(by_rate.iter().all(|&n| n == 38), "15 kbit/s = mode 5");
    assert_eq!(params.bit_rate, Some(15_000));

    // VBR + DTX: sizes vary, null frames appear, the header says VBR.
    let (vbr, params) = framework_encode(
        &x,
        8_000,
        &[
            ("quality", "6"),
            ("vbr", "true"),
            ("vbr_quality", "5.5"),
            ("dtx", "true"),
            ("complexity", "2"),
        ],
    )
    .unwrap();
    let mut distinct = vbr.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(distinct.len() >= 4, "VBR packet sizes {distinct:?}");
    assert!(vbr.contains(&1), "a null frame is one octet");
    assert!(vbr.iter().sum::<usize>() < plain.iter().sum::<usize>());
    assert_eq!(
        oxideav_speex::SpeexHeader::parse(&params.extradata)
            .unwrap()
            .vbr,
        1
    );

    // ABR: the target is the declared bit-rate and the stream meets it.
    let (abr, params) = framework_encode(&x, 8_000, &[("abr", "9000")]).unwrap();
    assert_eq!(params.bit_rate, Some(9_000));
    let bps = abr.iter().sum::<usize>() as f64 * 8.0 * 50.0 / abr.len() as f64;
    assert!((bps - 9_000.0).abs() < 900.0, "ABR 9000: {bps:.0} bit/s");

    // VAD alone: two sizes (speech and comfort noise).
    let (vad, _) = framework_encode(&x, 8_000, &[("quality", "8"), ("vad", "true")]).unwrap();
    let mut distinct = vad.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct, vec![6, 38]);

    // The plain search is selectable and changes the stream's content,
    // not its size.
    let (unweighted, _) =
        framework_encode(&x, 8_000, &[("quality", "6"), ("noise_weighting", "false")]).unwrap();
    assert_eq!(unweighted, plain);

    // Out-of-range settings are refused up front.
    for bad in [
        [("complexity", "0")],
        [("complexity", "11")],
        [("quality", "11")],
        [("vbr_quality", "10.5")],
        [("bitrate", "1000")],
    ] {
        assert!(
            framework_encode(&x[..160], 8_000, &bad).is_err(),
            "{bad:?} accepted"
        );
    }
}
