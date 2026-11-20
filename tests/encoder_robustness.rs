//! The encoder on hostile input — digital silence, full-scale square
//! waves, DC, white noise, single clicks, and abrupt switches between
//! them — in every rate class, at several qualities and complexities
//! and under every rate mode: it must never panic, every packet must
//! decode to a whole frame, and silence must stay silent.

use oxideav_speex::{SpeexStreamDecoder, SpeexStreamEncoder};

fn signals(n: usize) -> Vec<(&'static str, Vec<i16>)> {
    let mut seed = 0x1234_5678u32;
    let mut noise = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 16) as i16
    };
    vec![
        ("silence", vec![0; n]),
        (
            "full-scale square",
            (0..n)
                .map(|i| if (i / 9) % 2 == 0 { i16::MAX } else { i16::MIN })
                .collect(),
        ),
        ("dc", vec![12_345; n]),
        ("negative rail", vec![i16::MIN; n]),
        ("white noise", (0..n).map(|_| noise()).collect()),
        (
            "clicks",
            (0..n)
                .map(|i| if i % 211 == 0 { i16::MAX } else { 0 })
                .collect(),
        ),
        (
            "switching",
            (0..n)
                .map(|i| match (i / 700) % 4 {
                    0 => 0,
                    1 => noise(),
                    2 => ((i % 40) as i16 - 20) * 1_500,
                    _ => i16::MAX,
                })
                .collect(),
        ),
    ]
}

fn drive(tag: &str, enc: &mut SpeexStreamEncoder, x: &[i16]) -> Vec<i16> {
    let n = enc.frame_samples();
    let mut dec = SpeexStreamDecoder::for_header(&enc.header()).unwrap();
    let mut out = Vec::new();
    for f in x.chunks_exact(n) {
        let p = enc
            .encode_frame(f)
            .unwrap_or_else(|e| panic!("{tag}: encode failed: {e}"));
        let pcm = dec
            .decode_packet_pcm_i16(&p)
            .unwrap_or_else(|e| panic!("{tag}: own packet failed to decode: {e}"));
        assert_eq!(pcm.len(), n, "{tag}");
        out.extend(pcm);
    }
    out
}

#[test]
fn hostile_input_encodes_and_decodes_at_every_quality() {
    for rate in [8_000u32, 16_000, 32_000] {
        let n = (rate / 50) as usize;
        for (name, x) in signals(n * 8) {
            for q in [0u8, 1, 2, 4, 7, 10] {
                let mut enc = SpeexStreamEncoder::new(rate, q).unwrap();
                let out = drive(&format!("{rate} {name} q{q}"), &mut enc, &x);
                if name == "silence" {
                    let peak = out.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
                    // (The vocoder and comfort-noise generators keep a
                    // sub-LSB excitation floor, hence not exactly zero.)
                    assert!(peak <= 8, "{rate} q{q}: silence decoded to peak {peak}");
                }
            }
        }
    }
}

#[test]
fn hostile_input_survives_every_rate_mode_and_complexity() {
    type Setup = fn(&mut SpeexStreamEncoder);
    let setups: [(&str, Setup); 6] = [
        ("vbr", |e| e.set_vbr(Some(7.5))),
        ("abr", |e| e.set_abr(Some(9_000))),
        ("vad+dtx", |e| {
            e.set_vad(true);
            e.set_dtx(true)
        }),
        ("vbr+dtx+cap", |e| {
            e.set_vbr(Some(10.0));
            e.set_dtx(true);
            e.set_vbr_max_bitrate(Some(6_000))
        }),
        ("complexity 1", |e| e.set_complexity(1)),
        ("complexity 10, plain search", |e| {
            e.set_complexity(10);
            e.set_noise_weighting(None)
        }),
    ];
    for rate in [8_000u32, 16_000] {
        let n = (rate / 50) as usize;
        for (name, x) in signals(n * 6) {
            for (mode, setup) in setups {
                let mut enc = SpeexStreamEncoder::new(rate, 6).unwrap();
                setup(&mut enc);
                drive(&format!("{rate} {name} {mode}"), &mut enc, &x);
            }
        }
    }
    // Stereo with one dead and one clipping channel.
    let mut enc = SpeexStreamEncoder::new_stereo(16_000, 5).unwrap();
    enc.set_vbr(Some(5.0));
    let mut dec = SpeexStreamDecoder::for_header(&enc.header()).unwrap();
    let frame: Vec<i16> = (0..320)
        .flat_map(|i| [0, if i % 2 == 0 { i16::MAX } else { i16::MIN }])
        .collect();
    for _ in 0..6 {
        let p = enc.encode_frame(&frame).unwrap();
        assert_eq!(dec.decode_packet_stereo_i16(&p).unwrap().len(), 640);
    }
}
