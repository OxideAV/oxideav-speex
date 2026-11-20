//! Black-box cross-decode of the encoder's streams: every quality in
//! each rate class, every rate mode, stereo and the in-band side
//! channel, wrapped as Ogg/Speex and handed to two independent decoder
//! executables when they are installed. A missing tool skips its half;
//! the streams themselves are always checked against the crate's own
//! decoder.
//!
//! The first tool, run with its enhancer off, must reproduce the
//! crate's decode sample for sample (a few LSB of float drift), which
//! shows the encoder's decoder mirror and a real decoder agree on what
//! the stream means; the second must decode without a diagnostic.

mod common;

use std::process::Command;

use common::{have_tool, ogg_speex, scratch_dir, speech_like, wav_pcm};
use oxideav_speex::{SpeexStreamDecoder, SpeexStreamEncoder, inband_code_spec};

/// Encode `x`, decode it with the crate, and return (packets, pcm).
fn encode(enc: &mut SpeexStreamEncoder, x: &[i16]) -> (Vec<Vec<u8>>, Vec<i16>) {
    let n = enc.frame_samples() * usize::from(enc.channels());
    let mut dec = SpeexStreamDecoder::for_header(&enc.header()).unwrap();
    dec.set_enhancer(false);
    let (mut packets, mut pcm) = (Vec::new(), Vec::new());
    for f in x.chunks_exact(n) {
        let p = enc.encode_frame(f).unwrap();
        pcm.extend(if enc.channels() == 2 {
            dec.decode_packet_stereo_i16(&p).unwrap()
        } else {
            dec.decode_packet_pcm_i16(&p).unwrap()
        });
        packets.push(p);
    }
    (packets, pcm)
}

/// Hand the stream to the installed black-box decoders. `tolerance` is
/// the largest sample difference allowed between the first tool's
/// decode and the crate's own (`None`: only require a clean decode of
/// the right length).
fn cross_decode(
    tag: &str,
    enc: &SpeexStreamEncoder,
    packets: &[Vec<u8>],
    own: &[i16],
    tolerance: Option<i32>,
    tools: (bool, bool),
) {
    let dir = scratch_dir(&format!("bb-{}", tag.replace([' ', '/'], "-")));
    let spx = dir.join("s.spx");
    std::fs::write(
        &spx,
        ogg_speex(&enc.header().write_bytes(), packets, enc.frame_samples()),
    )
    .unwrap();
    if tools.0 {
        let wav = dir.join("speexdec.wav");
        let out = Command::new("speexdec")
            .arg("--no-enh")
            .arg(&spx)
            .arg(&wav)
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&out.stderr).to_lowercase();
        assert!(
            out.status.success() && !log.contains("corrupt") && !log.contains("error"),
            "{tag}: speexdec rejected the stream: {log}"
        );
        let y = wav_pcm(&std::fs::read(&wav).unwrap());
        assert_eq!(y.len(), own.len(), "{tag}: speexdec output length");
        if let Some(tol) = tolerance {
            let worst = y
                .iter()
                .zip(own)
                .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
                .max()
                .unwrap_or(0);
            assert!(
                worst <= tol,
                "{tag}: speexdec differs from the crate's decode by {worst} LSB"
            );
        }
    }
    if tools.1 {
        let wav = dir.join("ffmpeg.wav");
        let out = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(&spx)
            .arg(&wav)
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success() && log.trim().is_empty(),
            "{tag}: ffmpeg reported: {log}"
        );
        let y = wav_pcm(&std::fs::read(&wav).unwrap());
        assert!(
            y.len() + 2 * enc.frame_samples() * usize::from(enc.channels()) >= own.len(),
            "{tag}: ffmpeg decoded {} of {} samples",
            y.len(),
            own.len()
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

fn tools() -> Option<(bool, bool)> {
    let t = (have_tool("speexdec"), have_tool("ffmpeg"));
    if !t.0 && !t.1 {
        eprintln!("skipped: neither speexdec nor ffmpeg is installed");
        return None;
    }
    Some(t)
}

#[test]
fn every_quality_in_every_rate_class_decodes_in_black_box_decoders() {
    let Some(tools) = tools() else { return };
    for rate in [8_000u32, 16_000, 32_000] {
        let x = speech_like(rate, 1.2, 4_917);
        for q in 0..=10u8 {
            let mut enc = SpeexStreamEncoder::new(rate, q).unwrap();
            let (packets, own) = encode(&mut enc, &x);
            cross_decode(
                &format!("{rate} q{q}"),
                &enc,
                &packets,
                &own,
                Some(12),
                tools,
            );
        }
    }
}

#[test]
fn every_sub_mode_pair_decodes_in_black_box_decoders() {
    let Some(tools) = tools() else { return };
    let x = speech_like(16_000, 0.6, 31);
    for nb in 1..=8u8 {
        for hb in 0..=4u8 {
            let mut enc = SpeexStreamEncoder::new(16_000, 8).unwrap();
            enc.set_submodes(Some((nb, hb))).unwrap();
            let (packets, own) = encode(&mut enc, &x);
            cross_decode(
                &format!("wb nb{nb} hb{hb}"),
                &enc,
                &packets,
                &own,
                Some(12),
                tools,
            );
        }
    }
}

#[test]
fn rate_modes_decode_in_black_box_decoders() {
    let Some(tools) = tools() else { return };
    for rate in [8_000u32, 16_000, 32_000] {
        let x = speech_like(rate, 3.0, 4_917);
        type Setup = fn(&mut SpeexStreamEncoder);
        let cases: [(&str, Setup, bool); 5] = [
            ("vbr", |e| e.set_vbr(Some(6.5)), true),
            ("abr", |e| e.set_abr(Some(14_000)), true),
            ("vad", |e| e.set_vad(true), true),
            // Null frames hand the decoders an excitation memory the
            // staged material does not pin sample-exactly; require a
            // clean decode only.
            ("dtx", |e| e.set_dtx(true), false),
            (
                "vbr+dtx+cap",
                |e| {
                    e.set_vbr(Some(9.0));
                    e.set_dtx(true);
                    e.set_vbr_max_bitrate(Some(20_000));
                },
                false,
            ),
        ];
        for (name, setup, exact) in cases {
            let mut enc = SpeexStreamEncoder::new(rate, 8).unwrap();
            setup(&mut enc);
            let (packets, own) = encode(&mut enc, &x);
            cross_decode(
                &format!("{rate} {name}"),
                &enc,
                &packets,
                &own,
                exact.then_some(12),
                tools,
            );
        }
    }
}

#[test]
fn stereo_and_side_channel_decode_in_black_box_decoders() {
    let Some(tools) = tools() else { return };
    // Intensity stereo: the black-box stereo expansion is not sample
    // pinned against this crate's, so require a clean decode.
    let mono = speech_like(16_000, 1.0, 5);
    let inter: Vec<i16> = mono.iter().flat_map(|&s| [s, s / 2]).collect();
    let mut enc = SpeexStreamEncoder::new_stereo(16_000, 7).unwrap();
    let (packets, own) = encode(&mut enc, &inter);
    cross_decode("stereo wb", &enc, &packets, &own, None, tools);

    // In-band messages ahead of every fifth frame: decoders skip them
    // and the audio is untouched. (Application payloads — mode 13 — are
    // left out: the installed reference decoder does not skip them the
    // way manual §5.5 describes, see the README's open questions.)
    let x = speech_like(8_000, 1.0, 5);
    let mut enc = SpeexStreamEncoder::new(8_000, 6).unwrap();
    let mut dec = SpeexStreamDecoder::for_header(&enc.header()).unwrap();
    dec.set_enhancer(false);
    let (mut packets, mut own) = (Vec::new(), Vec::new());
    for (i, f) in x.as_chunks::<160>().0.iter().enumerate() {
        if i % 5 == 0 {
            for (code, payload) in [(8u8, u64::from(b'a')), (10, 3_000), (6, 2), (12, 77)] {
                enc.queue_inband(oxideav_speex::InbandMessage {
                    spec: inband_code_spec(code),
                    payload,
                });
            }
        }
        let p = enc.encode_frame(f).unwrap();
        own.extend(dec.decode_packet_pcm_i16(&p).unwrap());
        packets.push(p);
    }
    cross_decode("side channel", &enc, &packets, &own, Some(12), tools);
}
