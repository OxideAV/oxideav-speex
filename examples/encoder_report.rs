//! Encoder quality / speed report: every quality 0..=10 in each rate
//! class, on a deterministic speech-like source and the crate's staged
//! fixture sources.
//!
//! ```text
//! cargo run --release --example encoder_report -- [--ref] [--rates 8000,16000,32000]
//!     [--complexity N] [--weighting on|off] [--seconds S] [--quick]
//! ```
//!
//! Per row: the measured bit-rate, whole-signal SNR, segmental SNR and
//! noise-weighted segmental SNR of encode → decode against the input
//! (enhancer off), and the encode / decode speed as multiples of real
//! time. With `--ref`, the same measures for a black-box encoder at the
//! same quality (decoded by its own black-box decoder, enhancer off)
//! and a cross-check that the black-box decoder turns this crate's
//! stream into the same PCM as the crate's own decoder. The two
//! executables are looked up on `PATH`; rows read `–` without them.

#[path = "../tests/common/mod.rs"]
mod common;

use std::process::Command;
use std::time::Instant;

use common::{Fidelity, fidelity, ogg_speex, raw_pcm, speech_like, wav_bytes, wav_pcm};
use oxideav_speex::{SpeexStreamDecoder, SpeexStreamEncoder};

/// The input as the decoder's built-in 8 kHz output high-pass leaves
/// it — the part of a narrowband input any conforming decode can
/// reproduce.
fn nb_highpassed(x: &[i16]) -> Vec<i16> {
    let c = oxideav_speex::NB_OUTPUT_HIGHPASS;
    let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    x.iter()
        .map(|&v| {
            let v = f64::from(v);
            let y = c[0] * v + c[1] * x1 + c[2] * x2 - c[3] * y1 - c[4] * y2;
            (x2, x1, y2, y1) = (x1, v, y1, y);
            y.round().clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

struct Row {
    bitrate: f64,
    fid: Fidelity,
    enc_x: f64,
    dec_x: f64,
    packets: Vec<Vec<u8>>,
    header: [u8; 80],
    pcm: Vec<i16>,
}

/// Encoder settings the report runs with (`None` = the default).
#[derive(Clone, Copy, Default)]
struct Settings {
    complexity: Option<u8>,
    /// `Some(false)` searches on the plain waveform error,
    /// `Some(true)` in the manual's noise-weighted domain.
    weighting: Option<bool>,
}

fn run_ours(x: &[i16], rate: u32, quality: u8, settings: Settings) -> Row {
    let mut enc = SpeexStreamEncoder::new(rate, quality).expect("encoder");
    if let Some(c) = settings.complexity {
        enc.set_complexity(c);
    }
    match settings.weighting {
        Some(true) => enc.set_noise_weighting(Some((
            oxideav_speex::WEIGHT_GAMMA1,
            oxideav_speex::WEIGHT_GAMMA2,
        ))),
        Some(false) => enc.set_noise_weighting(None),
        None => {}
    }
    let n = enc.frame_samples();
    let header = enc.header().write_bytes();
    let frames: Vec<&[i16]> = x.chunks_exact(n).collect();
    let seconds = (frames.len() * n) as f64 / f64::from(rate);
    let t = Instant::now();
    let packets: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| enc.encode_frame(f).expect("encode"))
        .collect();
    let enc_s = t.elapsed().as_secs_f64();
    let mut dec = SpeexStreamDecoder::for_sample_rate(rate).expect("decoder");
    dec.set_enhancer(false);
    let t = Instant::now();
    let mut pcm = Vec::with_capacity(x.len());
    for p in &packets {
        pcm.extend(dec.decode_packet_pcm_i16(p).expect("decode"));
    }
    let dec_s = t.elapsed().as_secs_f64();
    let bits: usize = packets.iter().map(|p| p.len() * 8).sum();
    let target = if rate == 8_000 {
        nb_highpassed(x)
    } else {
        x.to_vec()
    };
    Row {
        bitrate: bits as f64 / seconds,
        fid: fidelity(&target, &pcm, rate, (rate / 20) as usize),
        enc_x: seconds / enc_s,
        dec_x: seconds / dec_s,
        packets,
        header,
        pcm,
    }
}

/// Black-box encode + decode of `x` at `quality`; `None` when the
/// executables are missing or fail.
fn run_reference(x: &[i16], rate: u32, quality: u8, dir: &std::path::Path) -> Option<Fidelity> {
    let wav = dir.join("in.wav");
    let spx = dir.join("ref.spx");
    let out = dir.join("ref.wav");
    std::fs::write(&wav, wav_bytes(x, rate, 1)).ok()?;
    let band = match rate {
        8_000 => "-n",
        16_000 => "-w",
        _ => "-u",
    };
    // (The black-box encoder's input high-pass is switched off: this
    // crate's encoder has none, and a second high-pass in front of the
    // decoder's own costs several dB on every waveform measure.)
    let ok = Command::new("speexenc")
        .args([band, "--no-highpass", "--quality", &quality.to_string()])
        .arg(&wav)
        .arg(&spx)
        .output()
        .ok()?
        .status
        .success();
    if !ok {
        return None;
    }
    let ok = Command::new("speexdec")
        .arg("--no-enh")
        .arg(&spx)
        .arg(&out)
        .output()
        .ok()?
        .status
        .success();
    if !ok {
        return None;
    }
    let y = wav_pcm(&std::fs::read(&out).ok()?);
    let target = if rate == 8_000 {
        nb_highpassed(x)
    } else {
        x.to_vec()
    };
    Some(fidelity(&target, &y, rate, (rate / 20) as usize))
}

/// Largest sample difference between the black-box decode of this
/// crate's stream and the crate's own decode (`None`: tool missing or
/// it rejected the stream).
fn cross_decode(row: &Row, rate: u32, dir: &std::path::Path) -> Option<i32> {
    let n = (rate / 50) as usize;
    let spx = dir.join("ours.spx");
    let out = dir.join("ours.wav");
    std::fs::write(&spx, ogg_speex(&row.header, &row.packets, n)).ok()?;
    let res = Command::new("speexdec")
        .arg("--no-enh")
        .arg(&spx)
        .arg(&out)
        .output()
        .ok()?;
    if !res.status.success() {
        return None;
    }
    let y = wav_pcm(&std::fs::read(&out).ok()?);
    // The black-box decode leads by its look-ahead; find the offset
    // that matches the crate's own decode.
    let mut best = i32::MAX;
    for lag in 0..n {
        let m = y.len().min(row.pcm.len().saturating_sub(lag));
        if m < n {
            break;
        }
        let d = (0..m)
            .map(|i| (i32::from(y[i]) - i32::from(row.pcm[i + lag])).abs())
            .max()
            .unwrap_or(i32::MAX);
        best = best.min(d);
    }
    Some(best)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let with_ref = flag("--ref");
    let quick = flag("--quick");
    let settings = Settings {
        complexity: value("--complexity").and_then(|v| v.parse().ok()),
        weighting: value("--weighting").map(|v| v == "on"),
    };
    let seconds: f64 = value("--seconds")
        .and_then(|v| v.parse().ok())
        .unwrap_or(6.0);
    let rates: Vec<u32> = value("--rates")
        .map(|v| v.split(',').filter_map(|r| r.parse().ok()).collect())
        .unwrap_or_else(|| vec![8_000, 16_000, 32_000]);
    let dir = common::scratch_dir("report");
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    for rate in rates {
        let mut sources: Vec<(String, Vec<i16>)> =
            vec![("speech-like".into(), speech_like(rate, seconds, 4_917))];
        let staged: &[&str] = match rate {
            8_000 => &["nb-conformance/speechy8k.raw"],
            16_000 => &[
                "wb-conformance/speechy16k.raw",
                "hb-mode4-wb-q10/source16k.wav",
            ],
            _ => &[
                "uwb-conformance/speechy32k.raw",
                "hb-mode4-uwb-q10/source32k.wav",
            ],
        };
        if !quick {
            for rel in staged {
                if let Ok(bytes) = std::fs::read(root.join(rel)) {
                    let pcm = if rel.ends_with(".wav") {
                        wav_pcm(&bytes)
                    } else {
                        raw_pcm(&bytes)
                    };
                    sources.push(((*rel).into(), pcm));
                }
            }
        }
        for (name, x) in &sources {
            println!(
                "\n### {rate} Hz — {name} ({:.1} s)\n",
                x.len() as f64 / f64::from(rate)
            );
            if with_ref {
                println!(
                    "| q | kbit/s | SNR | segSNR | wSNR | enc × | dec × | ref SNR | ref segSNR | ref wSNR | black-box decode Δ |"
                );
                println!("| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |");
            } else {
                println!("| q | kbit/s | SNR | segSNR | wSNR | enc × | dec × |");
                println!("| --: | --: | --: | --: | --: | --: | --: |");
            }
            for q in 0..=10u8 {
                let row = run_ours(x, rate, q, settings);
                print!(
                    "| {q} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1} | {:.0} |",
                    row.bitrate / 1000.0,
                    row.fid.snr,
                    row.fid.seg_snr,
                    row.fid.pw_snr,
                    row.enc_x,
                    row.dec_x
                );
                if with_ref {
                    match run_reference(x, rate, q, &dir) {
                        Some(f) => print!(" {:.2} | {:.2} | {:.2} |", f.snr, f.seg_snr, f.pw_snr),
                        None => print!(" – | – | – |"),
                    }
                    match cross_decode(&row, rate, &dir) {
                        Some(d) => print!(" ±{d} |"),
                        None => print!(" rejected |"),
                    }
                }
                println!();
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
