//! **Encoder fidelity, end to end** — the registry encoder → registry
//! decoder round trip on real-length deterministic material, measured
//! against the encoder's input (not against a reference stream).
//!
//! Locks the r480 encoder repairs:
//!
//! * the LPC analysis sign convention (`A(z) = 1 − Σ a z⁻ⁱ`): before
//!   it, every speech envelope reached the LSP root-finder as a
//!   non-minimum-phase polynomial and the narrowband / wideband /
//!   ultra-wideband encoders all silently transmitted a flat envelope
//!   (voiced frames came out 10–15 dB hot, unvoiced 7 dB quiet);
//! * the mode-7 frame-gain estimate (quality 10 picked the top gain);
//! * the exact high-band gain rules (`tracer-2026-09.md` §1.3, §5–§6),
//!   including wideband / ultra-wideband **quality 10** — high-band
//!   mode 4, the reporter's 32 kHz 44 kbit/s default in issue #7.
//!
//! Black-box cross-check (recorded in the CHANGELOG, not run here): the
//! same streams decoded by the reference `speexdec --no-enh` give the
//! same figures to within ±0.2 dB.

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, SampleFormat, TimeBase,
};

fn wav_pcm(wav: &[u8]) -> Vec<i16> {
    let p = wav.windows(4).position(|w| w == b"data").unwrap() + 8;
    wav[p..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect()
}

fn raw_pcm(raw: &[u8]) -> Vec<i16> {
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect()
}

/// Encode `pcm` at `rate` / `quality` through the registry factories and
/// decode it straight back.
fn round_trip(pcm: &[i16], rate: u32, quality: u8) -> Vec<i16> {
    let mut p = CodecParameters::audio(CodecId::new("speex"));
    p.sample_rate = Some(rate);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::S16);
    p.options.insert("quality", quality.to_string().as_str());
    let mut enc = oxideav_speex::make_encoder(&p).expect("encoder");
    let data: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    enc.send_frame(&Frame::Audio(AudioFrame {
        samples: pcm.len() as u32,
        pts: Some(0),
        data: vec![data],
    }))
    .unwrap();
    let mut packets: Vec<Packet> = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(pk) => packets.push(pk),
            Err(Error::NeedMore) => break,
            Err(e) => panic!("{e}"),
        }
    }
    let mut dec = oxideav_speex::make_decoder(enc.output_params()).expect("decoder");
    let mut out = Vec::new();
    for pk in &packets {
        dec.send_packet(&Packet::new(0, TimeBase::from_rate(rate), pk.data.clone()))
            .unwrap();
        while let Ok(Frame::Audio(a)) = dec.receive_frame() {
            out.extend(raw_pcm(&a.data[0]));
        }
    }
    out
}

/// (best-lag SNR dB, per-band energy offsets dB) of `y` against `x`.
/// Bands: 0–4 kHz, then 4–8, then 8–16 as the rate allows (FFT-free:
/// the QMF-free estimate is the energy of a first-difference-split
/// signal, so each band is measured with the crate's own analysis bank).
fn measure(x: &[i16], y: &[i16]) -> (f64, i64) {
    let n = x.len().min(y.len());
    let (mut best, mut lag) = (f64::MIN, 0i64);
    for l in 0..400i64 {
        let m = n - l as usize;
        let (mut xy, mut xx, mut yy) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..m {
            let a = f64::from(x[i]);
            let b = f64::from(y[i + l as usize]);
            xy += a * b;
            xx += a * a;
            yy += b * b;
        }
        let c = xy / (xx * yy).sqrt().max(1e-9);
        if c > best {
            best = c;
            lag = l;
        }
    }
    let m = n - lag as usize;
    let (mut s, mut e) = (0.0f64, 0.0f64);
    for i in 0..m {
        let a = f64::from(x[i]);
        let d = a - f64::from(y[i + lag as usize]);
        s += a * a;
        e += d * d;
    }
    (10.0 * (s / e).log10(), lag)
}

/// Energy (dB) of the low / high QMF half of a 16 kHz signal.
fn half_band_db(x: &[f64]) -> (f64, f64) {
    let mut qmf = oxideav_speex::QmfAnalysis::new();
    let n = x.len() / 320 * 320;
    let (mut el, mut eh) = (0.0f64, 0.0f64);
    let (mut lo, mut hi) = ([0.0f64; 160], [0.0f64; 160]);
    for k in (0..n).step_by(320) {
        qmf.split_slices(&x[k..k + 320], &mut lo, &mut hi);
        el += lo.iter().map(|v| v * v).sum::<f64>();
        eh += hi.iter().map(|v| v * v).sum::<f64>();
    }
    (10.0 * el.max(1.0).log10(), 10.0 * eh.max(1.0).log10())
}

fn band_offsets_wb(x: &[i16], y: &[i16], lag: i64) -> (f64, f64) {
    let m = x.len().min(y.len()) - lag as usize;
    let xs: Vec<f64> = x[..m].iter().map(|&v| f64::from(v)).collect();
    let ys: Vec<f64> = y[lag as usize..lag as usize + m]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let (xl, xh) = half_band_db(&xs);
    let (yl, yh) = half_band_db(&ys);
    (yl - xl, yh - xh)
}

#[test]
fn narrowband_speech_round_trip() {
    let x = raw_pcm(include_bytes!("fixtures/nb-conformance/speechy8k.raw"));
    for (q, floor) in [(4u8, 5.0f64), (8, 8.0), (10, 8.0)] {
        let y = round_trip(&x, 8_000, q);
        let (snr, lag) = measure(&x, &y);
        println!("NB speechy8k q{q}: snr {snr:.2} dB (lag {lag})");
        assert!(snr > floor, "NB q{q}: snr {snr:.2} dB <= {floor}");
    }
}

#[test]
fn wideband_quality_10_round_trip() {
    for (name, x) in [
        (
            "speechy16k",
            raw_pcm(include_bytes!("fixtures/wb-conformance/speechy16k.raw")),
        ),
        (
            "hb-mode4 source",
            wav_pcm(include_bytes!("fixtures/hb-mode4-wb-q10/source16k.wav")),
        ),
    ] {
        for q in [8u8, 10] {
            let y = round_trip(&x, 16_000, q);
            let (snr, lag) = measure(&x, &y);
            let (dl, dh) = band_offsets_wb(&x, &y, lag);
            println!(
                "WB {name} q{q}: snr {snr:.2} dB (lag {lag}) band offsets {dl:+.2} / {dh:+.2} dB"
            );
            // r480 measured: speechy16k 11.63 / 13.00 dB, hb-mode4 source
            // 5.23 / 8.95 dB (q8 / q10); the reference encoder on the
            // latter gives 9.25 dB at q10 through the same metric.
            let floor = if q == 10 { 7.5 } else { 4.5 };
            assert!(snr > floor, "WB {name} q{q}: snr {snr:.2} dB <= {floor}");
            assert!(
                dl.abs() < 1.5 && dh.abs() < 3.0,
                "WB {name} q{q}: bands {dl:+.2} / {dh:+.2}"
            );
        }
    }
}

#[test]
fn ultra_wideband_quality_10_round_trip() {
    // 32 kHz quality 10 = 44 kbit/s, the issue-#7 reporter's default.
    for (name, x) in [
        (
            "speechy32k",
            raw_pcm(include_bytes!("fixtures/uwb-conformance/speechy32k.raw")),
        ),
        (
            "hb-mode4 source",
            wav_pcm(include_bytes!("fixtures/hb-mode4-uwb-q10/source32k.wav")),
        ),
    ] {
        let y = round_trip(&x, 32_000, 10);
        let n = x.len().min(y.len());
        // The 8-16 kHz layer is a gain-only fold, so a waveform SNR is
        // meaningless above 8 kHz: measure SNR below 7 kHz (a 511-tap
        // windowed-sinc low-pass applied identically to both signals —
        // a QMF split would alias the folded band back in) and the
        // 8-16 kHz energy from the outer QMF high half.
        let to_f = |v: &[i16]| v.iter().map(|&s| f64::from(s)).collect::<Vec<f64>>();
        let (xf, yf) = (to_f(&x[..n]), to_f(&y[..n]));
        let (xl, yl) = (
            lowpass(&xf, 7_000.0 / 32_000.0),
            lowpass(&yf, 7_000.0 / 32_000.0),
        );
        let mut qa = oxideav_speex::QmfAnalysis::new();
        let mut qb = oxideav_speex::QmfAnalysis::new();
        let (mut xh, mut yh) = (0.0f64, 0.0f64);
        let (mut lo, mut hi) = ([0.0f64; 320], [0.0f64; 320]);
        for k in (0..n / 640 * 640).step_by(640) {
            qa.split_slices(&xf[k..k + 640], &mut lo, &mut hi);
            xh += hi.iter().map(|v| v * v).sum::<f64>();
            qb.split_slices(&yf[k..k + 640], &mut lo, &mut hi);
            yh += hi.iter().map(|v| v * v).sum::<f64>();
        }
        let (snr, lag) = measure(&xl, &yl);
        let m = n - lag as usize - 600;
        let lvl = |v: &[i16]| v.iter().map(|&s| f64::from(s) * f64::from(s)).sum::<f64>();
        let dl = 10.0 * (lvl(&yl[lag as usize..lag as usize + m]) / lvl(&xl[..m])).log10();
        let dh = 10.0 * (yh / xh).log10();
        println!(
            "UWB {name} q10: <7 kHz snr {snr:.2} dB (lag {lag}) level {dl:+.2} dB | 8-16k {dh:+.2} dB"
        );
        assert!(snr > 8.0, "UWB {name}: <7 kHz snr {snr:.2} dB");
        assert!(dl.abs() < 1.5, "UWB {name}: <7 kHz level {dl:+.2} dB");
        // The fold borrows the inner band's excitation, so the outer
        // layer under-reaches on speech whose 8-16 kHz band is louder
        // than its 4-8 kHz one: the reference itself lands at -7.8 dB
        // on speechy32k at q10 (black-box, `speexenc -u` + `speexdec`).
        assert!(
            dh > -6.0 && dh < 3.0,
            "UWB {name}: 8-16 kHz offset {dh:+.2} dB"
        );
    }
}

/// Windowed-sinc (Hann, 511 taps) low-pass at normalised cutoff `fc`
/// (cycles/sample), zero-phase (centred), rounded back to `i16`.
fn lowpass(x: &[f64], fc: f64) -> Vec<i16> {
    const M: i64 = 255;
    let h: Vec<f64> = (-M..=M)
        .map(|k| {
            let t = k as f64;
            let sinc = if k == 0 {
                2.0 * fc
            } else {
                (2.0 * std::f64::consts::PI * fc * t).sin() / (std::f64::consts::PI * t)
            };
            let w = 0.5 + 0.5 * (std::f64::consts::PI * t / (M as f64 + 1.0)).cos();
            sinc * w
        })
        .collect();
    (0..x.len() as i64)
        .map(|n| {
            let mut acc = 0.0f64;
            for (j, &hk) in h.iter().enumerate() {
                let i = n + j as i64 - M;
                if i >= 0 && (i as usize) < x.len() {
                    acc += hk * x[i as usize];
                }
            }
            acc.round().clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

#[test]
fn analysed_speech_envelopes_always_reach_the_lsp_domain() {
    // The LPC sign regression: every analysed frame of real speech must
    // convert to LSPs (it failed on 100 % of frames before r480).
    for x in [
        raw_pcm(include_bytes!("fixtures/nb-conformance/speechy8k.raw")),
        raw_pcm(include_bytes!("fixtures/nb-conformance/src8k.raw")),
    ] {
        let xf: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let mut frames = 0;
        for k in (0..xf.len() - 200).step_by(160) {
            let c = oxideav_speex::lpc_analyse(&xf[k..k + 200]).unwrap();
            if c.a.iter().all(|&v| v == 0.0) {
                continue; // digital silence
            }
            oxideav_speex::lpc_to_lsp(&c.a)
                .unwrap_or_else(|e| panic!("frame {}: {e} (a = {:?})", k / 160, c.a));
            frames += 1;
        }
        assert!(frames >= 40, "{frames} voiced/unvoiced frames analysed");
    }
}
