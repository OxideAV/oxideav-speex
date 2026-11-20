//! Shared measurement helpers for the encoder tests and the
//! `encoder_report` example: a deterministic speech-like source, the
//! fidelity measures, and an Ogg wrapper for handing streams to a
//! black-box decoder.
#![allow(dead_code)]

use std::f64::consts::PI;

/// Deterministic 31-bit linear congruential noise in `[-1, 1)`.
pub struct Noise(pub u32);

impl Noise {
    pub fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
        f64::from(self.0) / f64::from(1u32 << 30) - 1.0
    }
}

/// A two-pole resonator at `freq` Hz with bandwidth `bw` Hz, unit gain
/// at DC.
struct Resonator {
    b0: f64,
    a1: f64,
    a2: f64,
    y1: f64,
    y2: f64,
}

impl Resonator {
    fn new() -> Self {
        Self {
            b0: 1.0,
            a1: 0.0,
            a2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn tune(&mut self, freq: f64, bw: f64, rate: f64) {
        let r = (-PI * bw / rate).exp();
        self.a1 = 2.0 * r * (2.0 * PI * freq / rate).cos();
        self.a2 = -r * r;
        self.b0 = 1.0 - self.a1 - self.a2;
    }

    fn step(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.a1 * self.y1 + self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Deterministic speech-like material: voiced stretches (a gliding
/// pitch pulse train through four moving formant resonators), noisy
/// fricatives, short bursts and near-silent pauses over a faint noise
/// floor — the segment types a speech coder's mode decisions react to.
pub fn speech_like(rate: u32, seconds: f64, seed: u32) -> Vec<i16> {
    let fs = f64::from(rate);
    let n = (fs * seconds) as usize;
    let mut noise = Noise(seed);
    // Vowel formant targets (Hz), F1..F4.
    const VOWELS: [[f64; 4]; 5] = [
        [730.0, 1090.0, 2440.0, 3400.0],
        [270.0, 2290.0, 3010.0, 3600.0],
        [300.0, 870.0, 2240.0, 3300.0],
        [530.0, 1840.0, 2480.0, 3500.0],
        [570.0, 840.0, 2410.0, 3350.0],
    ];
    // Segment plan: (kind, seconds). 0 voiced, 1 fricative, 2 pause, 3 burst.
    const PLAN: [(u8, f64); 12] = [
        (2, 0.12),
        (0, 0.34),
        (1, 0.11),
        (0, 0.27),
        (2, 0.09),
        (3, 0.03),
        (0, 0.41),
        (1, 0.14),
        (2, 0.31),
        (0, 0.22),
        (3, 0.02),
        (0, 0.3),
    ];
    let mut res = [
        Resonator::new(),
        Resonator::new(),
        Resonator::new(),
        Resonator::new(),
    ];
    let mut fric = Resonator::new();
    let mut out: Vec<f64> = Vec::with_capacity(n);
    let (mut seg, mut seg_left) = (0usize, 0usize);
    let (mut kind, mut seg_len) = (2u8, 1usize);
    let mut vowel = 0usize;
    let mut phase = 0.0f64;
    let mut pitch_hz = 120.0f64;
    let mut env = 0.0f64;
    let mut lp = 0.0f64;
    for i in 0..n {
        if seg_left == 0 {
            let (k, d) = PLAN[seg % PLAN.len()];
            kind = k;
            seg_len = ((d * (0.8 + 0.4 * (noise.next() * 0.5 + 0.5))) * fs) as usize + 1;
            seg_left = seg_len;
            seg += 1;
            if kind == 0 {
                vowel = (vowel + 1 + (seg % 3)) % VOWELS.len();
                pitch_hz = 95.0 + 110.0 * (noise.next() * 0.5 + 0.5);
            }
        }
        let pos = (seg_len - seg_left) as f64 / seg_len as f64;
        seg_left -= 1;
        let t = i as f64 / fs;
        let target_env = match kind {
            0 => (PI * pos).sin().powf(0.35),
            1 => 0.25 * (PI * pos).sin().powf(0.5),
            3 => 0.6 * (1.0 - pos),
            _ => 0.0,
        };
        env += 0.004 * (8000.0 / fs) * (target_env - env);
        let mut s = 0.0f64;
        if kind == 0 {
            // Pitch declines through the segment with a slow vibrato.
            let f0 = pitch_hz * (1.0 - 0.18 * pos) * (1.0 + 0.02 * (2.0 * PI * 5.0 * t).sin());
            phase += f0 / fs;
            let mut pulse = 0.0;
            if phase >= 1.0 {
                phase -= 1.0;
                pulse = 1.0;
            }
            // A one-pole glottal shaping of the pulse train plus breath.
            lp = 0.9f64.powf(8000.0 / fs) * lp + pulse;
            let src = lp - 0.12 + 0.02 * noise.next();
            let next = VOWELS[(vowel + 1) % VOWELS.len()];
            let mut v = src;
            for (k, r) in res.iter_mut().enumerate() {
                let f = VOWELS[vowel][k] + (next[k] - VOWELS[vowel][k]) * pos * 0.6;
                if f < 0.46 * fs {
                    r.tune(f, 60.0 + 40.0 * k as f64, fs);
                    v = r.step(v) * 1.6;
                }
            }
            s = v;
        } else if kind == 1 || kind == 3 {
            let centre = if kind == 1 { 0.33 * fs } else { 0.2 * fs };
            fric.tune(centre.min(0.45 * fs), 0.12 * fs, fs);
            s = 6.0 * (fric.step(noise.next()) - 0.3 * noise.next());
        }
        out.push(env * s);
    }
    // Normalise the speech to a 14 000 peak and lay it over a faint
    // stationary noise floor.
    let peak = out.iter().fold(1e-9f64, |m, v| m.max(v.abs()));
    out.iter()
        .map(|v| (14_000.0 * v / peak + 6.0 * noise.next()).round() as i16)
        .collect()
}

/// Order-`p` LPC (`A(z) = 1 + Σ a_k z^{-k}`) of a Hamming-windowed
/// block by the autocorrelation method.
fn lpc(block: &[f64], p: usize) -> Vec<f64> {
    let n = block.len();
    let w: Vec<f64> = block
        .iter()
        .enumerate()
        .map(|(i, &v)| v * (0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect();
    let mut r = vec![0.0f64; p + 1];
    for (k, rk) in r.iter_mut().enumerate() {
        *rk = (k..n).map(|i| w[i] * w[i - k]).sum();
    }
    r[0] = r[0] * 1.0001 + 1.0;
    let mut a = vec![0.0f64; p + 1];
    a[0] = 1.0;
    let mut err = r[0];
    for i in 1..=p {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / err;
        let prev = a.clone();
        for j in 1..i {
            a[j] = prev[j] + k * prev[i - j];
        }
        a[i] = k;
        err *= 1.0 - k * k;
        if err <= 0.0 {
            break;
        }
    }
    a
}

/// The three fidelity measures of a decode `y` against its source `x`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fidelity {
    /// Whole-signal SNR at the best alignment, dB.
    pub snr: f64,
    /// Segmental SNR: mean of the per-20 ms SNR (each clamped to
    /// −10..=35 dB) over the frames that carry signal.
    pub seg_snr: f64,
    /// Noise-weighted segmental SNR: the same mean with source and
    /// error both passed through `W(z) = A(z/0.9)/A(z/0.6)` of the
    /// source frame's own LPC (manual Eq. 8.1), so error under a
    /// formant counts less than error in a spectral valley.
    pub pw_snr: f64,
    /// The alignment (decode delay in samples) the measures use.
    pub lag: usize,
}

/// Measure `y` against `x` (both at `rate`), searching the decode delay
/// over `0..max_lag`.
pub fn fidelity(x: &[i16], y: &[i16], rate: u32, max_lag: usize) -> Fidelity {
    let n = x.len().min(y.len());
    let (mut best, mut lag) = (f64::MIN, 0usize);
    for l in 0..max_lag.min(n / 2) {
        let m = n - l;
        let (mut xy, mut xx, mut yy) = (0.0f64, 0.0f64, 0.0f64);
        for i in (0..m).step_by(1) {
            let a = f64::from(x[i]);
            let b = f64::from(y[i + l]);
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
    fidelity_at(x, y, rate, lag)
}

/// The measures at a known alignment.
pub fn fidelity_at(x: &[i16], y: &[i16], rate: u32, lag: usize) -> Fidelity {
    let n = x.len().min(y.len().saturating_sub(lag));
    let xs: Vec<f64> = x[..n].iter().map(|&v| f64::from(v)).collect();
    let es: Vec<f64> = (0..n)
        .map(|i| f64::from(x[i]) - f64::from(y[i + lag]))
        .collect();
    let (s, e) = (
        xs.iter().map(|v| v * v).sum::<f64>(),
        es.iter().map(|v| v * v).sum::<f64>(),
    );
    let frame = (rate / 50) as usize;
    let order = if rate <= 8_000 { 10 } else { 16 };
    let floor = 60.0 * 60.0 * frame as f64; // frames quieter than ~ -55 dBFS are pauses
    let (mut seg, mut pw, mut frames) = (0.0f64, 0.0f64, 0usize);
    let (mut xm, mut em) = (vec![0.0f64; order], vec![0.0f64; order]); // FIR memories
    let (mut xo, mut eo) = (vec![0.0f64; order], vec![0.0f64; order]); // IIR memories
    for k in 0..n / frame {
        let xr = &xs[k * frame..(k + 1) * frame];
        let er = &es[k * frame..(k + 1) * frame];
        let a = lpc(xr, order);
        let weigh = |v: &[f64], fm: &mut Vec<f64>, om: &mut Vec<f64>| -> f64 {
            let mut acc = 0.0;
            for &s in v {
                let mut u = s;
                let (mut g1, mut g2) = (0.9f64, 0.6f64);
                for j in 0..order {
                    u += a[j + 1] * g1 * fm[j] - a[j + 1] * g2 * om[j];
                    g1 *= 0.9;
                    g2 *= 0.6;
                }
                fm.rotate_right(1);
                fm[0] = s;
                om.rotate_right(1);
                om[0] = u;
                acc += u * u;
            }
            acc
        };
        let (sw, ew) = (weigh(xr, &mut xm, &mut xo), weigh(er, &mut em, &mut eo));
        let (sf, ef) = (
            xr.iter().map(|v| v * v).sum::<f64>(),
            er.iter().map(|v| v * v).sum::<f64>(),
        );
        if sf < floor {
            continue;
        }
        frames += 1;
        seg += (10.0 * (sf / ef.max(1e-9)).log10()).clamp(-10.0, 35.0);
        pw += (10.0 * (sw / ew.max(1e-9)).log10()).clamp(-10.0, 35.0);
    }
    let f = frames.max(1) as f64;
    Fidelity {
        snr: 10.0 * (s / e.max(1e-9)).log10(),
        seg_snr: seg / f,
        pw_snr: pw / f,
        lag,
    }
}

/// Ogg page checksum: CRC-32 with generator 0x04C11DB7, zero initial
/// value, no bit reflection, over the page with a zeroed checksum field.
fn ogg_crc(page: &[u8]) -> u32 {
    let mut crc = 0u32;
    for &b in page {
        crc ^= u32::from(b) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04c1_1db7
            } else {
                crc << 1
            };
        }
    }
    crc
}

fn ogg_page(out: &mut Vec<u8>, flags: u8, granule: u64, serial: u32, seq: u32, packet: &[u8]) {
    let start = out.len();
    out.extend_from_slice(b"OggS");
    out.push(0);
    out.push(flags);
    out.extend_from_slice(&granule.to_le_bytes());
    out.extend_from_slice(&serial.to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    let lacing = packet.len() / 255 + 1;
    out.push(lacing as u8);
    for i in 0..lacing {
        out.push(if i + 1 == lacing {
            (packet.len() % 255) as u8
        } else {
            255
        });
    }
    out.extend_from_slice(packet);
    let crc = ogg_crc(&out[start..]);
    out[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());
}

/// Wrap a Speex packet sequence as an Ogg/Speex file: the 80-byte
/// stream header, an empty comment packet, then one packet per page.
pub fn ogg_speex(header: &[u8; 80], packets: &[Vec<u8>], frame_samples: usize) -> Vec<u8> {
    let serial = 0x5eed_0491;
    let mut out = Vec::new();
    ogg_page(&mut out, 2, 0, serial, 0, header);
    let vendor = b"oxideav-speex";
    let mut comment = Vec::new();
    comment.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    comment.extend_from_slice(vendor);
    comment.extend_from_slice(&0u32.to_le_bytes());
    ogg_page(&mut out, 0, 0, serial, 1, &comment);
    for (i, p) in packets.iter().enumerate() {
        let flags = if i + 1 == packets.len() { 4 } else { 0 };
        let granule = ((i + 1) * frame_samples) as u64;
        ogg_page(&mut out, flags, granule, serial, i as u32 + 2, p);
    }
    out
}

/// The packets of an Ogg file whose packets each fit one page run
/// (lacing reassembled, pages in order).
pub fn ogg_packets(ogg: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut cur = Vec::new();
    let mut p = 0usize;
    while p + 27 <= ogg.len() && &ogg[p..p + 4] == b"OggS" {
        let nseg = usize::from(ogg[p + 26]);
        let table = &ogg[p + 27..p + 27 + nseg];
        let mut q = p + 27 + nseg;
        for &l in table {
            cur.extend_from_slice(&ogg[q..q + usize::from(l)]);
            q += usize::from(l);
            if l < 255 {
                packets.push(core::mem::take(&mut cur));
            }
        }
        p = q;
    }
    packets
}

/// A 16-bit mono (or interleaved) PCM WAV file.
pub fn wav_bytes(pcm: &[i16], rate: u32, channels: u16) -> Vec<u8> {
    let data = pcm.len() as u32 * 2;
    let mut w = Vec::with_capacity(44 + data as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&channels.to_le_bytes());
    w.extend_from_slice(&rate.to_le_bytes());
    w.extend_from_slice(&(rate * u32::from(channels) * 2).to_le_bytes());
    w.extend_from_slice(&(channels * 2).to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data.to_le_bytes());
    for s in pcm {
        w.extend_from_slice(&s.to_le_bytes());
    }
    w
}

/// The samples of a 16-bit PCM WAV file's `data` chunk.
pub fn wav_pcm(wav: &[u8]) -> Vec<i16> {
    let Some(p) = wav.windows(4).position(|w| w == b"data") else {
        return Vec::new();
    };
    raw_pcm(&wav[p + 8..])
}

/// Headerless s16le samples.
pub fn raw_pcm(raw: &[u8]) -> Vec<i16> {
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect()
}

/// Whether an executable answers on this machine (black-box tools are
/// optional; callers skip when absent).
pub fn have_tool(name: &str) -> bool {
    std::process::Command::new(name)
        .arg("--help")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

/// A scratch directory for black-box tool I/O.
pub fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("oxideav-speex-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).ok();
    d
}
