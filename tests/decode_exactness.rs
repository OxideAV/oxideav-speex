//! **Decode exactness against the reference PCM** (round r483).
//!
//! Every staged reference decode is scored sample by sample against
//! this crate's decode of the same `input.spx` through the public
//! [`SpeexStreamDecoder`]: the share of samples that are identical,
//! the share within ±1 LSB, and the largest deviation. The reference
//! float decoder is itself only reproducible to ±2 LSB across hosts
//! (fixture notes), so "within ±1" is the meaningful exactness bar.
//!
//! The decoder's PCM is aligned with the reference's: the only offset
//! is the number of leading samples the reference toolchain trimmed
//! from its file (`lead`, a property of each fixture's capture).
//!
//! Run with `--nocapture` to see the per-fixture numbers.

use oxideav_speex::{SpeexHeader, SpeexStreamDecoder, StereoDecoder};

struct Fixture {
    name: &'static str,
    spx: &'static [u8],
    reference: &'static [u8],
    /// Perceptual enhancer on (the reference was decoded `--enh`).
    enhancer: bool,
    /// Leading samples the reference file trimmed (reference sample
    /// `i` is decode sample `i + lead`).
    lead: usize,
    /// Floors: share of identical samples, share within ±1 LSB, and the
    /// largest allowed deviation.
    min_exact: f64,
    min_within1: f64,
    max_abs: i32,
}

macro_rules! fx {
    ($name:literal, $dir:literal, $pcm:literal, $enh:expr, $lead:expr, $ex:expr, $w1:expr, $mx:expr) => {
        Fixture {
            name: $name,
            spx: include_bytes!(concat!("fixtures/", $dir, "/input.spx")),
            reference: include_bytes!(concat!("fixtures/", $dir, "/", $pcm)),
            enhancer: $enh,
            lead: $lead,
            min_exact: $ex,
            min_within1: $w1,
            max_abs: $mx,
        }
    };
}

const FIXTURES: &[Fixture] = &[
    // --- narrowband ---
    fx!(
        "nb q4 (submode 3)",
        "nb-enhancer-q4",
        "expected.pcm",
        false,
        0,
        0.95,
        1.0,
        1
    ),
    fx!(
        "nb q4 enhancer on",
        "nb-enhancer-q4",
        "expected-enh.pcm",
        true,
        0,
        0.95,
        1.0,
        1
    ),
    fx!(
        "nb q0 vocoder (submode 1)",
        "nb-mode1-vocoder-q0",
        "expected.pcm",
        false,
        0,
        0.97,
        1.0,
        1
    ),
    fx!(
        "nb q0 VAD+DTX (null frames)",
        "nb-mode1-dtx-q0",
        "expected.pcm",
        false,
        0,
        0.95,
        1.0,
        1
    ),
    // --- wideband ---
    fx!(
        "wb q10 (hb 4)",
        "hb-mode4-wb-q10",
        "expected.pcm",
        false,
        223,
        0.92,
        1.0,
        1
    ),
    fx!(
        "wb q10 -20 dB",
        "hb-mode4-wb-q10-lvl20",
        "expected.pcm",
        false,
        223,
        0.98,
        1.0,
        1
    ),
    fx!(
        "wb q10 tilt",
        "hb-mode4-wb-q10-tilt",
        "expected.pcm",
        false,
        223,
        0.98,
        1.0,
        1
    ),
    fx!(
        "wb q6 (hb 2)",
        "hb-mode2-wb-q6",
        "expected.pcm",
        false,
        0,
        0.94,
        1.0,
        1
    ),
    fx!(
        "wb q8 (hb 3)",
        "hb-mode3-wb-q8",
        "expected.pcm",
        false,
        223,
        0.94,
        1.0,
        1
    ),
    fx!(
        "wb q1 fold (hb 1)",
        "wb-mode1-folded",
        "expected.pcm",
        false,
        0,
        0.73,
        0.999,
        2
    ),
    // --- ultra-wideband ---
    fx!(
        "uwb q10 (hb 4 + fold)",
        "hb-mode4-uwb-q10",
        "expected.pcm",
        false,
        509,
        0.90,
        0.995,
        2
    ),
    fx!(
        "uwb q1 speech 3-layer",
        "uwb-speech-3layer",
        "expected.pcm",
        false,
        0,
        0.83,
        0.97,
        4
    ),
    fx!(
        "uwb q1 tone geometry",
        "uwb-fold-geometry",
        "expected.pcm",
        false,
        0,
        0.79,
        1.0,
        1
    ),
];

fn lift(buf: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos + 27 <= buf.len() && &buf[pos..pos + 4] == b"OggS" {
        let nseg = buf[pos + 26] as usize;
        let mut body = pos + 27 + nseg;
        for &ln in &buf[pos + 27..pos + 27 + nseg] {
            current.extend_from_slice(&buf[body..body + ln as usize]);
            body += ln as usize;
            if ln < 255 {
                packets.push(core::mem::take(&mut current));
            }
        }
        pos = body;
    }
    packets
}

fn pcm(bytes: &[u8]) -> Vec<i16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn decode(spx: &[u8], enhancer: bool) -> Vec<i16> {
    let packets = lift(spx);
    let header = SpeexHeader::parse(&packets[0]).expect("header");
    let mut dec = SpeexStreamDecoder::for_header(&header).expect("mode");
    dec.set_enhancer(enhancer);
    let mut out = Vec::new();
    for p in &packets[2..] {
        out.extend(dec.decode_packet_pcm_i16(p).expect("packet decodes"));
    }
    out
}

/// `(exact share, within-±1 share, max |error|, compared samples)`.
fn score(reference: &[i16], ours: &[i16], lead: usize) -> (f64, f64, i32, usize) {
    let n = reference.len().min(ours.len().saturating_sub(lead));
    let (mut exact, mut w1, mut mx) = (0usize, 0usize, 0i32);
    for i in 0..n {
        let d = (i32::from(reference[i]) - i32::from(ours[i + lead])).abs();
        exact += usize::from(d == 0);
        w1 += usize::from(d <= 1);
        mx = mx.max(d);
    }
    (exact as f64 / n as f64, w1 as f64 / n as f64, mx, n)
}

#[test]
fn mono_fixtures_decode_to_the_reference() {
    let mut failures = Vec::new();
    for fx in FIXTURES {
        let ours = decode(fx.spx, fx.enhancer);
        let reference = pcm(fx.reference);
        let (exact, w1, mx, n) = score(&reference, &ours, fx.lead);
        println!(
            "{:<30} {n:>6} samples: exact {:6.2} %  within ±1 {:7.3} %  max {mx}",
            fx.name,
            100.0 * exact,
            100.0 * w1
        );
        assert!(n > 10_000, "{}: comparison window", fx.name);
        if exact < fx.min_exact || w1 < fx.min_within1 || mx > fx.max_abs {
            failures.push(fx.name);
        }
    }
    assert!(
        failures.is_empty(),
        "below the exactness floor: {failures:?}"
    );
}

/// The alignment really is the trimmed lead: one sample either way
/// breaks it.
#[test]
fn reference_alignment_is_sample_exact() {
    for fx in FIXTURES.iter().take(5) {
        let ours = decode(fx.spx, fx.enhancer);
        let reference = pcm(fx.reference);
        let (at, ..) = score(&reference, &ours, fx.lead);
        let (next, ..) = score(&reference, &ours, fx.lead + 1);
        assert!(at > 0.9 && next < 0.2, "{}: {at} vs {next}", fx.name);
    }
}

/// `(name, input.spx, expected-mono.pcm, expected.pcm, mono floor, L/R floor)`.
type StereoCase = (
    &'static str,
    &'static [u8],
    &'static [u8],
    &'static [u8],
    f64,
    f64,
);

/// Intensity stereo: the transmitted mono decode, and the interleaved
/// L/R reconstruction against the reference's two-channel decode
/// (floors: mono exact share, L/R exact share).
#[test]
fn stereo_fixtures_decode_to_the_reference() {
    let cases: [StereoCase; 4] = [
        (
            "stereo nb q4",
            include_bytes!("fixtures/stereo-nb-ladder-q4/input.spx"),
            include_bytes!("fixtures/stereo-nb-ladder-q4/expected-mono.pcm"),
            include_bytes!("fixtures/stereo-nb-ladder-q4/expected.pcm"),
            0.95,
            0.96,
        ),
        (
            "stereo nb q8",
            include_bytes!("fixtures/stereo-nb-ladder-q8/input.spx"),
            include_bytes!("fixtures/stereo-nb-ladder-q8/expected-mono.pcm"),
            include_bytes!("fixtures/stereo-nb-ladder-q8/expected.pcm"),
            0.95,
            0.96,
        ),
        (
            "stereo wb q6",
            include_bytes!("fixtures/stereo-wb-ladder-q6/input.spx"),
            include_bytes!("fixtures/stereo-wb-ladder-q6/expected-mono.pcm"),
            include_bytes!("fixtures/stereo-wb-ladder-q6/expected.pcm"),
            0.96,
            0.98,
        ),
        (
            "stereo uwb q8",
            include_bytes!("fixtures/stereo-uwb-decorr-q8/input.spx"),
            include_bytes!("fixtures/stereo-uwb-decorr-q8/expected-mono.pcm"),
            include_bytes!("fixtures/stereo-uwb-decorr-q8/expected.pcm"),
            0.96,
            0.97,
        ),
    ];
    let mut failures = Vec::new();
    for (name, spx, ref_mono, ref_stereo, min_mono, min_stereo) in cases {
        let packets = lift(spx);
        let header = SpeexHeader::parse(&packets[0]).expect("header");
        let mut dec = SpeexStreamDecoder::for_header(&header).expect("mode");
        dec.set_enhancer(false);
        let mut sd = StereoDecoder::new();
        let (mut mono, mut stereo) = (Vec::new(), Vec::new());
        for p in &packets[2..] {
            for (m, payload) in dec.decode_packet_frames_stereo(p).expect("decodes") {
                stereo.extend(sd.interleave_frame(&m, payload.unwrap_or(0b0000_0011)));
                mono.extend(m);
            }
        }
        let ref_mono = pcm(ref_mono);
        let ref_stereo = pcm(ref_stereo);
        // Find the trimmed lead on the mono decode (whole samples).
        let lead = (0..800)
            .max_by(|&a, &b| {
                score(&ref_mono, &mono, a)
                    .0
                    .partial_cmp(&score(&ref_mono, &mono, b).0)
                    .unwrap()
            })
            .unwrap();
        let (m_ex, m_w1, m_mx, _) = score(&ref_mono, &mono, lead);
        let (s_ex, s_w1, s_mx, _) = score(&ref_stereo, &stereo, 2 * lead);
        println!(
            "{name:<14} lead {lead}: mono exact {:.2} % ±1 {:.3} % max {m_mx} | \
             L/R exact {:.2} % ±1 {:.3} % max {s_mx}",
            100.0 * m_ex,
            100.0 * m_w1,
            100.0 * s_ex,
            100.0 * s_w1
        );
        // The two-channel decode multiplies the 16-bit mono by gains up
        // to 2: a mono sample one LSB off is up to two LSB off per
        // channel, so ±2 bounds the stereo law itself.
        if m_ex < min_mono || m_w1 < 1.0 || s_ex < min_stereo || s_mx > 2 {
            failures.push(name);
        }
    }
    assert!(failures.is_empty(), "below the floor: {failures:?}");
}
