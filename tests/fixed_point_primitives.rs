//! **Fixed-point primitives against the staged fixed-point traces**
//! (round r483; `docs/audio/speex/tracer-2026-09.md` §1.2, §3, §4).
//!
//! * the Q13 LSP cosine and its stored 16-bit form over every input
//!   (`tables/lsp-cos-q13-full.csv`, 25 737 rows);
//! * the Q13 LSP decode + interpolation + LSP → LPC schedule of both
//!   bands, through the traced `A(−1)` values of every sub-frame of five
//!   wideband fixtures (`hb-gain-trace-fixed.csv`, 5 × 304 × 2);
//! * the high-band gain schedule (`g_c`, `R`, `S`) on the same rows;
//! * the integer noise draws and the generator state across the
//!   submode-1 and null frames (`prng-trace-fixed.csv`).

use oxideav_speex::{
    FixedNoise, PacketFrame, PacketFrames, Submode, WidebandSubmode, hb_gain_q14, hb_lsp_q13,
    interpolate_lsp_q13, lpc_at_minus_one_q13, lsp_cos_q13, lsp_cos_stored, lsp_to_lpc_q13,
    nb_lsp_q13,
};

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

fn rows(csv: &str) -> Vec<Vec<&str>> {
    csv.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("frame") && !l.starts_with("x_q13"))
        .map(|l| l.split(',').map(str::trim).collect())
        .collect()
}

fn int(s: &str) -> i64 {
    s.parse::<f64>().unwrap() as i64
}

#[test]
fn cosine_matches_every_tabulated_input() {
    let table = include_str!("fixtures/tables/lsp-cos-q13-full.csv");
    let mut n = 0;
    for r in rows(table) {
        let x = int(r[0]) as i32;
        assert_eq!(i64::from(lsp_cos_q13(x)), int(r[1]), "cos at {x}");
        assert_eq!(i64::from(lsp_cos_stored(x)), int(r[2]), "stored at {x}");
        n += 1;
    }
    assert_eq!(n, 25_737);
}

const WB: [(&str, &[u8], &str); 5] = [
    (
        "hb-mode4-wb-q10",
        include_bytes!("fixtures/hb-mode4-wb-q10/input.spx"),
        include_str!("fixtures/hb-mode4-wb-q10/hb-gain-trace-fixed.csv"),
    ),
    (
        "hb-mode4-wb-q10-tilt",
        include_bytes!("fixtures/hb-mode4-wb-q10-tilt/input.spx"),
        include_str!("fixtures/hb-mode4-wb-q10-tilt/hb-gain-trace-fixed.csv"),
    ),
    (
        "hb-mode4-wb-q10-lvl20",
        include_bytes!("fixtures/hb-mode4-wb-q10-lvl20/input.spx"),
        include_str!("fixtures/hb-mode4-wb-q10-lvl20/hb-gain-trace-fixed.csv"),
    ),
    (
        "hb-mode2-wb-q6",
        include_bytes!("fixtures/hb-mode2-wb-q6/input.spx"),
        include_str!("fixtures/hb-mode2-wb-q6/hb-gain-trace-fixed.csv"),
    ),
    (
        "hb-mode3-wb-q8",
        include_bytes!("fixtures/hb-mode3-wb-q8/input.spx"),
        include_str!("fixtures/hb-mode3-wb-q8/hb-gain-trace-fixed.csv"),
    ),
];

/// Per-sub-frame `(A_nb(−1), A_hb(−1))` in Q13 from the fixed-point
/// LSP path, frame-major.
fn crossover_q13(spx: &[u8]) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    let (mut prev_nb, mut prev_hb): (Option<[i32; 10]>, Option<[i32; 8]>) = (None, None);
    for p in &lift(spx)[2..] {
        for f in PacketFrames::new(p) {
            let PacketFrame::Wideband {
                header,
                narrowband,
                high_band_header,
                high_band,
            } = f.expect("frame parses")
            else {
                continue;
            };
            let (Submode::Celp(nsm), WidebandSubmode::Documented(hsm)) =
                (header.submode, high_band_header.submode)
            else {
                panic!("WB fixture frame");
            };
            let nb = nb_lsp_q13(&narrowband, &nsm).expect("nb lsp");
            let hb = hb_lsp_q13(&high_band, &hsm).expect("hb lsp");
            let (pn, ph) = (prev_nb.unwrap_or(nb), prev_hb.unwrap_or(hb));
            for s in 0..4 {
                let a_nb = lsp_to_lpc_q13(&interpolate_lsp_q13(&pn, &nb, s));
                let a_hb = lsp_to_lpc_q13(&interpolate_lsp_q13(&ph, &hb, s));
                out.push((lpc_at_minus_one_q13(&a_nb), lpc_at_minus_one_q13(&a_hb)));
            }
            prev_nb = Some(nb);
            prev_hb = Some(hb);
        }
    }
    out
}

#[test]
fn lsp_to_lpc_reproduces_traced_crossovers_in_both_bands() {
    for (name, spx, trace) in WB {
        let ours = crossover_q13(spx);
        let t = rows(trace);
        assert_eq!(t.len(), 304, "{name}");
        for r in &t {
            let i = int(r[0]) as usize * 4 + int(r[1]) as usize;
            assert_eq!(i64::from(ours[i].0), int(r[6]), "{name} {i}: A_nb(−1)");
            assert_eq!(i64::from(ours[i].1), int(r[7]), "{name} {i}: A_hb(−1)");
        }
    }
}

#[test]
fn high_band_gain_schedule_reproduces_the_trace() {
    for (name, _, trace) in WB {
        for r in rows(trace) {
            let (gc, ratio, s) = hb_gain_q14(
                int(r[3]) as u8,
                int(r[5]) as i32,
                int(r[6]) as i32,
                int(r[7]) as i32,
            );
            assert_eq!(
                (i64::from(gc), i64::from(ratio), i64::from(s)),
                (int(r[4]), int(r[8]), int(r[9])),
                "{name} frame {} sub {}",
                r[0],
                r[1]
            );
        }
    }
}

#[test]
fn integer_noise_draws_and_state_hand_over() {
    for (name, trace) in [
        (
            "vocoder",
            include_str!("fixtures/nb-mode1-vocoder-q0/prng-trace-fixed.csv"),
        ),
        (
            "dtx",
            include_str!("fixtures/nb-mode1-dtx-q0/prng-trace-fixed.csv"),
        ),
    ] {
        let mut g = FixedNoise::new();
        let mut draws = 0;
        for r in rows(trace) {
            assert_eq!(i64::from(g.state()), int(r[3]), "{name}: state in");
            let null = r[1] == "all";
            let std = if null { int(r[6]) as i32 } else { 1 };
            let n = if null { 160 } else { 40 };
            let first: Vec<i64> = (0..n)
                .map(|_| {
                    let v = i64::from(g.draw(std));
                    // Mode-1 innovation samples are traced in Q14.
                    if null { v } else { v << 14 }
                })
                .collect();
            for k in 0..8 {
                assert_eq!(first[k], int(r[9 + k]), "{name}: draw {k}");
                draws += 1;
            }
            assert_eq!(i64::from(g.state()), int(r[4]), "{name}: state out");
        }
        assert!(draws > 2_400, "{name}: {draws}");
    }
}
