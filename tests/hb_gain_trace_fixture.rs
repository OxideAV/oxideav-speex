//! **Exact high-band innovation gain gate** — the Tracer round's
//! per-sub-frame decoder traces (`docs/audio/speex/tracer-2026-09.md`
//! §1, mirrored under `tests/fixtures/`).
//!
//! For high-band sub-modes 2/3/4 the reference applies
//! `S = g_c · e_L / R` per sub-frame, `e_L = sqrt(0.1 + mean e_nb²)`
//! (the narrowband excitation of the same sub-frame) and
//! `R = (A_nb(−1)+0.01)/(A_hb(−1)+0.01)` from the interpolated quantised
//! LPC. The staged `hb-gain-trace.csv` carries every one of those
//! quantities for every sub-frame of six streams. This gate decodes each
//! stream with the crate and checks, sub-frame by sub-frame:
//!
//! 1. the trace satisfies [`oxideav_speex::hb_gc_crossover_gain`]
//!    exactly (the crate's law *is* the traced law);
//! 2. the crate's own decoder state reproduces the traced `A_nb(−1)`,
//!    `A_hb(−1)` and `R` (pure LSP/LPC arithmetic — near exact);
//! 3. the crate's applied scale `S` tracks the traced one (it inherits
//!    the narrowband excitation's non-bit-exact residual through `e_L`).

use oxideav_speex::{
    UltraWidebandDecoder, WidebandDecoder, hb_encoder_gain_index, hb_gain_lb_base,
    hb_gc_crossover_gain,
};

fn lift(buf: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos + 27 <= buf.len() {
        if &buf[pos..pos + 4] != b"OggS" {
            break;
        }
        let nseg = buf[pos + 26] as usize;
        let segments = &buf[pos + 27..pos + 27 + nseg];
        let mut body_pos = pos + 27 + nseg;
        for &ln in segments {
            let take = ln as usize;
            current.extend_from_slice(&buf[body_pos..body_pos + take]);
            body_pos += take;
            if take < 255 {
                packets.push(core::mem::take(&mut current));
            }
        }
        pos = body_pos;
    }
    packets
}

/// One `hb-gain-trace.csv` row (float build).
struct Row {
    frame: usize,
    sub: usize,
    gain_corr: f64,
    nb_rms: f64,
    nb_api: f64,
    hb_api: f64,
    ratio: f64,
    scale: f64,
}

fn rows(csv: &str) -> Vec<Row> {
    csv.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("frame"))
        .map(|l| {
            let c: Vec<&str> = l.split(',').collect();
            let f = |i: usize| -> f64 { c[i].parse().unwrap() };
            Row {
                frame: c[0].parse().unwrap(),
                sub: c[1].parse().unwrap(),
                gain_corr: f(4),
                nb_rms: f(5),
                nb_api: f(6),
                hb_api: f(7),
                ratio: f(8),
                scale: f(9),
            }
        })
        .collect()
}

/// Per sub-frame: (A_nb(−1), A_hb(−1), e_L / (A_nb(−1)+0.01)).
type State = (f64, f64, f64);

fn wb_state(spx: &[u8]) -> Vec<State> {
    let mut dec = WidebandDecoder::new();
    let mut out = Vec::new();
    for p in &lift(spx)[2..] {
        dec.decode_packet(p).expect("stream decodes");
        collect(&dec, &mut out);
    }
    out
}

fn uwb_state(spx: &[u8]) -> Vec<State> {
    let mut dec = UltraWidebandDecoder::new();
    let mut out = Vec::new();
    for p in &lift(spx)[2..] {
        dec.decode_packet(p).expect("stream decodes");
        collect(dec.wideband_decoder(), &mut out);
    }
    out
}

fn collect(dec: &WidebandDecoder, out: &mut Vec<State>) {
    let api = *dec.low_band_decoder().last_crossover_response();
    let env = *dec.last_hb_env();
    let exc = dec.low_band_decoder().last_frame_excitation();
    for sf in 0..4 {
        let base = hb_gain_lb_base(&exc[sf * 40..sf * 40 + 40], api[sf]);
        out.push((api[sf], env[sf].0, base));
    }
}

fn pct(v: &mut [f64], p: usize) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(v.len() - 1) * p / 100]
}

/// Gate one stream. `p90_db` / `max_db` bound the applied-scale error.
fn gate(name: &str, state: &[State], csv: &str, p90_db: f64, max_db: f64) {
    let rows = rows(csv);
    assert_eq!(rows.len(), 304, "{name}: 76 frames x 4 sub-frames");
    assert_eq!(state.len(), rows.len(), "{name}: decoded sub-frame count");
    let (mut r_err, mut s_db) = (Vec::new(), Vec::new());
    for r in &rows {
        // 1. The trace obeys the crate's law (identity on traced values).
        let lb = r.nb_rms / (r.nb_api + 0.01);
        let s_law = hb_gc_crossover_gain(r.gain_corr as f32, lb, r.hb_api);
        assert!(
            ((s_law - r.scale) / r.scale).abs() < 1e-5,
            "{name} f{} s{}: law {s_law} vs traced {}",
            r.frame,
            r.sub,
            r.scale
        );
        // 2. + 3. The crate's decoder state against the trace.
        let (nb_api, hb_api, base) = state[r.frame * 4 + r.sub];
        let ratio = (nb_api + 0.01) / (hb_api + 0.01);
        r_err.push(((ratio - r.ratio) / r.ratio).abs());
        let s = hb_gc_crossover_gain(r.gain_corr as f32, base, hb_api);
        s_db.push(20.0 * (s / r.scale).log10().abs());
    }
    let r_max = pct(&mut r_err, 100);
    let (s50, s90, smax) = (pct(&mut s_db, 50), pct(&mut s_db, 90), pct(&mut s_db, 100));
    println!(
        "{name}: R max rel {r_max:.2e} | S |err| dB median {s50:.3} p90 {s90:.3} max {smax:.3}"
    );
    assert!(
        r_max < 1e-3,
        "{name}: crossover ratio max rel err {r_max:.2e}"
    );
    assert!(s50 < 0.1, "{name}: median S error {s50:.3} dB");
    assert!(s90 < p90_db, "{name}: p90 S error {s90:.3} dB");
    assert!(smax < max_db, "{name}: max S error {smax:.3} dB");
}

#[test]
fn mode4_wb_q10_gain_trace() {
    let st = wb_state(include_bytes!("fixtures/hb-mode4-wb-q10/input.spx"));
    gate(
        "hb-mode4-wb-q10",
        &st,
        include_str!("fixtures/hb-mode4-wb-q10/hb-gain-trace.csv"),
        0.1,
        0.3,
    );
}

#[test]
fn mode4_wb_q10_lvl20_gain_trace() {
    // −20 dB twin: e_L moves ×0.1, R is unchanged (tracer §1.4).
    let st = wb_state(include_bytes!("fixtures/hb-mode4-wb-q10-lvl20/input.spx"));
    gate(
        "hb-mode4-wb-q10-lvl20",
        &st,
        include_str!("fixtures/hb-mode4-wb-q10-lvl20/hb-gain-trace.csv"),
        0.5,
        1.0,
    );
}

#[test]
fn mode4_wb_q10_tilt_gain_trace() {
    // Tilted twin: 13 distinct HB envelopes, R moves (tracer §1.4).
    let st = wb_state(include_bytes!("fixtures/hb-mode4-wb-q10-tilt/input.spx"));
    gate(
        "hb-mode4-wb-q10-tilt",
        &st,
        include_str!("fixtures/hb-mode4-wb-q10-tilt/hb-gain-trace.csv"),
        0.2,
        0.6,
    );
}

#[test]
fn mode4_uwb_q10_inner_gain_trace() {
    let st = uwb_state(include_bytes!("fixtures/hb-mode4-uwb-q10/input.spx"));
    gate(
        "hb-mode4-uwb-q10",
        &st,
        include_str!("fixtures/hb-mode4-uwb-q10/hb-gain-trace.csv"),
        0.2,
        0.6,
    );
}

#[test]
fn mode2_and_mode3_share_the_law() {
    let st = wb_state(include_bytes!("fixtures/hb-mode2-wb-q6/input.spx"));
    gate(
        "hb-mode2-wb-q6",
        &st,
        include_str!("fixtures/hb-mode2-wb-q6/hb-gain-trace.csv"),
        0.1,
        0.3,
    );
    let st = wb_state(include_bytes!("fixtures/hb-mode3-wb-q8/input.spx"));
    gate(
        "hb-mode3-wb-q8",
        &st,
        include_str!("fixtures/hb-mode3-wb-q8/hb-gain-trace.csv"),
        0.1,
        0.3,
    );
}

/// The encoder side (`tracer-2026-09.md` §1.3): the staged float
/// *encoder* traces record, per sub-frame, the inputs of the reference's
/// gain decision and the field it chose. [`hb_encoder_gain_index`] must
/// choose the same field from the same inputs on every row.
#[test]
fn encoder_gain_rule_reproduces_traced_indices() {
    let traces = [
        include_str!("fixtures/hb-mode4-wb-q10-lvl20/hb-encoder-gain-trace.csv"),
        include_str!("fixtures/hb-mode4-wb-q10-tilt/hb-encoder-gain-trace.csv"),
        include_str!("fixtures/hb-mode2-wb-q6/hb-encoder-gain-trace.csv"),
        include_str!("fixtures/hb-mode3-wb-q8/hb-encoder-gain-trace.csv"),
    ];
    let mut n = 0usize;
    for csv in traces {
        for l in csv
            .lines()
            .filter(|l| !l.starts_with('#') && !l.starts_with("frame"))
        {
            let c: Vec<f64> = l.split(',').map(|v| v.parse().unwrap()).collect();
            let (e_h, e_l, ratio, q) = (c[2], c[3], c[4], c[6] as u8);
            assert_eq!(
                hb_encoder_gain_index(ratio, e_h, e_l),
                q,
                "frame {} sub {}: ratio {ratio} e_H {e_h} e_L {e_l}",
                c[0],
                c[1]
            );
            n += 1;
        }
    }
    assert_eq!(n, 4 * 304, "four traced streams x 304 sub-frames");
}
