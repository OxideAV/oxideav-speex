//! **Exact folded (mode-1) gain gate** — the Tracer round's per-sub-frame
//! fold traces (`docs/audio/speex/tracer-2026-09.md` §5–§6, mirrored under
//! `tests/fixtures/`).
//!
//! Layer 1 (4–8 kHz): `x = 0.9·exp((q−10)/8)·(A_hb(−1)+0.01)/(A_nb(−1)+0.01)
//! ·(−1)ⁿ·inn_nb`. Layer 2 (8–16 kHz): `x = 0.7·exp((q−10)/8)
//! ·(A_l2(−1)+0.01)/(A_hb(+1)+0.01)·zerostuff(x_hb)`. For every traced
//! sub-frame this gate checks that the crate's decoder state reproduces
//! the traced crossover ratio (pure LSP/LPC arithmetic), that the crate's
//! scale equals the traced `fold_const·fold_gain`, and that the fold
//! source RMS tracks the traced one (it inherits the narrowband decode's
//! non-bit-exact residual).

use oxideav_speex::{
    HB_FOLD_CROSSOVER_SCALE, UWB_FOLD_CROSSOVER_SCALE, UltraWidebandDecoder, WidebandDecoder,
    floored_rms, fold_gain_from_index,
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

/// One fold-trace row (float build).
struct Row {
    frame: usize,
    sub: usize,
    layer: u8,
    q: u8,
    ratio: f64,
    scale: f64,
    src_rms: f64,
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
                layer: c[2].parse().unwrap(),
                q: c[3].parse().unwrap(),
                ratio: f(6),
                scale: f(7) * f(8),
                src_rms: f(9),
            }
        })
        .collect()
}

/// Per (frame, sub-frame, layer): (crossover ratio, fold scale, source RMS).
type Key = (usize, usize, u8);
type Val = (f64, f64, f64);

fn layer1(dec: &WidebandDecoder, frame: usize, out: &mut Vec<(Key, f64, f64, f64)>) {
    let api = dec.low_band_decoder().last_crossover_response();
    let env = dec.last_hb_env();
    let inn = dec.low_band_decoder().last_frame_innovation();
    for sf in 0..4 {
        let ratio = (api[sf] + 0.01) / (env[sf].0 + 0.01);
        let src = floored_rms(inn[sf * 40..sf * 40 + 40].iter().map(|&v| f64::from(v)));
        out.push(((frame, sf, 1), ratio, HB_FOLD_CROSSOVER_SCALE / ratio, src));
    }
}

fn wb_state(spx: &[u8]) -> Vec<(Key, f64, f64, f64)> {
    let mut dec = WidebandDecoder::new();
    let mut out = Vec::new();
    for (f, p) in lift(spx)[2..].iter().enumerate() {
        dec.decode_packet(p).expect("stream decodes");
        layer1(&dec, f, &mut out);
    }
    out
}

fn uwb_state(spx: &[u8]) -> Vec<(Key, f64, f64, f64)> {
    let mut dec = UltraWidebandDecoder::new();
    let mut out = Vec::new();
    for (f, p) in lift(spx)[2..].iter().enumerate() {
        dec.decode_packet(p).expect("stream decodes");
        let wb = dec.wideband_decoder();
        layer1(wb, f, &mut out);
        let env = wb.last_hb_env();
        let l2 = dec.last_layer2_api();
        let exc = wb.last_hb_excitation();
        for sf in 0..4 {
            let ratio = (env[sf].1 + 0.01) / (l2[sf] + 0.01);
            // Zero-stuffed: 40 samples carry energy over an 80-sample span.
            let src = floored_rms(
                exc[sf * 40..sf * 40 + 40]
                    .iter()
                    .flat_map(|&v| [f64::from(v), 0.0]),
            );
            out.push(((f, sf, 2), ratio, UWB_FOLD_CROSSOVER_SCALE / ratio, src));
        }
    }
    out
}

fn pct(v: &mut [f64], p: usize) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(v.len() - 1) * p / 100]
}

fn gate(name: &str, state: &[(Key, f64, f64, f64)], csv: &str, layer: u8, src_p90_db: f64) {
    let map: std::collections::HashMap<Key, Val> =
        state.iter().map(|&(k, r, s, e)| (k, (r, s, e))).collect();
    let rows: Vec<Row> = rows(csv).into_iter().filter(|r| r.layer == layer).collect();
    assert!(
        rows.len() >= 280,
        "{name}: {} layer-{layer} rows",
        rows.len()
    );
    let (mut r_err, mut s_err, mut e_db) = (Vec::new(), Vec::new(), Vec::new());
    for r in &rows {
        let &(ratio, unit, src) = map
            .get(&(r.frame, r.sub, r.layer))
            .unwrap_or_else(|| panic!("{name}: no decode for frame {} sub {}", r.frame, r.sub));
        r_err.push(((ratio - r.ratio) / r.ratio).abs());
        let s = unit * fold_gain_from_index(r.q);
        s_err.push(((s - r.scale) / r.scale).abs());
        if r.src_rms > 1.0 {
            e_db.push(20.0 * (src / r.src_rms).log10().abs());
        }
    }
    let (r_max, s_max) = (pct(&mut r_err, 100), pct(&mut s_err, 100));
    let (e50, e90) = (pct(&mut e_db, 50), pct(&mut e_db, 90));
    println!(
        "{name} layer {layer}: {} rows | R max rel {r_max:.2e} | scale max rel {s_max:.2e} | \
         source rms |err| dB median {e50:.3} p90 {e90:.3}",
        rows.len()
    );
    assert!(r_max < 2e-3, "{name}: ratio max rel err {r_max:.2e}");
    assert!(s_max < 2e-3, "{name}: scale max rel err {s_max:.2e}");
    assert!(e50 < 0.1, "{name}: median source error {e50:.3} dB");
    assert!(e90 < src_p90_db, "{name}: p90 source error {e90:.3} dB");
}

#[test]
fn wb_mode1_fold_trace() {
    let st = wb_state(include_bytes!("fixtures/wb-mode1-folded/input.spx"));
    gate(
        "wb-mode1-folded",
        &st,
        include_str!("fixtures/wb-mode1-folded/fold-gain-trace.csv"),
        1,
        0.5,
    );
}

#[test]
fn uwb_speech_both_fold_layers_trace() {
    let st = uwb_state(include_bytes!("fixtures/uwb-speech-3layer/input.spx"));
    let csv = include_str!("fixtures/uwb-speech-3layer/fold-gain-trace.csv");
    gate("uwb-speech-3layer", &st, csv, 1, 0.5);
    gate("uwb-speech-3layer", &st, csv, 2, 0.5);
}

#[test]
fn uwb_q10_outer_fold_over_mode4_source_trace() {
    // The outer layer folds an innovation-coded (HB mode 4) inner band.
    let st = uwb_state(include_bytes!("fixtures/hb-mode4-uwb-q10/input.spx"));
    gate(
        "hb-mode4-uwb-q10",
        &st,
        include_str!("fixtures/hb-mode4-uwb-q10/uwb-fold-trace.csv"),
        2,
        0.5,
    );
}
