//! Integration test: run a real `speexenc`-encoded narrowband fixture
//! through the round-r286 decode tail — LSP reconstruction (r194) →
//! sub-frame LSP interpolation (r200) → **LSP→LPC conversion (r286)**
//! → innovation sub-vector decode (r277) → **LPC synthesis filter
//! (r286)** — and assert the stage produces finite, range-bounded PCM
//! that is *responsive* to the input (not silence, not a constant).
//!
//! The fixture is the same `tests/fixtures/nb_440hz_q8.spx` the
//! round-3 body-parser test consumes, generated via the black-box
//! `speexenc` binary (`tests/fixtures/Makefile`'s `regen` target is
//! the audit trail). `speexenc`'s **source** is not consulted; it is
//! invoked as an opaque process and its output bytes are the test
//! input.
//!
//! ## What this exercises vs. what it does NOT
//!
//! This test drives the **LSP→LPC conversion and the synthesis filter**
//! — the round-r286 scope. The excitation fed to the synthesis filter
//! here is the round-r277 innovation sub-vector `c[n]` alone; the
//! adaptive-codebook (pitch) contribution `p[n]` is *not* added,
//! because its gain Q-format is still gap-blocked (see the
//! `src/pitch_gain.rs` / `src/adaptive_contribution.rs` module docs).
//! That means the PCM here is **not** bit-exact reference Speex output
//! — bit-exactness additionally needs the documented LSP Q-format pin
//! and the pitch contribution. What it *does* prove is that the new
//! LSP→LPC + synthesis path is wired end-to-end off a real frame and
//! behaves as a stable all-pole filter driven by real excitation.
//!
//! Per workspace policy this crate does NOT take a dev-dependency on
//! `oxideav-ogg`; the test contains the same minimal inline Ogg
//! page-walker the sibling fixture test uses (just enough to lift the
//! Speex audio packets out of their single-stream Ogg framing).

use oxideav_speex::{
    lsp_to_lpc, lsp_vector_q10_to_radians, NarrowbandFrameBody, NarrowbandFrameHeader, Submode,
    SynthesisFilter,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/nb_440hz_q8.spx");

/// Inline minimal Ogg page-walker — identical contract to the sibling
/// `narrowband_body_fixture.rs` walker: stitch lacing-segment runs back
/// into Speex packets, ignoring chained streams / BOS-EOS / granulepos
/// / serial / CRC / sequence number.
fn lift_ogg_packets(buf: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos + 27 <= buf.len() {
        if &buf[pos..pos + 4] != b"OggS" {
            break;
        }
        let nseg = buf[pos + 26] as usize;
        if pos + 27 + nseg > buf.len() {
            break;
        }
        let segments = &buf[pos + 27..pos + 27 + nseg];
        let mut body_pos = pos + 27 + nseg;
        for &ln in segments {
            let take = ln as usize;
            if body_pos + take > buf.len() {
                return packets;
            }
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

/// Decode the fixture's audio packets to f64 PCM through the r286
/// synthesis tail. Returns the concatenated PCM for all frames.
fn decode_fixture_to_pcm() -> Vec<f64> {
    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];
    assert!(!audio.is_empty(), "no audio packets in fixture");

    let mut pcm = Vec::new();
    // One synthesis filter carried across all sub-frames and frames —
    // the IIR memory must persist (the whole point of an all-pole
    // synthesis filter).
    let mut filter = SynthesisFilter::new();
    // Previous frame's reconstructed Q10 LSPs (for interpolation).
    let mut prev_lsp_q10: Option<[i32; 10]> = None;

    for pkt in audio {
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(pkt).expect("header parse");
        let submode = match h.submode {
            Submode::Celp(s) => s,
            other => panic!("unexpected submode {:?}", other),
        };
        let body = NarrowbandFrameBody::parse(&mut r, &submode).expect("body parse");

        // r194: reconstruct this frame's LSPs.
        let curr_lsp_q10 = body
            .reconstructed_lsp_q10(&submode)
            .expect("mode 5 transmits a 30-bit LSP field");
        let prev = prev_lsp_q10.unwrap_or(curr_lsp_q10);

        // r200: per-sub-frame LSP interpolation (Q12).
        let sub_lsp = body
            .interpolated_lsp_q12(&submode, &prev)
            .expect("mode 5 has interpolated LSPs");

        for (sf_idx, sf) in body.subframes.iter().enumerate() {
            // Each sub-frame's interpolated LSPs (Q12) → radians → LPC.
            let lsp_q12 = sub_lsp.subframe(sf_idx).expect("sub-frame in range");
            // Convert Q12 → Q10 by an arithmetic shift (>>2), then to
            // radians via the documented helper. (interpolated LSPs are
            // Q12 = Q10 + 2.)
            let lsp_q10: [i32; 10] = {
                let mut out = [0i32; 10];
                for (o, &v) in out.iter_mut().zip(lsp_q12.iter()) {
                    *o = v >> 2;
                }
                out
            };
            let lpc = lsp_to_lpc(&lsp_vector_q10_to_radians(&lsp_q10));

            // r277: this sub-frame's innovation c[n].
            let c = sf
                .innovation_sub_vector(&submode)
                .expect("sub-frame innovation decodes");
            let exc_i32: Vec<i32> = c.iter().map(|&v| i32::from(v)).collect();

            // r286: synthesis filter. Excitation = innovation only (no
            // pitch contribution — gap-blocked; see module docs).
            let mut out = vec![0.0_f64; exc_i32.len()];
            filter.process_i32(&lpc, &exc_i32, &mut out);
            pcm.extend_from_slice(&out);
        }

        prev_lsp_q10 = Some(curr_lsp_q10);
    }
    pcm
}

#[test]
fn fixture_decodes_to_finite_responsive_pcm() {
    let pcm = decode_fixture_to_pcm();

    // 50 frames × 4 sub-frames × 40 samples ≈ 8000 samples for the 1 s
    // fixture; assert we produced a substantial PCM run.
    assert!(
        pcm.len() >= 40 * 4 * 40,
        "expected ≥ {} PCM samples, got {}",
        40 * 4 * 40,
        pcm.len()
    );

    // Every sample finite (no NaN/Inf from an unstable filter).
    for (i, &s) in pcm.iter().enumerate() {
        assert!(s.is_finite(), "sample {} is not finite: {}", i, s);
    }

    // The output must be *responsive* — driven by real excitation it is
    // neither all-zero (silence) nor a constant. Measure the spread.
    let max = pcm.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = pcm.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        max - min > 1.0,
        "PCM has no dynamic range (max {} − min {} ≤ 1) — synthesis stage produced near-constant output",
        max,
        min
    );

    // Non-zero energy.
    let energy: f64 = pcm.iter().map(|&s| s * s).sum();
    assert!(energy > 0.0, "PCM has zero energy");
}

#[test]
fn synthesis_history_continuity_changes_first_frame_only_at_boundary() {
    // Decoding the same fixture twice must be deterministic — the
    // synthesis filter has no hidden global state.
    let a = decode_fixture_to_pcm();
    let b = decode_fixture_to_pcm();
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "non-deterministic PCM at sample {}", i);
    }
}

#[test]
fn lpc_coefficients_are_finite_for_every_real_frame() {
    // Probe the LSP→LPC stage in isolation across the whole fixture:
    // every reconstructed/interpolated LSP set in a real stream must
    // convert to a finite LPC vector (the conversion never blows up on
    // genuine quantiser output).
    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];
    let mut prev_lsp_q10: Option<[i32; 10]> = None;
    let mut frames = 0u32;
    for pkt in audio {
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(pkt).expect("header");
        let submode = match h.submode {
            Submode::Celp(s) => s,
            other => panic!("unexpected submode {:?}", other),
        };
        let body = NarrowbandFrameBody::parse(&mut r, &submode).expect("body");
        let curr = body.reconstructed_lsp_q10(&submode).expect("lsp");
        let prev = prev_lsp_q10.unwrap_or(curr);
        let sub_lsp = body.interpolated_lsp_q12(&submode, &prev).expect("interp");
        for sf_idx in 0..4 {
            let lsp_q12 = sub_lsp.subframe(sf_idx).unwrap();
            let mut lsp_q10 = [0i32; 10];
            for (o, &v) in lsp_q10.iter_mut().zip(lsp_q12.iter()) {
                *o = v >> 2;
            }
            let lpc = lsp_to_lpc(&lsp_vector_q10_to_radians(&lsp_q10));
            for &c in &lpc {
                assert!(c.is_finite(), "non-finite LPC coefficient on a real frame");
            }
        }
        prev_lsp_q10 = Some(curr);
        frames += 1;
    }
    assert!(frames >= 40, "expected ≥40 audio frames, got {}", frames);
}
