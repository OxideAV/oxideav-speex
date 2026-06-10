//! Integration test: parse every audio packet of a real
//! `speexenc`-encoded narrowband file end-to-end through round-3's
//! frame-body bit-reader.
//!
//! The fixture is generated via the black-box `speexenc` binary —
//! `tests/fixtures/Makefile`'s `regen` target is the audit trail.
//! `speexenc`'s **source** is not consulted; it is invoked as an
//! opaque process and its output bytes are the test input.
//!
//! Per workspace policy this crate does NOT take a dev-dependency on
//! `oxideav-ogg`. Instead, the test contains a minimal inline Ogg
//! page-walker (≈30 lines) that pulls the Speex audio packets back
//! out of their Ogg framing — just enough to feed the codec under
//! test. No Ogg encoding, no chained-stream support, no granulepos
//! handling, no metadata; only what's needed to lift packets out of a
//! single-stream `.spx` file produced by `speexenc`.
//!
//! What is asserted:
//!
//! * The first audio packet's leading 5-bit header parses as
//!   `wideband=0, mode=5` (`quality 8` per Speex Manual Table 9.2
//!   maps to narrowband sub-mode 5 — 15 kbps, 300 bits per 20 ms
//!   frame).
//! * For every audio packet, [`NarrowbandFrameBody::parse`] consumes
//!   exactly `total_bits - 5` further bits without underflow.
//! * The cumulative cursor position lands within the packet's byte
//!   span (with at most 7 bits of padding leftover, since the encoder
//!   rounds up to a whole-byte packet — per §5.5 the encoder
//!   appends a mode-15 terminator and/or trailing zeros to pad).

use oxideav_speex::{
    NarrowbandFrameBody, NarrowbandFrameHeader, NbLspStages, Submode, NARROWBAND_FRAME_PREFIX_BITS,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/nb_440hz_q8.spx");

/// Inline minimal Ogg page-walker. Iterates packets reassembled from
/// per-page segment tables; returns the packet bodies as owned `Vec`s.
///
/// This is NOT a general-purpose Ogg implementation — it deliberately
/// ignores chained streams, the BOS/EOS flags, the granule position,
/// the bitstream serial number, the CRC, and the page sequence
/// number. It does just one thing: stitch lacing-segment runs back
/// into Speex packets.
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

#[test]
fn audio_packets_round_trip_through_narrowband_body_parser() {
    let packets = lift_ogg_packets(FIXTURE);
    // Speex stream packets: [0] = Speex header, [1] = comment header,
    // [2..] = audio. The fixture is 1 s of 8 kHz mono so the encoder
    // emits 50 frames + (possibly) a small tail; assert there's at
    // least 40 audio packets to make the test robust to encoder-side
    // packing tweaks.
    assert!(
        packets.len() >= 42,
        "expected at least 42 Ogg packets (1 header + 1 comment + ≥40 audio), got {}",
        packets.len()
    );

    let audio = &packets[2..];
    assert!(!audio.is_empty(), "no audio packets in fixture");

    // First audio packet: 5-bit prefix → wideband=0, mode=5.
    let (h0, _) = NarrowbandFrameHeader::parse_bytes(&audio[0])
        .expect("first audio packet must parse a frame header");
    assert!(!h0.wideband, "fixture is pure narrowband");
    assert_eq!(h0.mode_id, 5, "speexenc quality 8 → narrowband mode 5");

    // Every audio packet: header + body must walk without underflow,
    // and the cursor must end inside the packet (mode 5 = 300 bits ≤
    // 38 bytes = 304 bits, leaving up to 4 padding bits).
    let mut parsed = 0u32;
    for (i, pkt) in audio.iter().enumerate() {
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(pkt)
            .unwrap_or_else(|e| panic!("packet {}: header parse failed: {:?}", i, e));
        let s = match h.submode {
            Submode::Celp(s) => s,
            other => panic!("packet {}: unexpected submode {:?}", i, other),
        };
        let body = NarrowbandFrameBody::parse(&mut r, &s)
            .unwrap_or_else(|e| panic!("packet {}: body parse failed: {:?}", i, e));

        // Field-width sanity for mode 5.
        assert!(body.lsp_index < (1u32 << 30));
        assert!(body.ol_exc_gain_index < (1u8 << 5));
        for (sf_idx, sf) in body.subframes.iter().enumerate() {
            assert!(
                sf.pitch_index < (1u8 << 7),
                "packet {} subframe {}: pitch_index out of range",
                i,
                sf_idx
            );
            assert!(
                sf.pitch_gain_index < (1u8 << 7),
                "packet {} subframe {}: pitch_gain_index out of range",
                i,
                sf_idx
            );
            assert!(
                sf.innovation_gain_index < (1u8 << 3),
                "packet {} subframe {}: innovation_gain_index out of range",
                i,
                sf_idx
            );
            assert!(
                sf.innovation_vq_index < (1u128 << 48),
                "packet {} subframe {}: innovation_vq_index out of range",
                i,
                sf_idx
            );
        }

        // Cursor must land within the packet (with <= 7 bits trailing
        // padding for the byte-align tail).
        let consumed = r.consumed_bits();
        let expected = u32::from(s.total_bits);
        assert_eq!(
            consumed, expected,
            "packet {}: cursor expected at {} bits, got {}",
            i, expected, consumed
        );
        let pkt_bits = (pkt.len() as u32) * 8;
        assert!(
            consumed <= pkt_bits,
            "packet {}: parse over-ran packet ({} > {} bits)",
            i,
            consumed,
            pkt_bits
        );
        assert!(
            pkt_bits - consumed < 8,
            "packet {}: more than one padding byte left ({} bits)",
            i,
            pkt_bits - consumed
        );
        parsed += 1;
    }
    assert_eq!(
        parsed as usize,
        audio.len(),
        "must successfully parse every audio packet"
    );
}

#[test]
fn lsp_index_splits_and_reconstructs_for_every_audio_packet() {
    // Round r194 wiring probe: every real mode-5 (30-bit LSP) frame in
    // the fixture must produce a non-degenerate per-stage split + a
    // ten-coefficient Q10 LSP reconstruction. We don't assert specific
    // PCM-bit-exact values (that requires the LSP→LPC + synthesis path
    // which lands in later rounds), only structural properties that
    // hold for any well-formed LSP frame:
    //
    //   * The packed `lsp_index` splits into five per-stage 6-bit
    //     indices, each in 0..64.
    //   * The reconstruction returns ten coefficients (no panic, no
    //     None for an in-range 30-bit field).
    //   * Across many real frames the reconstructed coefficients are
    //     not all-zero in aggregate — i.e. the staged codebooks are
    //     contributing actual signal, not silently returning zeros.
    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];
    let mut nonzero_frames = 0u32;
    let mut total_frames = 0u32;
    for (i, pkt) in audio.iter().enumerate() {
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(pkt).expect("header parse");
        let s = match h.submode {
            Submode::Celp(s) => s,
            _ => panic!("packet {}: expected CELP", i),
        };
        let body = NarrowbandFrameBody::parse(&mut r, &s).expect("body parse");

        let stages = body.lsp_stages(&s).expect("mode 5 has 30-bit LSP");
        assert!(stages.stage0 < 64, "packet {} stage0 out of range", i);
        assert!(stages.low1 < 64, "packet {} low1 out of range", i);
        assert!(stages.high1 < 64, "packet {} high1 out of range", i);
        // 30-bit regime → low2 + high2 must both be Some.
        let low2 = stages.low2.expect("30-bit LSP must carry low2");
        let high2 = stages.high2.expect("30-bit LSP must carry high2");
        assert!(low2 < 64, "packet {} low2 out of range", i);
        assert!(high2 < 64, "packet {} high2 out of range", i);

        let coeffs = body
            .reconstructed_lsp_q10(&s)
            .expect("30-bit LSP must reconstruct");
        assert_eq!(coeffs.len(), 10);
        if coeffs.iter().any(|&c| c != 0) {
            nonzero_frames += 1;
        }
        total_frames += 1;
    }
    assert!(
        total_frames >= 40,
        "fixture should contain ≥40 audio frames"
    );
    assert!(
        nonzero_frames * 10 > total_frames * 9,
        "expected ≥90% of frames to reconstruct non-zero LSPs, got {}/{}",
        nonzero_frames,
        total_frames
    );
}

#[test]
fn lsp_stages_are_none_for_silence_mode() {
    // Direct construction of a silence-mode body: confirm
    // `lsp_stages` propagates the `LspQuant::None` invariant.
    use oxideav_speex::{LspQuant, NarrowbandSubmode};
    let silence = NarrowbandSubmode::for_id(0).unwrap();
    assert_eq!(silence.lsp, LspQuant::None);
    // A minimum-byte mode-0 body produces zero indices.
    let buf = [0u8; 1];
    let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
    let s = match h.submode {
        Submode::Celp(s) => s,
        _ => unreachable!(),
    };
    let body = NarrowbandFrameBody::parse(&mut r, &s).unwrap();
    assert!(body.lsp_stages(&s).is_none());
    assert!(body.reconstructed_lsp_q10(&s).is_none());
}

#[test]
fn lsp_split_round_trip_for_18bit_and_30bit_modes() {
    // Spot-check the splitter API at both LSP widths. These are
    // construction-time checks (no fixture needed); they're in the
    // integration-test file because they exercise the publicly
    // re-exported `NbLspStages::from_packed`.
    use oxideav_speex::LspQuant;
    let s18 = NbLspStages::from_packed((7 << 12) | (11 << 6) | 23, LspQuant::Bits18).unwrap();
    assert_eq!(s18.stage0, 7);
    assert_eq!(s18.low1, 11);
    assert_eq!(s18.high1, 23);
    assert!(s18.low2.is_none());
    assert!(s18.high2.is_none());

    let s30 = NbLspStages::from_packed(
        (1 << 24) | (2 << 18) | (3 << 12) | (4 << 6) | 5,
        LspQuant::Bits30,
    )
    .unwrap();
    assert_eq!(s30.stage0, 1);
    assert_eq!(s30.low1, 2);
    assert_eq!(s30.low2, Some(3));
    assert_eq!(s30.high1, 4);
    assert_eq!(s30.high2, Some(5));
}

#[test]
fn first_audio_packet_uses_full_5_bit_prefix() {
    let packets = lift_ogg_packets(FIXTURE);
    let pkt0 = &packets[2];
    // First byte top 5 bits = wideband(1) | mode_id(4); the bottom 3
    // bits start the LSP VQ field for mode 5 (30 bits of LSP). They
    // can be anything, but the prefix-consumed-bit-count must equal
    // NARROWBAND_FRAME_PREFIX_BITS regardless.
    let (_, r) = NarrowbandFrameHeader::parse_bytes(pkt0).unwrap();
    assert_eq!(r.consumed_bits(), NARROWBAND_FRAME_PREFIX_BITS);
}

#[test]
fn sub_frame_lsp_interpolation_walks_real_audio_stream() {
    // Round r200 wiring probe: walk the entire fixture frame-by-frame,
    // threading the previous frame's reconstructed Q10 LSPs into the
    // current frame's interpolation, and assert structural properties
    // that hold for any well-formed CELP LSP stream:
    //
    //   * Each frame produces exactly 4 sub-frame vectors of 10
    //     coefficients each.
    //   * Sub-frame 4 (s=3) equals 4·curr in Q12 for every frame —
    //     i.e. §9.1's "associated to the 4th sub-frame" property holds
    //     end-to-end.
    //   * The first frame's interpolation envelope is flat (all four
    //     sub-frames equal 4·curr in Q12) — `first_frame` produces no
    //     transient.
    //   * After many frames the interpolation envelope is non-flat at
    //     least some of the time, confirming the previous-frame state
    //     is actually being used and not silently zeroed.
    use oxideav_speex::{NbSubFrameLsp, NB_LSP_INTERP_OUTPUT_Q, NB_LSP_SUBFRAMES_PER_FRAME};

    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];

    let mut prev_lsp_q10: Option<[i32; 10]> = None;
    let mut frames_walked = 0u32;
    let mut frames_with_envelope_change = 0u32;

    for (i, pkt) in audio.iter().enumerate() {
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(pkt).expect("header parse");
        let s = match h.submode {
            Submode::Celp(s) => s,
            _ => panic!("packet {}: expected CELP", i),
        };
        let body = NarrowbandFrameBody::parse(&mut r, &s).expect("body parse");
        let curr = body
            .reconstructed_lsp_q10(&s)
            .expect("mode 5 must reconstruct LSPs");

        // Build per-sub-frame interpolation — first-frame init if no
        // previous LSP state available, steady-state otherwise.
        let interp = match prev_lsp_q10 {
            None => NbSubFrameLsp::first_frame(&curr),
            Some(ref prev) => body
                .interpolated_lsp_q12(&s, prev)
                .expect("mode 5 must interpolate"),
        };

        // Sub-frame 4 (s=3) must equal 4·curr in Q12 by §9.1.
        for (k, &c) in curr.iter().enumerate() {
            assert_eq!(
                interp.subframes[3][k],
                4 * c,
                "frame {i} sub-frame 4 coeff {k} != 4·curr",
            );
        }
        // Output dimensions.
        assert_eq!(interp.subframes.len(), NB_LSP_SUBFRAMES_PER_FRAME);

        if prev_lsp_q10.is_none() {
            // First-frame: every sub-frame must equal 4·curr (flat envelope).
            for sf in &interp.subframes {
                for (k, &c) in curr.iter().enumerate() {
                    assert_eq!(sf[k], 4 * c, "first frame envelope not flat");
                }
            }
        }

        // Track envelope-change frames (steady-state only).
        if prev_lsp_q10.is_some() {
            let flat = (0..NB_LSP_SUBFRAMES_PER_FRAME)
                .all(|si| (0..10).all(|k| interp.subframes[si][k] == 4 * curr[k]));
            if !flat {
                frames_with_envelope_change += 1;
            }
        }

        prev_lsp_q10 = Some(curr);
        frames_walked += 1;
    }

    assert!(frames_walked >= 40, "fixture should have ≥40 frames");
    // After many frames at least *some* must show a non-flat envelope —
    // a 440 Hz tone fixture is mostly stationary but each frame's LSP
    // VQ index differs at least slightly from the previous frame's.
    assert!(
        frames_with_envelope_change > 0,
        "expected at least one steady-state frame to have a non-flat interpolation envelope; got 0 of {} steady-state frames",
        frames_walked.saturating_sub(1),
    );

    // Lock the Q-format contract (cheap sanity check against the
    // crate-public constant, run inside the integration harness so a
    // future Q-format change can't slip past a re-export rename).
    assert_eq!(NB_LSP_INTERP_OUTPUT_Q, 12);
}

#[test]
fn first_frame_initialisation_matches_steady_state_when_prev_equals_current() {
    // The `first_frame` constructor must produce the same result as
    // `new(prev, curr)` when `prev == curr` — that's its definition,
    // and the fixture round-trip relies on this equivalence for the
    // stream-start case.
    use oxideav_speex::NbSubFrameLsp;
    let packets = lift_ogg_packets(FIXTURE);
    let pkt0 = &packets[2];
    let (h, mut r) = NarrowbandFrameHeader::parse_bytes(pkt0).unwrap();
    let s = match h.submode {
        Submode::Celp(s) => s,
        _ => unreachable!(),
    };
    let body = NarrowbandFrameBody::parse(&mut r, &s).unwrap();
    let curr = body.reconstructed_lsp_q10(&s).unwrap();
    let via_first = NbSubFrameLsp::first_frame(&curr);
    let via_new = NbSubFrameLsp::new(&curr, &curr);
    assert_eq!(via_first, via_new);
}

#[test]
fn interpolated_lsp_q12_returns_none_for_silence_mode() {
    // Silence sub-mode (mode 0) carries no LSP field; the interpolator
    // wrapper on `NarrowbandFrameBody` must propagate `None` so the
    // caller knows to fall back to its own LSP state.
    use oxideav_speex::NarrowbandSubmode;
    let silence = NarrowbandSubmode::for_id(0).unwrap();
    let buf = [0u8; 1];
    let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
    let s = match h.submode {
        Submode::Celp(s) => s,
        _ => unreachable!(),
    };
    let body = NarrowbandFrameBody::parse(&mut r, &s).unwrap();
    let prev = [0i32; 10];
    assert!(body.interpolated_lsp_q12(&silence, &prev).is_none());
}

#[test]
fn pitch_gain_taps_resolve_for_every_audio_subframe() {
    // Round r208 wiring probe: walk every audio frame in the fixture
    // and assert every sub-frame's 7-bit pitch-gain VQ index resolves
    // through the new `pitch_gain` module without panic.
    //
    // Structural properties:
    //   * Every sub-frame yields three β tap coefficients (i16).
    //   * Each tap lands in the documented post-bias signed-byte range
    //     -96..=159 (signed-byte +32 bias).
    //   * Across many frames at least *one* sub-frame produces a
    //     non-zero tap — i.e. the codebook is contributing real
    //     β coefficients, not a silent all-zero stream.
    //
    // Mode 5 (the fixture's sub-mode) uses the 7-bit pitch-gain VQ
    // per Table 9.1, so every resolution hits `pitch_gain_7bit()`.
    use oxideav_speex::PITCH_GAIN_TAPS;
    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];
    let mut total_subframes = 0u32;
    let mut nonzero_subframes = 0u32;
    for (i, pkt) in audio.iter().enumerate() {
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(pkt).expect("header parse");
        let s = match h.submode {
            Submode::Celp(s) => s,
            _ => panic!("packet {}: expected CELP", i),
        };
        let body = NarrowbandFrameBody::parse(&mut r, &s).expect("body parse");
        for (sf_idx, sf) in body.subframes.iter().enumerate() {
            let taps = sf.pitch_gain_taps(&s).unwrap_or_else(|| {
                panic!(
                    "packet {} sub-frame {}: pitch_gain_taps returned None",
                    i, sf_idx
                )
            });
            assert_eq!(taps.taps.len(), PITCH_GAIN_TAPS);
            for (t_idx, &t) in taps.taps.iter().enumerate() {
                assert!(
                    (-96..=159).contains(&t),
                    "packet {} sub-frame {} tap {}: value {} out of documented +32-bias band",
                    i,
                    sf_idx,
                    t_idx,
                    t
                );
            }
            if taps.taps.iter().any(|&t| t != 0) {
                nonzero_subframes += 1;
            }
            total_subframes += 1;
        }
    }
    assert!(
        total_subframes >= 4 * 40,
        "fixture should have ≥40 frames × 4 sub-frames"
    );
    assert!(
        nonzero_subframes > 0,
        "expected at least one sub-frame to produce non-zero β taps; got 0 of {} sub-frames",
        total_subframes
    );
}

#[test]
fn pitch_gain_taps_is_silence_for_mode_0() {
    // Silence sub-mode (mode 0, `PitchGainQuant::None`) must surface
    // `PitchGainTaps::SILENCE` (all-zero β taps) regardless of the
    // defaulted `pitch_gain_index`.
    use oxideav_speex::{NarrowbandSubmode, PitchGainTaps};
    let silence = NarrowbandSubmode::for_id(0).unwrap();
    let buf = [0u8; 1];
    let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
    let s = match h.submode {
        Submode::Celp(s) => s,
        _ => unreachable!(),
    };
    let body = NarrowbandFrameBody::parse(&mut r, &s).unwrap();
    for sf in &body.subframes {
        let taps = sf.pitch_gain_taps(&silence).unwrap();
        assert_eq!(taps, PitchGainTaps::SILENCE);
        assert_eq!(taps.taps, [0, 0, 0]);
    }
}

#[test]
fn innovation_subvector_decodes_for_mode_5_fixture() {
    // r277: the r220-era `Undocumented` pin for mode 5 is retired — the
    // staged per-codebook innovation bit-rate annotations
    // (`docs/audio/speex/tables/*.meta` + tables README) bind mode 5's
    // 48-bit/sub-frame field (× 200 = 9600 bps) to 8 × `Sv5_64`
    // (6-bit index, 5-sample sub-vector). Every sub-frame of every
    // audio packet of the real `speexenc`-encoded mode-5 fixture must
    // now decode into a 40-sample c[n] vector, and the per-sub-vector
    // lookups must match a manual MSB-first walk of the raw 48-bit
    // `innovation_vq_index` field against the staged codebook.
    use oxideav_speex::{innovation_5_64, SUBFRAME_SAMPLES};
    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];
    let mut total = 0u32;
    let mut nonzero_subframes = 0u32;
    for (i, pkt) in audio.iter().enumerate() {
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(pkt).expect("header");
        let s = match h.submode {
            Submode::Celp(s) => s,
            _ => panic!("packet {i}: expected CELP"),
        };
        assert_eq!(s.mode_id, 5);
        let body = NarrowbandFrameBody::parse(&mut r, &s).expect("body");
        for (sf_idx, sf) in body.subframes.iter().enumerate() {
            let c = sf
                .innovation_sub_vector(&s)
                .unwrap_or_else(|e| panic!("packet {i} sf {sf_idx}: {e}"));
            assert_eq!(c.len(), SUBFRAME_SAMPLES);
            // Cross-check against a manual MSB-first walk of the raw
            // packed field: 8 successive 6-bit indices into Sv5_64.
            for sv in 0..8usize {
                let shift = (7 - sv) as u32 * 6;
                let idx = ((sf.innovation_vq_index >> shift) & 0x3f) as usize;
                let row = &innovation_5_64()[idx];
                assert_eq!(
                    &c[sv * 5..sv * 5 + 5],
                    &row[..],
                    "packet {i} sf {sf_idx} sub-vector {sv}"
                );
            }
            if c.iter().any(|&x| x != 0) {
                nonzero_subframes += 1;
            }
            total += 1;
        }
    }
    assert!(
        total >= 4 * 40,
        "fixture must have ≥40 frames × 4 sub-frames"
    );
    assert!(
        nonzero_subframes > 0,
        "expected at least one sub-frame with non-zero innovation; got 0 of {total}"
    );
}

#[test]
fn innovation_subvector_for_silence_mode_is_all_zero() {
    // Mode 0 (silence) carries no innovation field; the dispatcher
    // must surface the all-zero 40-sample c[n] vector regardless of
    // the (defaulted-to-zero) `innovation_vq_index`.
    use oxideav_speex::{NarrowbandSubmode, SUBFRAME_SAMPLES};
    let silence = NarrowbandSubmode::for_id(0).unwrap();
    let buf = [0u8; 1];
    let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
    let s = match h.submode {
        Submode::Celp(s) => s,
        _ => unreachable!(),
    };
    let body = NarrowbandFrameBody::parse(&mut r, &s).unwrap();
    assert_eq!(silence.mode_id, 0);
    for sf in &body.subframes {
        let v = sf.innovation_sub_vector(&silence).unwrap();
        assert_eq!(v.len(), SUBFRAME_SAMPLES);
        assert!(v.iter().all(|&x| x == 0));
    }
}

#[test]
fn fixed_codebook_gain_indices_compose_for_every_audio_packet() {
    // Round r261 wiring probe: every audio packet of the mode-5
    // narrowband fixture composes the typed §9.2 / CELP companion §2.3
    // fixed-codebook gain index pair without panic.
    //
    // Mode 5 budgets (Table 9.1): 5-bit frame OL Exc gain × 3-bit
    // per-sub-frame innovation gain. Both factors must surface as the
    // "present" enum variants for every sub-frame of every audio packet.
    use oxideav_speex::{FrameInnovationGainIndex, SubFrameInnovationGainCorrection};
    let packets = lift_ogg_packets(FIXTURE);
    let audio = &packets[2..];
    let mut total_subframes = 0u32;
    let mut frame_index_range = (u8::MAX, 0u8);
    let mut nonzero_correction = 0u32;
    for (i, pkt) in audio.iter().enumerate() {
        let (h, mut r) = NarrowbandFrameHeader::parse_bytes(pkt).expect("header parse");
        let s = match h.submode {
            Submode::Celp(s) => s,
            _ => panic!("packet {i}: expected CELP"),
        };
        assert_eq!(s.mode_id, 5);
        assert_eq!(s.ol_exc_gain_bits, 5);
        assert_eq!(s.innovation_gain_bits, 3);
        let body = NarrowbandFrameBody::parse(&mut r, &s).expect("body parse");
        let composed = body
            .fixed_codebook_gain_indices(&s)
            .expect("mode-5 budgets in spec");
        for (sf_idx, slot) in composed.iter().enumerate() {
            // Frame factor is always present in mode 5 (5-bit field).
            let frame_idx = match slot.frame {
                FrameInnovationGainIndex::Indexed(i) => i,
                FrameInnovationGainIndex::Silence => {
                    panic!("packet {i} sf {sf_idx}: mode 5 frame factor must be present")
                }
            };
            assert!(frame_idx < 32, "5-bit OL Exc gain index must fit in 0..=31");
            frame_index_range.0 = frame_index_range.0.min(frame_idx);
            frame_index_range.1 = frame_index_range.1.max(frame_idx);
            // Per-sub-frame correction must be 3-bit for mode 5.
            let sub_idx = match slot.subframe {
                SubFrameInnovationGainCorrection::ThreeBit(i) => i,
                other => panic!("packet {i} sf {sf_idx}: expected ThreeBit, got {:?}", other),
            };
            assert!(sub_idx < 8, "3-bit Innovation gain index must fit in 0..=7");
            if sub_idx != 0 {
                nonzero_correction += 1;
            }
            assert_eq!(slot.wire_bit_budget(), 5 + 3);
            assert!(!slot.is_absent());
            total_subframes += 1;
        }
    }
    assert!(
        total_subframes >= 4 * 40,
        "fixture must have ≥40 frames × 4 sub-frames"
    );
    // The recorded fixture is a 440 Hz tone; the encoder should pick a
    // non-trivial 5-bit OL Exc gain index across the stream and at least
    // one non-zero 3-bit correction.
    assert!(
        frame_index_range.0 != frame_index_range.1,
        "expected the frame-level OL Exc gain index to vary across packets ({} = {} everywhere)",
        frame_index_range.0,
        frame_index_range.1,
    );
    assert!(
        nonzero_correction > 0,
        "expected at least one non-zero 3-bit innovation-gain correction across {} sub-frames",
        total_subframes
    );
}

#[test]
fn fixed_codebook_gain_indices_for_silence_mode_is_absent() {
    // Mode 0 (silence) carries neither OL Exc gain nor innovation
    // gain — every per-sub-frame composed pair must flag `is_absent`.
    use oxideav_speex::{
        FrameInnovationGainIndex, NarrowbandSubmode, SubFrameInnovationGainCorrection,
    };
    let silence = NarrowbandSubmode::for_id(0).unwrap();
    let buf = [0u8; 1];
    let (h, mut r) = NarrowbandFrameHeader::parse_bytes(&buf).unwrap();
    let s = match h.submode {
        Submode::Celp(s) => s,
        _ => unreachable!(),
    };
    let body = NarrowbandFrameBody::parse(&mut r, &s).unwrap();
    assert_eq!(silence.mode_id, 0);
    let composed = body
        .fixed_codebook_gain_indices(&silence)
        .expect("silence budgets {0,0} in spec");
    for slot in &composed {
        assert!(slot.is_absent());
        assert_eq!(slot.frame, FrameInnovationGainIndex::Silence);
        assert_eq!(slot.subframe, SubFrameInnovationGainCorrection::Absent);
        assert_eq!(slot.wire_bit_budget(), 0);
    }
}
