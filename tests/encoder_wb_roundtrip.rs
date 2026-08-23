//! End-to-end wideband encode → decode round-trip (round r385).
//!
//! Drives the [`WidebandEncoder`] output straight into the
//! [`WidebandDecoder`] — the full §10 sub-band CELP cycle: QMF analysis
//! split → embedded narrowband encode + high-band envelope/gain/
//! innovation encode → §10.4 packing → wideband decode (NB loop + HB
//! synthesis + QMF recombination to 16 kHz PCM). Asserts the encoded
//! stream is decodable, finite, deterministic, and input-tracking, and
//! that the high band actually carries energy when the input has 4–8 kHz
//! content.

use oxideav_speex::{WidebandDecoder, WidebandEncoder, QMF_WIDEBAND_FRAME};

/// A 16 kHz frame with energy in both bands: a voiced-ish 300 Hz
/// fundamental plus a 6 kHz high-band tone.
fn dual_band_frame(amp: f64, frame_idx: usize) -> [i16; QMF_WIDEBAND_FRAME] {
    let mut f = [0i16; QMF_WIDEBAND_FRAME];
    for (n, s) in f.iter_mut().enumerate() {
        let t = (frame_idx * QMF_WIDEBAND_FRAME + n) as f64;
        let low = (2.0 * std::f64::consts::PI * t * 300.0 / 16_000.0).sin();
        let high = 0.6 * (2.0 * std::f64::consts::PI * t * 6_000.0 / 16_000.0).sin();
        let v = amp * (low + high);
        *s = v.round().clamp(-32768.0, 32767.0) as i16;
    }
    f
}

/// A pure high-band (6.5 kHz) 16 kHz frame — no low-band content.
fn high_only_frame(amp: f64, frame_idx: usize) -> [i16; QMF_WIDEBAND_FRAME] {
    let mut f = [0i16; QMF_WIDEBAND_FRAME];
    for (n, s) in f.iter_mut().enumerate() {
        let t = (frame_idx * QMF_WIDEBAND_FRAME + n) as f64;
        let v = amp * (2.0 * std::f64::consts::PI * t * 6_500.0 / 16_000.0).sin();
        *s = v.round().clamp(-32768.0, 32767.0) as i16;
    }
    f
}

fn energy(pcm: &[f64]) -> f64 {
    pcm.iter().map(|&s| s * s).sum()
}

#[test]
fn encode_decode_round_trip_all_documented_hb_modes() {
    // Every supported HB mode (0..=3) over an NB mode-3 low band must
    // produce a stream the wideband decoder accepts end-to-end, with
    // finite 16 kHz PCM.
    for hb_mode in [0u8, 1, 2, 3] {
        let mut enc = WidebandEncoder::new();
        let mut dec = WidebandDecoder::new();
        for i in 0..4 {
            let frame = dual_band_frame(6000.0, i);
            let bytes = enc
                .encode_frame(&frame, 3, hb_mode)
                .unwrap_or_else(|e| panic!("hb mode {hb_mode} frame {i}: encode: {e}"));
            let out = dec
                .decode_packet(&bytes)
                .unwrap_or_else(|e| panic!("hb mode {hb_mode} frame {i}: decode: {e}"));
            assert!(
                out.wideband_pcm.iter().all(|s| s.is_finite()),
                "hb mode {hb_mode} frame {i}: non-finite PCM"
            );
        }
    }
}

#[test]
fn decoded_output_tracks_input_energy() {
    // A loud dual-band input must decode to substantially more energy
    // than a quiet one (order-of-magnitude tracking, not bit-exactness).
    let run = |amp: f64| -> f64 {
        let mut enc = WidebandEncoder::new();
        let mut dec = WidebandDecoder::new();
        let mut total = 0.0f64;
        for i in 0..4 {
            let frame = dual_band_frame(amp, i);
            let bytes = enc.encode_frame(&frame, 3, 3).expect("encode");
            let out = dec.decode_packet(&bytes).expect("decode");
            // Skip the first frame (filterbank + envelope transients).
            if i > 0 {
                total += energy(&out.wideband_pcm);
            }
        }
        total
    };
    let loud = run(8000.0);
    let quiet = run(80.0);
    assert!(
        loud > 20.0 * quiet,
        "loud energy {loud} should dominate quiet energy {quiet}"
    );
}

#[test]
fn high_band_content_reaches_the_high_band_channel() {
    // A pure 6.5 kHz input folds into the high-band half-band signal;
    // the decoded high-band channel must carry (much) more energy than
    // the decoded low band once the innovation modes are active.
    let mut enc = WidebandEncoder::new();
    let mut dec = WidebandDecoder::new();
    let mut hb_energy = 0.0f64;
    let mut lb_energy = 0.0f64;
    for i in 0..4 {
        let frame = high_only_frame(8000.0, i);
        let bytes = enc.encode_frame(&frame, 3, 3).expect("encode");
        let out = dec.decode_packet(&bytes).expect("decode");
        if i > 0 {
            hb_energy += energy(&out.high_band);
            lb_energy += energy(&out.low_band);
        }
    }
    assert!(
        hb_energy > lb_energy,
        "high-band energy {hb_energy} should exceed low-band leakage {lb_energy}"
    );
    assert!(hb_energy > 0.0, "high band must be non-silent");
}

#[test]
fn silence_encodes_and_decodes_to_near_silence() {
    let mut enc = WidebandEncoder::new();
    let mut dec = WidebandDecoder::new();
    let frame = [0i16; QMF_WIDEBAND_FRAME];
    let mut total = 0.0f64;
    for _ in 0..3 {
        let bytes = enc.encode_frame(&frame, 3, 2).expect("encode");
        let out = dec.decode_packet(&bytes).expect("decode");
        total += energy(&out.wideband_pcm);
    }
    // The gain quantiser's lowest level is small but non-zero; the
    // decoded output must stay tiny relative to a real signal's scale.
    assert!(
        total < 1.0e4,
        "silent input decoded to energy {total} (should be near-silent)"
    );
}

#[test]
fn round_trip_is_deterministic() {
    let run = || -> Vec<Vec<u8>> {
        let mut enc = WidebandEncoder::new();
        (0..3)
            .map(|i| {
                let frame = dual_band_frame(5000.0, i);
                enc.encode_frame(&frame, 3, 2).expect("encode")
            })
            .collect()
    };
    assert_eq!(run(), run(), "encoder must be deterministic");
}

#[test]
fn hb_mode_1_gain_only_stream_decodes() {
    // Mode 1 conveys only the high-band envelope + gain; the decoded
    // high band is silent (no innovation) but the stream must decode
    // cleanly and the low band must be live.
    let mut enc = WidebandEncoder::new();
    let mut dec = WidebandDecoder::new();
    let mut lb_energy = 0.0f64;
    for i in 0..3 {
        let frame = dual_band_frame(6000.0, i);
        let bytes = enc.encode_frame(&frame, 3, 1).expect("encode");
        let out = dec.decode_packet(&bytes).expect("decode");
        lb_energy += energy(&out.low_band);
    }
    assert!(lb_energy > 0.0, "low band must be non-silent");
}

#[test]
fn hb_mode_4_round_trips_with_live_high_band() {
    // r450: mode 4 encodes (two-stage sv8-128 search + the
    // crossover-anchored gain law) and its decode reconstructs a
    // non-silent high band from a dual-band source.
    let mut enc = WidebandEncoder::new();
    let mut dec = WidebandDecoder::new();
    let mut hb_energy = 0.0f64;
    for k in 0..6usize {
        let frame = dual_band_frame(4000.0, k);
        let bytes = enc.encode_frame(&frame, 3, 4).expect("mode 4 encodes");
        let out = dec.decode_packet(&bytes).expect("mode 4 decodes");
        hb_energy += energy(&out.high_band);
    }
    assert!(
        hb_energy.is_finite() && hb_energy > 0.0,
        "mode-4 high band must be live"
    );
}
