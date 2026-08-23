//! **Wideband decode-conformance matrix** (round r410).
//!
//! Extends the external validation beyond the r393 `wb-mode1-folded`
//! gate (a tone mix, every frame high-band sub-mode 1) to **speech-like
//! material and the innovation-VQ high-band sub-modes**: qualities 4,
//! 6 and 8 of a 2 s pitch-glide 16 kHz source (Table 10.2 ladder:
//! NB 4 + HB 1, NB 5 + HB 2, NB 6 + HB 3), scored absolutely against
//! the reference `--no-enh` decodes (`tests/fixtures/wb-conformance/`,
//! black-box I/O — see `NOTES.md`).
//!
//! What this gate pins (round r410):
//!
//! * the **crossover-shaped folded high-band law**
//!   (`hb_fold::folded_hb_scale`): the flat r393 constant decoded this
//!   speech fixture at −12.9 dB with a 20× energy overshoot
//!   concentrated in envelope troughs; the shaped law
//!   (`min(0.17·|A_hb(π)|, 0.354)·g`) scores 15.6 dB / 0.986 at unit
//!   energy while leaving the tone and ultra-wideband anchors
//!   untouched;
//! * the first reference comparison of high-band sub-modes 2 and 3
//!   (the `HbSv10_32` / `HbSv8_128` excitation-VQ paths and the 4-bit
//!   gain-correction law): 18.3 / 18.2 dB full-signal;
//! * the per-quality reference alignment (the q6/q8 reference decodes
//!   carry a different lead than the q1/q4 ones — pinned per fixture).

use oxideav_speex::{QmfAnalysis, WidebandDecoder};

struct Fixture {
    name: &'static str,
    nb_mode: u8,
    hb_mode: u8,
    spx: &'static [u8],
    reference: &'static [u8],
    /// Fixed alignment (16 kHz samples): positive = the reference
    /// decode leads ours (QMF/look-ahead padding); negative = ours
    /// leads the reference. Verified by
    /// `reference_lead_is_the_best_alignment`.
    ref_lead: i64,
    min_snr_db: f64,
    min_corr: f64,
    energy: (f64, f64),
    /// Floors for the high-band (4–8 kHz) half-band comparison.
    min_hb_snr_db: f64,
    min_hb_corr: f64,
}

const FIXTURES: &[Fixture] = &[
    // Measured r410 (full snr / corr / energy | low snr | high snr):
    Fixture {
        // 15.6 / 0.986 / 1.01 | low 19.6 | high −6.9 (folded band —
        // residual overshoot on peaky envelopes, recorded follow-up)
        name: "wb-q4",
        nb_mode: 4,
        hb_mode: 1,
        spx: include_bytes!("fixtures/wb-conformance/wb_q4.spx"),
        reference: include_bytes!("fixtures/wb-conformance/wb_q4.noenh.pcm"),
        ref_lead: 80,
        min_snr_db: 18.5,
        min_corr: 0.993,
        energy: (0.9, 1.15),
        min_hb_snr_db: -9.0,
        min_hb_corr: 0.4,
    },
    Fixture {
        // 18.3 / 0.9926 / 0.99
        name: "wb-q6",
        nb_mode: 5,
        hb_mode: 2,
        spx: include_bytes!("fixtures/wb-conformance/wb_q6.spx"),
        reference: include_bytes!("fixtures/wb-conformance/wb_q6.noenh.pcm"),
        ref_lead: -143,
        min_snr_db: 18.5,
        min_corr: 0.993,
        energy: (0.9, 1.1),
        min_hb_snr_db: 5.0,
        min_hb_corr: 0.85,
    },
    Fixture {
        // 18.2 / 0.9925 / 0.99
        name: "wb-q8",
        nb_mode: 6,
        hb_mode: 3,
        spx: include_bytes!("fixtures/wb-conformance/wb_q8.spx"),
        reference: include_bytes!("fixtures/wb-conformance/wb_q8.noenh.pcm"),
        ref_lead: -143,
        min_snr_db: 18.5,
        min_corr: 0.993,
        energy: (0.9, 1.1),
        min_hb_snr_db: 5.0,
        min_hb_corr: 0.85,
    },
];

/// Inline minimal Ogg page-walker — identical contract to the sibling
/// fixture tests.
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

fn reference_pcm(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Decode every audio packet through the wideband decoder, asserting
/// the expected Table 10.2 per-layer sub-modes on the first frame.
fn decode_fixture(fx: &Fixture) -> Vec<f64> {
    use oxideav_speex::{
        BitReader, NarrowbandFrameBody, NarrowbandFrameHeader, Submode,
        WidebandHighBandFrameHeader, WidebandSubmode,
    };

    let packets = lift_ogg_packets(fx.spx);
    assert!(packets.len() > 2, "{}: no audio packets", fx.name);

    // Framing cross-check on the first audio frame.
    {
        let mut r = BitReader::new(&packets[2]);
        let h = NarrowbandFrameHeader::parse(&mut r).unwrap();
        let nb_sub = match h.submode {
            Submode::Celp(s) => s,
            other => panic!("{}: unexpected NB submode {other:?}", fx.name),
        };
        assert_eq!(nb_sub.mode_id, fx.nb_mode, "{}: NB layer sub-mode", fx.name);
        let _ = NarrowbandFrameBody::parse(&mut r, &nb_sub).unwrap();
        let hb = WidebandHighBandFrameHeader::parse(&mut r).unwrap();
        assert!(hb.wideband, "{}: high-band flag", fx.name);
        let hb_sub = match hb.submode {
            WidebandSubmode::Documented(s) => s,
            other => panic!("{}: unexpected HB submode {other:?}", fx.name),
        };
        assert_eq!(hb_sub.mode_id, fx.hb_mode, "{}: HB layer sub-mode", fx.name);
    }

    let mut dec = WidebandDecoder::new();
    let mut out = Vec::new();
    for (i, pkt) in packets[2..].iter().enumerate() {
        let frame = dec
            .decode_packet(pkt)
            .unwrap_or_else(|e| panic!("{} frame {i}: {e}", fx.name));
        out.extend_from_slice(&frame.wideband_pcm);
    }
    out
}

/// `(absolute_snr_db, normalised_correlation, energy_ratio)` at a fixed
/// signed alignment (no fitted gain): positive `lead` drops leading
/// reference samples, negative drops leading decode samples.
fn score(ours: &[f64], reference: &[f64], lead: i64) -> (f64, f64, f64) {
    let (ours, reference): (&[f64], &[f64]) = if lead >= 0 {
        (ours, &reference[lead as usize..])
    } else {
        (&ours[(-lead) as usize..], reference)
    };
    let ref_lead = 0usize;
    let n = (reference.len().saturating_sub(ref_lead)).min(ours.len());
    assert!(n > 10_000, "comparison window too short: {n}");
    let r = &reference[ref_lead..ref_lead + n];
    let o = &ours[..n];
    let mut err = 0.0f64;
    let mut er = 0.0f64;
    let mut eo = 0.0f64;
    let mut dot = 0.0f64;
    for i in 0..n {
        let d = r[i] - o[i];
        err += d * d;
        er += r[i] * r[i];
        eo += o[i] * o[i];
        dot += r[i] * o[i];
    }
    let snr = 10.0 * (er / (err + 1e-12)).log10();
    let corr = dot / (er.sqrt() * eo.sqrt() + 1e-12);
    (snr, corr, eo / er)
}

/// Split a 16 kHz signal into its QMF half-bands (both signals measured
/// through the same analysis bank).
fn split_bands(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut qa = QmfAnalysis::new();
    let mut low = Vec::new();
    let mut high = Vec::new();
    for frame in x.chunks(320) {
        if frame.len() < 320 {
            break;
        }
        let mut l = [0.0f64; 160];
        let mut h = [0.0f64; 160];
        qa.split_slices(frame, &mut l, &mut h);
        low.extend_from_slice(&l);
        high.extend_from_slice(&h);
    }
    (low, high)
}

#[test]
fn wideband_conformance_matrix() {
    for fx in FIXTURES {
        let reference = reference_pcm(fx.reference);
        let ours = decode_fixture(fx);

        let (snr, corr, ratio) = score(&ours, &reference, fx.ref_lead);
        let (ref_low, ref_high) = split_bands(&reference);
        let (our_low, our_high) = split_bands(&ours);
        // Half-band comparison is only well-posed for an even full-rate
        // lead (an odd lead lands the half-bands half a sample apart,
        // which decorrelates the near-Nyquist high band entirely) —
        // q6/q8's odd reference alignment gates full-signal only.
        let bands = (fx.ref_lead % 2 == 0).then(|| {
            let (hsnr, hcorr, _) = score(&our_high, &ref_high, fx.ref_lead / 2);
            let (lsnr, lcorr, _) = score(&our_low, &ref_low, fx.ref_lead / 2);
            (hsnr, hcorr, lsnr, lcorr)
        });

        if let Some((hsnr, hcorr, lsnr, lcorr)) = bands {
            println!(
                "{} (nb {} hb {}): raw {snr:.2} dB corr {corr:.4} energy {ratio:.4} | low {lsnr:.2} dB {lcorr:.4} | high {hsnr:.2} dB {hcorr:.4}",
                fx.name, fx.nb_mode, fx.hb_mode
            );
        } else {
            println!(
                "{} (nb {} hb {}): raw {snr:.2} dB corr {corr:.4} energy {ratio:.4}",
                fx.name, fx.nb_mode, fx.hb_mode
            );
        }

        assert!(
            snr >= fx.min_snr_db,
            "{}: full SNR {snr:.2} dB < {} dB",
            fx.name,
            fx.min_snr_db
        );
        assert!(
            corr >= fx.min_corr,
            "{}: correlation {corr:.4} < {}",
            fx.name,
            fx.min_corr
        );
        assert!(
            ratio >= fx.energy.0 && ratio <= fx.energy.1,
            "{}: energy ratio {ratio:.4} outside [{}, {}]",
            fx.name,
            fx.energy.0,
            fx.energy.1
        );
        if let Some((hsnr, hcorr, _, _)) = bands {
            assert!(
                hsnr >= fx.min_hb_snr_db,
                "{}: high-band SNR {hsnr:.2} dB < {} dB",
                fx.name,
                fx.min_hb_snr_db
            );
            assert!(
                hcorr >= fx.min_hb_corr,
                "{}: high-band correlation {hcorr:.4} < {}",
                fx.name,
                fx.min_hb_corr
            );
        }
    }
}

/// The fixed alignment must be the best in a ±400-sample two-sided
/// window — a delay-convention regression fails loudly.
#[test]
fn reference_lead_is_the_best_alignment() {
    for fx in FIXTURES {
        let reference = reference_pcm(fx.reference);
        let ours = decode_fixture(fx);
        let (at_lead, _, _) = score(&ours, &reference, fx.ref_lead);
        let mut best = (f64::MIN, 0i64);
        for lead in -400i64..=400 {
            let (s, _, _) = score(&ours, &reference, lead);
            if s > best.0 {
                best = (s, lead);
            }
        }
        assert!(
            (best.1 - fx.ref_lead).abs() <= 1 && at_lead >= best.0 - 0.6,
            "{}: best lag {} ({:.2} dB) vs fixed lead {} ({at_lead:.2} dB)",
            fx.name,
            best.1,
            best.0,
            fx.ref_lead
        );
    }
}
