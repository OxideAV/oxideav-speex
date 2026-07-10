//! Narrowband 3-tap pitch-gain VQ reconstruction (round r208 scope).
//!
//! Wires the 5-bit / 7-bit 3-tap pitch-gain VQ codebooks staged in
//! [`crate::codebooks`] into typed `(g0, g1, g2)` β-coefficient
//! triples. Given a per-sub-frame raw codebook index parsed by the
//! round-3 frame-body bit-reader and the sub-mode's pitch-gain
//! quantisation regime, this module resolves the index into the three
//! tap coefficients the §9.2 long-term predictor equation requires:
//!
//! ```text
//! ea[n] = g0·e[n − T − 1] + g1·e[n − T] + g2·e[n − T + 1]
//! ```
//! (Speex Codec Manual Eq. 9.1; CELP companion §2.2.)
//!
//! ## Codebook layout
//!
//! The staged tables (`docs/audio/speex/tables/`) carry each row as a
//! four-column tuple `[g0, g1, g2, search_aid]`:
//!
//! * `pitch-gain-cdbk-5bit` — 32 × 4 (5-bit index, low-bit-rate modes
//!   per the companion §2.2: ≤ 11 kbps narrowband).
//! * `pitch-gain-cdbk-7bit` — 128 × 4 (7-bit index, higher-rate
//!   modes: ≥ 15 kbps narrowband and above).
//!
//! Columns 0..=2 are the three β tap gains. Per the codebook `.meta`
//! sidecar (`docs/audio/speex/tables/pitch-gain-cdbk-{5bit,7bit}.meta`)
//! the stored gain bytes are biased: the decoder adds `+32` (see
//! [`crate::codebooks::PITCH_GAIN_BIAS`]) to each of `g0`, `g1`, `g2`
//! before applying them. Column 3 (`search_aid`) is an encoder-side
//! pre-computed term used during the codebook search; the decoder
//! does not need it.
//!
//! ## Column ↔ lag association (fixture-arbitrated, round r410)
//!
//! Neither the manual nor the `.meta` sidecars pin **which stored
//! column multiplies which lag** in Eq. 9.1. The two candidate
//! readings (column 0 ↔ `e[n−T−1]` vs column 0 ↔ `e[n−T+1]`) were
//! arbitrated black-box against the staged narrowband reference
//! decodes (`tests/fixtures/nb-conformance/`): the **reversed**
//! association — column 0 multiplies `e[n−T+1]`, column 2 multiplies
//! `e[n−T−1]` — scores 13.5–14.4 dB absolute SNR across the VQ
//! sub-modes (energy ratio ≈ 0.98) where the direct reading scores
//! 2.8–5.6 dB with half the reference energy. [`reconstruct`]
//! therefore maps `taps[0] = col2`, `taps[1] = col1`, `taps[2] = col0`
//! (post-bias), keeping the [`PitchGainTaps`] convention `taps[0]` ↔
//! `e[n−T−1]`, `taps[1]` ↔ `e[n−T]`, `taps[2]` ↔ `e[n−T+1]` for every
//! downstream consumer.
//!
//! ## Numeric range
//!
//! Stored entries are signed bytes (the CSVs widen them to `i16` for
//! sign-safety). With the documented `+32` bias the post-bias gains
//! land in `(-128 + 32) .. (127 + 32) = -96 .. 159`. The codebook
//! row-0 of `pitch-gain-cdbk-5bit` is `(-32, -32, -32, 0)`, which
//! after the bias becomes `(0, 0, 0)` — the documented "silence"
//! / no-pitch entry. The existing `codebooks::tests` sanity test
//! `pitch_gain_5bit_first_row_matches_csv` already pins that
//! relationship.
//!
//! ## Output type and Q-format
//!
//! This module surfaces the resolved triple as a typed `[i16; 3]`
//! tuple of post-bias integers — the raw codebook units. The
//! manual + CELP companion are silent on a documented Q-format for
//! the pitch-gain values themselves: the runtime gain magnitude
//! depends on how the long-term predictor convolution chooses to
//! interpret them (typically Q6 = `gain / 64`, but the manual does
//! not commit to that explicitly in either §8.3 or §9.2). The post-bias
//! integer surface keeps this module independent of any Q-format
//! interpretation, so the downstream long-term-predictor step can
//! pick its convention later without re-deriving the reconstruction
//! here. Recorded as a docs gap in the README.
//!
//! ## What this module does NOT do
//!
//! * No long-term-predictor convolution. The §9.2 / Eq. 9.1
//!   convolution `ea[n] = g0·e[n − T − 1] + g1·e[n − T] + g2·e[n −
//!   T + 1]` requires the per-sub-frame pitch period `T` (already
//!   surfaced by [`crate::NarrowbandSubFrameIndices::pitch_period`])
//!   plus the historical excitation buffer `e[·]`. Buffer
//!   management + the convolution itself land in a follow-up round
//!   once the innovation-codebook lookup is also wired (then the
//!   excitation `e[n] = p[n] + c[n]` can be assembled, per the
//!   companion §2.3).
//! * No encoder-side codebook search. The reverse direction
//!   ((g0, g1, g2) → quantised index) needs the column-3
//!   `search_aid` term and a perceptual-weighting metric.
//!   Out of scope for r208.
//! * No high-band pitch path. The wideband sub-band CELP high band
//!   carries no adaptive codebook (§10.4 / Table 10.1: the
//!   high-band has only LSP + excitation gain + excitation VQ — no
//!   pitch field). The high band never consults this module.

use crate::codebooks::{
    pitch_gain_5bit, pitch_gain_7bit, PITCH_GAIN_5BIT_ENTRIES, PITCH_GAIN_7BIT_ENTRIES,
    PITCH_GAIN_BIAS, PITCH_GAIN_COLS,
};
use crate::submode::PitchGainQuant;

/// Number of taps in the Speex long-term predictor (manual Eq. 9.1 has
/// three: `e[n − T − 1]`, `e[n − T]`, `e[n − T + 1]`).
pub const PITCH_GAIN_TAPS: usize = 3;

/// Three β tap coefficients for the long-term predictor convolution,
/// stored as post-bias signed integers in raw codebook units.
///
/// Mapping back to Speex Manual Eq. 9.1:
///
/// * `taps[0]` is `g0` (multiplies `e[n − T − 1]`)
/// * `taps[1]` is `g1` (multiplies `e[n − T]`)
/// * `taps[2]` is `g2` (multiplies `e[n − T + 1]`)
///
/// See module docs for the Q-format docs gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PitchGainTaps {
    /// Three β tap coefficients in codebook-row order, with the
    /// documented `+32` bias already applied.
    pub taps: [i16; PITCH_GAIN_TAPS],
}

impl PitchGainTaps {
    /// All-zero taps — corresponds to the documented "no pitch"
    /// silence row of the 5-bit codebook (row 0 = `(-32, -32, -32, …)`
    /// → `(0, 0, 0)` after bias). Useful for the mode-0 silence path
    /// where no pitch field is transmitted.
    pub const SILENCE: Self = Self {
        taps: [0; PITCH_GAIN_TAPS],
    };
}

/// Resolve a per-sub-frame pitch-gain VQ index against the
/// quantisation regime advertised by the active sub-mode.
///
/// * `quant = None` (mode 0, silence) → [`PitchGainTaps::SILENCE`]
///   regardless of `index`. The bit-stream carries no field; the
///   caller's `index` value (defaulted to `0` by the parser) is
///   ignored.
/// * `quant = Vq5Bit` → looks up the 5-bit codebook; rejects an
///   index that overflows the 32-entry codebook.
/// * `quant = Vq7Bit` → looks up the 7-bit codebook; rejects an
///   index that overflows the 128-entry codebook.
///
/// Returns `None` on an out-of-range index. Since the parser caps
/// the index by the sub-mode's documented field width (5 or 7 bits),
/// `None` here only fires if the caller hand-builds an
/// [`crate::NarrowbandSubFrameIndices`] with a stray high-bit set.
pub fn reconstruct(index: u8, quant: PitchGainQuant) -> Option<PitchGainTaps> {
    match quant {
        PitchGainQuant::None => Some(PitchGainTaps::SILENCE),
        PitchGainQuant::Vq5Bit => lookup_biased(pitch_gain_5bit(), index, PITCH_GAIN_5BIT_ENTRIES),
        PitchGainQuant::Vq7Bit => lookup_biased(pitch_gain_7bit(), index, PITCH_GAIN_7BIT_ENTRIES),
    }
}

/// Shared lookup-plus-bias helper used by both codebook regimes.
///
/// Applies the documented `+32` bias and the **fixture-arbitrated
/// column reversal** (module docs): stored column 0 is the `e[n−T+1]`
/// tap, so it lands in `taps[2]`; stored column 2 lands in `taps[0]`.
fn lookup_biased(
    codebook: &[[i16; PITCH_GAIN_COLS]],
    index: u8,
    expected_entries: usize,
) -> Option<PitchGainTaps> {
    let idx = index as usize;
    if idx >= codebook.len() || idx >= expected_entries {
        return None;
    }
    let row = codebook[idx];
    let mut taps = [0i16; PITCH_GAIN_TAPS];
    for (i, slot) in taps.iter_mut().enumerate() {
        *slot = row[PITCH_GAIN_TAPS - 1 - i] + PITCH_GAIN_BIAS;
    }
    Some(PitchGainTaps { taps })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebooks::{pitch_gain_5bit, pitch_gain_7bit};

    #[test]
    fn silence_quant_yields_zero_taps() {
        let taps = reconstruct(0, PitchGainQuant::None).unwrap();
        assert_eq!(taps, PitchGainTaps::SILENCE);
        assert_eq!(taps.taps, [0, 0, 0]);
    }

    #[test]
    fn silence_quant_ignores_index_value() {
        // mode 0 carries no pitch-gain field; the parser's defaulted
        // `0` index is what shows up in practice but the contract is
        // explicit: regardless of `index`, `None` quant → SILENCE.
        for idx in [0u8, 1, 31, 127, 255] {
            let taps = reconstruct(idx, PitchGainQuant::None).unwrap();
            assert_eq!(taps, PitchGainTaps::SILENCE);
        }
    }

    #[test]
    fn vq5bit_row_0_is_documented_silence_after_bias() {
        // Row 0 of `pitch-gain-cdbk-5bit` is (-32, -32, -32, 0); after
        // the +32 bias the three β taps become (0, 0, 0). This is the
        // documented low-bit-rate "no pitch" entry (companion §2.2 +
        // the existing `codebooks::tests` pin).
        let taps = reconstruct(0, PitchGainQuant::Vq5Bit).unwrap();
        assert_eq!(taps.taps, [0, 0, 0]);
    }

    #[test]
    fn vq5bit_row_1_matches_csv_with_bias() {
        // Row 1 of the 5-bit codebook (from the staged CSV) is
        // `-31, -58, -16, 22`. After +32 bias and the r410 column
        // reversal the taps are (16, -26, 1); column 3 (search aid 22)
        // is discarded.
        let raw = pitch_gain_5bit()[1];
        let expected = [
            raw[2] + PITCH_GAIN_BIAS,
            raw[1] + PITCH_GAIN_BIAS,
            raw[0] + PITCH_GAIN_BIAS,
        ];
        let taps = reconstruct(1, PitchGainQuant::Vq5Bit).unwrap();
        assert_eq!(taps.taps, expected);
        // And cross-check against the literal CSV values (regression
        // guard if the table extraction changes upstream).
        assert_eq!(taps.taps, [16, -26, 1]);
    }

    #[test]
    fn vq7bit_row_0_matches_csv_with_bias() {
        let raw = pitch_gain_7bit()[0];
        let expected = [
            raw[2] + PITCH_GAIN_BIAS,
            raw[1] + PITCH_GAIN_BIAS,
            raw[0] + PITCH_GAIN_BIAS,
        ];
        let taps = reconstruct(0, PitchGainQuant::Vq7Bit).unwrap();
        assert_eq!(taps.taps, expected);
    }

    #[test]
    fn vq5bit_max_index_is_31() {
        let taps = reconstruct(31, PitchGainQuant::Vq5Bit);
        assert!(taps.is_some(), "5-bit index 31 is the max valid entry");
        let raw_31 = pitch_gain_5bit()[31];
        assert_eq!(
            taps.unwrap().taps,
            [
                raw_31[2] + PITCH_GAIN_BIAS,
                raw_31[1] + PITCH_GAIN_BIAS,
                raw_31[0] + PITCH_GAIN_BIAS,
            ]
        );
    }

    #[test]
    fn vq7bit_max_index_is_127() {
        let taps = reconstruct(127, PitchGainQuant::Vq7Bit);
        assert!(taps.is_some(), "7-bit index 127 is the max valid entry");
        let raw_127 = pitch_gain_7bit()[127];
        assert_eq!(
            taps.unwrap().taps,
            [
                raw_127[2] + PITCH_GAIN_BIAS,
                raw_127[1] + PITCH_GAIN_BIAS,
                raw_127[0] + PITCH_GAIN_BIAS,
            ]
        );
    }

    #[test]
    fn vq5bit_rejects_out_of_range_index() {
        // 5-bit field can only address 0..32 on the wire, but a
        // hand-constructed index >= 32 must return `None` rather
        // than panic.
        assert!(reconstruct(32, PitchGainQuant::Vq5Bit).is_none());
        assert!(reconstruct(33, PitchGainQuant::Vq5Bit).is_none());
        assert!(reconstruct(127, PitchGainQuant::Vq5Bit).is_none());
        assert!(reconstruct(255, PitchGainQuant::Vq5Bit).is_none());
    }

    #[test]
    fn vq7bit_accepts_full_range() {
        // 7-bit field addresses 0..128 — every index in that range
        // must succeed.
        for idx in 0u8..=127 {
            assert!(
                reconstruct(idx, PitchGainQuant::Vq7Bit).is_some(),
                "7-bit index {idx} unexpectedly rejected"
            );
        }
        // ... but 128 and above are out of range.
        assert!(reconstruct(128, PitchGainQuant::Vq7Bit).is_none());
        assert!(reconstruct(255, PitchGainQuant::Vq7Bit).is_none());
    }

    #[test]
    fn vq5bit_accepts_full_range() {
        for idx in 0u8..32 {
            assert!(
                reconstruct(idx, PitchGainQuant::Vq5Bit).is_some(),
                "5-bit index {idx} unexpectedly rejected"
            );
        }
    }

    #[test]
    fn reconstruction_drops_search_aid_column() {
        // The taps array is 3-wide; the codebook row is 4-wide
        // (col 3 = search_aid). The struct intentionally surfaces
        // only the first three columns.
        assert_eq!(PITCH_GAIN_TAPS, 3);
        for idx in [0u8, 5, 31] {
            let taps = reconstruct(idx, PitchGainQuant::Vq5Bit).unwrap();
            assert_eq!(taps.taps.len(), PITCH_GAIN_TAPS);
        }
    }

    #[test]
    fn bias_is_applied_consistently_across_taps_and_rows() {
        // Every emitted tap = stored byte + PITCH_GAIN_BIAS; check the
        // invariant holds for every row of both codebooks.
        for (idx, raw) in pitch_gain_5bit().iter().enumerate() {
            let taps = reconstruct(idx as u8, PitchGainQuant::Vq5Bit).unwrap();
            for t in 0..PITCH_GAIN_TAPS {
                assert_eq!(
                    taps.taps[t],
                    raw[PITCH_GAIN_TAPS - 1 - t] + PITCH_GAIN_BIAS,
                    "5-bit row {idx} tap {t}"
                );
            }
        }
        for (idx, raw) in pitch_gain_7bit().iter().enumerate() {
            let taps = reconstruct(idx as u8, PitchGainQuant::Vq7Bit).unwrap();
            for t in 0..PITCH_GAIN_TAPS {
                assert_eq!(
                    taps.taps[t],
                    raw[PITCH_GAIN_TAPS - 1 - t] + PITCH_GAIN_BIAS,
                    "7-bit row {idx} tap {t}"
                );
            }
        }
    }

    #[test]
    fn silence_constant_matches_5bit_row_0() {
        // The const SILENCE must be byte-identical to what looking up
        // 5-bit row 0 produces (the documented "no pitch" entry).
        let row0 = reconstruct(0, PitchGainQuant::Vq5Bit).unwrap();
        assert_eq!(row0, PitchGainTaps::SILENCE);
    }

    #[test]
    fn taps_fit_in_documented_range() {
        // Codebook entries are signed bytes (i8 range -128..=127);
        // with +32 bias the post-bias range is -96..=159. Every row
        // of both codebooks must fall in that band — guards against a
        // future table extraction that widens the underlying type.
        for raw in pitch_gain_5bit() {
            for &v in raw.iter().take(PITCH_GAIN_TAPS) {
                let biased = v + PITCH_GAIN_BIAS;
                assert!(
                    (-96..=159).contains(&biased),
                    "5-bit codebook entry {biased} out of documented signed-byte+bias range"
                );
            }
        }
        for raw in pitch_gain_7bit() {
            for &v in raw.iter().take(PITCH_GAIN_TAPS) {
                let biased = v + PITCH_GAIN_BIAS;
                assert!(
                    (-96..=159).contains(&biased),
                    "7-bit codebook entry {biased} out of documented signed-byte+bias range"
                );
            }
        }
    }
}
