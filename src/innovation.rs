//! Narrowband **innovation sub-vector lookup** (round r220 scope;
//! per-mode binding extended to modes 2 / 3 / 4 / 5 in r277).
//!
//! Wires the six staged narrowband innovation (fixed) codebooks in
//! [`crate::codebooks`] into typed `&'static [i16]` slices keyed by a
//! codebook-shape selector + a raw sub-vector index.
//!
//! ## What lands this round
//!
//! Per Speex Codec Manual §9.2 (p.31) and the CELP companion §2.3, each
//! 40-sample CELP sub-frame's innovation signal `c[n]` is encoded as a
//! concatenation of **fixed-size sub-vectors**, each sub-vector picked
//! from a bit-rate-dependent codebook. Worked examples (companion §2.3):
//!
//! * 3.95 kbps narrowband mode → sub-vector 20 samples, 32 entries
//!   (5 bits) → 10 bits/sub-frame total = 2000 bps innovation.
//! * 18.2 kbps narrowband mode → sub-vector 5 samples, 256 entries
//!   (8 bits) → 64 bits/sub-frame total = 12 800 bps innovation.
//!
//! This module surfaces the **codebook-row primitive** every per-mode
//! dispatcher needs: given one of the six documented codebook shapes
//! and a row index, return the static slice of `[i16]` samples that the
//! decoder concatenates into the per-sub-frame `c[n]` buffer.
//!
//! ## The six documented shapes (§9.2 + table inventory)
//!
//! Inferred from the staged codebook dimensions
//! (`docs/audio/speex/tables/`) and confirmed against the bit-budget
//! arithmetic of Table 9.1:
//!
//! | Shape         | sub-vector length | index width | entries |
//! | ------------- | ----------------: | ----------: | ------: |
//! | `Sv5_64`      |                 5 |     6 bits  |      64 |
//! | `Sv5_256`     |                 5 |     8 bits  |     256 |
//! | `Sv8_128`     |                 8 |     7 bits  |     128 |
//! | `Sv10_16`     |                10 |     4 bits  |      16 |
//! | `Sv10_32`     |                10 |     5 bits  |      32 |
//! | `Sv20_32`     |                20 |     5 bits  |      32 |
//!
//! ## Per-mode dispatcher
//!
//! Six of the nine narrowband modes have **docs-grounded** innovation
//! codebook assignments. Two come from the companion §2.3 worked
//! examples (mode 6, 18.2 kbps → sv5/256; mode 8, 3.95 kbps → sv20/32;
//! bound in r220). Four more (r277) come from the staged per-codebook
//! **innovation bit-rate annotations**: each codebook's `.meta` sidecar
//! (`docs/audio/speex/tables/*.meta`, `role:` field) and the staged
//! `tables/README.md` inventory record the innovation bit-rate each
//! codebook serves (3200 / 4000 / 7000 / 9600 / 12800 / 2000 bps), and
//! Table 9.1's per-sub-frame "Innovation VQ" row converts directly to
//! that rate (`bits/sub-frame × 4 sub-frames / 20 ms = bits × 200 bps`):
//!
//! | Mode | innovation_vq_bits | bps (×200) | dispatch                  |
//! | ---: | -----------------: | ---------: | ------------------------- |
//! |    2 |                16  |      3200  | 4 × `Sv10_16` (4 bits ea) |
//! |    3 |                20  |      4000  | 4 × `Sv10_32` (5 bits ea) |
//! |    4 |                35  |      7000  | 5 × `Sv8_128` (7 bits ea) |
//! |    5 |                48  |      9600  | 8 × `Sv5_64`  (6 bits ea) |
//! |    6 |                64  |     12800  | 8 × `Sv5_256` (8 bits ea) |
//! |    8 |                10  |      2000  | 2 × `Sv20_32` (5 bits ea) |
//!
//! Each binding is unique (exactly one staged codebook is annotated at
//! each rate) and self-consistent: `index_bits × count` equals the
//! Table 9.1 budget and `sub_vector_len × count` covers the 40-sample
//! sub-frame, for every row. The Table 9.2 composite bit-rates
//! cross-check too (e.g. mode 3 = 8000 bps total carries the 4000 bps
//! innovation stream; mode 6 = 18 200 bps carries 12 800 bps).
//!
//! Two modes remain [`InnovationMapping::Undocumented`]:
//!
//! * **Mode 1** (2.15 kbps vocoder) has `innovation_vq_bits = 0` — no
//!   VQ field on the wire at all — yet is not the silent mode 0: per
//!   Table 9.2 it transmits "mostly comfort noise". The excitation
//!   *generation* rule for this vocoder mode (how `c[n]` is synthesised
//!   without a codebook index) is not in the staged material.
//! * **Mode 7** (24.6 kbps) has `innovation_vq_bits = 96` → 19 200 bps,
//!   a rate no staged codebook is annotated with; 96 bits admit no
//!   `index_bits × count` / 40-sample-consistent decomposition over a
//!   single staged shape (96 = 12 × 8 bits would need 12 sub-vectors of
//!   length 3⅓). A multi-stage scheme is plausible but undocumented.
//!
//! Both are recorded as docs gaps in the README.
//!
//! ## What this module does NOT do
//!
//! * No innovation-gain scaling. The §9.2 fixed-codebook gain
//!   `g_frame × g_subf` consumes a separate frame-level + sub-frame
//!   field already surfaced by [`crate::NarrowbandFrameBody`]; applying
//!   the gain to the looked-up sub-vector samples needs the gain
//!   reconstruction (a separate later round) plus the per-mode
//!   sub-vector dispatcher above.
//! * No excitation buffer composition. The final excitation
//!   `e[n] = p[n] + c[n]` per companion §2.3 also needs the adaptive
//!   (long-term predictor) contribution `p[n]`; both stay deferred.
//! * No encoder-side codebook search.
//! * No high-band innovation dispatch. High-band excitation is coded
//!   like narrowband innovation per §10.3 but with its own per-mode
//!   binding (two codebook shapes, see [`crate::codebooks::hb_innovation_8_128`]
//!   and `hb_innovation_10_32`); a follow-up round adds the wideband
//!   path.

use crate::codebooks::{
    innovation_10_16, innovation_10_32, innovation_20_32, innovation_5_256, innovation_5_64,
    innovation_8_128,
};
use crate::submode::NarrowbandSubmode;

/// Selector for one of the six documented narrowband innovation
/// codebook shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InnovationCodebook {
    /// 5-sample sub-vector, 6-bit index, 64 entries (9.6 kbps innovation).
    Sv5_64,
    /// 5-sample sub-vector, 8-bit index, 256 entries (12.8 kbps innovation).
    Sv5_256,
    /// 8-sample sub-vector, 7-bit index, 128 entries (7.0 kbps innovation).
    Sv8_128,
    /// 10-sample sub-vector, 4-bit index, 16 entries (3.2 kbps innovation).
    Sv10_16,
    /// 10-sample sub-vector, 5-bit index, 32 entries (4.0 kbps innovation).
    Sv10_32,
    /// 20-sample sub-vector, 5-bit index, 32 entries (2.0 kbps innovation,
    /// 3.95 kbps mode 8).
    Sv20_32,
}

impl InnovationCodebook {
    /// Samples per sub-vector.
    pub const fn sub_vector_len(self) -> usize {
        match self {
            InnovationCodebook::Sv5_64 | InnovationCodebook::Sv5_256 => 5,
            InnovationCodebook::Sv8_128 => 8,
            InnovationCodebook::Sv10_16 | InnovationCodebook::Sv10_32 => 10,
            InnovationCodebook::Sv20_32 => 20,
        }
    }

    /// Codebook index width on the wire.
    pub const fn index_bits(self) -> u8 {
        match self {
            InnovationCodebook::Sv10_16 => 4,
            InnovationCodebook::Sv10_32 => 5,
            InnovationCodebook::Sv20_32 => 5,
            InnovationCodebook::Sv5_64 => 6,
            InnovationCodebook::Sv8_128 => 7,
            InnovationCodebook::Sv5_256 => 8,
        }
    }

    /// Total number of entries (= `1 << index_bits()`).
    pub const fn entries(self) -> u32 {
        1u32 << self.index_bits()
    }
}

/// Resolved one-sub-vector lookup result.
///
/// The slice's length is the codebook's [`InnovationCodebook::sub_vector_len`]
/// (5 / 8 / 10 / 20 depending on the codebook). Each `i16` is the raw
/// stored sample from the staged codebook CSV — no gain scaling has
/// been applied; the eventual fixed-codebook-gain step combines this
/// sub-vector with the §9.2 frame-level + sub-frame innovation gains.
pub fn sub_vector(codebook: InnovationCodebook, index: u32) -> Option<&'static [i16]> {
    let idx = index as usize;
    match codebook {
        InnovationCodebook::Sv5_64 => {
            let table = innovation_5_64();
            table.get(idx).map(|row| row.as_slice())
        }
        InnovationCodebook::Sv5_256 => {
            let table = innovation_5_256();
            table.get(idx).map(|row| row.as_slice())
        }
        InnovationCodebook::Sv8_128 => {
            let table = innovation_8_128();
            table.get(idx).map(|row| row.as_slice())
        }
        InnovationCodebook::Sv10_16 => {
            let table = innovation_10_16();
            table.get(idx).map(|row| row.as_slice())
        }
        InnovationCodebook::Sv10_32 => {
            let table = innovation_10_32();
            table.get(idx).map(|row| row.as_slice())
        }
        InnovationCodebook::Sv20_32 => {
            let table = innovation_20_32();
            table.get(idx).map(|row| row.as_slice())
        }
    }
}

/// Number of samples in a Speex narrowband sub-frame (per Speex Manual
/// §9.1: *"subdivided in 4 sub-frames of 5 ms each (40 samples)"*).
pub const SUBFRAME_SAMPLES: usize = 40;

/// Per-mode innovation dispatch resolved from the staged material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InnovationMapping {
    /// Silence sub-mode (mode 0). No innovation field on the wire; the
    /// `c[n]` sub-vector is implicitly all zero (no fixed-codebook
    /// contribution per §9.3 row "Innovation VQ" = 0 bits).
    Silence,
    /// Documented per-mode binding: the sub-frame's 40 samples come
    /// from `count` concatenated lookups against `codebook`, taking
    /// successive `codebook.index_bits()`-wide indices off the
    /// `innovation_vq_index` field in most-significant-bit-first order
    /// (the same order [`crate::NarrowbandFrameBody`] read the raw
    /// `u128`).
    Documented {
        /// Which of the six codebook shapes is used.
        codebook: InnovationCodebook,
        /// How many sub-vectors are concatenated per 40-sample sub-frame
        /// (must satisfy `codebook.sub_vector_len() * count == SUBFRAME_SAMPLES`).
        count: u8,
    },
    /// The per-mode excitation handling is not documented in the
    /// staged `docs/audio/speex/` material. Two modes fall here: mode
    /// 1 (0-bit VQ field; the vocoder excitation-generation rule is
    /// unstaged) and mode 7 (96-bit field → 19 200 bps innovation
    /// rate, which no staged codebook is annotated with). Recorded
    /// in the README as docs gaps.
    Undocumented,
}

impl InnovationMapping {
    /// Resolve the per-mode innovation dispatcher from a narrowband
    /// sub-mode.
    ///
    /// Returns [`InnovationMapping::Silence`] for mode 0
    /// (no innovation field), [`InnovationMapping::Documented`] for
    /// the six modes whose codebook binding is grounded by the staged
    /// material (modes 2 / 3 / 4 / 5 via the per-codebook innovation
    /// bit-rate annotations, modes 6 / 8 via the companion §2.3 worked
    /// examples), and [`InnovationMapping::Undocumented`] for modes
    /// 1 and 7 (see module docs).
    pub const fn for_mode(submode: &NarrowbandSubmode) -> Self {
        match submode.mode_id {
            0 => InnovationMapping::Silence,
            2 => InnovationMapping::Documented {
                codebook: InnovationCodebook::Sv10_16,
                count: 4,
            },
            3 => InnovationMapping::Documented {
                codebook: InnovationCodebook::Sv10_32,
                count: 4,
            },
            4 => InnovationMapping::Documented {
                codebook: InnovationCodebook::Sv8_128,
                count: 5,
            },
            5 => InnovationMapping::Documented {
                codebook: InnovationCodebook::Sv5_64,
                count: 8,
            },
            6 => InnovationMapping::Documented {
                codebook: InnovationCodebook::Sv5_256,
                count: 8,
            },
            8 => InnovationMapping::Documented {
                codebook: InnovationCodebook::Sv20_32,
                count: 2,
            },
            _ => InnovationMapping::Undocumented,
        }
    }
}

/// Errors from [`decode_subframe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InnovationError {
    /// The per-mode codebook binding is not documented in the staged
    /// material. See [`InnovationMapping::Undocumented`].
    Undocumented,
    /// A per-sub-vector index extracted from the `innovation_vq_index`
    /// field exceeded the codebook entry count (only possible with a
    /// hand-built [`crate::NarrowbandSubFrameIndices`]; the parser caps
    /// the field to the sub-mode's documented width).
    IndexOutOfRange { sub_vector: u8, index: u32 },
}

impl core::fmt::Display for InnovationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InnovationError::Undocumented => write!(
                f,
                "innovation dispatcher: per-mode codebook binding not in staged docs"
            ),
            InnovationError::IndexOutOfRange { sub_vector, index } => write!(
                f,
                "innovation dispatcher: sub-vector {sub_vector} index {index} out of codebook range"
            ),
        }
    }
}

impl std::error::Error for InnovationError {}

/// Decode the 40-sample fixed-codebook sub-vector `c[n]` of a single
/// CELP sub-frame from the raw `innovation_vq_index` field.
///
/// Walks `count` successive `index_bits`-wide chunks off the top of the
/// packed `innovation_vq_index` (MSB-first — the same order the parser
/// read them into the `u128`), looks up each chunk against
/// `codebook`, and concatenates the resulting sub-vectors into a single
/// 40-element vector.
///
/// Returns [`InnovationError::Undocumented`] for sub-modes whose
/// dispatcher is not bound (see [`InnovationMapping::Undocumented`]);
/// returns [`InnovationError::IndexOutOfRange`] for a malformed index.
///
/// For mode 0 (silence), the all-zero 40-sample vector is returned —
/// no innovation field is transmitted and the fixed-codebook
/// contribution is implicitly zero per §9.3 row "Innovation VQ" = 0.
pub fn decode_subframe(
    submode: &NarrowbandSubmode,
    innovation_vq_index: u128,
) -> Result<[i16; SUBFRAME_SAMPLES], InnovationError> {
    let mapping = InnovationMapping::for_mode(submode);
    match mapping {
        InnovationMapping::Silence => Ok([0i16; SUBFRAME_SAMPLES]),
        InnovationMapping::Undocumented => Err(InnovationError::Undocumented),
        InnovationMapping::Documented { codebook, count } => {
            let sv_len = codebook.sub_vector_len();
            let bits = codebook.index_bits() as u32;
            let total_bits = bits * u32::from(count);
            debug_assert!(
                total_bits as usize <= 128,
                "innovation total bits exceeds u128 capacity"
            );
            debug_assert_eq!(
                sv_len * usize::from(count),
                SUBFRAME_SAMPLES,
                "innovation dispatch sub-vectors must cover 40 samples"
            );

            let mask = if bits == 0 { 0 } else { (1u128 << bits) - 1 };
            let mut out = [0i16; SUBFRAME_SAMPLES];
            // The packed field's MSB carries sub-vector 0 (since the
            // parser shifted earlier reads up by `chunk` for every
            // subsequent read — see `read_wide` in narrowband_body).
            for sv in 0..count {
                let shift = u32::from(count - 1 - sv) * bits;
                let idx = ((innovation_vq_index >> shift) & mask) as u32;
                let row = sub_vector(codebook, idx).ok_or(InnovationError::IndexOutOfRange {
                    sub_vector: sv,
                    index: idx,
                })?;
                let base = usize::from(sv) * sv_len;
                out[base..base + sv_len].copy_from_slice(row);
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebooks::{
        innovation_10_16, innovation_10_32, innovation_20_32, innovation_5_256, innovation_5_64,
        innovation_8_128,
    };
    use crate::submode::NARROWBAND_SUBMODES;

    #[test]
    fn sub_vector_dimensions_match_codebook_meta() {
        let cases: [(InnovationCodebook, usize, u8, u32); 6] = [
            (InnovationCodebook::Sv5_64, 5, 6, 64),
            (InnovationCodebook::Sv5_256, 5, 8, 256),
            (InnovationCodebook::Sv8_128, 8, 7, 128),
            (InnovationCodebook::Sv10_16, 10, 4, 16),
            (InnovationCodebook::Sv10_32, 10, 5, 32),
            (InnovationCodebook::Sv20_32, 20, 5, 32),
        ];
        for (cb, sv, bits, ent) in cases {
            assert_eq!(cb.sub_vector_len(), sv, "{cb:?} sub_vector_len");
            assert_eq!(cb.index_bits(), bits, "{cb:?} index_bits");
            assert_eq!(cb.entries(), ent, "{cb:?} entries");
        }
    }

    #[test]
    fn sub_vector_resolves_first_row_of_every_codebook() {
        // Hit every codebook accessor's row 0 to ensure the slice is
        // wired correctly and lengths match the codebook's declared
        // sub_vector_len. This also exercises the OnceLock init paths.
        let cases: [(InnovationCodebook, usize); 6] = [
            (InnovationCodebook::Sv5_64, 5),
            (InnovationCodebook::Sv5_256, 5),
            (InnovationCodebook::Sv8_128, 8),
            (InnovationCodebook::Sv10_16, 10),
            (InnovationCodebook::Sv10_32, 10),
            (InnovationCodebook::Sv20_32, 20),
        ];
        for (cb, sv) in cases {
            let row = sub_vector(cb, 0).expect("row 0 must exist");
            assert_eq!(row.len(), sv);
            // Cross-check the slice matches the underlying static slice.
            let expected: &[i16] = match cb {
                InnovationCodebook::Sv5_64 => &innovation_5_64()[0],
                InnovationCodebook::Sv5_256 => &innovation_5_256()[0],
                InnovationCodebook::Sv8_128 => &innovation_8_128()[0],
                InnovationCodebook::Sv10_16 => &innovation_10_16()[0],
                InnovationCodebook::Sv10_32 => &innovation_10_32()[0],
                InnovationCodebook::Sv20_32 => &innovation_20_32()[0],
            };
            assert_eq!(row, expected);
        }
    }

    #[test]
    fn sub_vector_rejects_out_of_range_index() {
        // Every codebook's `entries()` value is the first invalid
        // index (since indices run 0..entries).
        for cb in [
            InnovationCodebook::Sv5_64,
            InnovationCodebook::Sv5_256,
            InnovationCodebook::Sv8_128,
            InnovationCodebook::Sv10_16,
            InnovationCodebook::Sv10_32,
            InnovationCodebook::Sv20_32,
        ] {
            let last_valid = cb.entries() - 1;
            assert!(
                sub_vector(cb, last_valid).is_some(),
                "{cb:?} last valid index {last_valid} should resolve"
            );
            assert!(
                sub_vector(cb, cb.entries()).is_none(),
                "{cb:?} index == entries should be out of range"
            );
            assert!(sub_vector(cb, u32::MAX).is_none(), "{cb:?} u32::MAX");
        }
    }

    #[test]
    fn mapping_silence_for_mode_0() {
        let s = NarrowbandSubmode::for_id(0).unwrap();
        assert_eq!(InnovationMapping::for_mode(&s), InnovationMapping::Silence);
    }

    #[test]
    fn mapping_documented_for_mode_6_and_8() {
        let m6 = NarrowbandSubmode::for_id(6).unwrap();
        assert_eq!(
            InnovationMapping::for_mode(&m6),
            InnovationMapping::Documented {
                codebook: InnovationCodebook::Sv5_256,
                count: 8,
            }
        );
        let m8 = NarrowbandSubmode::for_id(8).unwrap();
        assert_eq!(
            InnovationMapping::for_mode(&m8),
            InnovationMapping::Documented {
                codebook: InnovationCodebook::Sv20_32,
                count: 2,
            }
        );
    }

    #[test]
    fn mapping_documented_for_modes_2_to_5_via_bitrate_annotations() {
        // r277: the staged per-codebook innovation bit-rate annotations
        // (`docs/audio/speex/tables/*.meta` role fields + tables README)
        // bind modes 2/3/4/5 uniquely — Table 9.1 "Innovation VQ"
        // bits/sub-frame × 200 = the annotated innovation bps.
        let cases: [(u8, InnovationCodebook, u8); 4] = [
            (2, InnovationCodebook::Sv10_16, 4), // 16 bits → 3200 bps
            (3, InnovationCodebook::Sv10_32, 4), // 20 bits → 4000 bps
            (4, InnovationCodebook::Sv8_128, 5), // 35 bits → 7000 bps
            (5, InnovationCodebook::Sv5_64, 8),  // 48 bits → 9600 bps
        ];
        for (mode_id, codebook, count) in cases {
            let s = NarrowbandSubmode::for_id(mode_id).unwrap();
            assert_eq!(
                InnovationMapping::for_mode(&s),
                InnovationMapping::Documented { codebook, count },
                "mode {mode_id}"
            );
        }
    }

    #[test]
    fn bitrate_annotation_arithmetic_pins_each_binding() {
        // The grounding arithmetic itself: bits/sub-frame × 4 sub-frames
        // per 20 ms frame = bits × 200 bps, and each documented binding's
        // codebook is annotated at exactly that innovation rate.
        let annotated_bps: [(InnovationCodebook, u32); 6] = [
            (InnovationCodebook::Sv10_16, 3200),
            (InnovationCodebook::Sv10_32, 4000),
            (InnovationCodebook::Sv8_128, 7000),
            (InnovationCodebook::Sv5_64, 9600),
            (InnovationCodebook::Sv5_256, 12800),
            (InnovationCodebook::Sv20_32, 2000),
        ];
        for s in NARROWBAND_SUBMODES {
            if let InnovationMapping::Documented { codebook, .. } = InnovationMapping::for_mode(&s)
            {
                let bps = u32::from(s.innovation_vq_bits) * 200;
                let annotated = annotated_bps
                    .iter()
                    .find(|(cb, _)| *cb == codebook)
                    .map(|&(_, b)| b)
                    .unwrap();
                assert_eq!(
                    bps, annotated,
                    "mode {}: Table 9.1 rate must match the staged annotation",
                    s.mode_id
                );
            }
        }
    }

    #[test]
    fn mapping_undocumented_for_modes_without_grounded_binding() {
        // Mode 1: 0-bit VQ field, vocoder excitation generation unstaged.
        // Mode 7: 96 bits → 19 200 bps, no staged codebook at that rate.
        for mode_id in [1u8, 7] {
            let s = NarrowbandSubmode::for_id(mode_id).unwrap();
            assert_eq!(
                InnovationMapping::for_mode(&s),
                InnovationMapping::Undocumented,
                "mode {mode_id}"
            );
        }
    }

    #[test]
    fn documented_mappings_satisfy_40_sample_constraint() {
        // Both documented mode dispatchers must yield exactly
        // sub_vector_len * count == 40 samples per sub-frame.
        for s in NARROWBAND_SUBMODES {
            if let InnovationMapping::Documented { codebook, count } =
                InnovationMapping::for_mode(&s)
            {
                assert_eq!(
                    codebook.sub_vector_len() * usize::from(count),
                    SUBFRAME_SAMPLES,
                    "mode {} sub-vector dispatch must cover 40 samples",
                    s.mode_id
                );
            }
        }
    }

    #[test]
    fn documented_mappings_match_bit_budget() {
        // Sum of per-sub-vector index widths times count must equal
        // the sub-mode's innovation_vq_bits.
        for s in NARROWBAND_SUBMODES {
            if let InnovationMapping::Documented { codebook, count } =
                InnovationMapping::for_mode(&s)
            {
                let total = u32::from(codebook.index_bits()) * u32::from(count);
                assert_eq!(
                    total,
                    u32::from(s.innovation_vq_bits),
                    "mode {} dispatch bits {total} != table {} bits",
                    s.mode_id,
                    s.innovation_vq_bits
                );
            }
        }
    }

    #[test]
    fn decode_subframe_silence_is_all_zero() {
        let s = NarrowbandSubmode::for_id(0).unwrap();
        let v = decode_subframe(&s, 0).unwrap();
        assert_eq!(v.len(), SUBFRAME_SAMPLES);
        assert!(v.iter().all(|&x| x == 0));
    }

    #[test]
    fn decode_subframe_undocumented_returns_error() {
        for mode_id in [1u8, 7] {
            let s = NarrowbandSubmode::for_id(mode_id).unwrap();
            let r = decode_subframe(&s, 0);
            assert_eq!(
                r,
                Err(InnovationError::Undocumented),
                "mode {mode_id} should report Undocumented"
            );
        }
    }

    #[test]
    fn decode_subframe_mode_5_eight_sub_vectors_concatenated() {
        // Mode 5: 8 × Sv5_64, MSB-first packed in a 48-bit field.
        let s = NarrowbandSubmode::for_id(5).unwrap();
        let indices: [u32; 8] = [0, 1, 2, 7, 13, 33, 50, 63];
        let mut packed: u128 = 0;
        for &idx in &indices {
            packed = (packed << 6) | u128::from(idx);
        }
        let v = decode_subframe(&s, packed).unwrap();
        assert_eq!(v.len(), SUBFRAME_SAMPLES);
        for (sv, &idx) in indices.iter().enumerate() {
            let row = &innovation_5_64()[idx as usize];
            assert_eq!(&v[sv * 5..sv * 5 + 5], &row[..], "sub-vector {sv}");
        }
    }

    #[test]
    fn decode_subframe_mode_4_five_sub_vectors_concatenated() {
        // Mode 4: 5 × Sv8_128, MSB-first packed in a 35-bit field.
        let s = NarrowbandSubmode::for_id(4).unwrap();
        let indices: [u32; 5] = [0, 1, 42, 100, 127];
        let mut packed: u128 = 0;
        for &idx in &indices {
            packed = (packed << 7) | u128::from(idx);
        }
        let v = decode_subframe(&s, packed).unwrap();
        for (sv, &idx) in indices.iter().enumerate() {
            let row = &innovation_8_128()[idx as usize];
            assert_eq!(&v[sv * 8..sv * 8 + 8], &row[..], "sub-vector {sv}");
        }
    }

    #[test]
    fn decode_subframe_mode_3_four_sub_vectors_concatenated() {
        // Mode 3: 4 × Sv10_32, MSB-first packed in a 20-bit field.
        let s = NarrowbandSubmode::for_id(3).unwrap();
        let indices: [u32; 4] = [3, 0, 17, 31];
        let mut packed: u128 = 0;
        for &idx in &indices {
            packed = (packed << 5) | u128::from(idx);
        }
        let v = decode_subframe(&s, packed).unwrap();
        for (sv, &idx) in indices.iter().enumerate() {
            let row = &innovation_10_32()[idx as usize];
            assert_eq!(&v[sv * 10..sv * 10 + 10], &row[..], "sub-vector {sv}");
        }
    }

    #[test]
    fn decode_subframe_mode_2_four_sub_vectors_concatenated() {
        // Mode 2: 4 × Sv10_16, MSB-first packed in a 16-bit field.
        let s = NarrowbandSubmode::for_id(2).unwrap();
        let indices: [u32; 4] = [15, 8, 1, 0];
        let mut packed: u128 = 0;
        for &idx in &indices {
            packed = (packed << 4) | u128::from(idx);
        }
        let v = decode_subframe(&s, packed).unwrap();
        for (sv, &idx) in indices.iter().enumerate() {
            let row = &innovation_10_16()[idx as usize];
            assert_eq!(&v[sv * 10..sv * 10 + 10], &row[..], "sub-vector {sv}");
        }
    }

    #[test]
    fn decode_subframe_mode_5_index_extraction_is_msb_first() {
        // Single non-zero sub-vector index at slot 0 of mode 5's 48-bit
        // field should land at samples 0..5.
        let s = NarrowbandSubmode::for_id(5).unwrap();
        let idx_sv0 = 33u128;
        let packed = idx_sv0 << (7 * 6);
        let v = decode_subframe(&s, packed).unwrap();
        let row = &innovation_5_64()[33];
        let row0 = &innovation_5_64()[0];
        assert_eq!(&v[0..5], &row[..]);
        for sv in 1..8 {
            assert_eq!(&v[sv * 5..sv * 5 + 5], &row0[..], "sub-vector {sv}");
        }
    }

    #[test]
    fn decode_subframe_modes_2_to_5_max_indices_all_resolve() {
        // All-ones packed fields (every per-sub-vector index at its
        // codebook maximum) must resolve for every newly bound mode.
        for mode_id in [2u8, 3, 4, 5] {
            let s = NarrowbandSubmode::for_id(mode_id).unwrap();
            let packed = (1u128 << s.innovation_vq_bits) - 1;
            let v = decode_subframe(&s, packed)
                .unwrap_or_else(|e| panic!("mode {mode_id} max indices: {e}"));
            assert_eq!(v.len(), SUBFRAME_SAMPLES);
        }
    }

    #[test]
    fn decode_subframe_mode_8_two_sub_vectors_concatenated() {
        // Mode 8: 2 × Sv20_32, MSB-first packed in a 10-bit field.
        // Pack sub-vector-0 index = 3 and sub-vector-1 index = 17.
        let s = NarrowbandSubmode::for_id(8).unwrap();
        let idx = (3u128 << 5) | 17u128;
        let v = decode_subframe(&s, idx).unwrap();
        assert_eq!(v.len(), SUBFRAME_SAMPLES);
        // First half should match row 3 of sv20-32, second half row 17.
        let row3 = &innovation_20_32()[3];
        let row17 = &innovation_20_32()[17];
        assert_eq!(&v[0..20], &row3[..]);
        assert_eq!(&v[20..40], &row17[..]);
    }

    #[test]
    fn decode_subframe_mode_6_eight_sub_vectors_concatenated() {
        // Mode 6: 8 × Sv5_256, MSB-first packed in a 64-bit field.
        // Build a deterministic per-sub-vector index sequence.
        let s = NarrowbandSubmode::for_id(6).unwrap();
        let indices: [u32; 8] = [0, 1, 2, 5, 13, 42, 100, 255];
        let mut packed: u128 = 0;
        for &idx in &indices {
            packed = (packed << 8) | u128::from(idx);
        }
        let v = decode_subframe(&s, packed).unwrap();
        assert_eq!(v.len(), SUBFRAME_SAMPLES);
        for (sv, &idx) in indices.iter().enumerate() {
            let row = &innovation_5_256()[idx as usize];
            assert_eq!(&v[sv * 5..sv * 5 + 5], &row[..], "sub-vector {sv}");
        }
    }

    #[test]
    fn decode_subframe_mode_6_index_extraction_is_msb_first() {
        // Single non-zero sub-vector at slot 0 should land at samples 0..5.
        let s = NarrowbandSubmode::for_id(6).unwrap();
        // Index for sub-vector 0 is in the top 8 bits of the 64-bit field.
        let idx_sv0 = 42u128;
        let packed = idx_sv0 << (7 * 8);
        let v = decode_subframe(&s, packed).unwrap();
        let row = &innovation_5_256()[42];
        assert_eq!(&v[0..5], &row[..]);
        // Sub-vectors 1..=7 should resolve to index 0 of sv5_256
        // (the all-zero row in the spec is row 0 by convention; we
        // don't assume it is zero, only that it matches row[0]).
        let row0 = &innovation_5_256()[0];
        for sv in 1..8 {
            assert_eq!(&v[sv * 5..sv * 5 + 5], &row0[..], "sub-vector {sv}");
        }
    }

    #[test]
    fn decode_subframe_mode_8_index_extraction_is_msb_first() {
        // Single non-zero sub-vector at slot 0 should land at samples 0..20.
        let s = NarrowbandSubmode::for_id(8).unwrap();
        let idx_sv0 = 7u128;
        let packed = idx_sv0 << 5;
        let v = decode_subframe(&s, packed).unwrap();
        let row7 = &innovation_20_32()[7];
        let row0 = &innovation_20_32()[0];
        assert_eq!(&v[0..20], &row7[..]);
        assert_eq!(&v[20..40], &row0[..]);
    }

    #[test]
    fn decode_subframe_mode_6_max_indices_all_resolve() {
        // 8 × index = 255 (max for sv5_256) — must resolve successfully.
        let s = NarrowbandSubmode::for_id(6).unwrap();
        let mut packed = 0u128;
        for _ in 0..8 {
            packed = (packed << 8) | 255u128;
        }
        let v = decode_subframe(&s, packed).unwrap();
        let max_row = &innovation_5_256()[255];
        for sv in 0..8 {
            assert_eq!(&v[sv * 5..sv * 5 + 5], &max_row[..]);
        }
    }

    #[test]
    fn decode_subframe_mode_8_max_indices_all_resolve() {
        let s = NarrowbandSubmode::for_id(8).unwrap();
        let packed = (31u128 << 5) | 31u128;
        let v = decode_subframe(&s, packed).unwrap();
        let max_row = &innovation_20_32()[31];
        assert_eq!(&v[0..20], &max_row[..]);
        assert_eq!(&v[20..40], &max_row[..]);
    }

    #[test]
    fn codebook_index_bits_match_powers_of_two() {
        for cb in [
            InnovationCodebook::Sv5_64,
            InnovationCodebook::Sv5_256,
            InnovationCodebook::Sv8_128,
            InnovationCodebook::Sv10_16,
            InnovationCodebook::Sv10_32,
            InnovationCodebook::Sv20_32,
        ] {
            assert_eq!(1u32 << cb.index_bits(), cb.entries());
        }
    }
}
