//! Wideband high-band **innovation sub-vector lookup + per-mode
//! dispatcher** (round r230 scope).
//!
//! Mirrors r220's narrowband [`crate::innovation`] path for the
//! sub-band-CELP high band: wires the two staged high-band fixed-codebook
//! (excitation) tables in [`crate::codebooks`] into typed `&'static [i16]`
//! slices keyed by a codebook-shape selector + a raw sub-vector index,
//! and binds each documented high-band sub-mode (Table 10.1, modes
//! 0..=4) to a per-sub-frame codebook dispatch.
//!
//! ## Spec basis
//!
//! In-tree clean-room sources only:
//!
//! * `docs/audio/speex/speex-manual.pdf` §10.3 — *"the encoding of the
//!   high-band excitation is done in a way similar to that of the
//!   narrowband innovation."*
//! * `docs/audio/speex/speex-manual.pdf` §10.4 / Table 10.1 — high-band
//!   excitation VQ widths per sub-mode: 0 / 0 / 20 / 40 / 80 bits per
//!   sub-frame for modes 0..=4 respectively.
//! * `docs/audio/speex/speex-celp-companion.md` §9 / `tables/README.md`
//!   — the two staged high-band excitation codebook shapes:
//!     - `hb-innovation-cdbk-sv8-128`: 128 × 8 (7-bit + sign,
//!       8 000 bps high-band excitation)
//!     - `hb-innovation-cdbk-sv10-32`: 32 × 10 (5-bit, 4 000 bps
//!       high-band excitation)
//!
//! Each high-band sub-frame carries 40 samples (the QMF splits the
//! 16 kHz wideband signal into two 8 kHz half-bands and each half is
//! processed by a narrowband-style CELP at the same 20 ms cadence of
//! 4 sub-frames × 40 samples — see [`crate::wideband`] module docs).
//!
//! ## Per-mode dispatcher
//!
//! Two documented high-band sub-modes carry an excitation-VQ field
//! with a bit budget that is **uniquely pinned** by the published
//! codebook inventory plus the 40-sample sub-frame constraint:
//!
//! | Mode | excitation_vq_bits | Dispatch                                                  |
//! | ---: | -----------------: | --------------------------------------------------------- |
//! |    0 |                  0 | [`HbInnovationMapping::Silence`] (no VQ field)            |
//! |    1 |                  0 | [`HbInnovationMapping::Silence`] (gain-only, no VQ field) |
//! |    2 |                 20 | 4 × `HbSv10_32` (4 × 5-bit indices × 10 samples)          |
//! |    3 |                 40 | 5 × `HbSv8_128` (5 × (7+1) bits × 8 samples)              |
//! |    4 |                 80 | two-stage 5 × `HbSv8_128`, stage 2 at weight 0.4          |
//!
//! Derivations:
//!
//! * Mode 2 = 20 bits / sub-frame. With `HbSv10_32` (5-bit index,
//!   10-sample sub-vector): `4 × 5 = 20` bits ✓, `4 × 10 = 40` samples ✓.
//!   `HbSv8_128` (8-bit composite width, 8-sample sub-vector) would need
//!   `20 / 8 = 2.5` sub-vectors per sub-frame — not an integer, ruled out.
//!   So mode 2 is uniquely `4 × HbSv10_32`.
//! * Mode 3 = 40 bits / sub-frame. With `HbSv8_128` (7-bit index plus
//!   a 1-bit sign per sub-vector = 8-bit composite, 8-sample sub-vector):
//!   `5 × 8 = 40` bits ✓, `5 × 8 = 40` samples ✓. `HbSv10_32` would
//!   yield `8 × 5 = 40` bits but `8 × 10 = 80` samples — overshoots
//!   the 40-sample sub-frame, ruled out. So mode 3 is uniquely
//!   `5 × HbSv8_128`.
//! * Mode 4 = 80 bits / sub-frame = **two stages** of the mode-3
//!   geometry (2 × 5 × 8 bits) over the same five 8-sample slots, stage
//!   2 added at weight 0.4 — pinned by
//!   `docs/audio/speex/hb-innovation-binding.md` §1/§2 (bit-flip
//!   staircase + codebook-prediction + the `‖d2‖/‖d1‖ = 0.4` stage
//!   test). Within an 8-bit group the leading (MSB) bit is a sign and
//!   the low 7 bits are the `sv8-128` index. See
//!   [`decode_hb_subframe_mode4_f32`]. A residual on the absolute
//!   per-frame gain/energy law remains (crate README).
//!
//! ## Sign bit (only `HbSv8_128`)
//!
//! Per the staged `docs/audio/speex/hb-innovation-binding.md` §1, each
//! 8-bit `HbSv8_128` group is, in wire order, a **leading 1-bit
//! polarity sign followed by the 7-bit codebook row index**:
//!
//! ```text
//!  bit 0    : sign   (the codeword or its negation)
//!  bits 1-7 : index into hb-innovation-cdbk-sv8-128  (0 .. 127)
//! ```
//!
//! The §2.2 pair-sum measurement pins the order: `out(V) + out(V ^
//! 0x80)` is the same signal for every group value `V` (to ≤ 1 output
//! LSB), i.e. flipping the group's **most-significant** bit negates the
//! contribution — the MSB is the sign, for **both** modes 3 and 4
//! ("Both modes 3 and 4 give the same result"). When the sign bit is
//! `1` the resolved 8-sample sub-vector is negated element-wise. The
//! 5-bit `HbSv10_32` shape has no sign bit.
//!
//! (Rounds ≤ r438 read mode 3's group as `[7-bit index][sign]` from the
//! staged `tables/README.md` "7-bit + sign" shorthand; the binding
//! doc's measured staircase supersedes that reading, and provenance/08
//! re-confirms the §1 layout from staged bytes alone at median
//! |ρ| ≈ 0.90 against the recovered high-band excitation.)
//!
//! ## What this module does NOT do
//!
//! * No fixed-codebook gain scaling. The §10.3 high-band fixed-codebook
//!   gain `g_frame × g_subf` consumes the separate per-sub-frame
//!   excitation_gain field already surfaced by
//!   [`crate::HighBandSubFrameIndices::excitation_gain_index`];
//!   applying the gain to the looked-up sub-vector samples needs the
//!   gain reconstruction (a separate later round).
//! * No excitation buffer composition. The final high-band excitation
//!   in §10.3 also needs the adaptive-codebook contribution — but
//!   §10.2 states *"no pitch prediction is used in the high band"*,
//!   so the high-band path is simpler than the narrowband
//!   `e[n] = p[n] + c[n]`. The per-sub-frame `c[n]` (already gain-
//!   scaled) is the entire high-band excitation. Composition lands
//!   alongside the gain step.
//! * No encoder-side codebook search.
//!
//! ## Module structure
//!
//! Mirrors r220's narrowband [`crate::innovation`] layout: a shape
//! selector ([`HbInnovationCodebook`]), a row-lookup primitive
//! ([`hb_innovation_sub_vector`]), a per-mode dispatcher
//! ([`HbInnovationMapping::for_mode`]), and a per-sub-frame decoder
//! ([`decode_hb_subframe`]) that walks the packed excitation-VQ index
//! MSB-first and concatenates the resolved sub-vectors into a single
//! 40-sample `[i16; 40]` vector.

use crate::codebooks::{hb_innovation_10_32, hb_innovation_8_128};
use crate::wideband::WidebandHighBandSubmode;

/// Selector for one of the two documented high-band innovation codebook
/// shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HbInnovationCodebook {
    /// 8-sample sub-vector, 7-bit index + 1-bit sign per lookup, 128
    /// entries (8 000 bps high-band excitation).
    HbSv8_128,
    /// 10-sample sub-vector, 5-bit index, 32 entries (4 000 bps
    /// high-band excitation).
    HbSv10_32,
}

impl HbInnovationCodebook {
    /// Samples per sub-vector.
    pub const fn sub_vector_len(self) -> usize {
        match self {
            HbInnovationCodebook::HbSv8_128 => 8,
            HbInnovationCodebook::HbSv10_32 => 10,
        }
    }

    /// Codebook-row index width on the wire (not including the sign
    /// bit, when present).
    pub const fn index_bits(self) -> u8 {
        match self {
            HbInnovationCodebook::HbSv8_128 => 7,
            HbInnovationCodebook::HbSv10_32 => 5,
        }
    }

    /// `true` if a 1-bit sign field accompanies every index lookup
    /// (only [`HbInnovationCodebook::HbSv8_128`]).
    pub const fn has_sign_bit(self) -> bool {
        matches!(self, HbInnovationCodebook::HbSv8_128)
    }

    /// Total wire width per sub-vector lookup, including the sign bit
    /// when present.
    pub const fn slot_bits(self) -> u8 {
        if self.has_sign_bit() {
            self.index_bits() + 1
        } else {
            self.index_bits()
        }
    }

    /// Total number of entries (= `1 << index_bits()`).
    pub const fn entries(self) -> u32 {
        1u32 << self.index_bits()
    }
}

/// Number of samples in a Speex wideband high-band sub-frame.
///
/// Same as the narrowband sub-frame size — the QMF analysis filter
/// splits the 16 kHz wideband signal into two 8 kHz half-bands and
/// each half is processed by a narrowband-style CELP at the same
/// 20 ms / 4-sub-frame cadence (Manual §10 / §10.1).
pub const HB_SUBFRAME_SAMPLES: usize = 40;

/// Resolved one-sub-vector lookup result for the high-band codebooks.
///
/// The slice's length is the codebook's
/// [`HbInnovationCodebook::sub_vector_len`] (8 or 10 depending on the
/// codebook). Each `i16` is the raw stored sample from the staged
/// codebook CSV — no sign-bit polarity has been applied (the sign is
/// applied separately in [`decode_hb_subframe`]), and no gain scaling
/// has been applied.
pub fn hb_innovation_sub_vector(
    codebook: HbInnovationCodebook,
    index: u32,
) -> Option<&'static [i16]> {
    let idx = index as usize;
    match codebook {
        HbInnovationCodebook::HbSv8_128 => {
            let table = hb_innovation_8_128();
            table.get(idx).map(|row| row.as_slice())
        }
        HbInnovationCodebook::HbSv10_32 => {
            let table = hb_innovation_10_32();
            table.get(idx).map(|row| row.as_slice())
        }
    }
}

/// Stage-2 refinement weight of the high-band **mode-4** two-stage
/// innovation (`docs/audio/speex/hb-innovation-binding.md` §1 / §2.4:
/// `‖d2‖/‖d1‖ = 0.4000`, `cos(d1,d2) = 1`). Mode 4 is mode 3 plus a
/// 0.4-weighted second pass of the *same* `sv8-128` table over the same
/// five 8-sample slots.
pub const HB_MODE4_STAGE2_WEIGHT: f32 = 0.4;

/// Decode the **mode-4** (80-bit, two-stage) high-band innovation
/// sub-frame off the raw `excitation_vq_index` field, returning the raw
/// codebook-domain shape as `[f32; 40]` (sign-applied, stage-2 added at
/// [`HB_MODE4_STAGE2_WEIGHT`]) — *before* the shared
/// `INNOVATION_CODEBOOK_SCALE` / gain scaling the caller applies.
///
/// Per `docs/audio/speex/hb-innovation-binding.md` §1/§2 the 80-bit
/// field is **two stages × five 8-bit groups**, each group an
/// `sv8-128` sub-vector (8 samples). Within an 8-bit group the leading
/// (most-significant) bit is a **sign** and the low 7 bits are the
/// 0..127 codebook index (the §2.2 pair-sum test: flipping `0x80`
/// negates the sub-vector). Stage 1 is the first five groups; stage 2
/// the next five, laid over the same five slots and added at weight
/// 0.4:
///
/// ```text
/// inn[40] = concat_k s1_k·cdbk8[i1_k] + 0.4·concat_k s2_k·cdbk8[i2_k]
/// ```
pub fn decode_hb_subframe_mode4_f32(excitation_vq_index: u128) -> [f32; HB_SUBFRAME_SAMPLES] {
    // 10 groups of 8 bits, MSB-first: groups 0..4 = stage 1, 5..9 = stage 2.
    let group = |j: u32| -> (u32, bool) {
        let slot = ((excitation_vq_index >> ((9 - j) * 8)) & 0xFF) as u32;
        let sign = (slot >> 7) & 1 == 1;
        (slot & 0x7F, sign)
    };
    let table = hb_innovation_8_128();
    let mut out = [0.0f32; HB_SUBFRAME_SAMPLES];
    for k in 0..5u32 {
        let (i1, s1) = group(k);
        let (i2, s2) = group(k + 5);
        let row1 = &table[i1 as usize];
        let row2 = &table[i2 as usize];
        let sgn1 = if s1 { -1.0f32 } else { 1.0 };
        let sgn2 = if s2 { -1.0f32 } else { 1.0 };
        let base = (k as usize) * 8;
        for m in 0..8usize {
            out[base + m] =
                sgn1 * f32::from(row1[m]) + HB_MODE4_STAGE2_WEIGHT * sgn2 * f32::from(row2[m]);
        }
    }
    out
}

/// Per-mode high-band innovation dispatch resolved from the staged
/// material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HbInnovationMapping {
    /// No excitation-VQ field on the wire (modes 0 and 1; high-band
    /// excitation is then implicitly zero / gain-only per Table 10.1's
    /// excitation_vq_bits = 0).
    Silence,
    /// Documented per-mode binding: each sub-frame's 40-sample
    /// excitation comes from `count` concatenated lookups against
    /// `codebook`, taking successive `codebook.slot_bits()`-wide
    /// slots off the `excitation_vq_index` field in
    /// most-significant-bit-first order (the same order the parser
    /// [`crate::WidebandHighBandBody::parse`] read them into the
    /// `u128`). When `codebook.has_sign_bit()` is `true`, each slot is
    /// a leading 1-bit sign followed by the 7-bit index
    /// (`hb-innovation-binding.md` §1), and a sign of `1` negates the
    /// looked-up sub-vector element-wise.
    Documented {
        /// Which of the two codebook shapes is used.
        codebook: HbInnovationCodebook,
        /// How many sub-vectors are concatenated per 40-sample
        /// sub-frame (must satisfy
        /// `codebook.sub_vector_len() * count == HB_SUBFRAME_SAMPLES`).
        count: u8,
    },
    /// The per-mode codebook binding is not uniquely pinned by the
    /// staged material — the bit budget plus the
    /// 40-sample-per-sub-frame constraint plus the two available
    /// codebook shapes do not yield a unique decomposition. Recorded
    /// in the README as a docs gap.
    Undocumented,
}

impl HbInnovationMapping {
    /// Resolve the per-mode high-band innovation dispatcher from a
    /// high-band sub-mode.
    ///
    /// Returns [`HbInnovationMapping::Silence`] for modes 0 and 1
    /// (no excitation-VQ field on the wire),
    /// [`HbInnovationMapping::Documented`] for modes 2 and 3 (whose
    /// codebook binding is uniquely pinned by the staged inventory),
    /// and [`HbInnovationMapping::Undocumented`] for mode 4.
    pub const fn for_mode(submode: &WidebandHighBandSubmode) -> Self {
        // Dispatch by the documented per-sub-frame excitation_vq_bits
        // budget (Table 10.1) rather than the mode_id directly — keeps
        // the dispatcher closure-of-arithmetic over the published
        // codebook inventory, and is the only consistent reading
        // without assuming material outside the staged tables.
        match submode.excitation_vq_bits {
            0 => HbInnovationMapping::Silence,
            20 => HbInnovationMapping::Documented {
                codebook: HbInnovationCodebook::HbSv10_32,
                count: 4,
            },
            40 => HbInnovationMapping::Documented {
                codebook: HbInnovationCodebook::HbSv8_128,
                count: 5,
            },
            _ => HbInnovationMapping::Undocumented,
        }
    }
}

/// Errors from [`decode_hb_subframe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HbInnovationError {
    /// The per-mode codebook binding is not documented in the staged
    /// material. See [`HbInnovationMapping::Undocumented`].
    Undocumented,
    /// A per-sub-vector index extracted from the `excitation_vq_index`
    /// field exceeded the codebook entry count (only possible with a
    /// hand-built [`crate::HighBandSubFrameIndices`]; the parser caps
    /// the field to the sub-mode's documented width).
    IndexOutOfRange { sub_vector: u8, index: u32 },
}

impl core::fmt::Display for HbInnovationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HbInnovationError::Undocumented => write!(
                f,
                "hb-innovation dispatcher: per-mode codebook binding not in staged docs"
            ),
            HbInnovationError::IndexOutOfRange { sub_vector, index } => write!(
                f,
                "hb-innovation dispatcher: sub-vector {sub_vector} index {index} out of codebook range"
            ),
        }
    }
}

impl std::error::Error for HbInnovationError {}

/// Decode the 40-sample fixed-codebook sub-vector for a single CELP
/// high-band sub-frame from the raw `excitation_vq_index` field.
///
/// Walks `count` successive `slot_bits`-wide chunks off the top of
/// the packed `excitation_vq_index` (MSB-first — the same order the
/// parser read them into the `u128`), splits each slot into its
/// optional leading 1-bit sign + `index_bits` index
/// (`hb-innovation-binding.md` §1), looks up each index against
/// `codebook`, applies the sign, and concatenates the resulting
/// sub-vectors into a single 40-element vector.
///
/// Returns [`HbInnovationError::Undocumented`] for sub-modes whose
/// dispatcher is not bound (see [`HbInnovationMapping::Undocumented`]);
/// returns [`HbInnovationError::IndexOutOfRange`] for a malformed
/// index.
///
/// For modes 0 / 1 (silence / gain-only), the all-zero 40-sample
/// vector is returned — no excitation-VQ field is transmitted and the
/// fixed-codebook contribution is implicitly zero per Table 10.1's
/// excitation_vq_bits = 0.
pub fn decode_hb_subframe(
    submode: &WidebandHighBandSubmode,
    excitation_vq_index: u128,
) -> Result<[i16; HB_SUBFRAME_SAMPLES], HbInnovationError> {
    let mapping = HbInnovationMapping::for_mode(submode);
    match mapping {
        HbInnovationMapping::Silence => Ok([0i16; HB_SUBFRAME_SAMPLES]),
        HbInnovationMapping::Undocumented => Err(HbInnovationError::Undocumented),
        HbInnovationMapping::Documented { codebook, count } => {
            let sv_len = codebook.sub_vector_len();
            let slot_bits = u32::from(codebook.slot_bits());
            let index_bits = u32::from(codebook.index_bits());
            let total_bits = slot_bits * u32::from(count);
            debug_assert!(
                total_bits as usize <= 128,
                "hb innovation total bits exceeds u128 capacity"
            );
            debug_assert_eq!(
                sv_len * usize::from(count),
                HB_SUBFRAME_SAMPLES,
                "hb innovation dispatch sub-vectors must cover 40 samples"
            );

            let slot_mask = if slot_bits == 0 {
                0
            } else {
                (1u128 << slot_bits) - 1
            };
            let index_mask = if index_bits == 0 {
                0
            } else {
                (1u128 << index_bits) - 1
            };
            let has_sign = codebook.has_sign_bit();
            let mut out = [0i16; HB_SUBFRAME_SAMPLES];
            for sv in 0..count {
                let shift = u32::from(count - 1 - sv) * slot_bits;
                let slot = (excitation_vq_index >> shift) & slot_mask;
                let (idx, sign) = if has_sign {
                    // Leading (most-significant) bit of the slot is the
                    // sign; the low `index_bits` are the codebook index —
                    // `hb-innovation-binding.md` §1 (§2.2's `V ^ 0x80`
                    // pair-sum pins the MSB as the sign for modes 3 and 4
                    // alike).
                    let i = slot & index_mask;
                    let s = (slot >> index_bits) & 1 != 0;
                    (i as u32, s)
                } else {
                    ((slot & index_mask) as u32, false)
                };
                let row = hb_innovation_sub_vector(codebook, idx).ok_or(
                    HbInnovationError::IndexOutOfRange {
                        sub_vector: sv,
                        index: idx,
                    },
                )?;
                let base = usize::from(sv) * sv_len;
                if sign {
                    for (out_sample, &src) in out[base..base + sv_len].iter_mut().zip(row.iter()) {
                        // wrapping_neg gives the same result as `-` for
                        // any in-range `i16` (codebook entries are
                        // `signed char` values, well inside the i16
                        // range) while sidestepping the i16::MIN
                        // edge-case lint.
                        *out_sample = src.wrapping_neg();
                    }
                } else {
                    out[base..base + sv_len].copy_from_slice(row);
                }
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebooks::{hb_innovation_10_32, hb_innovation_8_128};
    use crate::wideband::WIDEBAND_HIGH_BAND_SUBMODES;

    // ---- Codebook-shape primitive ----

    #[test]
    fn sub_vector_dimensions_match_codebook_meta() {
        let cases: [(HbInnovationCodebook, usize, u8, u32, bool, u8); 2] = [
            (HbInnovationCodebook::HbSv8_128, 8, 7, 128, true, 8),
            (HbInnovationCodebook::HbSv10_32, 10, 5, 32, false, 5),
        ];
        for (cb, sv, bits, ent, sign, slot) in cases {
            assert_eq!(cb.sub_vector_len(), sv, "{cb:?} sub_vector_len");
            assert_eq!(cb.index_bits(), bits, "{cb:?} index_bits");
            assert_eq!(cb.entries(), ent, "{cb:?} entries");
            assert_eq!(cb.has_sign_bit(), sign, "{cb:?} has_sign_bit");
            assert_eq!(cb.slot_bits(), slot, "{cb:?} slot_bits");
        }
    }

    #[test]
    fn sub_vector_resolves_first_row_of_every_codebook() {
        let row_8_128 = hb_innovation_sub_vector(HbInnovationCodebook::HbSv8_128, 0).unwrap();
        assert_eq!(row_8_128.len(), 8);
        assert_eq!(row_8_128, &hb_innovation_8_128()[0]);

        let row_10_32 = hb_innovation_sub_vector(HbInnovationCodebook::HbSv10_32, 0).unwrap();
        assert_eq!(row_10_32.len(), 10);
        assert_eq!(row_10_32, &hb_innovation_10_32()[0]);
    }

    #[test]
    fn sub_vector_rejects_out_of_range_index() {
        for cb in [
            HbInnovationCodebook::HbSv8_128,
            HbInnovationCodebook::HbSv10_32,
        ] {
            let last_valid = cb.entries() - 1;
            assert!(
                hb_innovation_sub_vector(cb, last_valid).is_some(),
                "{cb:?} last valid index {last_valid}"
            );
            assert!(
                hb_innovation_sub_vector(cb, cb.entries()).is_none(),
                "{cb:?} index == entries should be out of range"
            );
            assert!(
                hb_innovation_sub_vector(cb, u32::MAX).is_none(),
                "{cb:?} u32::MAX"
            );
        }
    }

    #[test]
    fn codebook_index_bits_match_powers_of_two() {
        for cb in [
            HbInnovationCodebook::HbSv8_128,
            HbInnovationCodebook::HbSv10_32,
        ] {
            assert_eq!(1u32 << cb.index_bits(), cb.entries());
        }
    }

    // ---- Per-mode dispatcher ----

    #[test]
    fn mapping_silence_for_modes_0_and_1() {
        for mode_id in [0u8, 1] {
            let s = WidebandHighBandSubmode::for_id(mode_id).unwrap();
            assert_eq!(
                HbInnovationMapping::for_mode(&s),
                HbInnovationMapping::Silence,
                "mode {mode_id}"
            );
        }
    }

    #[test]
    fn mapping_documented_for_mode_2() {
        let s = WidebandHighBandSubmode::for_id(2).unwrap();
        assert_eq!(
            HbInnovationMapping::for_mode(&s),
            HbInnovationMapping::Documented {
                codebook: HbInnovationCodebook::HbSv10_32,
                count: 4,
            }
        );
    }

    #[test]
    fn mapping_documented_for_mode_3() {
        let s = WidebandHighBandSubmode::for_id(3).unwrap();
        assert_eq!(
            HbInnovationMapping::for_mode(&s),
            HbInnovationMapping::Documented {
                codebook: HbInnovationCodebook::HbSv8_128,
                count: 5,
            }
        );
    }

    #[test]
    fn mapping_undocumented_for_mode_4() {
        let s = WidebandHighBandSubmode::for_id(4).unwrap();
        assert_eq!(
            HbInnovationMapping::for_mode(&s),
            HbInnovationMapping::Undocumented
        );
    }

    #[test]
    fn documented_mappings_satisfy_40_sample_constraint() {
        for s in WIDEBAND_HIGH_BAND_SUBMODES {
            if let HbInnovationMapping::Documented { codebook, count } =
                HbInnovationMapping::for_mode(&s)
            {
                assert_eq!(
                    codebook.sub_vector_len() * usize::from(count),
                    HB_SUBFRAME_SAMPLES,
                    "mode {} sub-vector dispatch must cover 40 samples",
                    s.mode_id
                );
            }
        }
    }

    #[test]
    fn documented_mappings_match_bit_budget() {
        // Sum of per-sub-vector slot widths × count must equal the
        // sub-mode's excitation_vq_bits.
        for s in WIDEBAND_HIGH_BAND_SUBMODES {
            if let HbInnovationMapping::Documented { codebook, count } =
                HbInnovationMapping::for_mode(&s)
            {
                let total = u32::from(codebook.slot_bits()) * u32::from(count);
                assert_eq!(
                    total,
                    u32::from(s.excitation_vq_bits),
                    "mode {} dispatch bits {total} != table {} bits",
                    s.mode_id,
                    s.excitation_vq_bits
                );
            }
        }
    }

    // ---- Per-sub-frame decode ----

    #[test]
    fn decode_hb_subframe_silence_is_all_zero() {
        for mode_id in [0u8, 1] {
            let s = WidebandHighBandSubmode::for_id(mode_id).unwrap();
            let v = decode_hb_subframe(&s, 0).unwrap();
            assert_eq!(v.len(), HB_SUBFRAME_SAMPLES);
            assert!(v.iter().all(|&x| x == 0), "mode {mode_id}");
        }
    }

    #[test]
    fn decode_hb_subframe_mode_4_returns_undocumented() {
        let s = WidebandHighBandSubmode::for_id(4).unwrap();
        let r = decode_hb_subframe(&s, 0);
        assert_eq!(r, Err(HbInnovationError::Undocumented));
    }

    #[test]
    fn decode_hb_subframe_mode_2_four_sub_vectors_concatenated() {
        // Mode 2: 4 × HbSv10_32, MSB-first packed in a 20-bit field.
        let s = WidebandHighBandSubmode::for_id(2).unwrap();
        let indices: [u32; 4] = [3, 17, 0, 31];
        let mut packed: u128 = 0;
        for &idx in &indices {
            packed = (packed << 5) | u128::from(idx);
        }
        let v = decode_hb_subframe(&s, packed).unwrap();
        assert_eq!(v.len(), HB_SUBFRAME_SAMPLES);
        for (sv, &idx) in indices.iter().enumerate() {
            let row = &hb_innovation_10_32()[idx as usize];
            assert_eq!(&v[sv * 10..sv * 10 + 10], &row[..], "sub-vector {sv}");
        }
    }

    #[test]
    fn decode_hb_subframe_mode_2_index_extraction_is_msb_first() {
        // Single non-zero sub-vector at slot 0 should land at samples 0..10.
        let s = WidebandHighBandSubmode::for_id(2).unwrap();
        let idx_sv0 = 7u128;
        let packed = idx_sv0 << (3 * 5);
        let v = decode_hb_subframe(&s, packed).unwrap();
        let row7 = &hb_innovation_10_32()[7];
        let row0 = &hb_innovation_10_32()[0];
        assert_eq!(&v[0..10], &row7[..]);
        for sv in 1..4 {
            assert_eq!(&v[sv * 10..sv * 10 + 10], &row0[..], "sub-vector {sv}");
        }
    }

    #[test]
    fn decode_hb_subframe_mode_2_max_indices_all_resolve() {
        // 4 × index = 31 (max for HbSv10_32) — must resolve successfully.
        let s = WidebandHighBandSubmode::for_id(2).unwrap();
        let mut packed = 0u128;
        for _ in 0..4 {
            packed = (packed << 5) | 31u128;
        }
        let v = decode_hb_subframe(&s, packed).unwrap();
        let max_row = &hb_innovation_10_32()[31];
        for sv in 0..4 {
            assert_eq!(&v[sv * 10..sv * 10 + 10], &max_row[..]);
        }
    }

    #[test]
    fn decode_hb_subframe_mode_3_five_sub_vectors_concatenated_no_sign() {
        // Mode 3: 5 × HbSv8_128 with sign bit cleared, MSB-first packed
        // in a 40-bit field (5 × 8 = 40 bits per sub-frame).
        let s = WidebandHighBandSubmode::for_id(3).unwrap();
        let indices: [u32; 5] = [0, 1, 2, 5, 127];
        // Each slot is [sign=0][7-bit index] (binding §1) — sign cleared.
        let mut packed: u128 = 0;
        for &idx in &indices {
            let slot = u128::from(idx);
            packed = (packed << 8) | slot;
        }
        let v = decode_hb_subframe(&s, packed).unwrap();
        assert_eq!(v.len(), HB_SUBFRAME_SAMPLES);
        for (sv, &idx) in indices.iter().enumerate() {
            let row = &hb_innovation_8_128()[idx as usize];
            assert_eq!(&v[sv * 8..sv * 8 + 8], &row[..], "sub-vector {sv}");
        }
    }

    #[test]
    fn decode_hb_subframe_mode_3_sign_bit_negates_sub_vector() {
        // Mode 3 with sign bit set on every slot — every sub-vector
        // sample should be negated element-wise.
        let s = WidebandHighBandSubmode::for_id(3).unwrap();
        let indices: [u32; 5] = [1, 2, 3, 4, 5];
        let mut packed: u128 = 0;
        for &idx in &indices {
            // [sign=1][7-bit index] → leading sign bit set (0x80).
            let slot = 0x80u128 | u128::from(idx);
            packed = (packed << 8) | slot;
        }
        let v = decode_hb_subframe(&s, packed).unwrap();
        for (sv, &idx) in indices.iter().enumerate() {
            let row = &hb_innovation_8_128()[idx as usize];
            for (k, &expect_positive) in row.iter().enumerate() {
                assert_eq!(
                    v[sv * 8 + k],
                    expect_positive.wrapping_neg(),
                    "sub-vector {sv} sample {k}"
                );
            }
        }
    }

    #[test]
    fn decode_hb_subframe_mode_3_mixed_sign_bits() {
        // Mode 3 with sign bits in a known pattern (alternating).
        let s = WidebandHighBandSubmode::for_id(3).unwrap();
        let indices: [u32; 5] = [10, 20, 30, 40, 50];
        let signs: [u32; 5] = [0, 1, 0, 1, 0];
        let mut packed: u128 = 0;
        for k in 0..5 {
            let slot = (u128::from(signs[k]) << 7) | u128::from(indices[k]);
            packed = (packed << 8) | slot;
        }
        let v = decode_hb_subframe(&s, packed).unwrap();
        for sv in 0..5 {
            let row = &hb_innovation_8_128()[indices[sv] as usize];
            let negated = signs[sv] == 1;
            for k in 0..8 {
                let want = if negated {
                    row[k].wrapping_neg()
                } else {
                    row[k]
                };
                assert_eq!(v[sv * 8 + k], want, "sub-vector {sv} sample {k}");
            }
        }
    }

    #[test]
    fn decode_hb_subframe_mode_3_max_index_with_sign_set_resolves() {
        // Mode 3, every slot at index 127 with sign set — must resolve
        // and every sample must be negated.
        let s = WidebandHighBandSubmode::for_id(3).unwrap();
        let mut packed: u128 = 0;
        for _ in 0..5 {
            let slot = 0x80u128 | 127u128;
            packed = (packed << 8) | slot;
        }
        let v = decode_hb_subframe(&s, packed).unwrap();
        let max_row = &hb_innovation_8_128()[127];
        for sv in 0..5 {
            for k in 0..8 {
                assert_eq!(v[sv * 8 + k], max_row[k].wrapping_neg());
            }
        }
    }

    /// The binding doc's §2.2 pair-sum measurement, restated on the
    /// crate: for any 8-bit group value `V`, the contributions of `V`
    /// and `V ^ 0x80` are exact negations (flipping the **leading** bit
    /// of the group flips only the polarity), for mode 3 and for both
    /// mode-4 stages — the group layout is `[sign][7-bit index]`.
    #[test]
    fn group_msb_flip_negates_contribution_modes_3_and_4() {
        let s3 = WidebandHighBandSubmode::for_id(3).unwrap();
        for v in [0u128, 7, 33, 90, 127, 0x85, 0xFF] {
            // Mode 3: place the group in sub-vector 0, rest zero.
            let a = decode_hb_subframe(&s3, v << 32).unwrap();
            let b = decode_hb_subframe(&s3, (v ^ 0x80) << 32).unwrap();
            for k in 0..8 {
                assert_eq!(a[k], b[k].wrapping_neg(), "mode 3 V={v:#x} k={k}");
            }
            // Mode 4 stage 1 (top group) and stage 2 (sixth group).
            // Park the *other* stage's slot-0 group on index 43 — the
            // unique all-zero row — so the probed group's contribution
            // is isolated within samples 0..8.
            let zero = 43u128;
            let a1 = decode_hb_subframe_mode4_f32((v << 72) | (zero << 32));
            let b1 = decode_hb_subframe_mode4_f32(((v ^ 0x80) << 72) | (zero << 32));
            let a2 = decode_hb_subframe_mode4_f32((zero << 72) | (v << 32));
            let b2 = decode_hb_subframe_mode4_f32((zero << 72) | ((v ^ 0x80) << 32));
            for k in 0..8 {
                assert_eq!(a1[k], -b1[k], "mode 4 stage 1 V={v:#x} k={k}");
                assert_eq!(a2[k], -b2[k], "mode 4 stage 2 V={v:#x} k={k}");
            }
        }
    }

    /// Mode 3's group split equals mode 4's stage-1 group split — the
    /// same `[sign][index]` reading serves both, per binding §2.2
    /// ("Both modes 3 and 4 give the same result").
    #[test]
    fn mode_3_group_split_matches_mode_4_stage_1() {
        let s3 = WidebandHighBandSubmode::for_id(3).unwrap();
        for v in [1u128, 0x2A, 0x80, 0xAB, 0x7F, 0xFF] {
            let m3 = decode_hb_subframe(&s3, v << 32).unwrap();
            // Mode 4 with stage-1 group 0 = V and stage-2 group 0 parked
            // on the zero row (43): the first 8 samples are the pure
            // stage-1 group contribution.
            let m4 = decode_hb_subframe_mode4_f32((v << 72) | (43u128 << 32));
            for k in 0..8 {
                assert_eq!(f32::from(m3[k]), m4[k], "V={v:#x} k={k}");
            }
        }
    }

    /// Codebook index 43 is the unique all-zero `sv8-128` row — the
    /// "no excitation here" symbol (provenance/08; a mis-indexed
    /// codebook would emit noise where the reference emits silence).
    #[test]
    fn sv8_128_row_43_is_the_only_zero_row() {
        let table = hb_innovation_8_128();
        for (i, row) in table.iter().enumerate() {
            let all_zero = row.iter().all(|&v| v == 0);
            assert_eq!(all_zero, i == 43, "row {i}");
        }
    }

    #[test]
    fn decode_hb_subframe_mode_3_index_extraction_is_msb_first() {
        // Single non-zero sub-vector at slot 0 should land at samples
        // 0..8 only; remaining slots are index 0 with sign cleared.
        let s = WidebandHighBandSubmode::for_id(3).unwrap();
        let idx_sv0 = 42u128;
        // Sub-vector 0 occupies bits 39..32; place [sign=0][42] there.
        let packed = idx_sv0 << (4 * 8);
        let v = decode_hb_subframe(&s, packed).unwrap();
        let row42 = &hb_innovation_8_128()[42];
        let row0 = &hb_innovation_8_128()[0];
        assert_eq!(&v[0..8], &row42[..]);
        for sv in 1..5 {
            assert_eq!(&v[sv * 8..sv * 8 + 8], &row0[..], "sub-vector {sv}");
        }
    }
}
