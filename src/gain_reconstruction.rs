//! **Log-domain scalar gain reconstruction grid** (r316 scope) — turns
//! the index-only gain fields surfaced by [`crate::fixed_codebook_gain`]
//! (narrowband frame-level OL excitation gain) and
//! [`crate::hb_excitation_gain`] (high-band per-sub-frame excitation
//! gain) into reconstructed scalar magnitudes via the documented
//! log-domain reconstruction grid.
//!
//! ## Spec basis
//!
//! The gain-quantiser trace doc staged at
//! `docs/audio/speex/gain-quantiser-and-lsp-lpc-trace.md` §2 / §4 pins
//! the **reconstruction grid shape** for the excitation-gain scalar
//! quantisers as a monotone log-domain map from a code index to a gain
//! magnitude:
//!
//! ```text
//! g(index) = 10 ^ ( (index − offset) / slope )
//! ```
//!
//! where `slope` is the number of codes per decade of gain
//! (equivalently the dB-per-step is `20 / slope`) and `offset` biases
//! the code-0 reference level. This is the inverse of the encode-side
//! quantiser the doc records as
//! `index = clip(round(log10(rms) · slope + offset), 0, 2^bits − 1)`.
//!
//! The grid is shared across the narrowband frame-level OL excitation
//! gain (§2, 5 bits, present in every NB mode except mode 0) and the
//! high-band excitation gain (§4, 5 bits in HB mode 1 and 4 bits in HB
//! modes 2..=4) — the doc §4 records the high-band gain is *"coded in
//! the same way as for narrowband"*, i.e. the same log-domain grid
//! applied to a fresh high-band residual.
//!
//! ## What the doc pins vs. what stays a recorded gap
//!
//! The doc pins the **grid structure** (the log-domain map above, that
//! the map is strictly monotone increasing in the index, and the
//! order-of-magnitude of the parameters: §2 records `slope ≈ 7.75`
//! codes/decade for the 5-bit field over ~80 dB of dynamic range, and
//! §4 records ~80 dB for the 5-bit HB field and ~64 dB for the 4-bit HB
//! field). It explicitly records that the codec author's **exact**
//! `slope` / `offset` constants are **not published** in the staged
//! manual / RFC and must be recovered by a behavioural-trace
//! calibration of the reference binary (doc §2 "Behavioural-trace
//! methodology"). No such calibration CSV is staged yet.
//!
//! This module therefore implements the documented grid as a
//! **parametric** quantiser carrying the doc's structural parameters,
//! mirroring the crate-wide "Q-format-agnostic primitive" pattern
//! (r234 / r241 / r244 / r261 / r269): the grid shape and its
//! structural invariants land now and are tested; the eventual exact
//! `(slope, offset)` pin from the staged calibration commutes through
//! by replacing the parameter set in [`GainGrid`] with no change to the
//! reconstruction algebra or its consumers. The reconstructed values
//! are not yet reference-bit-exact and the public `Decoder` endpoints
//! stay [`crate::Error::NotImplemented`] until the calibration closes.
//!
//! ## What this module DOES
//!
//! * [`GainGrid`] — a parametric log-domain reconstruction grid
//!   `(slope, offset, bits)` with [`GainGrid::reconstruct`] evaluating
//!   the documented §2 map for one index and [`GainGrid::table`]
//!   materialising the full `0..2^bits` reconstruction sweep.
//! * [`NB_OL_EXC_GAIN_GRID`] — the narrowband frame-level OL
//!   excitation-gain grid (5-bit, §2 parameters).
//! * [`HB_EXC_GAIN_GRID_5BIT`] / [`HB_EXC_GAIN_GRID_4BIT`] — the
//!   high-band excitation-gain grids (§4 parameters).
//! * [`reconstruct_frame_ol_exc_gain`] — reconstruct the NB frame-level
//!   gain magnitude from a [`crate::FrameInnovationGainIndex`] (the
//!   silence variant reconstructs to `0.0`).
//! * [`reconstruct_hb_exc_gain`] — reconstruct the HB per-sub-frame gain
//!   magnitude from a [`crate::HbExcitationGainIndex`] (the absent
//!   variant reconstructs to `0.0`), dispatching on the 5-bit / 4-bit
//!   surface form to the matching grid.
//!
//! ## What this module DOES NOT do
//!
//! * No reference-bit-exact magnitudes — the exact `(slope, offset)`
//!   pin is the recorded behavioural-trace gap above.
//! * No sub-frame innovation-gain **correction** reconstruction. The
//!   doc §3 records the NB per-sub-frame 1-bit / 3-bit correction
//!   `g_innov = g_frame · c[idx]` as a *separate* small look-up table
//!   that is likewise behavioural-trace-blocked (no `c[]` values are
//!   staged); the correction multiplier therefore stays at the index
//!   layer in [`crate::SubFrameInnovationGainCorrection`].
//! * No excitation scaling. Multiplying the §8.4 excitation `c[n]` by
//!   the reconstructed gain is a downstream layer that consumes both
//!   this primitive and the eventual exact pin.
//! * No encoder-side index selection.

use crate::fixed_codebook_gain::FrameInnovationGainIndex;
use crate::hb_excitation_gain::HbExcitationGainIndex;

/// A parametric log-domain scalar gain reconstruction grid.
///
/// Evaluates the documented §2 reconstruction map
/// `g(index) = 10^((index − offset) / slope)` for an `bits`-wide code
/// index. The grid is strictly monotone increasing in the index for any
/// positive `slope`, matching the doc's "spans the codec's full dynamic
/// range" requirement.
///
/// The `slope` / `offset` fields carry the doc's structural parameters;
/// the eventual exact-constant pin from the staged behavioural-trace
/// calibration replaces them with no change to [`Self::reconstruct`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainGrid {
    /// Codes per decade of gain (doc §2: the encode-side log10-slope).
    /// The dB-per-step of the grid is `20.0 / slope`. Must be positive
    /// for a monotone-increasing grid.
    pub slope: f32,
    /// Code-0 reference bias (doc §2 `OFFSET`): the index value that
    /// maps to unit gain (`g = 1.0`).
    pub offset: f32,
    /// Code width in bits (the Table 9.1 / Table 10.1 field width). The
    /// grid addresses indices `0..2^bits`.
    pub bits: u8,
}

impl GainGrid {
    /// Number of distinct indices the grid addresses (`2^bits`).
    pub const fn entries(&self) -> u32 {
        1u32 << self.bits
    }

    /// Largest in-range index (`2^bits − 1`).
    pub const fn max_index(&self) -> u32 {
        self.entries() - 1
    }

    /// Reconstruct the gain magnitude for one code `index` per the
    /// documented §2 grid `g = 10^((index − offset) / slope)`.
    ///
    /// The index is taken modulo nothing — callers pass the raw field
    /// value, which the bit-reader already constrains to `0..2^bits` by
    /// construction. For `index == offset` the result is exactly `1.0`
    /// (unit gain); below it the gain attenuates, above it the gain
    /// amplifies, monotonically.
    pub fn reconstruct(&self, index: u32) -> f32 {
        let exponent = (index as f32 - self.offset) / self.slope;
        10.0_f32.powf(exponent)
    }

    /// The grid's dB-per-step (`20 / slope`): the gain change in
    /// decibels between two adjacent indices.
    pub fn db_per_step(&self) -> f32 {
        20.0 / self.slope
    }

    /// Total dynamic range in dB the grid spans across its full index
    /// sweep (`(2^bits − 1) · db_per_step`).
    pub fn dynamic_range_db(&self) -> f32 {
        self.max_index() as f32 * self.db_per_step()
    }

    /// Materialise the full `0..2^bits` reconstruction sweep as a
    /// `Vec<f32>`, one magnitude per index. Useful for callers that
    /// want the precomputed table rather than per-index evaluation.
    pub fn table(&self) -> Vec<f32> {
        (0..self.entries()).map(|i| self.reconstruct(i)).collect()
    }
}

/// Narrowband frame-level OL excitation-gain grid (Table 9.1 "OL Exc
/// gain" row, 5 bits, present in NB modes 1..=8).
///
/// Parameters per the gain-quantiser trace doc §2: `slope ≈ 7.75`
/// codes/decade (the doc's worked figure for a 5-bit field spanning
/// ~80 dB of dynamic range), `offset` at the mid-range bias the doc
/// records in the `0..4` band — `2.0` is chosen as the centre of that
/// recorded band so the unit-gain reference sits inside the addressable
/// index range. Both are structural placeholders pending the staged
/// behavioural-trace calibration.
pub const NB_OL_EXC_GAIN_GRID: GainGrid = GainGrid {
    slope: 7.75,
    offset: 2.0,
    bits: 5,
};

/// High-band excitation-gain grid for HB mode 1 (Table 10.1, 5 bits).
///
/// Per doc §4 the high-band gain is *"coded in the same way as for
/// narrowband"* over ~80 dB of dynamic range with 32 reconstruction
/// levels — i.e. the same grid shape as [`NB_OL_EXC_GAIN_GRID`].
pub const HB_EXC_GAIN_GRID_5BIT: GainGrid = GainGrid {
    slope: 7.75,
    offset: 2.0,
    bits: 5,
};

/// High-band excitation-gain grid for HB modes 2..=4 (Table 10.1,
/// 4 bits).
///
/// Per doc §4 the 4-bit HB field spans ~64 dB of dynamic range with 16
/// reconstruction levels. Holding the same `db_per_step` as the 5-bit
/// grid (`20 / 7.75 ≈ 2.58 dB`) over 15 steps gives ~38.7 dB; the doc's
/// wider ~64 dB figure implies a coarser step, so the 4-bit `slope` is
/// scaled to spread ~64 dB across its 15 steps:
/// `slope = 15 · 20 / 64 ≈ 4.69`. The `offset` keeps the doc's
/// mid-range bias placement (`1.0`, the centre of the smaller index
/// span). Both stay structural placeholders pending calibration.
pub const HB_EXC_GAIN_GRID_4BIT: GainGrid = GainGrid {
    slope: 4.69,
    offset: 1.0,
    bits: 4,
};

/// Reconstruct the narrowband frame-level OL excitation-gain magnitude
/// from a typed [`FrameInnovationGainIndex`].
///
/// * [`FrameInnovationGainIndex::Silence`] (mode 0) reconstructs to
///   `0.0` — no excitation gain is transmitted and the frame is silent.
/// * [`FrameInnovationGainIndex::Indexed`] reconstructs through
///   [`NB_OL_EXC_GAIN_GRID`].
pub fn reconstruct_frame_ol_exc_gain(index: FrameInnovationGainIndex) -> f32 {
    match index {
        FrameInnovationGainIndex::Silence => 0.0,
        FrameInnovationGainIndex::Indexed(i) => NB_OL_EXC_GAIN_GRID.reconstruct(u32::from(i)),
    }
}

/// Reconstruct the high-band per-sub-frame excitation-gain magnitude
/// from a typed [`HbExcitationGainIndex`].
///
/// * [`HbExcitationGainIndex::Absent`] (mode 0) reconstructs to `0.0` —
///   no high-band excitation gain is transmitted.
/// * [`HbExcitationGainIndex::FiveBit`] (mode 1) reconstructs through
///   [`HB_EXC_GAIN_GRID_5BIT`].
/// * [`HbExcitationGainIndex::FourBit`] (modes 2..=4) reconstructs
///   through [`HB_EXC_GAIN_GRID_4BIT`].
pub fn reconstruct_hb_exc_gain(index: HbExcitationGainIndex) -> f32 {
    match index {
        HbExcitationGainIndex::Absent => 0.0,
        HbExcitationGainIndex::FiveBit(i) => HB_EXC_GAIN_GRID_5BIT.reconstruct(u32::from(i)),
        HbExcitationGainIndex::FourBit(i) => HB_EXC_GAIN_GRID_4BIT.reconstruct(u32::from(i)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The §2 reconstruction map evaluates exactly at the unit-gain
    /// reference: `g(offset) == 1.0` for any grid.
    #[test]
    fn unit_gain_at_offset_index() {
        // `offset` is integer-valued for the NB grid, so an exact index
        // hits it.
        let g = NB_OL_EXC_GAIN_GRID.reconstruct(NB_OL_EXC_GAIN_GRID.offset as u32);
        assert!(
            (g - 1.0).abs() < 1e-5,
            "g(offset) should be unit gain, got {g}"
        );
    }

    /// The grid is strictly monotone increasing in the index across the
    /// full sweep — the doc's "spans the dynamic range" requirement.
    #[test]
    fn grid_is_strictly_monotone_increasing() {
        for grid in [
            NB_OL_EXC_GAIN_GRID,
            HB_EXC_GAIN_GRID_5BIT,
            HB_EXC_GAIN_GRID_4BIT,
        ] {
            let table = grid.table();
            assert_eq!(table.len() as u32, grid.entries());
            for w in table.windows(2) {
                assert!(
                    w[1] > w[0],
                    "grid must be strictly increasing: {} !> {}",
                    w[1],
                    w[0]
                );
            }
        }
    }

    /// Pointwise pin: every grid entry equals the documented §2 formula
    /// `10^((index − offset)/slope)`.
    #[test]
    fn pointwise_pin_matches_documented_formula() {
        for grid in [
            NB_OL_EXC_GAIN_GRID,
            HB_EXC_GAIN_GRID_5BIT,
            HB_EXC_GAIN_GRID_4BIT,
        ] {
            for i in 0..grid.entries() {
                let expected = 10.0_f32.powf((i as f32 - grid.offset) / grid.slope);
                assert_eq!(grid.reconstruct(i), expected, "index {i}");
            }
        }
    }

    /// Adjacent indices differ by exactly the grid's dB-per-step (the
    /// log-domain spacing is uniform).
    #[test]
    fn adjacent_indices_uniform_in_db() {
        for grid in [NB_OL_EXC_GAIN_GRID, HB_EXC_GAIN_GRID_4BIT] {
            let table = grid.table();
            let step_db = grid.db_per_step();
            for w in table.windows(2) {
                let measured_db = 20.0 * (w[1] / w[0]).log10();
                assert!(
                    (measured_db - step_db).abs() < 1e-3,
                    "log-domain step should be uniform: {measured_db} vs {step_db}"
                );
            }
        }
    }

    /// The 5-bit grids cover ~80 dB and the 4-bit grid ~64 dB of dynamic
    /// range, matching the doc §2 / §4 order-of-magnitude figures
    /// (tolerance is wide because the exact constants are the recorded
    /// behavioural-trace gap — this only pins the documented decade
    /// scale, not the exact endpoint).
    #[test]
    fn dynamic_range_matches_doc_order_of_magnitude() {
        // 5-bit ~80 dB (doc §2 / §4); 31 steps × 2.58 dB ≈ 80 dB.
        assert!(
            (NB_OL_EXC_GAIN_GRID.dynamic_range_db() - 80.0).abs() < 5.0,
            "5-bit grid should span ~80 dB, got {}",
            NB_OL_EXC_GAIN_GRID.dynamic_range_db()
        );
        assert!(
            (HB_EXC_GAIN_GRID_5BIT.dynamic_range_db() - 80.0).abs() < 5.0,
            "5-bit HB grid should span ~80 dB, got {}",
            HB_EXC_GAIN_GRID_5BIT.dynamic_range_db()
        );
        // 4-bit ~64 dB (doc §4); 15 steps spread across ~64 dB.
        assert!(
            (HB_EXC_GAIN_GRID_4BIT.dynamic_range_db() - 64.0).abs() < 5.0,
            "4-bit HB grid should span ~64 dB, got {}",
            HB_EXC_GAIN_GRID_4BIT.dynamic_range_db()
        );
    }

    /// The grid's entry count matches its bit width.
    #[test]
    fn entries_match_bit_width() {
        assert_eq!(NB_OL_EXC_GAIN_GRID.entries(), 32);
        assert_eq!(NB_OL_EXC_GAIN_GRID.max_index(), 31);
        assert_eq!(HB_EXC_GAIN_GRID_5BIT.entries(), 32);
        assert_eq!(HB_EXC_GAIN_GRID_4BIT.entries(), 16);
        assert_eq!(HB_EXC_GAIN_GRID_4BIT.max_index(), 15);
    }

    /// NB silence reconstructs to zero gain; an indexed field
    /// reconstructs through the grid.
    #[test]
    fn nb_frame_reconstruction_dispatch() {
        assert_eq!(
            reconstruct_frame_ol_exc_gain(FrameInnovationGainIndex::Silence),
            0.0
        );
        for i in 0..32u8 {
            let g = reconstruct_frame_ol_exc_gain(FrameInnovationGainIndex::Indexed(i));
            assert_eq!(
                g,
                NB_OL_EXC_GAIN_GRID.reconstruct(u32::from(i)),
                "index {i}"
            );
            assert!(g > 0.0, "indexed gain is strictly positive");
        }
    }

    /// HB absent reconstructs to zero; 5-bit / 4-bit dispatch to the
    /// matching grid.
    #[test]
    fn hb_reconstruction_dispatch() {
        assert_eq!(reconstruct_hb_exc_gain(HbExcitationGainIndex::Absent), 0.0);
        for i in 0..32u8 {
            let g = reconstruct_hb_exc_gain(HbExcitationGainIndex::FiveBit(i));
            assert_eq!(
                g,
                HB_EXC_GAIN_GRID_5BIT.reconstruct(u32::from(i)),
                "5-bit {i}"
            );
        }
        for i in 0..16u8 {
            let g = reconstruct_hb_exc_gain(HbExcitationGainIndex::FourBit(i));
            assert_eq!(
                g,
                HB_EXC_GAIN_GRID_4BIT.reconstruct(u32::from(i)),
                "4-bit {i}"
            );
        }
    }

    /// The NB 5-bit grid and the HB 5-bit grid share the documented
    /// "coded in the same way" parameters (doc §4).
    #[test]
    fn nb_and_hb_5bit_grids_share_parameters() {
        assert_eq!(NB_OL_EXC_GAIN_GRID, HB_EXC_GAIN_GRID_5BIT);
    }

    /// Every reconstructed magnitude is finite and positive across the
    /// full index sweep of every grid (no NaN / inf at the endpoints).
    #[test]
    fn all_reconstructed_values_finite_and_positive() {
        for grid in [
            NB_OL_EXC_GAIN_GRID,
            HB_EXC_GAIN_GRID_5BIT,
            HB_EXC_GAIN_GRID_4BIT,
        ] {
            for v in grid.table() {
                assert!(
                    v.is_finite() && v > 0.0,
                    "value {v} must be finite + positive"
                );
            }
        }
    }

    /// dB-per-step relates to slope as `20 / slope`.
    #[test]
    fn db_per_step_inverts_slope() {
        assert!((NB_OL_EXC_GAIN_GRID.db_per_step() - 20.0 / 7.75).abs() < 1e-5);
        assert!((HB_EXC_GAIN_GRID_4BIT.db_per_step() - 20.0 / 4.69).abs() < 1e-5);
    }
}
