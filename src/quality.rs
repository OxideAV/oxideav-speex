//! Quality → sub-mode ladders for the three Speex rate classes —
//! the encode-side mapping from the 0–10 quality knob (*The Speex Codec
//! Manual* §2.1) to the per-layer sub-mode IDs a conformant encoder
//! emits (round r389 scope).
//!
//! ## What the staged material pins
//!
//! * **Narrowband** — Table 9.2 (manual §9.4 p.33) lists each narrowband
//!   sub-mode's quality setting directly: mode 1 ↔ quality 0, mode 8 ↔
//!   quality 1, mode 2 ↔ quality 2, mode 3 ↔ qualities 3–4, mode 4 ↔
//!   5–6, mode 5 ↔ 7–8, mode 6 ↔ 9, mode 7 ↔ 10. RFC 5574 Table 1
//!   repeats the same association.
//! * **Wideband** — Table 10.2 (manual §10.4 p.35) gives the *total*
//!   bit-rate per quality (embedded narrowband layer + high band). At
//!   the fixed 50 frames/s (20 ms frames, §2.1) each rate is an exact
//!   bits-per-frame total, and every total decomposes as
//!   `Table 9.1 narrowband total + Table 10.1 high-band total`:
//!
//!   ```text
//!    q:  bps    bits  = NB(mode) + HB(mode)
//!    0:  3,950 →  79  =  43 (1)  +  36 (1)
//!    1:  5,750 → 115  =  79 (8)  +  36 (1)
//!    2:  7,750 → 155  = 119 (2)  +  36 (1)
//!    3:  9,800 → 196  = 160 (3)  +  36 (1)
//!    4: 12,800 → 256  = 220 (4)  +  36 (1)
//!    5: 16,800 → 336  = 300 (5)  +  36 (1)
//!    6: 20,600 → 412  = 300 (5)  + 112 (2)
//!    7: 23,800 → 476  = 364 (6)  + 112 (2)
//!    8: 27,800 → 556  = 364 (6)  + 192 (3)
//!    9: 34,200 → 684  = 492 (7)  + 192 (3)
//!   10: 42,200 → 844  = 492 (7)  + 352 (4)
//!   ```
//!
//!   Nine of the eleven decompositions are **unique** over the two
//!   staged tables. The two ambiguous totals (155 = 119+36 = 43+112 at
//!   quality 2; 412 = 300+112 = 220+192 at quality 6) are resolved by
//!   layer monotonicity: a higher quality never *lowers* an individual
//!   layer's sub-mode (the alternative readings would drop the
//!   narrowband layer below the previous quality's — 43 < 79 and
//!   220 < 300 respectively). The pinning test below re-derives every
//!   row arithmetically from the sub-mode bit totals.
//! * **Ultra-wideband** — RFC 5574 Table 2 lists the ultra-wideband
//!   bit-rate next to the wideband one for every quality 0..=10, and the
//!   difference is a **constant +1,800 bit/s = +36 bits per frame** at
//!   every quality — exactly the Table 10.1 **mode 1** high-band total
//!   (1 wideband bit + 3 mode-ID bits + 12 LSP bits + 4 × 5 gain bits),
//!   and no other combination of Table 10.1 totals. The second
//!   (8–16 kHz) sub-band layer of a conformant ultra-wideband stream is
//!   therefore the Table 10.1 **gain-only mode-1 layer at every
//!   quality**: [`UWB_HIGH_BAND_MODE`].
//!
//! The quality knob itself is an *encoder* selection — the decoder
//! learns each frame's modes from the in-band mode IDs — so this module
//! is pure data + arithmetic; nothing here touches the bit-stream.

/// Number of Speex frames per second (20 ms frames, manual §2.1) — the
/// factor between a bits-per-frame total and its bit-rate.
pub const FRAMES_PER_SECOND: u32 = 50;

/// Highest quality setting (§2.1: *"a quality parameter that ranges
/// from 0 to 10"*).
pub const MAX_QUALITY: u8 = 10;

/// The Table 10.1 high-band sub-mode of the ultra-wideband second
/// (8–16 kHz) sub-band layer in a conformant stream — **mode 1**
/// (gain-only, 36 bits/frame), pinned at every quality by the constant
/// +1,800 bit/s ultra-wideband column of RFC 5574 Table 2 (see module
/// docs).
pub const UWB_HIGH_BAND_MODE: u8 = 1;

/// Narrowband quality → sub-mode (Table 9.2 quality column; RFC 5574
/// Table 1). Index = quality 0..=10.
const NB_QUALITY_MODE: [u8; 11] = [1, 8, 2, 3, 3, 4, 4, 5, 5, 6, 7];

/// Wideband quality → (narrowband layer sub-mode, high-band layer
/// sub-mode), derived from Table 10.2's per-quality totals (module
/// docs). Index = quality 0..=10.
const WB_QUALITY_MODES: [(u8, u8); 11] = [
    (1, 1),
    (8, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (5, 2),
    (6, 2),
    (6, 3),
    (7, 3),
    (7, 4),
];

/// Wideband total bit-rate per quality — Table 10.2 (manual §10.4).
const WB_BITRATE_BPS: [u32; 11] = [
    3_950, 5_750, 7_750, 9_800, 12_800, 16_800, 20_600, 23_800, 27_800, 34_200, 42_200,
];

/// Ultra-wideband total bit-rate per quality — RFC 5574 Table 2
/// ("ultra wideband bit-rate" column).
const UWB_BITRATE_BPS: [u32; 11] = [
    5_750, 7_550, 9_550, 11_600, 14_600, 18_600, 22_400, 25_600, 29_600, 36_000, 44_000,
];

/// The per-layer sub-modes a conformant **wideband** encoder emits for
/// a quality setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WbQualityModes {
    /// Embedded narrowband (low band) Table 9.1 sub-mode ID.
    pub nb_mode: u8,
    /// High-band Table 10.1 sub-mode ID.
    pub hb_mode: u8,
}

/// The per-layer sub-modes a conformant **ultra-wideband** encoder
/// emits for a quality setting: the wideband ladder plus the fixed
/// mode-1 second high-band layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UwbQualityModes {
    /// Embedded narrowband (low band) Table 9.1 sub-mode ID.
    pub nb_mode: u8,
    /// First (4–8 kHz) high-band Table 10.1 sub-mode ID.
    pub hb_mode: u8,
    /// Second (8–16 kHz) high-band layer sub-mode ID — always
    /// [`UWB_HIGH_BAND_MODE`] (see module docs).
    pub uwb_hb_mode: u8,
}

/// Narrowband sub-mode for a quality setting `0..=10` (Table 9.2 /
/// RFC 5574 Table 1). `None` above [`MAX_QUALITY`].
pub fn nb_mode_for_quality(quality: u8) -> Option<u8> {
    NB_QUALITY_MODE.get(usize::from(quality)).copied()
}

/// Wideband per-layer sub-modes for a quality setting `0..=10`,
/// derived from Table 10.2 (module docs). `None` above [`MAX_QUALITY`].
pub fn wb_modes_for_quality(quality: u8) -> Option<WbQualityModes> {
    WB_QUALITY_MODES
        .get(usize::from(quality))
        .map(|&(nb_mode, hb_mode)| WbQualityModes { nb_mode, hb_mode })
}

/// Ultra-wideband per-layer sub-modes for a quality setting `0..=10`:
/// the wideband ladder plus the RFC-pinned mode-1 second high-band
/// layer. `None` above [`MAX_QUALITY`].
pub fn uwb_modes_for_quality(quality: u8) -> Option<UwbQualityModes> {
    wb_modes_for_quality(quality).map(|wb| UwbQualityModes {
        nb_mode: wb.nb_mode,
        hb_mode: wb.hb_mode,
        uwb_hb_mode: UWB_HIGH_BAND_MODE,
    })
}

/// Nominal wideband total bit-rate for a quality setting — Table 10.2.
pub fn wb_bitrate_bps(quality: u8) -> Option<u32> {
    WB_BITRATE_BPS.get(usize::from(quality)).copied()
}

/// Nominal ultra-wideband total bit-rate for a quality setting —
/// RFC 5574 Table 2.
pub fn uwb_bitrate_bps(quality: u8) -> Option<u32> {
    UWB_BITRATE_BPS.get(usize::from(quality)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::submode::NarrowbandSubmode;
    use crate::wideband::WIDEBAND_HIGH_BAND_SUBMODES;

    /// Bits per frame of a narrowband sub-mode (Table 9.1 total).
    fn nb_bits(mode: u8) -> u32 {
        u32::from(NarrowbandSubmode::for_id(mode).unwrap().total_bits)
    }

    /// Bits per frame of a high-band sub-mode (Table 10.1 total).
    fn hb_bits(mode: u8) -> u32 {
        u32::from(WIDEBAND_HIGH_BAND_SUBMODES[mode as usize].total_bits)
    }

    /// The decisive arithmetic pin: for every quality, the ladder's
    /// per-layer bit totals × 50 frames/s reproduce Table 10.2's
    /// wideband bit-rate **exactly**.
    #[test]
    fn wb_ladder_reproduces_table_10_2_rates_exactly() {
        for q in 0..=MAX_QUALITY {
            let m = wb_modes_for_quality(q).unwrap();
            let bits = nb_bits(m.nb_mode) + hb_bits(m.hb_mode);
            assert_eq!(
                bits * FRAMES_PER_SECOND,
                wb_bitrate_bps(q).unwrap(),
                "quality {q}: ({}, {})",
                m.nb_mode,
                m.hb_mode
            );
        }
    }

    /// RFC 5574 Table 2's ultra-wideband column is the wideband column
    /// plus one Table 10.1 mode-1 layer (36 bits/frame = 1,800 bit/s) at
    /// **every** quality — the pin behind [`UWB_HIGH_BAND_MODE`].
    #[test]
    fn uwb_ladder_reproduces_rfc_table_2_rates_exactly() {
        for q in 0..=MAX_QUALITY {
            let m = uwb_modes_for_quality(q).unwrap();
            assert_eq!(m.uwb_hb_mode, UWB_HIGH_BAND_MODE);
            let bits = nb_bits(m.nb_mode) + hb_bits(m.hb_mode) + hb_bits(m.uwb_hb_mode);
            assert_eq!(
                bits * FRAMES_PER_SECOND,
                uwb_bitrate_bps(q).unwrap(),
                "quality {q}"
            );
            // Constant-offset restatement of the same fact.
            assert_eq!(
                uwb_bitrate_bps(q).unwrap() - wb_bitrate_bps(q).unwrap(),
                hb_bits(UWB_HIGH_BAND_MODE) * FRAMES_PER_SECOND
            );
        }
    }

    /// No Table 10.1 total other than mode 1's equals 36 bits — the
    /// uniqueness half of the UWB layer pin.
    #[test]
    fn only_mode_1_high_band_total_is_36_bits() {
        for (mode, sub) in WIDEBAND_HIGH_BAND_SUBMODES.iter().enumerate() {
            if mode == usize::from(UWB_HIGH_BAND_MODE) {
                assert_eq!(sub.total_bits, 36);
            } else {
                assert_ne!(sub.total_bits, 36, "mode {mode}");
            }
        }
    }

    /// Narrowband quality ladder matches Table 9.2's quality column:
    /// each quality's mode carries Table 9.2's bit-rate for that mode.
    #[test]
    fn nb_ladder_matches_table_9_2() {
        // Table 9.2 bit-rates by narrowband mode (bps).
        let mode_rate = |m: u8| nb_bits(m) * FRAMES_PER_SECOND;
        let expected: [(u8, u32); 11] = [
            (1, 2_150),
            (8, 3_950),
            (2, 5_950),
            (3, 8_000),
            (3, 8_000),
            (4, 11_000),
            (4, 11_000),
            (5, 15_000),
            (5, 15_000),
            (6, 18_200),
            (7, 24_600),
        ];
        for (q, (mode, rate)) in expected.iter().enumerate() {
            let got = nb_mode_for_quality(q as u8).unwrap();
            assert_eq!(got, *mode, "quality {q}");
            assert_eq!(mode_rate(got), *rate, "quality {q} rate");
        }
    }

    /// Layer monotonicity — the disambiguation rule used for the two
    /// non-unique totals (module docs): neither layer's bits-per-frame
    /// ever decreases as quality rises.
    #[test]
    fn wb_ladder_layers_are_monotonic() {
        for q in 1..=MAX_QUALITY {
            let lo = wb_modes_for_quality(q - 1).unwrap();
            let hi = wb_modes_for_quality(q).unwrap();
            assert!(
                nb_bits(hi.nb_mode) >= nb_bits(lo.nb_mode),
                "NB layer shrank at quality {q}"
            );
            assert!(
                hb_bits(hi.hb_mode) >= hb_bits(lo.hb_mode),
                "HB layer shrank at quality {q}"
            );
        }
    }

    /// Out-of-range qualities are rejected across the board.
    #[test]
    fn qualities_above_10_are_none() {
        for q in 11..=255u8 {
            assert_eq!(nb_mode_for_quality(q), None);
            assert_eq!(wb_modes_for_quality(q), None);
            assert!(uwb_modes_for_quality(q).is_none());
            assert_eq!(wb_bitrate_bps(q), None);
            assert_eq!(uwb_bitrate_bps(q), None);
        }
    }
}
