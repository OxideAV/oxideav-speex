//! Typed narrowband sub-mode (mode-ID) records.
//!
//! Per *The Speex Codec Manual* §9.3 ("Bit allocation") and Table 9.1
//! "Bit allocation for narrowband modes", the narrowband encoder may
//! pack any of nine documented bit budgets per 20 ms frame, indexed by
//! a 4-bit `mode_ID`. Each column of Table 9.1 gives the per-parameter
//! bit count; the parameters are emitted in the order listed
//! (frame-level fields first, then sub-frame fields concatenated for
//! all four sub-frames).
//!
//! Round-2 scope: surface the per-column bit budgets as a typed
//! `NarrowbandSubmode` so the round-3 CELP decoder body has a
//! structural anchor to dispatch against. No bit-stream parsing yet
//! beyond what `frame.rs` already does — this module is the *table*,
//! not the codepath that consumes it.
//!
//! ## Mode-ID semantics (§9.3 + §5.5)
//!
//! Per §9.3: *"There are 7 different narrowband bit-rates defined for
//! Speex, ranging from 250 bps to 24.6 kbps … Each frame starts with
//! the mode ID encoded with 4 bits which allows a range from 0 to 15,
//! though only the first 7 values are used (the others are
//! reserved)."* Table 9.1 itself lists nine columns (0 through 8), so
//! the "first 7" wording in the prose conflicts with the table — we
//! follow the *table* and surface modes 0..=8.
//!
//! Mode 0 (250 bps) is the "silence" / comfort-noise frame; its
//! Innovation VQ is empty (0 bits) and the 4-bit mode ID, prefixed by
//! the 1-bit wideband flag, completes the 5-bit frame total reported
//! in Table 9.1.
//!
//! Mode 8 (3.95 kbps) is the documented low-bit-rate addition between
//! "silence" (mode 0) and the regular 5.95 kbps mode 1 — see the
//! "modes below 5.9 kbps should not be used for speech" note in §9.3.
//!
//! Reserved-for-signalling slots not in Table 9.1 — modes 13, 14,
//! 15 — are exposed via [`Submode::special`] rather than
//! [`NarrowbandSubmode`] because they don't carry CELP frame bodies:
//!
//! * **Mode 13** — custom in-band message; size field follows (§5.5).
//! * **Mode 14** — in-band signalling; one of the entries from
//!   Table 5.1 follows.
//! * **Mode 15** — terminator pseudo-frame used by the encoder to
//!   pad out the last byte of a packet (§5.5).

/// Quantisation regime used by a sub-mode's pitch-gain field.
///
/// Distilled from §9.2 ("the βi coefficients are vector-quantized
/// using 7 bits at higher bit-rates (15 kbps narrowband and above) and
/// 5 bits at lower bit-rates (11 kbps narrowband and below)").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchGainQuant {
    /// No pitch contribution for this sub-mode (mode 0 — silence).
    None,
    /// 5-bit VQ codebook for the 3-tap pitch gains.
    Vq5Bit,
    /// 7-bit VQ codebook (high-rate modes).
    Vq7Bit,
}

/// LSP quantisation regime used by a sub-mode's frame-level LSP field.
///
/// §9.1: *"The LSPs are encoded using vector quantizatin (VQ) with 30
/// bits for higher quality modes and 18 bits for lower quality."* The
/// silence mode skips LSPs entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LspQuant {
    /// LSPs not transmitted for this sub-mode.
    None,
    /// 18-bit "low-quality" LSP VQ.
    Bits18,
    /// 30-bit "high-quality" LSP VQ.
    Bits30,
}

/// Per-column record from Table 9.1, "Bit allocation for narrowband
/// modes".
///
/// Field bit counts are stored as `u8` because none exceeds 96 (the
/// largest Table 9.1 entry, Innovation VQ for mode 7). The
/// `total_bits` field matches the table's `Total` row exactly, so the
/// table-row math is auditable from a single struct literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NarrowbandSubmode {
    /// Mode ID (4-bit, 0..=8 as listed in Table 9.1).
    pub mode_id: u8,

    /// Nominal bit-rate in bit/s, rounded as in Table 9.2. `0` for
    /// silence (mode 0 spends only 5 bits per 20 ms frame → 250 bps,
    /// kept as `Some(250)`); `None` is not used.
    pub nominal_bps: u32,

    /// Frame-level: LSP VQ width.
    pub lsp: LspQuant,
    /// Frame-level: open-loop pitch (bits, 0 if absent).
    pub ol_pitch_bits: u8,
    /// Frame-level: open-loop pitch gain (bits).
    pub ol_pitch_gain_bits: u8,
    /// Frame-level: open-loop excitation gain (bits).
    pub ol_exc_gain_bits: u8,

    /// Sub-frame: fine pitch period (bits).
    pub fine_pitch_bits: u8,
    /// Sub-frame: pitch-gain VQ regime.
    pub pitch_gain: PitchGainQuant,
    /// Sub-frame: innovation-gain (bits per sub-frame).
    pub innovation_gain_bits: u8,
    /// Sub-frame: innovation VQ (bits per sub-frame).
    pub innovation_vq_bits: u8,

    /// `Total` row from Table 9.1, in bits per 20 ms frame, including
    /// the 1-bit wideband flag and 4-bit mode ID.
    pub total_bits: u16,
}

/// Number of CELP sub-frames per narrowband frame (`= 4`), in `usize`
/// for indexing convenience. Mirrors [`NarrowbandSubmode::SUBFRAMES_PER_FRAME`]
/// (which is `u32` for the bit-budget arithmetic in
/// [`NarrowbandSubmode::computed_total_bits`]).
pub const SUBFRAMES_PER_FRAME: usize = 4;

impl NarrowbandSubmode {
    /// Number of CELP sub-frames per frame (constant 4 for narrowband
    /// per §9.1: "subdivided in 4 sub-frames of 5 ms each").
    pub const SUBFRAMES_PER_FRAME: u32 = 4;

    /// Returns the sub-mode record for `mode_id`, or `None` if the ID
    /// is not one of the documented narrowband sub-modes (`0..=8`).
    pub const fn for_id(mode_id: u8) -> Option<Self> {
        let idx = mode_id as usize;
        if idx >= NARROWBAND_SUBMODES.len() {
            return None;
        }
        // Const-fn doesn't allow `?` over Option in a slice index.
        Some(NARROWBAND_SUBMODES[idx])
    }

    /// Re-compute the table's `Total` row from the individual field
    /// counts. The 1-bit wideband flag + 4-bit mode ID prefix is added
    /// here so the sum should equal [`Self::total_bits`].
    pub fn computed_total_bits(&self) -> u32 {
        let lsp_bits = match self.lsp {
            LspQuant::None => 0,
            LspQuant::Bits18 => 18,
            LspQuant::Bits30 => 30,
        };
        let pitch_gain_bits = match self.pitch_gain {
            PitchGainQuant::None => 0,
            PitchGainQuant::Vq5Bit => 5,
            PitchGainQuant::Vq7Bit => 7,
        };
        let per_subframe = u32::from(self.fine_pitch_bits)
            + pitch_gain_bits
            + u32::from(self.innovation_gain_bits)
            + u32::from(self.innovation_vq_bits);
        // 1 wideband flag + 4 mode ID + frame-level fields + 4 * sub-frame fields.
        1 + 4
            + lsp_bits
            + u32::from(self.ol_pitch_bits)
            + u32::from(self.ol_pitch_gain_bits)
            + u32::from(self.ol_exc_gain_bits)
            + Self::SUBFRAMES_PER_FRAME * per_subframe
    }
}

/// Distinguishes a regular CELP sub-mode from one of the three
/// reserved-for-signalling pseudo-frame slots described in §5.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Submode {
    /// Regular CELP-coded sub-mode from Table 9.1 (modes 0..=8).
    Celp(NarrowbandSubmode),
    /// Custom in-band message (mode 13). The 5-bit size field that
    /// follows tells the decoder how many bytes to skip if it doesn't
    /// understand the payload.
    CustomInband,
    /// In-band signalling (mode 14). The 4-bit code that follows
    /// indexes Table 5.1.
    InbandSignalling,
    /// Terminator pseudo-frame (mode 15). Indicates end-of-packet.
    Terminator,
}

impl Submode {
    /// Maps a raw 4-bit mode ID to either a regular sub-mode record or
    /// one of the §5.5 special slots. Returns `None` for IDs that are
    /// neither listed in Table 9.1 nor reserved for signalling
    /// (currently 9..=12, which the spec calls out as "reserved").
    pub fn for_id(mode_id: u8) -> Option<Self> {
        match mode_id {
            0..=8 => NarrowbandSubmode::for_id(mode_id).map(Submode::Celp),
            13 => Some(Submode::CustomInband),
            14 => Some(Submode::InbandSignalling),
            15 => Some(Submode::Terminator),
            // 9..=12 are the spec's "reserved" range.
            _ => None,
        }
    }

    /// Convenience predicate for the regular CELP sub-modes.
    pub fn is_celp(&self) -> bool {
        matches!(self, Submode::Celp(_))
    }
}

/// All nine documented narrowband sub-modes, indexed by `mode_id`.
///
/// Each entry mirrors one column of Table 9.1 verbatim. The
/// `total_bits` field is the table's `Total` row; the
/// [`NarrowbandSubmode::computed_total_bits`] method re-derives it
/// from the individual counts as a self-consistency check (verified
/// in tests below).
pub const NARROWBAND_SUBMODES: [NarrowbandSubmode; 9] = [
    // Mode 0 — silence / comfort-noise (250 bps).
    NarrowbandSubmode {
        mode_id: 0,
        nominal_bps: 250,
        lsp: LspQuant::None,
        ol_pitch_bits: 0,
        ol_pitch_gain_bits: 0,
        ol_exc_gain_bits: 0,
        fine_pitch_bits: 0,
        pitch_gain: PitchGainQuant::None,
        innovation_gain_bits: 0,
        innovation_vq_bits: 0,
        total_bits: 5,
    },
    // Mode 1 — 2.15 kbps (Table 9.1 column 1: total 43 bits / 20 ms).
    NarrowbandSubmode {
        mode_id: 1,
        nominal_bps: 2_150,
        lsp: LspQuant::Bits18,
        ol_pitch_bits: 7,
        ol_pitch_gain_bits: 4,
        ol_exc_gain_bits: 5,
        fine_pitch_bits: 0,
        pitch_gain: PitchGainQuant::None,
        innovation_gain_bits: 1,
        innovation_vq_bits: 0,
        total_bits: 43,
    },
    // Mode 2 — 5.95 kbps (Table 9.1 column 2: total 119 bits / 20 ms).
    NarrowbandSubmode {
        mode_id: 2,
        nominal_bps: 5_950,
        lsp: LspQuant::Bits18,
        ol_pitch_bits: 7,
        ol_pitch_gain_bits: 0,
        ol_exc_gain_bits: 5,
        fine_pitch_bits: 0,
        pitch_gain: PitchGainQuant::Vq5Bit,
        innovation_gain_bits: 0,
        innovation_vq_bits: 16,
        total_bits: 119,
    },
    // Mode 3 — 8 kbps (Table 9.1 column 3: total 160 bits / 20 ms).
    NarrowbandSubmode {
        mode_id: 3,
        nominal_bps: 8_000,
        lsp: LspQuant::Bits18,
        ol_pitch_bits: 0,
        ol_pitch_gain_bits: 0,
        ol_exc_gain_bits: 5,
        fine_pitch_bits: 7,
        pitch_gain: PitchGainQuant::Vq5Bit,
        innovation_gain_bits: 1,
        innovation_vq_bits: 20,
        total_bits: 160,
    },
    // Mode 4 — 11 kbps (Table 9.1 column 4: total 220 bits / 20 ms).
    NarrowbandSubmode {
        mode_id: 4,
        nominal_bps: 11_000,
        lsp: LspQuant::Bits18,
        ol_pitch_bits: 0,
        ol_pitch_gain_bits: 0,
        ol_exc_gain_bits: 5,
        fine_pitch_bits: 7,
        pitch_gain: PitchGainQuant::Vq5Bit,
        innovation_gain_bits: 1,
        innovation_vq_bits: 35,
        total_bits: 220,
    },
    // Mode 5 — 15 kbps (Table 9.1 column 5: total 300 bits / 20 ms).
    NarrowbandSubmode {
        mode_id: 5,
        nominal_bps: 15_000,
        lsp: LspQuant::Bits30,
        ol_pitch_bits: 0,
        ol_pitch_gain_bits: 0,
        ol_exc_gain_bits: 5,
        fine_pitch_bits: 7,
        pitch_gain: PitchGainQuant::Vq7Bit,
        innovation_gain_bits: 3,
        innovation_vq_bits: 48,
        total_bits: 300,
    },
    // Mode 6 — 18.2 kbps (Table 9.1 column 6: total 364 bits / 20 ms).
    NarrowbandSubmode {
        mode_id: 6,
        nominal_bps: 18_200,
        lsp: LspQuant::Bits30,
        ol_pitch_bits: 0,
        ol_pitch_gain_bits: 0,
        ol_exc_gain_bits: 5,
        fine_pitch_bits: 7,
        pitch_gain: PitchGainQuant::Vq7Bit,
        innovation_gain_bits: 3,
        innovation_vq_bits: 64,
        total_bits: 364,
    },
    // Mode 7 — 24.6 kbps (Table 9.1 column 7: total 492 bits / 20 ms).
    NarrowbandSubmode {
        mode_id: 7,
        nominal_bps: 24_600,
        lsp: LspQuant::Bits30,
        ol_pitch_bits: 0,
        ol_pitch_gain_bits: 0,
        ol_exc_gain_bits: 5,
        fine_pitch_bits: 7,
        pitch_gain: PitchGainQuant::Vq7Bit,
        innovation_gain_bits: 3,
        innovation_vq_bits: 96,
        total_bits: 492,
    },
    // Mode 8 — 3.95 kbps low-rate extension
    // (Table 9.1 column 8: total 79 bits / 20 ms).
    NarrowbandSubmode {
        mode_id: 8,
        nominal_bps: 3_950,
        lsp: LspQuant::Bits18,
        ol_pitch_bits: 7,
        ol_pitch_gain_bits: 4,
        ol_exc_gain_bits: 5,
        fine_pitch_bits: 0,
        pitch_gain: PitchGainQuant::None,
        innovation_gain_bits: 0,
        innovation_vq_bits: 10,
        total_bits: 79,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_9_1_totals_match_field_breakdown() {
        // Every column of Table 9.1 must satisfy
        //   1 (wideband flag) + 4 (mode ID) + frame fields
        //                     + 4 * sub-frame fields  ==  Total
        // — this catches any future copy-paste error in the table.
        for s in NARROWBAND_SUBMODES {
            let computed = s.computed_total_bits();
            assert_eq!(
                computed,
                u32::from(s.total_bits),
                "mode {} total: spec {}, computed {}",
                s.mode_id,
                s.total_bits,
                computed,
            );
        }
    }

    #[test]
    fn table_9_1_totals_are_table_values() {
        // Hard-coded spot-check: Table 9.1's Total row, mode 0..=8.
        let expected = [5u16, 43, 119, 160, 220, 300, 364, 492, 79];
        for (i, want) in expected.iter().enumerate() {
            let got = NARROWBAND_SUBMODES[i].total_bits;
            assert_eq!(got, *want, "mode {} total mismatch", i);
        }
    }

    #[test]
    fn for_id_covers_0_through_8() {
        for id in 0u8..=8 {
            let s = NarrowbandSubmode::for_id(id).expect("must exist");
            assert_eq!(s.mode_id, id);
        }
    }

    #[test]
    fn for_id_rejects_above_8() {
        for id in 9u8..=15 {
            assert!(
                NarrowbandSubmode::for_id(id).is_none(),
                "mode {} should not be a regular CELP sub-mode",
                id
            );
        }
    }

    #[test]
    fn submode_dispatch_for_signalling_slots() {
        assert!(matches!(Submode::for_id(0), Some(Submode::Celp(_))));
        assert!(matches!(Submode::for_id(8), Some(Submode::Celp(_))));
        // 9..=12 are "reserved" with no documented role.
        for id in 9u8..=12 {
            assert!(Submode::for_id(id).is_none(), "mode {} reserved", id);
        }
        assert_eq!(Submode::for_id(13), Some(Submode::CustomInband));
        assert_eq!(Submode::for_id(14), Some(Submode::InbandSignalling));
        assert_eq!(Submode::for_id(15), Some(Submode::Terminator));
    }

    #[test]
    fn silence_mode_has_no_celp_payload() {
        let s = NarrowbandSubmode::for_id(0).unwrap();
        assert_eq!(s.nominal_bps, 250);
        assert_eq!(s.lsp, LspQuant::None);
        assert_eq!(s.pitch_gain, PitchGainQuant::None);
        assert_eq!(s.total_bits, 5);
        assert_eq!(s.computed_total_bits(), 5);
    }

    #[test]
    fn highest_mode_uses_30_bit_lsp_and_7_bit_pitch_gain() {
        // §9.2: 7-bit pitch-gain VQ at 15 kbps+; §9.1: 30-bit LSP at
        // "higher quality modes". Both should hold for mode 7.
        let s = NarrowbandSubmode::for_id(7).unwrap();
        assert_eq!(s.lsp, LspQuant::Bits30);
        assert_eq!(s.pitch_gain, PitchGainQuant::Vq7Bit);
    }

    #[test]
    fn is_celp_matches_table_membership() {
        assert!(Submode::for_id(0).unwrap().is_celp());
        assert!(Submode::for_id(7).unwrap().is_celp());
        assert!(!Submode::for_id(13).unwrap().is_celp());
        assert!(!Submode::for_id(14).unwrap().is_celp());
        assert!(!Submode::for_id(15).unwrap().is_celp());
    }
}
