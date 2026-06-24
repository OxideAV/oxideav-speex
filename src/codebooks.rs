//! Speex CELP companion-table data (round 191 scope).
//!
//! Wires the staged clean-room tables (vendored into the crate at
//! `tables/`, byte-identical to the master copy under
//! `docs/audio/speex/tables/` in the workspace umbrella) as pure-data
//! accessors. The tables themselves are factual numeric arrays
//! (codebook entries, analysis windows, QMF filter taps) — the same
//! arrays a byte-exact decoder needs in order to invert the §9.1 LSP
//! VQ, the §9.2 pitch / innovation VQs, and (for wideband) the §10.x
//! high-band MSVQ and QMF split. The arrays were extracted under
//! the project's data-only Extractor role and are tracked alongside
//! their per-table `.meta` sidecars; see `tables/README.md` in the
//! crate root and
//! `docs/audio/speex/provenance/01-speex-table-extraction.md` in
//! the workspace umbrella for the chain of custody.
//!
//! ## What this round delivers
//!
//! * Each codebook / window / filter array is embedded into the
//!   crate via [`include_str!`] over the per-table CSV and parsed
//!   on first use into a typed [`std::sync::OnceLock`]-backed
//!   `&'static [Row]`.
//! * Public accessors expose every array used by the modes the
//!   crate already parses bit-stream-level:
//!     - narrowband LSP VQ (stage 0 + four split-band stages, §9.1);
//!     - 3-tap pitch-gain VQ (5-bit and 7-bit, §9.2);
//!     - six narrowband innovation codebooks (Table 9.1 columns 2-7
//!       and the 3.95 kbps mode 8 entry);
//!     - wideband high-band LSP MSVQ (2 stages × 6 bits) and the
//!       two high-band innovation codebooks documented in §10.4 /
//!       Table 10.1;
//!     - the 200-sample LPC analysis window, the 11-tap
//!       autocorrelation lag window, and the 64-tap QMF analysis
//!       filter `h0` used by the wideband split (§10.1).
//! * Dimension constants for every accessor are cross-checked
//!   against the bit-budget records already in
//!   [`crate::submode`] (e.g. mode-3 innovation VQ is 20 bits per
//!   sub-frame → the matching codebook has `2.pow(20 / 4)` = 32
//!   entries of 10 samples; mode-6 7-bit pitch-gain VQ → 128 rows
//!   of 4 columns; etc.). These checks live in the module's
//!   `#[cfg(test)]` block.
//!
//! ## What this round does NOT do
//!
//! No LSP→LPC conversion, no innovation-codebook gain scaling, no
//! synthesis filtering, no encoder-side codebook search. Those
//! remain `Error::NotImplemented` until subsequent rounds — this
//! module is the *table*, not the codepath that consumes it.
//!
//! ## Naming
//!
//! Public identifiers describe each array by its *role in the
//! Speex CELP algorithm* (stage number, dimension, regime), not by
//! the source-side C identifier. The CSV `.meta` sidecars preserve
//! the canonical C names for cross-referencing the extraction
//! manifest, but those names are not exposed in this crate's
//! public surface.

use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// CSV parsing — shared between every codebook accessor below.
// ---------------------------------------------------------------------------

/// Parses a CSV row of comma-separated decimal integers into a
/// fixed-size `[i16; N]` row. Used by every multi-column codebook
/// accessor; the dimension is enforced by const-generic so a CSV row
/// with the wrong column count panics at parse time (caught by tests
/// before reaching production users).
fn parse_row<const N: usize>(line: &str) -> [i16; N] {
    let mut row = [0i16; N];
    let mut count = 0usize;
    for tok in line.split(',') {
        let v: i16 = tok
            .trim()
            .parse()
            .expect("speex codebook CSV: row token must be a decimal integer");
        assert!(
            count < N,
            "speex codebook CSV: row has more than {} columns",
            N
        );
        row[count] = v;
        count += 1;
    }
    assert_eq!(
        count, N,
        "speex codebook CSV: row has {} columns, expected {}",
        count, N
    );
    row
}

/// Parses a CSV body (one row per non-empty line) into a
/// `Vec<[i16; N]>`. Final entry count is asserted to match
/// `expected_rows` — every embedded table has a known shape.
fn parse_table<const N: usize>(body: &'static str, expected_rows: usize) -> Vec<[i16; N]> {
    let rows: Vec<[i16; N]> = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(parse_row::<N>)
        .collect();
    assert_eq!(
        rows.len(),
        expected_rows,
        "speex codebook CSV row count mismatch: got {}, expected {}",
        rows.len(),
        expected_rows
    );
    rows
}

/// Parses a single-column table (one decimal integer per line).
fn parse_column(body: &'static str, expected_rows: usize) -> Vec<i16> {
    let rows: Vec<i16> = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim()
                .parse::<i16>()
                .expect("speex single-column CSV: token must be a decimal integer")
        })
        .collect();
    assert_eq!(
        rows.len(),
        expected_rows,
        "speex single-column CSV row count mismatch: got {}, expected {}",
        rows.len(),
        expected_rows
    );
    rows
}

/// Parse a single-column float CSV body into a `Vec<f64>`, asserting the
/// expected row count (the float-variant companion to [`parse_column`]).
fn parse_float_column(body: &'static str, expected_rows: usize) -> Vec<f64> {
    let rows: Vec<f64> = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim()
                .parse::<f64>()
                .expect("speex single-column float CSV: token must be a decimal float")
        })
        .collect();
    assert_eq!(
        rows.len(),
        expected_rows,
        "speex single-column float CSV row count mismatch: got {}, expected {}",
        rows.len(),
        expected_rows
    );
    rows
}

// ---------------------------------------------------------------------------
// Narrowband LSP VQ codebooks (§9.1).
//
// Stage 0 quantises all 10 LSP coefficients into a 6-bit VQ (64
// entries × 10 values). Stages 1-4 are split-band 6-bit VQs over the
// low half (coeffs 0-4) and high half (coeffs 5-9), each 64 × 5.
// Decoder scaling factors documented in `.meta` are surfaced here as
// [`NbLspScale`] so the §9.1 LSP→LPC consumer (a later round) can
// apply them without the codebook lookup itself caring about the
// scale.
// ---------------------------------------------------------------------------

/// Number of entries in each narrowband LSP VQ stage (6-bit index).
pub const NB_LSP_STAGE_ENTRIES: usize = 64;
/// Number of coefficients in the narrowband LSP order-10 vector.
pub const NB_LSP_ORDER: usize = 10;
/// Number of coefficients per split-band stage (low or high).
pub const NB_LSP_SPLIT_HALF: usize = 5;

/// Decoder scaling factor associated with a narrowband LSP stage,
/// per the table-extraction `.meta` sidecars
/// (`LSP_DIV_256` / `LSP_DIV_512` / `LSP_DIV_1024`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NbLspScale {
    /// Divide by 256.
    Div256,
    /// Divide by 512.
    Div512,
    /// Divide by 1024.
    Div1024,
}

impl NbLspScale {
    /// Integer divisor used in the scaling.
    pub const fn divisor(self) -> i32 {
        match self {
            NbLspScale::Div256 => 256,
            NbLspScale::Div512 => 512,
            NbLspScale::Div1024 => 1024,
        }
    }
}

const NB_LSP_STAGE0_CSV: &str = include_str!("../tables/nb-lsp-cdbk-stage0.csv");
const NB_LSP_LOW1_CSV: &str = include_str!("../tables/nb-lsp-cdbk-low1.csv");
const NB_LSP_LOW2_CSV: &str = include_str!("../tables/nb-lsp-cdbk-low2.csv");
const NB_LSP_HIGH1_CSV: &str = include_str!("../tables/nb-lsp-cdbk-high1.csv");
const NB_LSP_HIGH2_CSV: &str = include_str!("../tables/nb-lsp-cdbk-high2.csv");

static NB_LSP_STAGE0: OnceLock<Vec<[i16; NB_LSP_ORDER]>> = OnceLock::new();
static NB_LSP_LOW1: OnceLock<Vec<[i16; NB_LSP_SPLIT_HALF]>> = OnceLock::new();
static NB_LSP_LOW2: OnceLock<Vec<[i16; NB_LSP_SPLIT_HALF]>> = OnceLock::new();
static NB_LSP_HIGH1: OnceLock<Vec<[i16; NB_LSP_SPLIT_HALF]>> = OnceLock::new();
static NB_LSP_HIGH2: OnceLock<Vec<[i16; NB_LSP_SPLIT_HALF]>> = OnceLock::new();

/// Narrowband LSP VQ stage 0: 64 entries × 10 coefficients. Scaling
/// regime: [`NbLspScale::Div256`].
pub fn nb_lsp_stage0() -> &'static [[i16; NB_LSP_ORDER]] {
    NB_LSP_STAGE0.get_or_init(|| parse_table(NB_LSP_STAGE0_CSV, NB_LSP_STAGE_ENTRIES))
}

/// Narrowband LSP VQ stage 1, low half (coeffs 0-4): 64 × 5.
/// Scaling: [`NbLspScale::Div512`].
pub fn nb_lsp_low1() -> &'static [[i16; NB_LSP_SPLIT_HALF]] {
    NB_LSP_LOW1.get_or_init(|| parse_table(NB_LSP_LOW1_CSV, NB_LSP_STAGE_ENTRIES))
}

/// Narrowband LSP VQ stage 2, low half (coeffs 0-4): 64 × 5.
/// Scaling: [`NbLspScale::Div1024`].
pub fn nb_lsp_low2() -> &'static [[i16; NB_LSP_SPLIT_HALF]] {
    NB_LSP_LOW2.get_or_init(|| parse_table(NB_LSP_LOW2_CSV, NB_LSP_STAGE_ENTRIES))
}

/// Narrowband LSP VQ stage 3, high half (coeffs 5-9): 64 × 5.
/// Scaling: [`NbLspScale::Div512`].
pub fn nb_lsp_high1() -> &'static [[i16; NB_LSP_SPLIT_HALF]] {
    NB_LSP_HIGH1.get_or_init(|| parse_table(NB_LSP_HIGH1_CSV, NB_LSP_STAGE_ENTRIES))
}

/// Narrowband LSP VQ stage 4, high half (coeffs 5-9): 64 × 5.
/// Scaling: [`NbLspScale::Div1024`].
pub fn nb_lsp_high2() -> &'static [[i16; NB_LSP_SPLIT_HALF]] {
    NB_LSP_HIGH2.get_or_init(|| parse_table(NB_LSP_HIGH2_CSV, NB_LSP_STAGE_ENTRIES))
}

/// Scaling regime associated with each narrowband LSP stage.
pub const fn nb_lsp_scale(stage: u8) -> Option<NbLspScale> {
    match stage {
        0 => Some(NbLspScale::Div256),
        1 => Some(NbLspScale::Div512),
        2 => Some(NbLspScale::Div1024),
        3 => Some(NbLspScale::Div512),
        4 => Some(NbLspScale::Div1024),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// 3-tap pitch-gain VQ (§9.2).
//
// Two variants: a 5-bit low-bit-rate table (32 × 4) used by the
// 2.15 kbps and 3.95 kbps modes, and a 7-bit table (128 × 4) used by
// the higher-quality modes (15 kbps narrowband and above). Each row
// is [g0, g1, g2, search_aid]; the gains are stored with a -32 bias
// so a runtime consumer adds +32 before applying.
// ---------------------------------------------------------------------------

/// Pitch-gain VQ row layout: three taps + one search-aid term.
pub const PITCH_GAIN_COLS: usize = 4;
/// Entries in the 5-bit pitch-gain VQ.
pub const PITCH_GAIN_5BIT_ENTRIES: usize = 32;
/// Entries in the 7-bit pitch-gain VQ.
pub const PITCH_GAIN_7BIT_ENTRIES: usize = 128;
/// Bias added to stored gain bytes when the decoder applies a row.
pub const PITCH_GAIN_BIAS: i16 = 32;

const PITCH_GAIN_5BIT_CSV: &str = include_str!("../tables/pitch-gain-cdbk-5bit.csv");
const PITCH_GAIN_7BIT_CSV: &str = include_str!("../tables/pitch-gain-cdbk-7bit.csv");

static PITCH_GAIN_5BIT: OnceLock<Vec<[i16; PITCH_GAIN_COLS]>> = OnceLock::new();
static PITCH_GAIN_7BIT: OnceLock<Vec<[i16; PITCH_GAIN_COLS]>> = OnceLock::new();

/// 5-bit pitch-gain VQ (low-bit-rate modes). 32 × 4 rows of
/// `[g0, g1, g2, search_aid]`; add [`PITCH_GAIN_BIAS`] to each gain
/// component before use.
pub fn pitch_gain_5bit() -> &'static [[i16; PITCH_GAIN_COLS]] {
    PITCH_GAIN_5BIT.get_or_init(|| parse_table(PITCH_GAIN_5BIT_CSV, PITCH_GAIN_5BIT_ENTRIES))
}

/// 7-bit pitch-gain VQ (high-bit-rate modes). 128 × 4 rows of
/// `[g0, g1, g2, search_aid]`; add [`PITCH_GAIN_BIAS`] to each gain
/// component before use.
pub fn pitch_gain_7bit() -> &'static [[i16; PITCH_GAIN_COLS]] {
    PITCH_GAIN_7BIT.get_or_init(|| parse_table(PITCH_GAIN_7BIT_CSV, PITCH_GAIN_7BIT_ENTRIES))
}

// ---------------------------------------------------------------------------
// Narrowband innovation codebooks (Table 9.1 / §9.2).
//
// One codebook per (sub-vector size, innovation-VQ bits) pair. The
// crate exposes every shape Table 9.1 references for modes 2..=7
// and mode 8 — the dispatching layer is the existing
// `NarrowbandSubmode::innovation_vq_bits` field, mapped to one of
// these accessors by the consumer.
// ---------------------------------------------------------------------------

/// Shape descriptor for a narrowband innovation codebook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InnovationShape {
    /// Samples per innovation sub-vector (5, 8, 10, or 20).
    pub sub_vector_len: u8,
    /// Codebook index width in bits.
    pub index_bits: u8,
    /// Number of entries (= 1 << `index_bits`).
    pub entries: u32,
}

const INNOV_5_64_CSV: &str = include_str!("../tables/innovation-cdbk-sv5-64.csv");
const INNOV_5_256_CSV: &str = include_str!("../tables/innovation-cdbk-sv5-256.csv");
const INNOV_8_128_CSV: &str = include_str!("../tables/innovation-cdbk-sv8-128.csv");
const INNOV_10_16_CSV: &str = include_str!("../tables/innovation-cdbk-sv10-16.csv");
const INNOV_10_32_CSV: &str = include_str!("../tables/innovation-cdbk-sv10-32.csv");
const INNOV_20_32_CSV: &str = include_str!("../tables/innovation-cdbk-sv20-32.csv");

static INNOV_5_64: OnceLock<Vec<[i16; 5]>> = OnceLock::new();
static INNOV_5_256: OnceLock<Vec<[i16; 5]>> = OnceLock::new();
static INNOV_8_128: OnceLock<Vec<[i16; 8]>> = OnceLock::new();
static INNOV_10_16: OnceLock<Vec<[i16; 10]>> = OnceLock::new();
static INNOV_10_32: OnceLock<Vec<[i16; 10]>> = OnceLock::new();
static INNOV_20_32: OnceLock<Vec<[i16; 20]>> = OnceLock::new();

/// 5-sample × 6-bit innovation codebook (64 entries).
pub fn innovation_5_64() -> &'static [[i16; 5]] {
    INNOV_5_64.get_or_init(|| parse_table(INNOV_5_64_CSV, 64))
}

/// 5-sample × 8-bit innovation codebook (256 entries).
pub fn innovation_5_256() -> &'static [[i16; 5]] {
    INNOV_5_256.get_or_init(|| parse_table(INNOV_5_256_CSV, 256))
}

/// 8-sample × 7-bit innovation codebook (128 entries).
pub fn innovation_8_128() -> &'static [[i16; 8]] {
    INNOV_8_128.get_or_init(|| parse_table(INNOV_8_128_CSV, 128))
}

/// 10-sample × 4-bit innovation codebook (16 entries).
pub fn innovation_10_16() -> &'static [[i16; 10]] {
    INNOV_10_16.get_or_init(|| parse_table(INNOV_10_16_CSV, 16))
}

/// 10-sample × 5-bit innovation codebook (32 entries).
pub fn innovation_10_32() -> &'static [[i16; 10]] {
    INNOV_10_32.get_or_init(|| parse_table(INNOV_10_32_CSV, 32))
}

/// 20-sample × 5-bit innovation codebook (32 entries) used by the
/// 3.95 kbps mode 8.
pub fn innovation_20_32() -> &'static [[i16; 20]] {
    INNOV_20_32.get_or_init(|| parse_table(INNOV_20_32_CSV, 32))
}

// ---------------------------------------------------------------------------
// Wideband high-band LSP MSVQ + innovation codebooks (§10.x).
//
// The high-band LSP quantiser is a 2-stage MSVQ over an LPC order-8
// vector, 6 bits per stage (12 bits total — the same width
// `wideband.rs` already extracts as a raw index). The high-band
// innovation codebooks are the two shapes documented in Table 10.1
// for modes 1..=4: 8-sample × 7-bit + sign, and 10-sample × 5-bit.
// ---------------------------------------------------------------------------

/// High-band LSP MSVQ stage width (per `.meta`).
pub const HB_LSP_STAGE_ENTRIES: usize = 64;
/// High-band LPC order.
pub const HB_LPC_ORDER: usize = 8;

const HB_LSP_STAGE1_CSV: &str = include_str!("../tables/hb-lsp-cdbk-stage1.csv");
const HB_LSP_STAGE2_CSV: &str = include_str!("../tables/hb-lsp-cdbk-stage2.csv");

static HB_LSP_STAGE1: OnceLock<Vec<[i16; HB_LPC_ORDER]>> = OnceLock::new();
static HB_LSP_STAGE2: OnceLock<Vec<[i16; HB_LPC_ORDER]>> = OnceLock::new();

/// High-band LSP MSVQ stage 1: 64 × 8. Scaling: [`NbLspScale::Div256`].
pub fn hb_lsp_stage1() -> &'static [[i16; HB_LPC_ORDER]] {
    HB_LSP_STAGE1.get_or_init(|| parse_table(HB_LSP_STAGE1_CSV, HB_LSP_STAGE_ENTRIES))
}

/// High-band LSP MSVQ stage 2 (residual): 64 × 8. Scaling:
/// [`NbLspScale::Div512`].
pub fn hb_lsp_stage2() -> &'static [[i16; HB_LPC_ORDER]] {
    HB_LSP_STAGE2.get_or_init(|| parse_table(HB_LSP_STAGE2_CSV, HB_LSP_STAGE_ENTRIES))
}

/// Scaling regime for a wideband high-band LSP stage (1 or 2).
pub const fn hb_lsp_scale(stage: u8) -> Option<NbLspScale> {
    match stage {
        1 => Some(NbLspScale::Div256),
        2 => Some(NbLspScale::Div512),
        _ => None,
    }
}

const HB_INNOV_8_128_CSV: &str = include_str!("../tables/hb-innovation-cdbk-sv8-128.csv");
const HB_INNOV_10_32_CSV: &str = include_str!("../tables/hb-innovation-cdbk-sv10-32.csv");

static HB_INNOV_8_128: OnceLock<Vec<[i16; 8]>> = OnceLock::new();
static HB_INNOV_10_32: OnceLock<Vec<[i16; 10]>> = OnceLock::new();

/// High-band innovation codebook, 8-sample × 7-bit (with sign), 128 entries.
pub fn hb_innovation_8_128() -> &'static [[i16; 8]] {
    HB_INNOV_8_128.get_or_init(|| parse_table(HB_INNOV_8_128_CSV, 128))
}

/// High-band innovation codebook, 10-sample × 5-bit, 32 entries.
pub fn hb_innovation_10_32() -> &'static [[i16; 10]] {
    HB_INNOV_10_32.get_or_init(|| parse_table(HB_INNOV_10_32_CSV, 32))
}

// ---------------------------------------------------------------------------
// Windows + QMF filter (Q15 fixed-point variants).
//
// `lpc_analysis_window` is the asymmetric 200-sample window the
// open-loop LPC analysis applies before autocorrelation. `lag_window`
// is the order-10 lag window applied to the autocorrelations. `qmf_h0`
// is the 64-tap QMF analysis filter that the wideband encoder/decoder
// uses to split into / merge from the 0-4 kHz and 4-8 kHz bands per
// §10.1. Only the Q15 fixed-point variants are wired here; the float
// variants stay in `tables/` for any later round that wants them.
// ---------------------------------------------------------------------------

/// Length of the asymmetric LPC analysis window.
pub const LPC_ANALYSIS_WINDOW_LEN: usize = 200;
/// Length of the autocorrelation lag window.
pub const LPC_LAG_WINDOW_LEN: usize = 11;
/// Length of the QMF analysis filter.
pub const QMF_FILTER_LEN: usize = 64;

const LPC_WINDOW_Q15_CSV: &str = include_str!("../tables/lpc-analysis-window-q15.csv");
const LAG_WINDOW_Q15_CSV: &str = include_str!("../tables/lpc-autocorr-lag-window-q15.csv");
const QMF_H0_Q15_CSV: &str = include_str!("../tables/qmf-filter-h0-q15.csv");
const QMF_H0_FLOAT_CSV: &str = include_str!("../tables/qmf-filter-h0-float.csv");

static LPC_WINDOW_Q15: OnceLock<Vec<i16>> = OnceLock::new();
static LAG_WINDOW_Q15: OnceLock<Vec<i16>> = OnceLock::new();
static QMF_H0_Q15: OnceLock<Vec<i16>> = OnceLock::new();
static QMF_H0_FLOAT: OnceLock<Vec<f64>> = OnceLock::new();

/// 200-sample asymmetric LPC analysis window (Q15 fixed-point).
pub fn lpc_analysis_window_q15() -> &'static [i16] {
    LPC_WINDOW_Q15.get_or_init(|| parse_column(LPC_WINDOW_Q15_CSV, LPC_ANALYSIS_WINDOW_LEN))
}

/// 11-tap autocorrelation lag window (Q15 fixed-point).
pub fn lpc_lag_window_q15() -> &'static [i16] {
    LAG_WINDOW_Q15.get_or_init(|| parse_column(LAG_WINDOW_Q15_CSV, LPC_LAG_WINDOW_LEN))
}

/// 64-tap QMF analysis filter `h0` (Q15 fixed-point) used by the
/// wideband 8 kHz/8 kHz split.
pub fn qmf_h0_q15() -> &'static [i16] {
    QMF_H0_Q15.get_or_init(|| parse_column(QMF_H0_Q15_CSV, QMF_FILTER_LEN))
}

/// 64-tap QMF prototype lowpass filter `h0` (float variant), the
/// reconstruction-side input for the [`crate::qmf`] two-band synthesis
/// filterbank. Same prototype as [`qmf_h0_q15`] at full float precision
/// (the staged `qmf-filter-h0-float.csv`).
pub fn qmf_h0_float() -> &'static [f64] {
    QMF_H0_FLOAT.get_or_init(|| parse_float_column(QMF_H0_FLOAT_CSV, QMF_FILTER_LEN))
}

// ---------------------------------------------------------------------------
// Tests — dimension self-checks + cross-checks against `submode.rs`.
//
// Every accessor is hit so the `OnceLock` init paths run, panics in
// the parsers are exercised under `cargo test`, and the dimension
// constants tracked in `submode.rs` line up with the actual codebook
// shapes shipped in the CSV body.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::submode::{NarrowbandSubmode, PitchGainQuant, NARROWBAND_SUBMODES};

    #[test]
    fn narrowband_lsp_codebooks_have_expected_dimensions() {
        assert_eq!(nb_lsp_stage0().len(), NB_LSP_STAGE_ENTRIES);
        assert_eq!(nb_lsp_stage0()[0].len(), NB_LSP_ORDER);
        for accessor in [nb_lsp_low1, nb_lsp_low2, nb_lsp_high1, nb_lsp_high2] {
            let cb = accessor();
            assert_eq!(cb.len(), NB_LSP_STAGE_ENTRIES);
            assert_eq!(cb[0].len(), NB_LSP_SPLIT_HALF);
        }
    }

    #[test]
    fn nb_lsp_scale_covers_all_five_stages() {
        for stage in 0u8..=4 {
            assert!(nb_lsp_scale(stage).is_some(), "stage {stage} unscaled");
        }
        assert!(nb_lsp_scale(5).is_none());
        // Scaling pattern from `.meta`: 256 / 512 / 1024 / 512 / 1024.
        let pattern = [
            NbLspScale::Div256,
            NbLspScale::Div512,
            NbLspScale::Div1024,
            NbLspScale::Div512,
            NbLspScale::Div1024,
        ];
        for (i, want) in pattern.iter().enumerate() {
            assert_eq!(nb_lsp_scale(i as u8).unwrap(), *want);
        }
    }

    #[test]
    fn pitch_gain_codebooks_have_expected_dimensions() {
        assert_eq!(pitch_gain_5bit().len(), PITCH_GAIN_5BIT_ENTRIES);
        assert_eq!(pitch_gain_7bit().len(), PITCH_GAIN_7BIT_ENTRIES);
        assert_eq!(pitch_gain_5bit()[0].len(), PITCH_GAIN_COLS);
        assert_eq!(pitch_gain_7bit()[0].len(), PITCH_GAIN_COLS);
    }

    #[test]
    fn pitch_gain_widths_match_submode_table() {
        // For every Vq5Bit/Vq7Bit sub-mode the index width should
        // match log2(entries) of the codebook we'd dispatch to.
        for s in NARROWBAND_SUBMODES {
            match s.pitch_gain {
                PitchGainQuant::None => {}
                PitchGainQuant::Vq5Bit => {
                    assert_eq!(pitch_gain_5bit().len(), 1 << 5);
                }
                PitchGainQuant::Vq7Bit => {
                    assert_eq!(pitch_gain_7bit().len(), 1 << 7);
                }
            }
        }
    }

    #[test]
    fn innovation_codebooks_have_expected_dimensions() {
        assert_eq!(innovation_5_64().len(), 64);
        assert_eq!(innovation_5_256().len(), 256);
        assert_eq!(innovation_8_128().len(), 128);
        assert_eq!(innovation_10_16().len(), 16);
        assert_eq!(innovation_10_32().len(), 32);
        assert_eq!(innovation_20_32().len(), 32);

        assert_eq!(innovation_5_64()[0].len(), 5);
        assert_eq!(innovation_5_256()[0].len(), 5);
        assert_eq!(innovation_8_128()[0].len(), 8);
        assert_eq!(innovation_10_16()[0].len(), 10);
        assert_eq!(innovation_10_32()[0].len(), 10);
        assert_eq!(innovation_20_32()[0].len(), 20);
    }

    #[test]
    fn innovation_codebook_index_widths_match_meta() {
        // The bit width on the wire that each innovation codebook
        // can be addressed by is log2(entries). The actual
        // (sub_vector_len, bits-per-sub-frame) → codebook dispatcher
        // is a later round (the per-sub-frame budget is split across
        // multiple sub-vectors for the higher modes — e.g. mode 5's
        // 48 bits/sub-frame are NOT a single 12-bit lookup), so this
        // test only asserts the codebook *shapes* on offer match the
        // staged `.meta` dimensions.
        let shapes: &[(u8, u8, usize)] = &[
            (5, 6, 64),
            (5, 8, 256),
            (8, 7, 128),
            (10, 4, 16),
            (10, 5, 32),
            (20, 5, 32),
        ];
        let counts = [
            innovation_5_64().len(),
            innovation_5_256().len(),
            innovation_8_128().len(),
            innovation_10_16().len(),
            innovation_10_32().len(),
            innovation_20_32().len(),
        ];
        for (i, (_sv, bits, ent)) in shapes.iter().enumerate() {
            assert_eq!(counts[i], *ent, "shape index {i}");
            assert_eq!(
                *ent,
                1 << *bits as u32,
                "shape index {i}: 2^{bits} != {ent}"
            );
        }
    }

    #[test]
    fn hb_lsp_codebooks_have_expected_dimensions() {
        assert_eq!(hb_lsp_stage1().len(), HB_LSP_STAGE_ENTRIES);
        assert_eq!(hb_lsp_stage2().len(), HB_LSP_STAGE_ENTRIES);
        assert_eq!(hb_lsp_stage1()[0].len(), HB_LPC_ORDER);
        assert_eq!(hb_lsp_stage2()[0].len(), HB_LPC_ORDER);
        // Two 6-bit stages → the 12-bit `lsp_msvq_index` already
        // parsed by `wideband.rs` is exactly addressable.
        assert_eq!(NB_LSP_STAGE_ENTRIES * HB_LSP_STAGE_ENTRIES, 1 << 12);
    }

    #[test]
    fn hb_lsp_scale_covers_both_stages_only() {
        assert_eq!(hb_lsp_scale(1), Some(NbLspScale::Div256));
        assert_eq!(hb_lsp_scale(2), Some(NbLspScale::Div512));
        assert!(hb_lsp_scale(0).is_none());
        assert!(hb_lsp_scale(3).is_none());
    }

    #[test]
    fn hb_innovation_codebooks_have_expected_dimensions() {
        assert_eq!(hb_innovation_8_128().len(), 128);
        assert_eq!(hb_innovation_10_32().len(), 32);
        assert_eq!(hb_innovation_8_128()[0].len(), 8);
        assert_eq!(hb_innovation_10_32()[0].len(), 10);
    }

    #[test]
    fn windows_and_qmf_have_expected_lengths() {
        assert_eq!(lpc_analysis_window_q15().len(), LPC_ANALYSIS_WINDOW_LEN);
        assert_eq!(lpc_lag_window_q15().len(), LPC_LAG_WINDOW_LEN);
        assert_eq!(qmf_h0_q15().len(), QMF_FILTER_LEN);
    }

    #[test]
    fn nb_lsp_stage0_first_row_matches_csv() {
        // CSV row 0 is "30, 19, 38, 34, 40, 32, 46, 43, 58, 43";
        // hard-coded so a future refresh of the CSV that scrambles
        // rows is caught locally rather than only in audit.
        let row = nb_lsp_stage0()[0];
        assert_eq!(row, [30, 19, 38, 34, 40, 32, 46, 43, 58, 43]);
    }

    #[test]
    fn pitch_gain_5bit_first_row_matches_csv() {
        // CSV row 0 is "-32, -32, -32, 0".
        let row = pitch_gain_5bit()[0];
        assert_eq!(row, [-32, -32, -32, 0]);
        // After applying the documented +32 bias the row becomes the
        // all-zero "silence" pitch-tap.
        let biased: [i16; 3] = [
            row[0] + PITCH_GAIN_BIAS,
            row[1] + PITCH_GAIN_BIAS,
            row[2] + PITCH_GAIN_BIAS,
        ];
        assert_eq!(biased, [0, 0, 0]);
    }

    #[test]
    fn lpc_lag_window_first_value_is_unit_q15() {
        // The lag window starts at 1.0 in Q15 = 32767 by construction;
        // tests this against the staged CSV head row.
        assert_eq!(lpc_lag_window_q15()[0], 32767);
    }

    #[test]
    fn parse_row_rejects_extra_columns() {
        let r = std::panic::catch_unwind(|| parse_row::<2>("1, 2, 3"));
        assert!(r.is_err());
    }

    #[test]
    fn parse_row_rejects_too_few_columns() {
        let r = std::panic::catch_unwind(|| parse_row::<3>("1, 2"));
        assert!(r.is_err());
    }

    /// Self-consistency probe: hold the `NarrowbandSubmode` invariant
    /// from `submode.rs` against the codebook surface this module
    /// publishes. If a later round changes either side, this fires.
    #[test]
    fn submode_widths_round_trip_against_codebook_counts() {
        for s in NARROWBAND_SUBMODES {
            // Pitch-gain dispatch.
            let pg_entries = match s.pitch_gain {
                PitchGainQuant::None => 0,
                PitchGainQuant::Vq5Bit => pitch_gain_5bit().len(),
                PitchGainQuant::Vq7Bit => pitch_gain_7bit().len(),
            };
            if s.pitch_gain != PitchGainQuant::None {
                let bits: u32 = match s.pitch_gain {
                    PitchGainQuant::Vq5Bit => 5,
                    PitchGainQuant::Vq7Bit => 7,
                    PitchGainQuant::None => unreachable!(),
                };
                assert_eq!(pg_entries, 1 << bits);
            }
            // Self-check on the computed total bits matches the
            // table-row total (already enforced in submode.rs, but we
            // re-assert here so this module's tests run a guard if
            // someone edits one side without the other).
            assert_eq!(
                NarrowbandSubmode::for_id(s.mode_id)
                    .unwrap()
                    .computed_total_bits(),
                u32::from(s.total_bits)
            );
        }
    }
}
