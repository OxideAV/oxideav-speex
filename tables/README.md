# oxideav-speex/tables/ — Speex CELP bit-exact codebook tables

Numeric codebook/table data the Speex algorithm narrative describes
structurally but does **not** print. The manual states the bit widths,
sub-vector sizes, and stage wiring of each quantiser; the actual
numeric arrays land here as **pure data** (facts; see the workspace's
clean-room policy), extracted by a dedicated data-only Extractor pass.

The CSVs in this directory are vendored into the crate so `cargo
build` against the standalone repository can read them via
`include_str!`. Each array is addressed by a role-named accessor in
`src/codebooks.rs` (LSP stage number, codebook sub-vector length /
index width, etc.). The full extraction manifest — per-table
dimension records, line ranges, source SHA-256 hashes, and the
data-only Extractor narrative — lives in the workspace umbrella
under `docs/audio/speex/tables/` and
`docs/audio/speex/provenance/01-speex-table-extraction.md`. The CSV
bytes in this directory are identical to the master copy there.

## Table inventory

### Narrowband LSP VQ codebooks

5 stages × 6 bits — stage 0 spans all 10 LSP coefficients; stages
1-4 are split-band 6-bit VQs over the low (coeffs 0-4) and high
(coeffs 5-9) halves. Per-stage decoder scaling regimes (divide by
256 / 512 / 1024) are documented in `src/codebooks.rs` via the
`NbLspScale` enum + `nb_lsp_scale(stage)` accessor.

- `nb-lsp-cdbk-stage0.csv` — 64 × 10
- `nb-lsp-cdbk-low1.csv` — 64 × 5
- `nb-lsp-cdbk-low2.csv` — 64 × 5
- `nb-lsp-cdbk-high1.csv` — 64 × 5
- `nb-lsp-cdbk-high2.csv` — 64 × 5

### High-band LSP MSVQ codebooks (wideband, SB-CELP)

2-stage MSVQ × 6 bits/stage (12 bits total), LPC order = 8.

- `hb-lsp-cdbk-stage1.csv` — 64 × 8
- `hb-lsp-cdbk-stage2.csv` — 64 × 8

### 3-tap pitch (adaptive-codebook) gain VQ

Rows are `[g0, g1, g2, search_aid]`; decoder adds a +32 bias to each
gain component before applying.

- `pitch-gain-cdbk-5bit.csv` — 32 × 4
- `pitch-gain-cdbk-7bit.csv` — 128 × 4

### Narrowband innovation codebooks

One CSV per (sub-vector length, codebook width) shape Table 9.1
references.

- `innovation-cdbk-sv5-64.csv` — 64 × 5 (6-bit)
- `innovation-cdbk-sv5-256.csv` — 256 × 5 (8-bit)
- `innovation-cdbk-sv8-128.csv` — 128 × 8 (7-bit)
- `innovation-cdbk-sv10-16.csv` — 16 × 10 (4-bit)
- `innovation-cdbk-sv10-32.csv` — 32 × 10 (5-bit)
- `innovation-cdbk-sv20-32.csv` — 32 × 20 (5-bit) — 3.95 kbps mode 8

### Wideband high-band innovation codebooks

- `hb-innovation-cdbk-sv8-128.csv` — 128 × 8 (7-bit with sign)
- `hb-innovation-cdbk-sv10-32.csv` — 32 × 10 (5-bit)

### Scalar excitation-gain quantiser tables

Wired through `src/gain_reconstruction.rs`. The full extraction
manifest is `docs/audio/speex/provenance/02-speex-gain-quant.md`.

- `nb-ol-gain-table-q15.csv` — 32 levels (5-bit NB OL excitation gain;
  float magnitude = `28406 · level / 2^15 / 16384` ≈ `exp(qe/3.5)`)
- `nb-exc-gain-scal3-float.csv` — 8 levels (3-bit NB sub-frame gain
  correction); `nb-exc-gain-scal3-bound-float.csv` — 7 boundaries
- `nb-exc-gain-scal1-float.csv` — 2 levels (1-bit NB sub-frame gain
  correction); `nb-exc-gain-scal1-bound-float.csv` — 1 boundary
- `hb-gc-quant-bound-float.csv` — 16 boundaries (4-bit HB gain
  correction; level = `0.87360 · bound`)
- `hb-fold-quant-bound-float.csv` — 32 boundaries/levels (5-bit HB
  folded gain)

### LPC / lag / QMF fixtures (Q15 + float variants)

- `lpc-analysis-window-{q15,float}.csv` — 200 samples (asymmetric)
- `lpc-autocorr-lag-window-{q15,float}.csv` — 11 taps (order+1)
- `qmf-filter-h0-{q15,float}.csv` — 64-tap QMF analysis filter

Only the Q15 variants are wired through `src/codebooks.rs` today;
the float variants are kept here for any future round that needs
them.
