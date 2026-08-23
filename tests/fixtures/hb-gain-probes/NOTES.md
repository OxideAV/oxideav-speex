# Crafted-bitstream probe measurements (round r450)

Method audit trail for the r450 law measurements. Unlike the r446
probes (synthetic *audio* through `speexenc`), these probes craft the
**bitstreams themselves** with the crate's own frame writers
(`write_narrowband_body` / `write_high_band_frame` — every transmitted
field chosen directly), wrap them in Ogg with a scratch muxer reusing a
committed fixture's header pages, and decode them with `speexdec
--no-enh` invoked as an **opaque binary** (workspace clean-room policy;
its source is NOT consulted). Because the encoder is out of the loop,
decoder laws are observed without encoder compensation — the confound
that limited every earlier black-box round.

- Toolchain: `speexdec` **1.2.1** (Homebrew build, macOS arm64).
- Per r446 convention the generator/analysis programs are
  session-scratch tooling (not tracked); this note plus the updated
  gates are the audit trail. All streams are deterministic (fixed field
  values; PRNG streams use the LCG
  `x ← 6364136223846793005·x + 1442695040888963407`, top-33-bit
  output, seeds `0x5EEC_2450…0x5EEC_2457`).
- Analysis: the reference decode is QMF-split with the staged 64-tap
  prototype (the provenance/08 instrument); per-sub-frame gains are
  least-squares fits of the recovered high band against the known
  innovation waveforms run through the crate's own synthesis filter
  (banded normal equations, alignment swept). Narrowband streams are
  decoded to 8 kHz directly, and the reference excitation is recovered
  by inverse-filtering with the known transmitted envelope.

## What the r450 probe families measured

1. **Alignment/parity.** With aperiodic (random-field) streams the
   reference output sits exactly **one sub-frame (40 samples at 8 kHz,
   both bands equally)** later than this crate's decode — a pure global
   delay. Because the QMF high band is recovered through a `(−1)ⁿ`
   modulation, a one-sample full-rate parity error *negates* the
   recovered high band: the r440 sub-band gate's even-only delay sweep
   (2×half-band) had pinned 142 where the truth is **143**, and the
   apparent mode-4 polarity inversion was that parity error. At odd
   parity the direct (positive) polarity matches for HB modes 2/3/4.
2. **HB modes 2/3/4 absolute gain law** (constant-gc grids over the
   full 4-bit range; gc steps at frame and sub-frame boundaries;
   random gc; innovation-row energy sweeps; zero-innovation gaps;
   low-band level/innovation/envelope/pitch variants; valid-codepoint
   NB LSP variants; an 8-point high-band LSP sweep; per-sub-frame
   isolation): `g = gc_recon · |A_hb(π)| · rms(e_lb)/|A_lb(π)|`, with
   `gc_recon = 0.87360·gc_quant_bound[q]` the staged reconstruction and
   **no further constant** (measured 0.852…0.884 across a 66×
   `|A_hb(π)|` range, mean ≈ 0.873 ≈ the staged multiplier). Linear in
   the correction (flat to 0.3 %), no innovation-energy term (< 1 %),
   no memory (single-sub-frame settling; unchanged after 32 zero
   frames), same-sub-frame drivers, one law for modes 2, 3 and 4
   (mode-2/mode-4 constants agree to 0.5 %). See
   `oxideav_speex::hb_gc_crossover_gain`.
3. **NB innovation path is exact.** Pure-innovation narrowband streams
   (random fields, zero pitch taps) decode with band-limited
   (300–3400 Hz) reference-vs-crate innovation scale **0.999** across
   modes 2/3/4/5/6/8 and both sub-frame gain-correction tables; the
   broadband residual is the reference's default output high-pass
   (measured separately: bilinear 2nd-order HP, `fc ≈ 80.7 Hz,
   Q ≈ 0.87` at 8 kHz; `fc ≈ 41.5 Hz, Q ≈ 1.12` at 16 kHz).
4. **NB mode-7 stage-2 weight = 0.455.** Two-regressor band-limited
   fits give stage-1 weight 0.9990…1.0003 and stage-2 weight
   0.4545…0.4555, constant across the 3-bit gain grid
   (`oxideav_speex::NB_MODE7_STAGE2_WEIGHT`). The high-band mode-4
   stage-2 weight re-measures as exactly the staged **0.4** (stage-2-only
   crafted streams fit at 1.0006 of stage-1-only through the
   0.4-weighted decode).
5. **Clipping guard.** Strong-resonance probes must keep reference
   peaks well inside i16 — saturated reference output masquerades as a
   unity-pole pitch loop (first tap-probe battery discarded for this).
