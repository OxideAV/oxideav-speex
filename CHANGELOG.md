# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Mode-1 comfort noise at the measured level** (r450 probes). The
  reference decodes the vocoder mode's zero-innovation frames to a
  deterministic decoder-side noise sequence scaled linearly by the
  frame OL gain (output rms `1.051·exp(qe/3.5)` through the lsp-0
  envelope, exact over a 31× grid; the sequence is identical across
  streams and not frame-periodic). The decoder now injects its own
  deterministic white sequence at that level
  (`NB_MODE1_NOISE_SCALE = 0.854` in the excitation domain,
  variance-compensated under the forced pitch loop) — mode-1 output is
  no longer noise-free; the reference's exact PRNG stream is
  unrecoverable black-box and is deliberately not carried over.
- The intensity-stereo README/gates reflect the measured §4.1 block
  phase; the stale stereo "block phase not reproduced" caveats are
  gone.

### Changed

- **Intensity stereo: the §4.1 block phase is measured and reproduced**
  (r450 crafted stereo probes — the stream header's channel field
  patched to 2 and every per-frame code-9 payload chosen directly).
  A sign-flip probe locates the channel switch **sample-exactly at
  `frame_start − N/4`**: the reference's gain block leads its decoded
  audio by one sub-frame (its mono path is buffered, its stereo gains
  are not), and a balance-step probe traces the in-block blend as
  exactly the documented `g(i) = g_new·(1−a^{N−i}) + g_prev·a^{N−i}`,
  `a = 0.980`. `StereoDecoder` now carries the mono signal one
  sub-frame so the interpolation lands on the measured phase. The
  steady laws confirm to 0.3 %: `ln(gL/gR) = bal/8` across the balance
  grid, and `gL² + gR²` reproduces the staged `1/e_ratio` table
  `{0.25, 0.315, 0.397, 0.5}` at every index. The stereo fixture's
  interleaved decode now tracks its (r450-lifted, 20.7 dB) mono decode
  within 1.5 dB (floors tightened).
- The ultra-wideband second layer's excitation-VQ modes are measured as
  **rejected by the reference decoder itself** (crafted 3-layer
  streams; see the uwb_decoder module docs) — the crate's typed
  `UwbLayerUndocumented` error is the conformant surface and the
  former "sub-frame geometry" gap closes as a negative result.

### Added

- **Wideband / ultra-wideband quality-10 (44 kbit/s) encoding lands**
  (r450). With the mode-4 binding, its exactly-0.4 stage-2 weight and
  the crossover-anchored absolute gain law all pinned, the encoder's
  high-band search now covers mode 4: stage 1 greedy over the five
  `sv8-128` slots at the reconstructed gain, stage 2 over the residual
  at weight 0.4, joint with the 4-bit gain grid — and the whole
  gain-correction family (modes 2/3/4) searches at the decode law's
  per-sub-frame base `|A_hb(π)|·rms(e_lb)/|A_lb(π)|` (the previous
  search scored without the base, mismatching the decoder).
  `HbInnovationMapping` gains the `DocumentedTwoStage` mode-4 variant
  (`decode_hb_subframe` rounds the exact f32 shape on its i16
  surface); `encode_packet_quality(10)` packs to the staged Table-10.2
  budget on both the wideband and ultra-wideband ladders.

### Changed

- **Ultra-wideband second-layer fold — exact outer crossover law, and
  the source is zero-stuffed** (r450 crafted 3-layer probes,
  `tests/fixtures/hb-gain-probes/NOTES.md`). Crafted UWB streams with
  known random first-high-band innovation and one-at-a-time sweeps of
  the second layer's 5-bit gain, its envelope, and the first band's
  envelope measure
  `e_l2 = 0.664 · fold_quant_bound[g5] · |A_l2(π)|/|A_hb1(0)| · (−1)ⁿ ·
  zerostuff(e_hb1)` — linear over the whole staged gain table, and the
  fold source is the first high band's excitation **zero-stuffed** to
  the 16 kHz half-band rate, both spectral images kept (fit score
  0.98–0.997 vs 0.61–0.67 for the r403 linear-interpolated source).
  Spectral continuity at the **8 kHz join**: after the two QMF folds
  the second layer's `π` edge and the first band's `0` edge both land
  at 8 kHz, completing the crossover-anchored architecture at every
  band boundary. Supersedes the r403 source and its
  `UWB_FOLD_RECONSTRUCTION_MULT = 1/16` flat constant; the r403-era
  "tone oracle vs speech oracle conflict" dissolves — the tone
  fixture's "independent" outer tones are the zero-stuffed source's
  images, which linear interpolation was suppressing.
- Conformance deltas (floors tightened): `hb-mode4-uwb-q10` full
  6.6→**20.2 dB** (alignment corr 0.9967), per-band mean error
  **0.10 / 0.49 / 0.70 dB**; `uwb-fold-geometry` tone full
  19.4→**22.6 dB / corr 0.9973**, outer band 8.8→**46.8 dB / corr
  1.00000**; `uwb-speech` bands **0.85 / 0.84 / 1.12 dB** (campaign-A
  divergence closed), its q4 tracking gate 2→**21.9 dB / 0.9968**;
  UWB output-high-pass row 26.9 dB / 0.99899.

### Changed

- **Narrowband pitch: the exact short-lag rules and the measured
  output high-pass land** (r450 crafted-bitstream probes). Three
  measured supersessions of r393/r410 fitted readings:
  - the 3-tap VQ adaptive codebook applies the §9.2 substitution
    **once** (common-`T` folding of the history read; a
    twice-substituted tail position, reachable only for `T < 21`,
    contributes zero) — crafted probe grids at `T = 22/33/57/61`
    recover the staged tables' taps **exactly** under this rule
    (`gain_scaled_pitch_subframe_repeat`), where the r410 in-sub-frame
    recursion misfits by 16–60 %;
  - the forced OL-pitch modes 1/8 run a true **unbounded centre-tap
    recursion** over the pitch partial with exact **float**
    coefficients `0.066667·quant` **capped at 0.99** for index 15
    (`T = 50` grid: fitted taps 0.2666…0.9333 and 0.9900, resid
    ≤ 0.04 %) — `gain_scaled_pitch_subframe_forced`; the r410
    recursion's 0.9 bound and the Q6 tap rounding are gone from this
    path;
  - the reference's default output high-pass is **measured** by
    cross-spectral transfer over the probe streams (the innovation path
    being reference-exact to 0.1 % isolates it): 8 kHz — bilinear
    biquad `fc ≈ 80.7 Hz, Q ≈ 0.87`; 16/32 kHz — third-order
    (biquad `fc ≈ 41.75 Hz, Q ≈ 1.38` × first-order ≈ 33 Hz), the
    32 kHz response matching the 16 kHz filter in absolute Hz.
    `OutputHighpass` now ships the measured transfers (still opt-in);
    the r393 30 Hz-Butterworth reading is superseded.
- Conformance deltas: the NB matrix **through the measured high-pass**
  jumps from 13.1–19.5 dB to **25.6–33.3 dB at corr 0.9986–0.9998**
  (tones 32–33.3 dB, speech 25.6–27.3 dB); raw rows hold or improve
  (speech-q2 10.5→11.6 dB) with energy ratios landing 0.98–0.99; the
  `hb-mode4-wb-q10` low band (NB mode 7) reaches **25.6 dB / corr
  0.9986**; the stereo fixture's mono decode lifts 13.8→20.7 dB (the
  §4.1 stereo block-phase approximation is now the visible term of the
  interleaved gate — recorded follow-up).

### Changed

- **Mode-1 folded high band — exact crossover-anchored law, and the
  fold source is the innovation-only excitation** (r450
  crafted-bitstream probes, `tests/fixtures/hb-gain-probes/NOTES.md`).
  Mode-1 streams sweeping the 5-bit folded-gain grid, the high-band
  envelope, the low-band envelope and the low-band level one at a time
  measure `s = 0.8518 · fold_quant_bound[g5] · |A_hb(π)| / |A_lb(π)|` —
  **linear over the whole 32-level staged table** (flat to 0.2 %; the
  r393/r410 `min(C·|Â|, 1/(2√2))` ceiling was an artefact of fitting
  without the `1/|A_lb(π)|` term) — and a two-source fit on a
  pitch-plus-random-innovation stream splits the fold source cleanly:
  innovation weight matches the law exactly, **pitch weight −0.0001**.
  The r393/r403 composed-excitation source (`e = p + c`) is superseded:
  only `g·c[n]` folds ([`NarrowbandDecoder::last_frame_innovation`],
  mirrored in the encoder's mode-1 gain selection). Same
  spectral-continuity architecture as the modes-2/3/4 law.
- Conformance deltas (floors tightened): `wb-q4` speech
  15.6→**20.5 dB** full-signal, its high band **−6.9 dB / corr 0.49 →
  +16.5 dB / corr 0.989** (the fitted law folded the pitch contribution
  too, up to ≈10× hot in energy on pitchy frames); `wb-mode1-folded`
  tone high band 38.9→**44.1 dB / corr 1.00000**; `uwb-speech` bands
  1.12/4.77/7.06→**0.96/0.92/3.17 dB** (the campaign-A "outer-layer
  divergence" was mostly the inner fold's law+source; the remaining
  8–16 kHz tail is the r403 outer-source law, still open); UWB tone
  full 19.1→19.4 dB.

### Changed

- **High-band modes 2/3/4 absolute innovation gain — the exact law
  lands** (r450, crafted-bitstream probes,
  `tests/fixtures/hb-gain-probes/NOTES.md`). The crate's frame writers
  now double as a probe generator: streams whose per-sub-frame gain
  index, innovation content, low-band level / envelope / innovation /
  pitch, and high-band envelope are varied **one at a time** were
  decoded by the reference binary (black-box), and the per-sub-frame
  high-band gain measures as
  `g = gc_recon · |A_hb(π)| · rms(e_lb) / |A_lb(π)|` — the transmitted
  4-bit correction times the ratio anchoring the reconstructed high
  band to the low band's spectral amplitude at the 4 kHz QMF crossover,
  with `gc_recon`'s staged `0.87360` multiplier as the **only**
  constant (measured 0.852…0.884 across a 66× envelope range). Linear
  in the correction, no innovation-energy term, no backward-adaptive
  memory, one law for modes 2/3/4 (`hb_gc_crossover_gain`, wired
  per-sub-frame through `NarrowbandDecoder::last_crossover_response`).
  Supersedes the r440 fixture-fitted `(gc·lb_rms)²` reading and the
  r446 "backward-adaptive memory" interpretation (natural-speech
  co-variation and envelope motion, respectively). The mode-4
  innovation polarity flips back to direct (`HB_INNOVATION_POLARITY =
  1.0`): the r440 flip compensated a one-sample **parity** error in the
  fixture alignment (an odd full-rate offset negates a QMF-recovered
  high band), not a real inversion.
- **Narrowband mode-7 stage-2 weight = 0.455** (r450, measured
  0.4545…0.4555 across the gain grid, constant;
  `NB_MODE7_STAGE2_WEIGHT`, exact on the new `decode_subframe_f32`
  path). The unweighted stage sum made every quality-10 low band
  ≈ +1.6 dB hot. The analogous stage-isolation probe re-confirms the
  high-band mode-4 stage-2 weight as exactly the staged 0.4.
- Conformance deltas from the two laws (gates tightened accordingly):
  `hb-mode4-wb-q10` isolated 4–8 kHz sub-band **−7.1 dB / corr −0.54 →
  +21.1 dB / corr 0.998** (at the parity-corrected 143-sample
  alignment; energy ratio 1.12), low band 13.7 dB / 0.979 / energy
  0.99; `hb-mode4-uwb-q10` per-band mean error 1.29→**0.29** /
  5.55→**0.81** / 7.98→6.47 dB; wideband speech matrix `wb-q6`
  15.6(18.3)→**20.1 dB** raw full-signal (energy 0.987), `wb-q8` →
  **20.8 dB**; the r446 gain-probe pair decodes at ≤ 0.4 dB per-segment
  band error in both bands (was a pinned 6…27 dB divergence).
- The r446 probe conclusions are re-read by the r450 gate docs: the
  "backward-adaptive base" evidence was the transmitted high-band
  envelope moving `|A_hb(π)|` under encoder control — with the encoder
  out of the loop, every driver is same-sub-frame and memoryless.

### Added

- **QMF-recovered high-band excitation — full crate-machinery
  replication of provenance/08** (r446,
  `tests/hb_mode4_recovered_gain_table.rs`). Docs round 8 recovered the
  isolated 4–8 kHz sub-band of the `hb-mode4-wb-q10` oracle from staged
  bytes alone and staged the result as
  `tables/hb-mode4-recovered-gain.csv` (299 sub-frame rows); this round
  re-runs the whole measurement through the crate's own machinery —
  `QmfAnalysis` as the instrument, `decode_hb_subframe_mode4_f32` as
  the innovation rebuild (80-bit group packing + leading sign bit +
  exact 0.4 stage-2 weight all exercised), `reconstruct_hb_exc_gain`
  for the correction column — and gates:
  - the **binding confirmation, oracle-free**: 299 rows, mean |ρ|
    **0.8839** (doc: 0.8617), median 0.9042, 93.3 % above 0.8, sign
    positive on 97.0 %, with the alignment sweep peaking **uniquely at
    the staged −40** (3.2× margin over every other delay in
    −260…+59);
  - the **per-row staged-table cross-check** (docs checkout,
    skip-if-absent): the filter-free `lb_frame_rms` column reproduces
    to < 3 × 10⁻⁵ log — the crate's analysis bank *is* the staged
    table's instrument, including its window convention (all windows
    at the −40 offset, zero-padded, /160-normalised, real extent
    `(pcm_len + 63)/2`) — and the LPC-derived ρ / RMS / projection
    columns agree to mean |Δρ| 0.026 with a small tail on sparse
    `gc = 0` sub-frames below the doc's own |ρ| > 0.6 regression cut;
  - the **gain-law direction**: on the staged rows the crate's
    fixed-exponent reading (`HB_GC_STATE_EXP_GC = HB_GC_STATE_EXP_LB
    = 2`) lands the doc's regression **exactly** — R² 0.791 /
    rms 8.89 dB, the transmitted correction alone R² 0.005, both-at-1
    12.86 dB — with the absolute intercept left free (provenance/08
    deliberately asserts no closed-form law; the exact law remains the
    recorded docs gap).

  `frame-trace.txt` is mirrored into `tests/fixtures/hb-mode4-wb-q10/`
  (existing oracle-mirroring precedent); the mode-4 shape decoder and
  stage-2 weight are re-exported as `#[doc(hidden)]` test plumbing.

- **QMF-recovered mode-4 excitation on the SECOND fixture — the
  replication provenance/08 asked for** (r446,
  `tests/hb_mode4_uwb_recovered_excitation.rs`). Provenance/08 names
  "the `hb-mode4-uwb-q10` trace re-emitted in the WB format" as one of
  the three asks that would close residual 1, blocked only on its own
  trace reader; the staged trace parses fine under this crate's
  reader, so the second replication runs now. The 4–8 kHz sub-band is
  recovered through a **two-stage** crate-`QmfAnalysis` split
  (32 kHz → 16 kHz low half → 8 kHz sub-bands) with a **per-frame**
  order-8 LPC fit (this fixture's transmitted envelope varies, so the
  doc's global fit is not legitimate here). Results, all gated:
  - **binding replication #2, oracle-free**: 298 sub-frames, mean |ρ|
    **0.9316** (above the WB fixture's 0.86–0.88), positive on
    99.7 %, alignment peak **uniquely at the same −40** with a 3.8×
    margin — the §1 binding now rests on two independent fixtures,
    one with a varying LSP envelope, through two analysis stages;
  - the r440 fixed-2 gain reading stays serviceable on rows it was
    never fitted to (R² 0.746);
  - **new docs-gap evidence, pinned**: the fixed-2 residual is
    **56 % explained by which LSP envelope the frame transmits**
    (17 classes) — an envelope term the constant-LSP WB fixture could
    not show — and a decoder-**state** term (previous sub-frame's
    recovered excitation RMS) beats the same-frame low-band level
    (free-fit R² 0.900 vs 0.815; the same ordering holds on the WB
    fixture's rows, 0.875 vs 0.796). **No law is adopted**: the
    free-fit exponents are not stable across the two fixtures
    (gc 0.21↔0.92, state 0.66↔0.82), and fitting a loose formula is
    exactly what provenance/07/08 decline to do. The refined
    discriminating ask is recorded in the README.

- **Mode-4 gain-base discrimination — the provenance/08 fixture pair,
  generated and gated** (r446, `tests/hb_mode4_gain_probe_fixture.rs`
  + `tests/fixtures/hb-mode4-gain-probes/`, black-box per the r410
  conformance-fixture precedent). Provenance/08 names "a fixture pair
  differing only in the low-band content at fixed high-band bits, or
  vice versa" as what would close #329 residual 1; this round
  generated exactly that pair (one deterministic source, five
  ×{0.05…1.9} amplitude segments applied to one band at a time,
  `speexenc -w --quality 10`, all-mode-4 streams) and ran the
  QMF-recovered measurement over it. **The drivers are settled:**
  - `lbvar`: across a **31.6 dB** low-band sweep at fixed high-band
    content, the recovered per-sub-frame gain moves ≈ **2 dB** — the
    reference's gain base is **not** the same frame's low-band level
    (a causal `lb²` base would swing ≈ 63 dB); provenance/08's
    low-band R² was natural-speech co-variation.
  - `hbvar`: across an 18 dB high-band sweep at fixed low band, the
    recovered gain tracks the level while the transmitted 4-bit
    correction stays **parked at the grid bottom** (median index 0–2
    per segment) — the base is **backward-adaptive decoder state**
    (recent high-band excitation memory, ≈ 1–2-frame settling at
    segment steps), not the transmitted field.
  - A third stream (envelope sweep at fixed levels, not committed —
    NOTES.md) shows only a weak non-monotonic envelope effect: the
    LSP envelope is not the driver either.
  The exact predictor **update rule** (time constant, domain,
  cold-start, correction feedback) is not recoverable from
  steady-state black-box probing — a backward-adaptive loop with
  wrong constants accumulates multiplicative drift — so the decode
  law is **unchanged** and the update rule is the recorded docs ask.
  The gate additionally pins the r440 fitted law's off-manifold
  behaviour as a **known divergence** (4–8 kHz per-segment mean band
  error 5.9–23 dB with per-segment ceilings; low band unaffected,
  ≤ 2.2 dB): when the update rule lands, this gate is the immediate
  validation target.

- **Ultra-wideband quality-10 (stacked mode-4) conformance gate**
  (r446, `tests/hb_mode4_uwb_fixture.rs` on the staged
  `docs/audio/speex/fixtures/hb-mode4-uwb-q10/` oracle, mirrored per
  precedent). The 32 kHz quality-10 default is the stacked case no
  prior gate covered: inner (4–8 kHz) high-band layer in **submode 4**
  (352-bit layer), outer (8–16 kHz) layer in folded submode 1, so the
  outer fold's source is an innovation-coded high band for the first
  time. The gate pins:
  - **bit-exact framing across all 76 frames** — NB submode 7, the
    inner layer's LSP MSVQ pair, per-sub-frame 4-bit gain correction
    and the packed **80-bit mode-4 excitation field verbatim**, and
    the outer layer's LSP pair + four 5-bit folded gains, all against
    the reference decoder's own `frame-trace.txt` (the first framing
    validation of submode 4 inside the embedded UWB recursion);
  - **the stream decodes** (a q10 UWB stream was undecodable before
    campaign B, and no gate had ever decoded one) — decoder delay
    pinned at 351 samples / 32 kHz against the source-length-trimmed
    reference;
  - **per-band tracking**: 0–4 kHz **1.29 dB** mean |err|
    (reference-tracking), 4–8 kHz **5.55 dB** — the r440
    state-derived mode-4 gain base replicating its wideband-fixture
    figure (≈6.1 dB) on a second stream whose high-band LSP pair
    **varies frame to frame** (the WB oracle holds it constant), so
    the level-tracking law is not a one-fixture artifact — and
    8–16 kHz **7.98 dB**, the same documented campaign-A outer-fold
    residual as `uwb-speech-3layer` (`hb-folded-gain.md` §7.4 docs
    gap).

- **QMF band-isolation exactness gate** (`tests/qmf_band_isolation.rs`,
  r440). `provenance/08-qmf-recovered-hb-excitation.md` validates the
  staged 64-tap prototype as a measurement instrument (88–95 dB
  sub-band isolation on six probe tones); the crate's `QmfAnalysis`
  reproduces all six documented figures **to their 0.1 dB print
  precision** (−95.10 / −95.61 / −87.94 / +87.94 / +95.61 / +95.10 dB),
  and the gate pins the match at ±0.5 dB plus mirror-pair symmetry —
  the crate's analysis bank *is* the provenance instrument, tap for tap
  and sign for sign.
- **QMF-route sub-band conformance gate** for the mode-4 q10 oracle
  (`tests/hb_mode4_fixture.rs::mode4_q10_qmf_subband_gate`): splits
  reference and crate decode into their true 8 kHz sub-bands with the
  staged prototype and scores each absolutely; also measures + pins the
  decoder's alignment against the source-length-trimmed `expected.pcm`
  (≈142 full-rate samples of codec look-ahead the toolchain trimmed).
- **High-band mode-4 absolute gain — state-derived base**
  (provenance/08, r440). The gc-only reading (gain =
  `0.87360·gc_bound[q]`) is measured by the staged material as *not*
  the gain (`R² = 0.005`); the base tracks the same frame's
  reconstructed **low-band level**. Mode 4 now decodes with
  `g = HB_GC_STATE_SCALE · (gc_recon · lb_frame_rms)²` — the doc's
  nearly-free fixed-exponent reading with the absolute scale
  fixture-calibrated (documented fitted; the exact law remains the
  recorded docs gap). Threaded as
  `synthesise_high_band_frame_leveled` / 
  `gain_scaled_hb_innovation_from_body_leveled` (legacy entries keep
  the old behaviour); scoped to mode 4 — adopting the
  mode-4-calibrated law for modes 2/3 measurably regresses the staged
  `wb_q6` oracle, so those await their own fixture.
- **High-band mode-4 innovation polarity pinned**
  (`HB_INNOVATION_POLARITY = −1` through this crate's QMF conventions)
  — the one-bit trial `hb-innovation-binding.md` §4 prescribes, run
  against `expected.pcm`: the direct reading correlates the isolated
  4–8 kHz sub-band **negatively** (−0.27…−0.47, strengthening with
  gain scale), the flip positively at the same magnitude. Paired with
  the crate's `g1[n] = −2·(−1)ⁿ·h0[n]` synthesis convention, like the
  r393 fold sign.
- **Measured mode-4 delta** (staged `hb-mode4-wb-q10`): 4–8 kHz band
  mean |err| **13.11 → 6.11 dB**, isolated high sub-band correlation
  **≈0 → +0.44**, energy ratio 0.086 → 0.74; low band unchanged
  (0–4 kHz 2.2 dB). All prior anchors unchanged (wb-q4/q6/q8 15.59 /
  18.30 / 18.31 dB, mode-1 fold 38.87 dB / 0.99994, UWB 19.15 dB,
  stereo ladder).

- **§7.6 crossover-response parity + compression model surfaced**
  (r440, `hb-folded-gain.md` §7.6 + the newly staged
  `tables/hb-fold-envelope-vs-transmitted.csv`).
  `hb_crossover_response_from_lsp` is the pinned odd-parity closed form
  `|Â(π)| = Π 4·cos²(ωᵢ/2)` over the odd-indexed LSP class, verified
  equal to the crate's direct polynomial evaluation on every swept
  envelope (≤0.008 log10); `hb_crossover_response_bwexp` +
  `HB_FOLD_ENVELOPE_COMPRESSION_GAMMA = 0.944` realise §7.6's
  bandwidth-expansion compression model (surfaced as *inferred*, per
  the doc — not wired into the default fold scale, whose ceiling law
  remains the best the crate realises against the anchor oracles).
  `tests/fold_envelope_sweep.rs` now cross-checks the crate's chain
  against the staged CSV's `dlog10_Api_odd_parity` and
  `dlog10_Api_odd_bwexp_0p944` columns — reproduced to ≤0.007 log10 on
  all six settings (skip-if-absent on standalone checkouts).

- **Exact float-build OL-gain quantiser law**
  (`quantise_frame_ol_exc_gain_exact`, r440). The staged provenance/02
  records the float encoder's exact expression —
  `qe = floor(0.5 + 3.5·ln(gain))`, clamped `[0, 31]`, level
  `exp(qe/3.5)` — i.e. round-to-nearest in the log-gain domain. The
  narrowband encoder's first frame-gain estimate now uses that exact
  law instead of the Q15 `scal_quant32` threshold walk (which resolves
  in-cell gains upward); the closed-loop refinement is unchanged and
  every encoder round-trip / conformance gate holds. Unit gates pin
  the level round-trip, log-domain midpoint rounding, clamps, and
  ≤1-step agreement with the threshold path.

### Fixed

- **High-band mode-3 sign/index wire order** now matches the staged
  binding (`docs/audio/speex/hb-innovation-binding.md` §1): each 8-bit
  `sv8-128` group is a **leading 1-bit polarity sign followed by the
  7-bit codebook index**, for mode 3 exactly as for mode 4 (the §2.2
  `V ^ 0x80` pair-sum measurement pins the group MSB as the sign for
  both modes; provenance/08 re-confirms the layout from staged bytes
  alone). Rounds ≤ r438 read mode 3's group as `[7-bit index][sign]`
  from the tables/README shorthand — encoder and decoder agreed with
  each other, so round-trips passed, but reference mode-3 streams
  decoded with the index and sign fields crossed. `decode_hb_subframe`
  and `search_hb_innovation` both now use the measured `[sign][index]`
  split; new unit gates pin the §2.2 pair-sum negation, the
  mode-3-equals-mode-4-stage-1 group split, and row 43 as the unique
  all-zero `sv8-128` sub-vector (the "no excitation" symbol).
  Measured on the staged `wb_q8` speech fixture (HB submode 3 in every
  frame): full-signal 18.22 → 18.31 dB.

### Added

- Campaign B: **intensity stereo — true in-band decode and encode**
  (`src/stereo.rs`, wired through `SpeexStreamDecoder` +
  `SpeexFrameworkDecoder`/`Encoder`). Implements the clean-room law of
  `docs/audio/speex/intensity-stereo.md`: the 8-bit code-9 payload
  (`[sign][5-bal][2-e_ratio]`) reconstructs an `(gL,gR)` gain pair
  (`b=exp(bal/8)`, `F=√(0.5/e_ratio)`, `e_ratio={0.25,0.315,0.397,0.5}`)
  with the §4 intra-frame interpolation (`a=0.980`), producing
  interleaved L/R from the single mono decode; the encoder emits the
  `(L+R)/2` downmix with the per-frame code-9 message prefixed and
  declares 2 channels. New gate `tests/intensity_stereo_fixture.rs` on
  the staged `stereo-nb-ladder-q4` oracle shows the interleaved SNR
  **tracks the mono decode within ~0.4 dB** (13.40 vs 13.83 dB) — the
  L/R reconstruction adds no material error; the §4.1 sub-frame
  block-phase offset is not reproduced (bounds byte-exactness).
- Campaign B: **high-band mode 4 (WB/UWB quality 10) now decodes.**
  Bound the 80-bit two-stage innovation per
  `docs/audio/speex/hb-innovation-binding.md` (2 × 5 × 8-bit groups over
  the same five `sv8-128` 8-sample slots, MSB sign bit, stage 2 at
  weight 0.4; `hb_innovation::decode_hb_subframe_mode4_f32`). A q10
  wideband stream that previously surfaced a docs-gap error now decodes;
  new gate `tests/hb_mode4_fixture.rs` pins the low band reference-
  tracking (0–4 kHz ≈ 2.2 dB) and the mode-4 high band as a documented
  residual floor (4–8 kHz ≈ 13 dB, see below).
- Campaign B: **folded-band scale-law validation** against the newly
  staged `fold-envelope-sweep` material (`tests/fold_envelope_sweep.rs`).
  A decode-free check that the crate's own `|Â(π)|` reproduces the
  reference decoder's measured per-band scale ratios: it matches to
  ~0.5 dB (≤0.031 log10) for the shallow/mid envelopes `i1∈{0,8,20}`,
  confirming the kneeless `s=C·|Â(π)|` law of `hb-folded-gain.md`
  §7.3/§7.5 where it is pinned, and diverging only at the near-
  degenerate deep envelopes `i1∈{33,49,63}` (`|Â(π)|≲0.05`) the doc
  itself flags.

### Fixed / investigated

- Campaign B **fold reconciliation (no net-positive decode change).**
  Both newly-staged reconciliation laws were implemented behind switches
  and measured against the staged oracles' `expected.pcm`: (a) the
  kneeless linear inner law `C·|Â(π)|` (no ceiling) regresses the
  `wb-mode1-folded` tone fixture (best 32.6 dB at C=0.15 vs 38.9 dB
  flat); (b) the §7.4 synthesized-WB-HB-signal outer source with a
  crossover-shaped scale regresses **both** the tone oracle (19→6 dB)
  and the speech 8–16 kHz band-mean (7.06→7.9+ dB) at every C. The
  `fold_envelope_sweep` validation shows *why* the two coexist: the
  crate's `|Â(π)|` is the reference's normalising response only in the
  shallow/mid range, so the kneeless law over-amplifies at the high
  `|Â(π)|` the anchor tone fixtures operate at. The default decode path
  is therefore unchanged (the ceiling law remains the best the crate
  realises); the residual is the high-`|Â(π)|` crossover-response
  normalisation + the unpinned outer image weighting — precise docs
  gaps (README).
- Campaign B **HB-innovation-mode residual (modes 2/3/4).** The mode-4
  q10 fixture is the first reference validation of an HB *innovation*
  mode: the two-stage shape decodes but the 4–8 kHz band tracks to
  ~13 dB (doc-faithful magnitude) / ~8 dB (best constant magnitude fit).
  A constant magnitude cannot close it, localising the residual to the
  **absolute per-frame HB-innovation gain/energy law**, which the staged
  evidence (codebook *shape* + the 0.4 stage weight, isolated via
  sign-difference) does not pin. The WB/UWB stereo ladders confirm the
  same: their mono decode is bounded by this same HB-innovation residual,
  not the stereo law. Recorded as a precise docs gap. WB/UWB quality-10
  *encoding* stays declined (decode works; encode search + the gain law
  are unpinned).

- Campaign A: **ultra-wideband 3-layer speech conformance gate**
  (`tests/uwb_speech_3layer_fixture.rs`) on the staged
  `docs/audio/speex/fixtures/uwb-speech-3layer/` oracle. Its
  `framing_matches_reference_trace` re-parses the stream and asserts —
  **bit-exactly across all 126 frames** — every NB / high-band /
  ultra-wideband sub-mode, both high-band layers' 12-bit LSP MSVQ stage
  indices `(i1,i2)`, and all eight 5-bit folded-gain indices against the
  reference decoder's own per-frame `frame-trace.txt`, proving the whole
  three-layer parse + index-extraction path is reference-exact.
  `decode_tracks_reference` pins the 32 kHz decode's full-signal SNR
  (16.33 dB) and per-band mean |error| (0–4 kHz ≈ 1.1 dB accurate;
  4–8 / 8–16 kHz ≈ 4.8 / 7.1 dB tracking) so a reconstruction fix shows
  up as a floor raise and a regression fails loudly.

### Fixed / investigated

- Campaign A **differential-debug of the UWB speech divergence**. The
  bit-exact framing (above) localises the residual to the folded
  high-band *reconstruction*, not the parse. `hb-folded-gain.md` §7.4's
  outer-layer law (fold source = the wideband layer's **synthesized**
  high-band signal, 2×-upsampled with the zero-stuff image pair, scaled
  by the same kneeless `C·|Â_uwb(π)|` law as the WB layer, `C = 0.17`
  coinciding with the inner slope) drives the 8–16 kHz band from ≈7 dB
  to **<1 dB** on the frames whose fold source is itself accurate — but
  it **conflicts with the `uwb-fold-geometry` tone oracle** (independent
  10/13.5 kHz outer tones a gain-only mode-1 fold cannot reproduce; the
  synthesized-signal source over-amplifies its resonant envelope,
  regressing that fixture 19 → 6 dB). No single scale reconciles the two
  staged oracles, so the default decode path is unchanged and the exact
  outer source normalisation is recorded as a docs gap (README). Also
  confirmed: **no staged fixture carries high-band mode-4 material** —
  all three high-band fixtures are all-mode-1 — so the mode-4 codebook
  binding (WB/UWB quality 10) cannot be observer-probed and stays a
  precise docs gap.

## [0.0.9](https://github.com/OxideAV/oxideav-speex/compare/v0.0.8...v0.0.9) - 2026-08-06

### Other

- README + CHANGELOG for the r438 framework wiring, mode 1/7 binding, and stereo/HB-mode-4 gap posture
- wire the oxideav-core registry surface — real Decoder/Encoder factories, dual-API make_ functions, header writer
- bind modes 1 and 7 from the staged innovation-binding doc — every Table 9.1 mode now encodes and decodes

### Added

- Round r438: **the `oxideav-core` framework surface is wired** — the
  generic codec entry points no longer return `NotImplemented`.
  `register()` installs real decoder + encoder factories under the id
  `"speex"` (claiming the Ogg payload magic `Speex   `), and the
  dual-API `make_decoder` / `make_encoder` free functions expose the
  same factories directly. The framework decoder drives
  `SpeexStreamDecoder` (header from `extradata`, from `sample_rate`, or
  from the in-band `Speex   ` header packet with comment/extra-header
  skip) and emits interleaved-S16 `AudioFrame`s; the framework encoder
  re-blocks arbitrary-length S16 input into 20 ms frames, emits one
  self-contained packet per frame, honours a `quality` (0..=10) option,
  and carries the 80-byte stream header in `output_params().extradata`
  (`SpeexHeader::write_bytes`, the new exact Table 7.1 serialiser).
- Round r438: **narrowband modes 1 and 7 decode and encode**, bound by
  the staged `docs/audio/speex/nb-innovation-binding.md`: mode 1 (2.15
  kbps vocoder, quality 0) has **no innovation codebook** — its four
  1-bit innovation-gain fields are read and discarded (they are inert
  in the reference decoder) and the excitation is the frame-level
  forced pitch path, which the encoder now drives with a real
  open-loop pitch estimate and the staged `provenance/02` forced-gain
  law (0.9 damping, `15·coef` on the 4-bit grid); mode 7 (24.6 kbps,
  quality 10) is **two independent 48-bit innovation stages** of eight
  6-bit `sv5-64` lookups summed by the decoder, with the encoder
  searching stage 2 on the residual stage 1 leaves. Every narrowband
  quality 0..=10 — and wideband/ultra-wideband 0..=9 — now encodes;
  WB/UWB quality 10 stays gated on the high-band mode-4
  innovation-binding docs gap and is rejected with that gap named.
- Round r438: **2-channel stream handling (documented fallback, not
  intensity stereo)**. The staged material defines the in-band
  intensity-stereo message's existence and 8-bit size (Table 5.1 code
  9) but not its payload semantics or L/R reconstruction law — a
  recorded docs gap — so the framework decoder emits shape-correct
  interleaved 2-channel PCM with both channels carrying the
  transmitted signal, and the framework encoder accepts 2-channel
  input by `(L+R)/2` downmix to a mono stream.

### Changed

- Mode-2 encoding now searches its 3-tap pitch-gain VQ at the
  transmitted frame-level open-loop period (the wire carries no
  per-sub-frame lag for that mode), and mode 8 transmits a real forced
  open-loop pitch gain instead of a hardwired `0` — both previously
  paired gains with untransmitted or zeroed lags.
- Removed `Error::NotImplemented`: no code path returns it since the
  framework wiring landed.

## [0.0.8](https://github.com/OxideAV/oxideav-speex/compare/v0.0.7...v0.0.8) - 2026-07-17

### Other

- doc(hidden) the internal CELP plumbing re-exports
- speech-material tracking gate exposes a distinct 3-layer divergence
- docs(readme) + gates: record r410 conformance matrix; raise WB low-band floors
- crossover-shaped folded high-band law, oracle-arbitrated + speech-fixture conformance gate
- pitch VQ lag-order + short-pitch recursion arbitrated against reference decodes
- *(readme)* record r403 ultra-wideband 3-layer external validation
- add decoder robustness / fuzz gate
- gate the output high-pass improvement on the 3-layer fixture
- add 3-layer fold-geometry fixture conformance gate
- pin second-layer fold source against staged 3-layer fixture

### Changed

- Marked the internal CELP-chain plumbing re-exports (LSP/LPC conversion,
  codebook tables, gain grids, QMF/synthesis filters, pitch/innovation
  search — 225 crate-root items) `#[doc(hidden)]`, so semver tooling
  tracks only the documented decode/encode/framing surface; no semantic
  or signature changes.

### Fixed

- Round r410: **3-tap pitch-gain VQ column ↔ lag association reversed**
  (fixture-arbitrated). The staged pitch-gain codebook `.meta` labels
  columns 0–2 `g0, g1, g2` but does not pin which column multiplies
  which Eq. 9.1 lag; the crate's original direct reading (column 0 ↔
  `e[n−T−1]`) decoded the VQ sub-modes at 2.8–5.6 dB absolute SNR with
  **half** the reference energy. Black-box arbitration against the new
  `tests/fixtures/nb-conformance/` reference decodes pins the reversed
  association (column 0 ↔ `e[n−T+1]`): `pitch_gain::reconstruct` now
  maps `taps[0] = col2, taps[1] = col1, taps[2] = col0`, lifting the VQ
  sub-modes to 13.4–14.4 dB / 0.98 energy ratio.
- Round r410: **in-sub-frame pitch recursion for short periods**
  (`gain_scaled_pitch_subframe_recursive`, now used by
  `NarrowbandDecoder`). For `T < 40` the §9.2 reads at `n − T + off ≥ 0`
  resolve to the **pitch-only partial** already produced inside the
  current sub-frame (with the tap gains bounded to a 0.9 total for
  those recursive reads — the staged `provenance/02` 0.9
  pitch-coefficient constant family), not to the manual's
  `n − 2T + off` repeat rule. Arbitrated black-box across ten staged
  reference decodes: the repeat rule scores 6.8 / 9.3 dB on the
  OL-pitch sub-modes 8 / 2 where the recursion scores 12.7–13.4 dB;
  recursing over the composed excitation instead of the pitch-only
  partial overshoots energy 2.8×. Fine-pitch sub-modes are within
  ±0.4 dB of the repeat rule (recorded residual ambiguity).

- Round r410: **crossover-shaped folded high-band law**
  (`hb_fold::folded_hb_scale`, superseding the flat r393 constant in
  the decode + encode paths). The r393 `K = 1/(2·√2)` matched the
  near-stationary tone fixture but overshot real speech by up to 130×
  per frame in envelope troughs (`wb_q4` fixture: −12.9 dB full-signal,
  20× energy). Synthetic oracle streams sweeping the 12-bit high-band
  LSP envelope against the reference decoder pin the missing factor as
  the envelope's **magnitude response at the 4 kHz QMF crossover**:
  `scale = min(0.17·|A_hb(π)|, 0.35355)·g` — linear (slope
  0.171…0.189) over the probed `|A_hb(π)| ≤ 1.4` region, saturating at
  the r393 constant where both real-stream anchors sit
  (`wb-mode1-folded` 2.4…3.6, `uwb-fold-geometry` embedded layer 3.56;
  both unchanged to within 0.01 dB). `wb_q4` moves to 15.6 dB /
  1.01× energy. A residual per-frame factor ≤ ≈6× on some speech
  frames remains unattributed (follow-up).

### Added

- Round r410: **ultra-wideband speech tracking gate**
  (`tests/uwb_conformance_fixture.rs` + `tests/fixtures/uwb-conformance/`)
  — a 32 kHz speech-like quality-4 fixture pinning a newly measured
  **known divergence** of the 3-layer path on non-stationary material
  (2.0 dB / 0.78 corr / 1.66× energy full-signal; the embedded 0–8 kHz
  half scores 2.3 dB where the same code scores 15.6 dB on the
  standalone wideband speech fixture, and the 8–16 kHz folded layer
  overshoots ≈10×). The divergence is *not* the inner fold law (shaping
  the outer fold by analogy regresses the tone anchor without fixing
  this) — recorded follow-up; the gate holds tracking floors.

- Round r410: **wideband decode-conformance matrix**
  (`tests/wb_conformance_fixture.rs` + `tests/fixtures/wb-conformance/`)
  — speech-like (pitch-glide) 16 kHz fixtures at qualities 4/6/8
  (Table 10.2 ladders NB 4 + HB 1, NB 5 + HB 2, NB 6 + HB 3), the first
  reference comparison of the high-band excitation-VQ sub-modes 2/3
  (18.3 / 18.2 dB full-signal) and of the folded sub-mode on speech
  (15.6 dB), CI-gated with per-fixture alignment pins.
- Round r410: **narrowband decode-conformance matrix**
  (`tests/nb_conformance_fixture.rs` + `tests/fixtures/nb-conformance/`)
  — ten black-box `speexenc`/`speexdec --no-enh` reference fixtures
  (tone-mix at qualities 1/2/3/5/7/9 = Table 9.2 sub-modes
  8/2/3/4/5/6, plus a pitch-gliding speech-like source at qualities
  1/2/3/7), each CI-gated on absolute SNR / correlation / energy-ratio
  floors, raw and through the fitted `OutputHighpass`, with the
  40-sample look-ahead alignment pinned. Measured r410: raw
  10.5–14.4 dB (energy 0.95–1.08), high-passed 13.1–19.5 dB — up from
  2.8–6.8 dB (energy 0.50–1.02) before the two pitch-path fixes.

- Round r393: **WB mode-1 folded high-band excitation — externally
  arbitrated and wired** (`hb_fold` module; closes the #170 fold-law
  gap for the wideband layer). The staged
  `docs/audio/speex/fixtures/wb-mode1-folded/` reference-decode fixture
  pins the reconstruction law `e_hb[n] = K·g·(−1)ⁿ·e_lb[n]`: the fold
  source is the embedded narrowband frame's composed excitation
  (`NarrowbandDecoder::last_frame_excitation`, new accessor), the
  `(−1)ⁿ` modulation is the sample-level spectral fold (candidate
  conventions without it score ≤ 0.31 high-band correlation vs 0.9999),
  `g` is the staged 32-level `fold_quant_bound` level, and
  `K = HB_FOLD_RECONSTRUCTION_MULT` (adopted `1/(2·√2)`, inside the
  fixture-measured `0.3516…0.3549` window). Wired through
  `synthesise_high_band_frame_folded` into `WidebandDecoder` (and the
  top-level `SpeexDecoder` / `SpeexStreamDecoder` paths).
- Round r393: **`INNOVATION_CODEBOOK_SCALE` (1/32) Q5 codebook-row
  normalisation** — the same fixture calibrates the absolute signal
  domain: the `signed char` innovation rows enter the excitation as Q5
  fractions (`c[n] = g·c_raw[n]/32`; measured `0.03154` ≈ `1/32` by
  least squares against the reference low band). Applied uniformly in
  the narrowband + high-band gain-scaled innovation paths and mirrored
  in the encoder gain selection, so decoded PCM now lands at the
  reference's absolute level (fixture energy ratio 0.97) instead of
  32× hot.
- Round r393: **ultra-wideband second-layer fold wired** — the 8–16 kHz
  half-band, previously reconstructed as zero under the fold gap, now
  decodes through the fixture-pinned law with the crate's
  recursion-consistent fold source (the embedded wideband layer's two
  excitation tracks — `WidebandDecoder::last_hb_excitation` +
  the narrowband excitation — recombined to 16 kHz via a dedicated QMF
  synthesis bank), per-sub-frame interpolated order-8 LPC and a
  persistent second-layer synthesis filter. `UltraWidebandEncoder` now
  picks its 5-bit gains against the decoder's exact fold source
  (`g = rms(residual)/(K·rms(source))`) through a local
  analysis-by-synthesis wideband decode, so UWB round-trips reconstruct
  the 8–16 kHz energy envelope (pinned by a new tracking test). The
  reference's own 16 kHz fold-source geometry stays a recorded gap
  (the staged fixture is wideband-only) — the source choice is
  documented as the crate's convention.
- Round r393: **on-wire layer-prefix grammar arbitrated** — the fixture
  pins the embedded-frame grammar: **every layer** is introduced by the
  1-bit wideband flag, `0` for the narrowband layer (so a real wideband
  frame's leading 5-bit prefix starts with `0`) and `1` for each
  Table 10.1 high-band layer; a high-band extension is announced by the
  bit *after* the narrowband body. `PacketFrames` previously required a
  leading `1` to detect a wideband frame and mis-walked real `speexenc`
  wideband packets (the fixture surfaces this immediately); it now
  peeks the post-body bit (next frame prefix / terminator / §5.5
  padding all start `0`), still accepting the legacy leading-`1`
  layout this crate's earlier encoders produced. All encoders
  (wideband, ultra-wideband, DTX) now write the reference convention.
- Round r393: **`SpeexDecoder` wideband path delegated to
  `WidebandDecoder`** (new body-level `decode_frame_bodies` entry +
  `low_band_decoder_mut` shared-state accessor) — the top-level decoder
  previously duplicated the wideband assembly and had not picked up the
  folded high band; the two public paths are now bit-identical, pinned
  by a fixture gate through the header-driven `SpeexStreamDecoder`.
- Round r393: **fold-consistent wideband encoder mode-1 gains** — the
  wideband encoder's gain-only high-band sub-mode now transmits
  `g = rms(residual)/(K·rms(e_lb))` against the embedded narrowband
  encoder's locally reconstructed excitation
  (`NarrowbandEncoder::last_frame_excitation`, the analysis-by-synthesis
  mirror of the decoder's fold source) instead of the raw residual RMS,
  so encode → decode round-trips reconstruct the 4–8 kHz energy
  envelope through the pinned law (new tracking test).
- Round r393: **opt-in decoder output high-pass** (`OutputHighpass`) —
  the manual's codec-control table documents a default-on high-pass
  whose transfer is not staged; the fixture's behavioural trace (phase
  lead ∝ 1/f, attenuation only below ≈ 50 Hz) fits a 2nd-order
  Butterworth at 30 Hz (flat optimum 28–45 Hz — documented as fitted,
  not reference-pinned). Applying it moves the fixture decode from
  16.7 dB to 18.3 dB; pinned by a dedicated gate. Opt-in: no decoder
  output changes unless the caller applies it.
- Round r393: **conformance gate** `tests/wb_mode1_folded_fixture.rs` —
  absolute (no scale freedom) scoring of the full fixture decode against
  the reference PCM at the fixed 80-sample reference lead: full-signal
  SNR ≥ 14 dB (measured 16.7), high-band SNR ≥ 30 dB / correlation
  ≥ 0.999 (measured 38.9 dB / 0.99994), energy ratios pinned; plus a
  framing re-verification of the staged notes (101 frames, NB mode 8,
  HB sub-mode 1).

- Round r389: **Ultra-wideband (32 kHz) subsystem — decode + encode +
  quality ladders + VAD/DTX.** Six landings driving the §2.2 embedded
  recursion one level above wideband:
  - **Slice-generic QMF filterbanks** (`QmfAnalysis::split_slices` /
    `QmfSynthesis::reconstruct_slices`) — the two-band mirror bank's
    carried state is only the filter-length band tails, so the same
    structure now serves the wideband inner filterbank (320 → 2×160)
    and the ultra-wideband **outer** filterbank (640 → 2×320). Pinned
    bit-identical to the fixed-length paths and perfect-reconstructing
    at the outer geometry.
  - **Quality → sub-mode ladders** (`quality` module) for all three
    rate classes. Narrowband from Table 9.2's quality column; wideband
    derived arithmetically from Table 10.2 (every per-quality bit-rate
    at 50 frames/s decomposes as one Table 9.1 + one Table 10.1 total;
    the two ambiguous decompositions resolve by layer monotonicity);
    ultra-wideband from RFC 5574 Table 2, whose UWB column sits a
    constant +1,800 bit/s = +36 bits/frame above the wideband column at
    every quality — exactly (and uniquely) the Table 10.1 **mode-1**
    total, pinning the conformant second (8–16 kHz) sub-band layer to
    the gain-only mode-1 frame (`UWB_HIGH_BAND_MODE`). Tests re-derive
    every staged rate exactly from the sub-mode bit totals.
  - **`UltraWidebandDecoder`** — full §5.5 packet walk (multi-frame,
    control pseudo-frames, terminator, padding) of the three-layer UWB
    frame: the embedded wideband layers through the new
    `WidebandDecoder::decode_frame_reader` / `decode_frame_after_header`
    cursor-level entries, then the second Table 10.1 high-band layer
    (12-bit LSP MSVQ envelope + staged 32-level folded-gain track,
    surfaced on the typed `UltraWidebandFrame`), recombined by the
    outer QMF into 640-sample 32 kHz PCM. The mode-1 folded-excitation
    *source* stays the recorded docs gap (#170), so the 8–16 kHz band
    reconstructs as zero and UWB output degrades gracefully to the
    embedded wideband content — pinned byte-identical to a standalone
    `WidebandDecoder` on the same packet (the scalable-bit-stream
    contract). Layer-2 VQ modes 2..=4 (geometry unpinned at the 16 kHz
    half-band) and reserved 5..=7 surface typed errors.
  - **`UltraWidebandEncoder`** — the encode mirror: outer QMF split,
    embedded `WidebandEncoder` low half, and the RFC-pinned 36-bit
    mode-1 second layer (order-8 LPC envelope over the staged
    200-sample window frame-end aligned — documented encoder freedom —
    through the 12-bit 2-stage MSVQ, plus four per-80-sample-sub-frame
    residual-RMS gains through the staged 5-bit folded-gain grid).
    `encode_frame` / `encode_packet` / `encode_packet_quality`
    (qualities 1..=8 given the NB mode-1/7 + HB mode-4 encode gaps);
    per-quality packet sizes pinned to the staged bits-per-frame
    totals; full encode → decode round trips.
  - **`SpeexStreamDecoder`** — header-driven dispatch: binds a parsed
    `SpeexHeader.mode` to the decode path it selects (NB/WB through the
    mixed-stream `SpeexDecoder`, UWB through `UltraWidebandDecoder` —
    the second high-band layer needs the out-of-band rate-class
    context), with `output_rate_hz` / `frame_samples` and flat
    `decode_packet_pcm_i16`.
  - **Adaptive frame-gain refinement** (narrowband encoder) — the
    frame-level OL excitation gain is now chosen **closed-loop**: the
    magnitude estimate's quantised neighbourhood (`{est−1, est, est+1}`
    on the staged 32-level `ol_gain` grid) is evaluated by running the
    full sub-frame encode at each candidate's reconstructed gain and
    keeping the one whose decoded excitation matches the residual best
    (encoder-side search freedom; the decode law is untouched). Never
    worse than the single-pass estimate — pinned by test. NB/WB
    `encode_packet_quality` entry points complete the quality-API
    symmetry with the UWB encoder.
  - **VAD/DTX** (`vad` module) — §2.1's pinned DTX frame format ("only
    5 bits … 250 bps" = the Table 9.1 mode-0 frame; 9/13 bits for the
    WB/UWB all-mode-0 frames) behind `encode_packet_dtx` on all three
    encoders, driven by `EnergyVad`, an RMS-threshold + hangover
    detector (the decision algorithm is unpinned by the manual —
    documented encoder freedom). Silent-packet sizes pinned (4 NB DTX
    frames + terminator = 4 bytes); frame count / 20 ms timing
    preserved through decode.
- Round r385: **Wideband (sub-band CELP) encoder — end-to-end.** A
  subsystem sweep drove the encode direction through the whole §10
  wideband path, mirroring the wideband decoder stage for stage. Seven
  landings, each round-trip-tested:
  - `QmfAnalysis` (in `qmf`) — the encode-direction two-band QMF
    **analysis** split: one 320-sample 16 kHz frame → the two 8 kHz
    half-bands (`lb = downsample2(h0·x)`, `hb = downsample2((−1)ⁿh0·x)`,
    the classical mirror-filter relations over the staged 64-tap
    prototype), with streaming FIR input history. The r365
    perfect-reconstruction test now runs through the public filterbank;
    the streaming split is pinned sample-identical to the direct
    whole-signal reference, and pure low/high tones pin the band-split
    direction.
  - Order-8 **high-band LPC analysis + LPC→LSP** (`analyse_hb` /
    `HbLpcCoefficients` in `lpc_analysis`; `hb_lpc_to_lsp` in
    `lpc_to_lsp`) — the §10.1 "very similar to narrowband" envelope
    front-end at the high-band order 8, via order-generic
    autocorrelation / stabilisation / Levinson-Durbin /
    auxiliary-polynomial root-find cores (the order-10 entry points
    delegate unchanged). Round-trip pinned against the decoder's
    `hb_lsp_to_lpc`.
  - `hb_encode` — the Table 10.1 **high-band frame writer**
    (`write_high_band_body` / `write_high_band_frame`,
    `parse(write(body)) == body` for HB modes 0..=4 including the 80-bit
    mode-4 excitation-VQ fields) and the §10.4 **wideband frame
    assembly** `encode_wideband_frame` (NB prefix with the wideband flag
    set + Table 9.1 body + HB frame — the exact reader chain the
    wideband decoder walks).
  - `hb_innovation_search` — the high-band fixed-codebook search
    (§10.3): per-sub-vector nearest-neighbour at a fixed gain over both
    staged codebook shapes, scoring `HbSv8_128` rows in **both
    polarities** and packing each slot as `index << 1 | sign` MSB-first,
    exactly the decoder's split.
  - `encoder_wb` — the top-level **`WidebandEncoder`**: QMF split →
    embedded r382 narrowband encode of the low band (shared NB state) →
    high-band envelope (order-8 LPC → LSP → Q10 − pinned HB base →
    2-stage MSVQ → 12-bit `lsp_index`, with the *quantised* LSPs driving
    per-sub-frame LPC through the §9.1 interpolation exactly as the
    decoder does) → per-sub-frame excitation → §10.4 packing. HB modes
    0/1/2/3 supported; mode 4 rejected as the recorded
    innovation-binding docs gap. `encode_frame` / `encode_frame_bodies`
    / `WbEncodeError`. A 7-test `encoder_wb_roundtrip` integration suite
    drives `WidebandEncoder → WidebandDecoder` over multi-frame streams:
    finite decode for all HB modes, loud-vs-quiet energy tracking, a
    6.5 kHz input landing its energy in the decoded high-band channel,
    near-silence round-trip, determinism.
  - **Packet-level encode** — `NarrowbandEncoder::encode_packet` /
    `WidebandEncoder::encode_packet` pack consecutive frames
    back-to-back and close with the new shared `write_packet_terminator`
    (the §5.5 5-bit mode-15 terminator + byte padding), so the padding
    tail can never misparse. A 5-test `encoder_packet_roundtrip` suite
    drives the packets through the **top-level `SpeexDecoder`**:
    3-frame NB packets → exactly 3 frames at 8 kHz, 2-frame WB packets
    → 2 frames at 16 kHz (the walker skips high-band parts per §10.4),
    terminator-only packets → zero frames, and packetisation is pinned
    decode-transparent (N frames in one packet ≡ N single-frame
    packets, bit-identical PCM).
  - **Closed-loop high-band gain selection** — the HB gain field is only
    4/5 bits, so `hb_quantise_gain_and_search` tries **every** level of
    the staged gain grid, runs the greedy shape search at each level's
    reconstructed gain, measures each `(gain, shape)` pair's **decoded**
    error through the exact decoder path, and keeps the argmin — never
    worse than any single open-loop pass (pinned), and immune to the
    overshooting-guess → zero-row failure mode.

  Functional, not bit-exact — same posture as the r382 narrowband
  encoder (the reference gain normalisation remains the documented
  gain-Q-format gap). The mode-1 high-band *reconstruction* law (the
  staged table inventory names a "folded" 5-bit gain, but no folding
  algorithm is staged; manual §10.3 says only "coded in the same way as
  for narrowband") stays a recorded docs gap — the encoder transmits
  the quantised residual-RMS gain and the decoder reconstructs mode-1
  high bands as silence. 640 lib tests (up from 631) + 12 new
  integration tests.

- Round r382: **Narrowband CELP encoder — end-to-end excitation encode.**
  A subsystem sweep drove the encoder from the r372 envelope chain to a
  full narrowband encode. New modules, each round-trip-tested:
  - `weighting` — the perceptual noise-weighting filter
    `W(z) = A(z/γ1)/A(z/γ2)` (manual §8.5 Eq. 8.1, γ1 = 0.9 / γ2 = 0.6),
    with bandwidth expansion `aw[j] = a[j]·γ^(j+1)` and a persistent
    pole-zero FIR/IIR filter.
  - `ol_pitch` — open-loop pitch estimation (§9.2): the integer lag
    `T ∈ [17,144]` maximising the energy-normalised squared
    cross-correlation, packed into the 7-bit OL-pitch field.
  - `abs_search` — analysis-by-synthesis primitives:
    `weighted_synthesis_zero_state` (filter through `1/A(z)` then `W(z)`)
    and `closed_loop_pitch_search` (per-period 3-tap basis filtering +
    gain-VQ scoring in the weighted-error domain).
  - `innovation_search` — the fixed-codebook sub-vector search (§9.2 /
    companion §2.3): per-sub-vector nearest-neighbour selection at the
    open-loop gain, packed MSB-first into `innovation_vq` exactly as the
    decoder parses it.
  - `nb_encode` — the Table 9.1 frame writer (`encode_narrowband_frame` /
    `write_narrowband_body`), the exact inverse of the body parser
    (`parse(write(body)) == body` for all nine modes).
  - `encoder_nb` — the top-level `NarrowbandEncoder`: LPC analysis → LSP
    VQ → per-sub-frame residual `A(z)·input` → pitch search → innovation
    search → gain quantisation → frame packing, with live excitation
    history feeding the pitch predictor. `encode_frame` /
    `encode_frame_body`; modes 2/3/4/5/6/8 (1/7 rejected as the
    undocumented-innovation gap). A `tests/encoder_nb_roundtrip.rs`
    integration test drives encode → wire → parse → `NarrowbandDecoder`
    and asserts finite PCM, input-energy tracking, and an exact
    pack/parse body round-trip. Functional (not bit-exact): the reference
    gain normalisation remains the documented gain-Q-format gap.

- Round r372: **Encoder bootstrap — high-band LSP-VQ quantiser.** The
  wideband counterpart of the narrowband `lsp_quant`: `hb_lsp::quantise_q10`
  (exported `quantise_hb_lsp_q10`) inverts the r214 high-band LSP
  reconstruction `hb_lsp::reconstruct_q10`. The high-band LSP quantiser is
  a 2-stage MSVQ (manual §10.1 / the staged `hb-lsp-cdbk-stage1/stage2`):
  stage 1 a full 8-coefficient VQ (scale 1/256 → ×4 Q10), stage 2 a
  residual VQ (scale 1/512 → ×2). The search is sequential greedy — pick
  the stage-1 row minimising squared error to the target, subtract, then
  pick the stage-2 row on the residual — with the exact Q10 scaling the
  decoder uses, so the indices reconstruct through the existing decoder
  path. New `pack_hb_lsp_index` packs the two 6-bit indices into the
  on-wire 12-bit `lsp_index` field (the layout `HbLspStages::from_packed`
  parses). 5 new tests (monotone refinement, pack↔from_packed, the full
  quantise→pack→decode round-trip, exact-row search). 572 lib tests, up
  from 567.

- Round r372: **Encoder bootstrap — radian→Q10 LSP bridge + full envelope
  encode chain.** `radians_to_lsp_q10` / `lsp_vector_radians_to_q10` are
  the encode-direction inverse of the decoder's `lsp_q10_to_radians` /
  `lsp_vector_q10_to_radians` (`value = round(ω · 2¹⁰)`), the bridge from
  the `lpc_to_lsp` root-finder output (radians) into the Q10 domain the
  `lsp_quant` VQ search consumes. This closes the complete short-term
  envelope encode path end-to-end:
  `signal → lpc_analyse → lpc_to_lsp → radians→Q10 → (− base) →
  quantise_lsp_q10 → pack_lsp_index → [wire] → from_packed →
  reconstruct_q10 → (+ base) → lsp_to_lpc`. New
  `tests/encoder_envelope_chain.rs` drives the whole chain on an analysed
  AR signal through both the 18-bit and 30-bit regimes and asserts the
  reconstructed LPC envelope is a faithful match to the analysed one (VQ
  envelope error below the coefficient energy), plus the on-wire
  `pack → from_packed` round-trip and the silence-mode skip. Exports:
  `radians_to_lsp_q10`, `lsp_vector_radians_to_q10`. 3 new unit tests +
  3 integration tests; 567 lib tests, up from 564.

- Round r372: **Encoder bootstrap — narrowband LSP-VQ quantiser
  (`lsp_quant` module).** The encode inverse of the r194 decoder LSP
  reconstruction (`lsp::reconstruct_q10`). Given the order-10 Q10 LSP
  vector from `lpc_to_lsp`, `quantise_lsp_q10` runs the multi-stage VQ
  search that picks the per-stage 6-bit indices (`NbLspStages`) best
  representing it: a sequential greedy search — stage 0 (full 10-coeff VQ,
  scale 1/256) → split low/high first refinement (stages 1/3, scale 1/512)
  → split low/high second refinement (stages 2/4, scale 1/1024, 30-bit
  regime only). Each stage minimises the squared error of the residual it
  sees against the staged `nb_lsp_*` codebooks, with the exact per-stage
  Q10 scaling the decoder uses, so the chosen indices reconstruct through
  the existing decoder path. `LspQuant::None` (mode 0) returns `None`. New
  `pack_lsp_index` packs the per-stage indices into the on-wire
  `lsp_index` field (MSB-first, the layout `NbLspStages::from_packed`
  parses) — closing the LSP encode→pack→decode round-trip. Grounded on
  §9.1's multi-stage-VQ structure + the staged codebook data; no external
  source consulted. 8 new tests (the monotone-refinement invariant, the
  pack↔from_packed round-trips for both regimes, the full
  quantise→pack→decode chain, and an error-reduction-vs-arbitrary-index
  check). 564 lib tests, up from 556.

- Round r372: **Encoder bootstrap — LPC→LSP conversion (`lpc_to_lsp`
  module).** The encode-direction inverse of the decoder's `lsp_to_lpc`:
  after `lpc_analysis` produces the order-10 LPC predictor, convert it to
  the Line Spectral Pair frequencies for quantisation (manual §9.1: *"LPC
  are converted to Line Spectral Pairs for quantisation"*). `lpc_to_lsp`
  forms the symmetric / antisymmetric auxiliary polynomials
  `P(z) = A(z) + z⁻⁽ᴺ⁺¹⁾A(1/z)` and `Q(z) = A(z) − z⁻⁽ᴺ⁺¹⁾A(1/z)`,
  deflates the `(1 ± z⁻¹)` boundary roots, and locates each polynomial's
  unit-circle roots by a dense `ω`-grid sign-change scan + bisection
  refinement — the `LPC_ORDER` LSP angles `ωₖ ∈ (0, π)` in ascending
  order, exactly the input convention `lsp_to_lpc` consumes. Round-trip
  validated against the in-tree `lsp_to_lpc` (`lsp_to_lpc(lpc_to_lsp(a))
  ≈ a` to < 1e-3 for a minimum-phase filter), so the encoder front-end's
  envelope path is closed end-to-end (analyse → LPC → LSP →
  [eventually quantise] → decoder reconstructs the same envelope). New
  `LpcToLspError`, `LSP_SCAN_STEPS` / `LSP_BISECT_ITERS` constants.
  Grounded on §9.1's structural prose + the textbook auxiliary-polynomial
  root-find; no external library source consulted. 6 new tests including
  the known-LSP round-trip, the flat-filter even-spacing pin, and a real
  analysed-signal envelope round-trip. 556 lib tests, up from 550.

- Round r372: **Encoder bootstrap — LPC analysis front-end (`lpc_analysis`
  module).** The first stage of the Speex *encoder* (the inverse of the
  decoder's LSP→LPC / synthesis path): turn a frame of input speech into
  the order-10 short-term LPC predictor. Per *The Speex Codec Manual* §8.2
  / §9.1 the chain is window → autocorrelate → stabilise → Levinson-Durbin.
  New functions: `apply_analysis_window` (the staged 200-sample asymmetric
  analysis window centred on the 4th sub-frame), `autocorrelate`
  (`R(m) = Σ x[i]·x[i−m]` to order 10 — the Toeplitz right-hand side),
  `stabilise_autocorrelation` (the `R(0) *= 1.0001` white-noise floor — the
  manual's worked value — plus the staged 11-tap autocorrelation lag
  window), `levinson_durbin` (solves the symmetric Toeplitz system
  `R·a = r` in `O(N²)`, returning the `LpcCoefficients { a, error }`), and
  `analyse` running the full pipeline. The output `a[0..10]` uses the same
  `A(z) = 1 − Σ aᵢ z⁻ⁱ` convention the decoder's `lsp_to_lpc` produces,
  so the encoder front-end is round-trippable against the existing
  synthesis path. Grounded entirely on §8.2's structural prose + the staged
  analysis/lag window **data** (new float accessors
  `codebooks::lpc_analysis_window_float` / `lpc_lag_window_float`);
  windowing / autocorrelation / Levinson-Durbin are the textbook
  linear-prediction primitives the manual names. New constants
  `LPC_NOISE_FLOOR` (1.0001) / `AUTOCORR_LAGS` (11), `LpcAnalysisError`.
  13 new unit tests including a known-AR-process recovery check (the
  recovered predictor's leading coefficients match the generating filter)
  and the residual-error-in-`[0, R(0)]` invariant. 550 lib tests, up
  from 538.

- Round r372: **Packet frame classification + structural summary.**
  A consumer that wanted to know "what is in this packet" before / instead
  of a full decode had to match on the `PacketFrame` enum by hand. New
  `FrameKind` enum (`Narrowband` / `Wideband` / `InbandSignalling` /
  `CustomInband`) with `is_audio()` / `is_control()`, plus
  `PacketFrame::kind()` / `is_audio()` / `is_control()` / `mode_id()`
  classification accessors. New `PacketSummary` walks a packet once
  (without decoding audio) and tallies per-kind frame counts:
  `narrowband` / `wideband` / `inband_signalling` / `custom_inband`,
  with `audio_frames()` / `control_frames()` / `total_frames()` /
  `is_wideband()` aggregates. `PacketSummary::walk` halts on the first
  `PacketError` (mirroring `PacketFrames`) and returns the accumulated
  summary on a clean terminator / padding tail. Pairs with
  `SpeexHeader::frames_per_packet` for a header-vs-payload cross-check
  (§7.3) and surfaces a packet's rate class (8 kHz vs 16 kHz) and its
  audio/control split for routing or DTX detection. Spec basis: *The
  Speex Codec Manual* §5.5. 8 new packet tests; 538 lib tests, up from
  530. Exports: `FrameKind`, `PacketSummary`.

- Round r372: **In-band signalling semantic interpretation +
  decoder control-frame surfacing.** The §5.5 mode-14 in-band messages
  were parsed to a raw `(code, payload)` `InbandMessage` but their
  meaning was never decoded and the top-level decoder discarded them
  (`DecodedFrame::Control` was a unit variant). This round adds the
  semantic layer over Table 5.1's "Content" column. New
  `InbandMessage::interpret() -> InbandRequest` decodes the raw payload
  into the typed request the peer is making: `PerceptualEnhancement(bool)`
  / `LessAggressive(bool)` / `SwitchMode(u8)` / `SwitchModeLowBand(u8)` /
  `SwitchModeHighBand(u8)` / `SwitchQualityVbr(u8)` /
  `RequestAcknowledge(AcknowledgePolicy)` / `SetRateMode(RateModeConfig)`
  / `TransmitCharacter(u8)` / `IntensityStereo(u8)` /
  `AnnounceMaxBitrate(u16)` / `AcknowledgePacket(u32)` / `Reserved{..}`.
  Two new typed helpers: `AcknowledgePolicy` (code 6: `None` / `All` /
  `InbandOnly` / `Other(u8)`) and `RateModeConfig` (code 7: decodes the
  documented `CBR(0) / VAD(1) / DTX(3) / VBR(5) / VBR+DTX(7)` bitmask
  into independent `vbr` / `vad` / `dtx` flags — bit 2 = VBR, bit 0 =
  VAD, bit 1 = DTX-with-VAD — reproducing every enumerated value exactly,
  plus `is_cbr()`). The interpret mapping is *total* over all sixteen
  codes and reads no further bits. Spec basis: *The Speex Codec Manual*
  §5.5 + Table 5.1 (staged in `docs/audio/speex/speex-manual.pdf`).
  The top-level `SpeexDecoder` now surfaces this end-to-end:
  `DecodedFrame::Control` carries a typed `ControlMessage`
  (`Inband { message, request }` for mode 14 / `Custom { size_bytes }`
  for mode 13), so a consumer can act on a mode-switch / rate-mode /
  intensity-stereo request without re-parsing the bit-stream. New
  `DecodedFrame::control_message()` accessor + `ControlMessage` export.
  13 new unit tests in `signalling::tests` + 3 new decoder tests.
  530 lib tests, up from 517. The intensity-stereo *decode* algorithm
  (balance byte → L/R reconstruction) remains a recorded docs gap — the
  manual stages only the 8-bit field, not the panning law.

- Round r369: **`UwbFrameLayout::for_header_mode` — header → sub-band
  framing descriptor.** Maps a `SpeexHeader::mode` field value (`0` /
  `1` / `2` = the `SPEEX_MODE_NARROWBAND` / `_WIDEBAND` / `_ULTRAWIDEBAND`
  constants, manual §7.3) to its embedded sub-band recursion layout,
  returning `None` for the unknown modes `SpeexHeader::is_known_mode`
  rejects. Links a parsed stream header to the layer-ladder descriptor so
  a caller can size the embedded recursion depth and the reconstructed
  rate from the header alone, before walking any packet; a test pins the
  layout's reconstructed rate against the header's canonical mode-rate.

- Round r369: **Header mode-class accessors + canonical-rate cross-check
  (`SpeexHeader`).** New `is_narrowband` / `is_wideband` /
  `is_ultrawideband` predicates over the `mode` field, plus
  `mode_sampling_rate_hz` — the canonical output rate the mode class
  implies (NB 8 kHz / WB 16 kHz / UWB 32 kHz, manual §2.2 embedded
  sub-band layering / §7.3), derived from the mode class independent of
  the self-declared `rate` field — and `rate_matches_mode`, which flags a
  header whose declared `rate` contradicts its mode class. The canonical
  rate matches the per-frame full-rate output the decoder produces
  (`DecodedFrame::sample_rate_hz`), so a consumer can default playback
  rate from the header alone or validate a stream before decoding.

- Round r369: **Whole-packet flat-`i16` decode convenience
  (`SpeexDecoder::decode_packet_pcm_i16`).** Decodes a Speex packet
  straight to a flat `Vec<i16>`, concatenating every audio frame's
  full-rate output in packet order and skipping the non-audio control
  pseudo-frames — the natural "decode this packet to playable samples"
  entry point over the typed [`decode_packet`] output. A narrowband
  packet yields a flat mono 8 kHz buffer, a wideband packet a flat mono
  16 kHz buffer (a packet is one sampling-rate class in every conformant
  Speex stream, §7.3). Tests pin the flat output against the flattened
  `pcm_i16()` of the typed decode and the control-frame-dropped split.

- Round r369: **`i16` PCM convenience across the wideband + top-level
  decode paths.** The narrowband path already had
  `NarrowbandDecoder::decode_frame_i16`; this round extends the same
  rounding/saturation convention to the rest of the public surface so a
  caller can collect ready-to-play `i16` PCM without re-implementing the
  float→`i16` quantiser. The previously-private narrowband `saturate_i16`
  (round-to-nearest, clamp to `[i16::MIN, i16::MAX]`) is promoted to a
  crate-root `pub fn` and reused everywhere: `WidebandFrame::wideband_pcm_i16`
  + `WidebandDecoder::decode_packet_i16` quantise the 320-sample 16 kHz
  wideband output, and `DecodedFrame::pcm_i16` returns the band-correct
  full-rate PCM (mono 8 kHz for a narrowband frame, mono 16 kHz for a
  wideband frame, `None` for a control pseudo-frame) with a matching
  `DecodedFrame::sample_rate_hz`. Centralising one quantiser keeps a
  low-band sample bit-identical whether it is read from the narrowband or
  the wideband `i16` output. Tests pin every new path against
  `saturate_i16` of the `f64` reference, the saturation clamp at the
  `i16` extremes (ties-away rounding), and the control-frame `None` case.

- Round r365: **QMF synthesis filterbank — wideband half-band → 16 kHz
  PCM recombination (`qmf` module).** New `QmfSynthesis` recombines the
  two reconstructed 8 kHz half-band signals (low band 0–4 kHz + high
  band 4–8 kHz folded) into a single 16 kHz wideband PCM frame, the
  final stage of the §10 sub-band CELP decode path that earlier rounds
  stopped short of. The reconstruction is the **classical two-band
  quadrature-mirror filterbank** (Croisier–Esteban–Galand) driven by the
  staged 64-tap prototype `h0` (`qmf-filter-h0-float.csv`) — a textbook
  multirate-DSP construction, the same clean-room category the staged
  LSP→LPC trace grants for the LSP polynomial reconstruction. Synthesis
  relations `g0 = 2·h0`, `g1 = -2·(-1)ⁿ·h0`, implemented in **polyphase**
  form and unit-pinned identical to the direct upsample-filter-sum
  reference. The staged prototype is normalised `Σh0 ≈ 1` /
  `Σh0_even = Σh0_odd ≈ 0.5`, so the factor-2 synthesis gain yields a
  **unity passband** (a constant low band reconstructs to the same
  constant — pinned by `constant_low_band_reconstructs_unity_passband`).
  FIR band histories persist across frames via the `QmfSynthesis` state
  for seamless streaming. New `qmf_h0_float()` accessor +
  `parse_float_column` helper surface the float prototype. The §10.2
  frequency-axis fold is intrinsic to the `(-1)ⁿ` synthesis modulation
  (no explicit half-band reversal). This closes the long-standing QMF
  synthesis docs gap for the **sample-correct** (textbook-DSP) path; the
  bit-exact polyphase delay convention the reference uses remains
  unpinned by the staged manual.
- Round r365: **QMF perfect-reconstruction sample-correctness pin.** A
  test-only two-band QMF *analysis* reference (`lb = downsample2(h0*x)`,
  `hb = downsample2((-1)ⁿh0*x)`) splits a synthetic 16 kHz full-band
  signal into the two half-bands; pushing those back through the
  `QmfSynthesis` filterbank recovers the original signal (relative error
  `< 1e-3` in the steady-state region, up to the filterbank group delay).
  This validates the CEG mirror relation's alias-cancellation +
  amplitude-distortion-free property for the staged prototype — the
  decisive sample-correctness check that the synthesis sign conventions,
  factor-2 gain, and polyphase decomposition are all correct.
- Round r365: **`WidebandDecoder` now emits 16 kHz wideband PCM.**
  `WidebandFrame` gains a `wideband_pcm: [f64; 320]` field carrying the
  QMF-recombined full-band signal; `WidebandDecoder` holds a persistent
  `QmfSynthesis` so a multi-frame wideband stream reconstructs
  continuously. The `low_band` / `high_band` half-band fields are
  retained for diagnostics. The top-level `SpeexDecoder` wideband path
  surfaces the recombined PCM end-to-end.
- Round r359: **`LSP_PI` angular-domain pin — closes the LSP
  angular-unit gap.** The Speex decoder stores LSP frequencies in a
  fixed-point angular domain where the staged constant `LSP_PI = 25736`
  (`docs/audio/speex/provenance/02-speex-gain-quant.md`, "LSP→LPC
  reconstruction path constants") measures the angle `π`. Earlier rounds
  flagged the angular interpretation of a stored LSP value as a
  *documented assumption* (`lsp_q*_to_radians` treated the stored value as
  `ω = value / 2^Q` rad without a staged confirmation). New
  `lsp_pi_domain` module pins this as a **provenance-confirmed fact**:
  `ω = v_storage · π / LSP_PI` exactly. It exposes `LSP_PI`, the
  storage↔radian conversions (`lsp_storage_to_radians` /
  `radians_to_lsp_storage`), the storage→Q10-radian bridge
  (`lsp_storage_to_q10`), and the Q15-storage `LSP_LINEAR` base vectors
  (`nb_lsp_linear_storage` = `(i+1)·2048`, `hb_lsp_linear_storage` =
  `i·2560 + 6144`). Seven module tests cross-check that the
  `LSP_PI`-domain conversion of the storage base vector lands on the
  **same** Q10-radian base vector the `lsp_base` module derived
  independently from the float `LSP_LINEAR` form — two staged numeric
  facts (`LSP_LINEAR`, `LSP_PI`) pinning one angle, the provenance
  cross-check that closes the angular-unit half of the bit-exactness gap.
  `lsp_to_lpc::lsp_qn_to_radians` doc updated to record the conversion as
  confirmed, not assumed. The remaining bit-exactness gap is now isolated
  purely to the reference decoder's fixed-point `lsp_cos` lookup-table
  evaluation order (table not staged) — independent of the now-pinned
  angular unit.
- Round r356: **Forced (open-loop) pitch-gain reconstruction (modes 1 /
  8)** — narrowband Table 9.1 modes 1 and 8 carry a frame-level 4-bit
  *forced* pitch-gain field (the `OL pitch gain` row) instead of the
  per-sub-frame 3-tap pitch-gain VQ. The field was parsed
  (`NarrowbandFrameBody::ol_pitch_gain_index`) but discarded, so those
  modes fell back to `PitchGainTaps::SILENCE` and their
  adaptive-codebook contribution `p[n]` was always zero — only the
  innovation `c[n]` drove synthesis. New `forced_pitch_gain` module
  reconstructs the forced coefficient from the numeric law pinned in
  `docs/audio/speex/provenance/02-speex-gain-quant.md`: decode
  `pitch_coef = 0.066667 · quant` (`quant ∈ 0..=15`, unit gain at 15).
  The forced path is a single-tap predictor `p[n] = pitch_coef · e[n−T]`,
  so the coefficient is expressed as a Q6 centre tap
  `round(0.066667 · quant · 64)` with zero side taps, reusing the
  existing §9.2 `gain_scaled_pitch` convolution (which divides the Q6
  dot product by 64). `NarrowbandDecoder::pitch_taps` now dispatches on
  the two mutually-exclusive Table 9.1 pitch-gain rows: per-sub-frame VQ
  (modes 2..=7), frame-level forced gain (modes 1, 8), or true silence
  (mode 0). New `forced_pitch_coef` / `forced_pitch_gain_taps` public
  helpers + `FORCED_PITCH_GAIN_{BITS,LEVELS,STEP}` constants. 7 module
  unit tests + 3 decoder tests (the load-bearing one proves a non-zero
  forced gain changes the mode-8 second-frame output versus a zeroed
  gain).
- Round r350: **High-band per-sub-frame LSP interpolation** — the
  wideband high-band synthesis now reconstructs a **separate order-8 LPC
  set per sub-frame** from the §9.1 four-way linear interpolation of the
  previous and current frame's high-band LSP vectors, instead of one
  frame-level set applied to all four sub-frames. Spec basis: *The Speex
  Codec Manual* §10.1 — the high-band linear prediction is *"very similar
  to narrowband. The only difference is that we use only 12 bits to
  encode the high-band LSP's"*; the manual names exactly one difference,
  so the §9.1 per-sub-frame interpolation applies verbatim to the high
  band (this resolves the earlier over-cautious "§10 is silent on
  high-band LSP interpolation" docs-gap note). New `hb_lsp_interp` module
  (`HbSubFrameLsp`, the high-band analogue of `NbSubFrameLsp`, Q10→Q12
  exact-integer four-way interpolation with the `prev = curr` first-frame
  convention); new base-aware per-sub-frame LSP→LPC helpers
  `lpc_from_hb_subframe_lsp_q12` / `hb_subframe_lpc_set_with_base` (add
  the pinned HB base vector + `.05`-rad HB `LSP_MARGIN`); new
  `synthesise_high_band_frame_interp(.., &mut prev_hb_lsp)` synthesis
  entry that threads the previous frame's high-band LSP for cross-frame
  continuity. `WidebandDecoder` / `SpeexDecoder` carry the prev-LSP state
  so the interpolation is continuous across wideband frames; high-band
  silence frames (no LSP field) leave the previous envelope untouched.
  The stateless `synthesise_high_band_frame` delegates to the
  interpolating path with the first-frame convention, reproducing the
  prior frame-level-LPC behaviour exactly (locked by a consistency
  test). 13 new lib unit tests + a 4-test `wb_high_band_lsp_interp`
  integration suite.
- Round r347: **LSP base-vector / Q-format pin (NB + HB boundedness)** —
  new `lsp_base` module pins the documented Speex LSP linear-init base
  vector (`LSP_LINEAR(i) = .25·i + .25` rad NB; `LSP_LINEAR_HIGH(i) =
  .3125·i + .75` rad HB) recorded as numeric facts in
  `docs/audio/speex/provenance/02-speex-gain-quant.md`. The r194 LSP
  reconstruction emits only the per-stage VQ codebook **deltas**; this
  round adds the base offset so the reconstructed LSP angles land inside
  the conformant `(0, π)` band **by construction** rather than via the
  radian-clamp fallback. The Q10-radian base vector is exact (both the
  `LSP_PI = 25736` Q15-domain path and the `M_PI` float path pin the
  same integers — cross-checked in tests). New base-aware LSP→LPC
  helpers `lpc_from_lsp_delta_q10` / `lpc_from_hb_lsp_delta_q10` /
  `subframe_lpc_set_with_base` / `nb_lsp_with_base_q10` /
  `hb_lsp_with_base_q10`; the narrowband decoder loop and the wideband
  `hb_lpc` accessor now reconstruct LPC through the bounded base-aware
  path. Exposes `nb_lsp_base_q10`, `hb_lsp_base_q10`,
  `nb_lsp_base_radians`, `hb_lsp_base_radians`, `add_nb_lsp_base`,
  `add_hb_lsp_base`, and the slope/intercept constants. This is the
  boundedness half of the LSP Q-format milestone; full bit-exactness
  against reference Speex output additionally requires the cosine-series
  fixed-point evaluation order, which is not pinned by the staged manual
  prose (recorded docs gap).
- Round r347: **LSP_MARGIN minimum-spacing enforcement** — `lsp_base`
  now pins the documented Speex `LSP_MARGIN` constant (`.002` rad NB /
  `.05` rad HB, recorded in
  `docs/audio/speex/provenance/02-speex-gain-quant.md`) and applies the
  order-preserving minimum-spacing safeguard
  (`enforce_lsp_margin_radians`) inside the base-aware NB / HB /
  per-sub-frame LSP→LPC paths. A forward + backward clamp pass keeps
  every reconstructed LSP angle strictly inside `[margin, π − margin]`,
  ascending with ≥ `2·margin` spacing, so the auxiliary-polynomial root
  split always yields a stable filter even for a degenerate / corrupt
  quantiser output. Exposes `enforce_lsp_margin_radians`,
  `nb/hb_lsp_margin_radians`, and the `*_LSP_MARGIN_Q10` /
  `*_LSP_MARGIN_RADIANS` constants.
- Round r347: **closed-loop boundedness validation** — new
  `tests/speex_decoder_fixture.rs::closed_loop_decode_is_non_divergent`
  drives the real `speexenc` mode-5 (q8) fixture through the top-level
  `SpeexDecoder` and asserts the closed excitation-feedback loop stays
  **non-divergent** (finite, frame-to-frame non-growing) with the LSP
  base vector + `LSP_MARGIN` pinned — the runaway an out-of-band LSP set
  would produce under live feedback can no longer happen. Output is
  bounded but not yet at reference *amplitude* (the cosine-series
  fixed-point Q-format + absolute gain calibration remain a documented
  docs gap).
- Round r340: **wideband high-band synthesis assembly + UWB framing
  recursion** — new `wb_synthesis` module composes the high-band
  primitives into the complete high-band branch of the wideband decode
  path: `synthesise_high_band_frame` reconstructs the order-8 high-band
  LPC (`hb_lpc`), and for each of the four sub-frames decodes +
  gain-scales the excitation (`e_hb[n] = g·c_hb[n]`, §10.2 no high-band
  pitch) and runs it through `1/A_hb(z)`, concatenating into the
  160-sample high-band 8 kHz half-band signal `x_hb[n]` — the second
  input to the QMF synthesis filterbank. Also adds the `UwbFrameLayout`
  / `SubBandLayer` typed descriptor for the embedded, scalable framing
  recursion (§2.2 "Embedded wideband structure"): narrowband →
  wideband → ultra-wideband adds one high-band layer (one wideband-flag
  recursion marker) per step, doubling the reconstructed rate
  (8/16/32 kHz). Exposes `synthesise_high_band_frame`, `UwbFrameLayout`,
  `SubBandLayer`, `HB_FRAME_SAMPLES`, `HB_SUBFRAMES_PER_FRAME`. New
  `tests/wb_high_band_synthesis.rs` drives the full chain on a synthetic
  mode-2 frame (finite, responsive, history-continuous). **Docs gaps**
  (not fished): (1) the QMF *synthesis* filterbank algorithm (polyphase
  recombination / `h0→{h0,h1}` pair / 2× factors / delay) is not in the
  staged manual — only the `h0` prototype is staged as data; (2) the
  per-sub-frame high-band LSP interpolation primitive is a follow-up
  (frame-level LPC used for all four sub-frames meanwhile); (3) the
  per-mode UWB high-band bit allocation (a "Table 11.x" analogue) is not
  in the staged manual. 9 lib tests + 4 integration tests.
- Round r340: **high-band LPC synthesis filter** — new `hb_synthesis`
  module runs the gain-scaled high-band excitation `e_hb[n]` through the
  order-8 all-pole synthesis filter `1/A_hb(z)`, producing the high-band
  8 kHz half-band signal that the QMF synthesis filterbank recombines
  with the narrowband (low-band) signal into 16 kHz wideband PCM. Per
  *The Speex Codec Manual* §10.1 the high-band linear prediction is *"very
  similar to what is done for narrowband"* — the synthesis recurrence
  `x[n] = e[n] + Σ a[i]·x[n−i]` is identical to the r286 narrowband
  `SynthesisFilter`, only at the high-band order 8 (`HB_LPC_ORDER`). The
  IIR history persists across sub-frame and frame boundaries. Consumes
  the `lpc_from_hb_lsp_q10` order-8 coefficients and the
  `gain_scaled_hb_innovation_subframe` excitation. Exposes
  `HbSynthesisFilter` with `process` / `process_subframe`. 7 new tests.
- Round r340: **gain-scaled high-band excitation** — new
  `gain_scaled_hb_innovation` module folds the reconstructed high-band
  `Excitation gain` factor (`reconstruct_hb_exc_gain`) into the raw
  high-band innovation sub-vector (`decode_hb_subframe`), producing the
  magnitude-correct high-band excitation `e_hb[n]` as `[f32; 40]`. Per
  *The Speex Codec Manual* §10.2 there is **no pitch prediction in the
  high band**, so the §8.4 composition `e[n] = p[n] + c[n]` collapses to
  `e_hb[n] = c_hb[n]` — this gain-scaled innovation is the *entire*
  high-band excitation that the high-band synthesis filter `1/A_hb(z)`
  consumes (the high-band counterpart of the r326
  `gain_scaled_innovation`, minus the adaptive-codebook term). Exposes
  `gain_scaled_hb_innovation_subframe`, `gain_scaled_hb_innovation_sample`,
  `gain_scaled_hb_innovation_from_body`, and
  `GAIN_SCALED_HB_INNOVATION_SAMPLES`. Modes 0/1 (silence/gain-only)
  vanish to `0.0`; mode 4 surfaces the documented `Undocumented`
  codebook-binding gap. 8 new tests.
- Round r337: **float-domain excitation composition** — new
  `gain_scaled_excitation` module joins the two gain-scaled contributions
  into the final per-sub-frame excitation `e[n] = p[n] + c[n]` of *The
  Speex Codec Manual* §8.4 / CELP companion §2.3, the float analogue of
  the r244 raw-integer `raw_excitation_subframe`. Both inputs are
  `[f32; 40]` already in the **same normalised float signal domain** (the
  r331 gain-scaled pitch contribution `p[n]` and the r326 gain-scaled
  innovation contribution `c[n]`), so the composition is a plain
  elementwise `f32` sum — no Q-format shift, matching the floating-point
  posture of the downstream `SynthesisFilter`. Unlike the r244 raw sum
  (whose terms carried different un-divided Q-formats), this sum is
  magnitude-correct: it closes the README "Not yet supported" tail's
  flagged composition step. Exposes `gain_scaled_excitation_subframe`,
  `gain_scaled_excitation_sample`, and `GAIN_SCALED_EXCITATION_SAMPLES`.
  Stream-start (empty buffer → `p[n] = 0.0`, `e[n] = c[n]`) and silence
  (both terms → `0.0`) cases behave per spec. 10 new tests (406 lib
  tests, up from 396).
- Round r335: **open-loop / scalar gain quantiser (encode direction)** —
  the `gain_reconstruction` module now ships the encode half that mirrors
  its existing reconstruction lookups. The new `scal_quant` core is the
  textbook sorted-threshold search (companion §2.3): the index is the
  count of decision boundaries a target gain meets-or-exceeds, saturated
  to the field width. Exposes `quantise_frame_ol_exc_gain`
  (NB 5-bit `OL Exc gain`, `scal_quant32` against the 32 normalised
  levels; non-positive/non-finite → `Silence`),
  `quantise_subframe_gain_correction` (NB 1-/3-bit innovation-gain
  correction over `scal1_bound`/`scal3_bound`; 0-bit → `Absent`,
  unknown budget → `None`), and `quantise_hb_exc_gain` (HB 5-bit folded
  gain over `fold_quant_bound`; HB 4-bit gain-correction over
  `gc_quant_bound` with the `0.87360` reconstruction multiplier divided
  out of the target). Each quantiser returns the same typed index enum
  the parser produces, so it is the exact inverse of the matching
  reconstruction function at every cell. Also exposes the
  `hb_gc_quant_bound` accessor. Spec basis: the staged decision-boundary
  arrays + `scal_quant`/`scal_quant32` semantics indexed in
  `provenance/02-speex-gain-quant.md`. 7 new tests covering per-cell
  round-trips, boundary placement, clamping, and the `scal_quant`
  threshold contract.
- Round r331: **gain-scaled adaptive-codebook (pitch) contribution** —
  new `gain_scaled_pitch` module divides the §9.2 long-term-predictor
  dot product (`adaptive_contribution_subframe`) by the now-staged
  **Q6** pitch-gain scaling (`GAIN_SCALING = 64`, `GAIN_SHIFT = 6`, from
  `provenance/02-speex-gain-quant.md` "Scalar constants" — the previously
  un-pinned pitch-gain Q-format), producing the pitch contribution
  `p[n]` as `[f32; 40]` in the **same normalised float signal domain**
  as the r326 gain-scaled `c[n]`. The two §8.4 contributions now share
  one domain, so the composition `e[n] = p[n] + c[n]` is well-posed.
  Exposes `gain_scaled_pitch_subframe`, `gain_scaled_pitch_sample`,
  `GAIN_SCALED_PITCH_SAMPLES`, and the `PITCH_GAIN_SCALING` constant.
  Stream-start and silence-tap cases vanish to `0.0`. 7 new tests (389
  lib tests, up from 382).
- Round r326: **gain-scaled fixed-codebook contribution** — new
  `gain_scaled_innovation` module folds the reconstructed fixed-codebook
  gain `g = g_frame · g_subf` (from `reconstruct_fixed_codebook_gain`)
  into the raw `[i16; 40]` innovation sub-vector, producing the
  magnitude-correct `c[n]` (`[f32; 40]`) that enters the Speex Codec
  Manual §8.4 excitation composition `e[n] = p[n] + c[n]`. Exposes
  `gain_scaled_innovation_subframe` (apply a reconstructed scalar gain),
  `gain_scaled_innovation_from_indices` (reconstruct + apply from typed
  `FixedCodebookGainIndices` in one call), `gain_scaled_innovation_sample`
  (single-sample helper), and `GAIN_SCALED_INNOVATION_SAMPLES`. Spec
  basis: §8.4 `e[n] = p[n] + c[n]` (manual) + companion §2.3
  fixed-codebook-gain product structure + the staged
  `provenance/02-speex-gain-quant.md` decoder application law
  `ener = MULT16_32_Q14(scal[q], ol_gain)` (the reconstructed gain
  multiplies the innovation). A silent frame drives the gain to `0.0`,
  vanishing the contribution. 11 new tests (382 lib tests, up from 371).
- Round r316: **log-domain scalar gain reconstruction grid** wiring the
  previously index-only excitation-gain fields into reconstructed scalar
  magnitudes. New `gain_reconstruction` module exposes the parametric
  `GainGrid` (`reconstruct` / `table` / `db_per_step` /
  `dynamic_range_db`), the three documented grids `NB_OL_EXC_GAIN_GRID`
  / `HB_EXC_GAIN_GRID_5BIT` / `HB_EXC_GAIN_GRID_4BIT`, and the dispatch
  helpers `reconstruct_frame_ol_exc_gain` / `reconstruct_hb_exc_gain`
  (silence / absent index variants reconstruct to `0.0`). Spec basis:
  the staged `docs/audio/speex/gain-quantiser-and-lsp-lpc-trace.md`
  §2 / §4 reconstruction grid `g(index) = 10^((index − offset) / slope)`
  shared between the narrowband frame-level OL excitation gain (5 bits)
  and the high-band excitation gain (5 bits HB mode 1, 4 bits HB modes
  2..=4, *"coded in the same way as for narrowband"*). The grid shape
  (monotone log-domain, uniform dB-per-step, ~80 dB / ~64 dB decade
  scale) is pinned and tested; the codec author's exact `(slope, offset)`
  constants are the recorded behavioural-trace gap, so the values are
  not yet reference-bit-exact and `GainGrid`'s parameters stay the
  single fix-site for the eventual calibration pin. 11 new tests
  (407 total, up from 396).

- Round r302: wideband **high-band (sub-band CELP) LSP→LPC conversion**,
  the high-band counterpart of the r286 narrowband path. The
  `lsp_to_lpc` polynomial core is refactored into an order-generic
  slice-based helper (`A(z) = (P(z) + Q(z)) / 2` for any even order;
  the order-10 narrowband path delegates to it unchanged) and a new
  order-8 high-band entry point is added: `hb_lsp_to_lpc` (radian
  input), `hb_lsp_q10_to_radians` (the shared Q10 angular-unit
  assumption applied at the high-band scale, since
  `HB_LSP_OUTPUT_Q == NB_LSP_OUTPUT_Q == 10`), and `lpc_from_hb_lsp_q10`
  composing the two. Wired through `WidebandHighBandBody::hb_lpc`, which
  composes the r214 high-band LSP MSVQ reconstruction with the new
  conversion to yield eight signed high-band LPC coefficients (returns
  `None` for silence mode 0). Spec basis: *The Speex Codec Manual* §10.1
  (high-band LSPs converted back to the LPC filter exactly as §9.1
  describes, at the order-8 high-band LPC order reconciled on
  `HB_LPC_ORDER`). 10 new lib tests (359 total, up from 349) + 4 new
  integration tests in `tests/hb_lsp_to_lpc.rs`. The numeric LSP
  fixed-point pin for bit-exactness remains the same recorded docs gap
  as the narrowband conversion.

- Round r296: narrowband **per-sub-frame LSP→LPC conversion for a full
  frame**, bridging the r200 sub-frame LSP interpolation (Q12
  `NbSubFrameLsp`) and the r286 LSP→LPC core. The r286 `lpc_from_lsp_q10`
  path only consumed the per-frame Q10 vector; the interpolated
  per-sub-frame vectors carry two extra sub-binary-point bits (Q12),
  so a Q-shift-aware conversion was missing. Generalises
  `lsp_q10_to_radians` into the Q-shift-parameterised
  `lsp_qn_to_radians` (Q10 helper now delegates), adds
  `lpc_from_subframe_lsp_q12` for one Q12 sub-frame vector, and
  `subframe_lpc_set` returning the four per-sub-frame LPC sets the
  synthesis filter consumes (manual §9.1: each sub-frame is filtered
  with its own interpolated LPC set; the 4th sub-frame carries the
  current LSPs unchanged). The angular-unit assumption stays pinned in
  the single shared `lsp_qn_to_radians` helper, keeping the recorded
  LSP Q-format docs gap to one fix-site.

- Round r286: narrowband **LSP→LPC conversion + the LPC synthesis
  filter**, closing the decoder "lacks LSP→LPC + synthesis filter"
  tail. New `lsp_to_lpc` module: the standard auxiliary-polynomial
  LSP reconstruction `A(z) = (P(z) + Q(z)) / 2` (P/Q built from the
  per-LSP second-order sections `[1, −2cos(ωₖ), 1]` with the
  `(1 ± z⁻¹)` boundary factors), grounded in *The Speex Codec
  Manual* §9.1 ("converted back to the LPC filter Â(z)") + §9.4
  ("S(z) = 1/A(z)"). `lsp_to_lpc` is the general radian-input float
  transform; `lsp_q10_to_radians` / `lsp_vector_q10_to_radians` /
  `lpc_from_lsp_q10` bridge the r194/r200 Q10 reconstruction under a
  documented angular-unit assumption (the LSP fixed-point pin for
  bit-exactness is a recorded docs gap). New `synthesis` module:
  `SynthesisFilter` runs the decoder recurrence
  `x[n] = e[n] + Σ a[i]·x[n−1−i]` (manual §8.2 prediction-error
  inverse) with persistent IIR history across sub-frame and frame
  boundaries, emitting `f64` or rounded/saturated `i16` PCM. New
  integration test `tests/synthesis_pcm_fixture.rs` runs the real
  mode-5 `speexenc` fixture end-to-end (LSP → interp → LSP→LPC →
  innovation `c[n]` → synthesis) to finite, responsive, non-silent,
  deterministic PCM — the crate's first real audio output. The
  adaptive-codebook (pitch) contribution is not yet folded into the
  excitation (gain Q-format still gap-blocked), so the PCM is
  correct-by-construction but not yet bit-exact. 17 new unit tests +
  3 integration tests (343 lib tests total).
- Round r277: narrowband **per-mode innovation codebook binding for
  modes 2 / 3 / 4 / 5**, retiring the r220 `Undocumented` dispatch
  for four of the six previously unbound modes. Grounding: the
  staged per-codebook innovation bit-rate annotations
  (`docs/audio/speex/tables/innovation-cdbk-*.meta` `role:` fields +
  the staged `tables/README.md` inventory) combined with Table 9.1's
  "Innovation VQ" row (`bits/sub-frame × 200 = innovation bps`)
  uniquely pin mode 2 → 4 × `Sv10_16` (3 200 bps), mode 3 → 4 ×
  `Sv10_32` (4 000 bps), mode 4 → 5 × `Sv8_128` (7 000 bps), mode 5
  → 8 × `Sv5_64` (9 600 bps); every binding satisfies
  `index_bits × count == innovation_vq_bits` and
  `sub_vector_len × count == 40` samples and cross-checks against
  Table 9.2's composite bit-rates. `InnovationMapping::for_mode` now
  surfaces `Documented` for modes 2 / 3 / 4 / 5 / 6 / 8;
  `Undocumented` narrows to mode 1 (0-bit VQ field; vocoder
  excitation-generation rule unstaged) and mode 7 (96 bits → 19 200
  bps, no annotated codebook). The real `speexenc`-encoded mode-5
  fixture now decodes every sub-frame's 40-sample `c[n]` innovation
  vector end-to-end; the r220 docs-gap-pin integration test is
  rewritten as `innovation_subvector_decodes_for_mode_5_fixture`
  with a manual MSB-first cross-check of all 8 per-sub-frame
  lookups against the raw 48-bit field. 8 new unit tests in
  `innovation::tests`; the r234 `resolve_lookback` doc-example is
  promoted from an ignored snippet to a compiled doc-test. 356
  tests total (326 unit + 29 integration + 1 doc), up from 347.
  The round's prime candidate (NB/HB excitation-gain magnitudes)
  remains DOCS-GAP-blocked: no open-loop scalar gain quantiser
  specification is staged under `docs/audio/speex/`.
- Round r269: wideband **high-band fixed-codebook gain index
  primitive** — the high-band counterpart of r261's narrowband
  composition. Per Speex Codec Manual §10.4 / Table 10.1 (CELP
  companion §5.1) the high band carries exactly one gain field per
  sub-frame (the `Excitation gain` row: `0 / 5 / 4 / 4 / 4` bits for
  modes 0..=4) and no frame-level factor, so the §9.2
  `g_frame × g_subf` product structure reduces to a single typed
  index. New `hb_excitation_gain` module exposes
  `HbExcitationGainIndex` (`Absent` for mode 0 / `FiveBit(0..=31)`
  for mode 1 / `FourBit(0..=15)` for modes 2..=4) with `resolve` /
  `from_body` / `is_absent` / `bit_budget` / `entries` / `raw_index`
  / `Display` helpers, the `hb_excitation_gain_indices(body, submode)`
  batch helper returning `[HbExcitationGainIndex; 4]` per high-band
  frame, and a new
  `WidebandHighBandBody::hb_excitation_gain_indices(submode)`
  convenience method mirroring r261's
  `NarrowbandFrameBody::fixed_codebook_gain_indices`. New public
  constants `HB_EXC_GAIN_BITS_MODE_1` (5) and
  `HB_EXC_GAIN_BITS_MODES_2_TO_4` (4). The numeric gain magnitude is
  gap-blocked behind the documented "computed, not a lookup array"
  open-loop scalar quantiser note (staged tables README "Not
  extracted" subsection), so the primitive surfaces only the typed
  index algebra per the r234 / r241 / r244 / r261 Q-format-agnostic
  pattern. 13 new unit tests in `hb_excitation_gain::tests` (318 lib
  tests, up from 305): mode-0 absent everywhere, mode-1 5-bit
  surface, modes-2..=4 4-bit surface, per-frame gain footprint equals
  `4 × budget` with no frame-level term for every documented mode,
  full 5-bit / 4-bit index ranges, non-conforming budget rejected,
  out-of-range slot rejected, `raw_index` / `entries` / `Display`
  surfaces, batch-vs-per-slot agreement, `is_absent` flags mode 0
  only, width constants match the staged sub-mode table. Plus 3 new
  integration tests in `tests/hb_excitation_gain_indices.rs`:
  synthetic high-band bodies for every documented mode 0..=4 built
  via the public `BitWriter` round-trip the written gain indices
  through `WidebandHighBandBody::parse` + the new accessor; the
  mode-0 body consumes zero bits and resolves `Absent`; resolution is
  independent of the LSP + excitation-VQ field contents. 347 tests
  total (318 unit + 29 integration), up from 331. README "Spec gaps
  noted" gains a **high-band excitation-gain quantiser** entry.

- Round r261: narrowband **fixed-codebook gain index composition
  primitive** surfacing the Speex Codec Manual §9.2 / CELP companion
  §2.3 product structure `fixed-codebook gain = g_frame × g_subf` at
  the typed-index layer. New `fixed_codebook_gain` module exposes
  `FrameInnovationGainIndex` (typed wrapper over the 5-bit frame-level
  OL excitation-gain field with a `Silence` variant for mode 0),
  `SubFrameInnovationGainCorrection` (typed wrapper over the 0 / 1 /
  3-bit per-sub-frame correction with an `Absent` variant for the
  0-bit-budget modes 0, 2, 8), and the composed `FixedCodebookGainIndices`
  pair. `fixed_codebook_gain_indices(body, submode)` returns
  `[FixedCodebookGainIndices; 4]` per frame; a new
  `NarrowbandFrameBody::fixed_codebook_gain_indices(submode)`
  convenience method wires it off the existing parsed body. Helpers:
  `FixedCodebookGainIndices::is_absent()` (silence-mode short-circuit),
  `wire_bit_budget()` (per-pair `5+0 / 5+1 / 5+3 / 0+0` budget pinning
  the spec's per-mode footprint), `from_body(body, submode, sub_idx)`
  (single-sub-frame helper). The numeric `g_frame × g_subf`
  reconstruction is gap-blocked behind the CELP companion §9
  "computed, not a lookup array" open-loop scalar quantiser note; this
  primitive surfaces only the algebra of the index composition,
  matching the r234 / r241 / r244 Q-format-agnostic design pattern.
  Public re-exports: `fixed_codebook_gain_indices`,
  `FixedCodebookGainIndices`, `FrameInnovationGainIndex`,
  `SubFrameInnovationGainCorrection`, `FRAME_OL_EXC_GAIN_BITS`,
  `FRAME_OL_EXC_GAIN_ENTRIES`. New `pub const SUBFRAMES_PER_FRAME:
  usize = 4` in `crate::submode` for index-typed callers (mirrors the
  existing `NarrowbandSubmode::SUBFRAMES_PER_FRAME: u32` used in the
  bit-budget arithmetic). 15 new unit tests in
  `fixed_codebook_gain::tests` (305 lib tests, up from 290 in r244):
  mode-0 silence flags absent everywhere, mode-1 1-bit correction
  surface, mode-2 frame factor present + correction absent, mode-5
  3-bit correction surface, mode-8 special low-bitrate pattern, every
  documented mode 0..=8 hits the in-spec budget pair `(0|5, 0|1|3)`,
  out-of-range sub-frame slot returns `None`, hand-built non-conforming
  budgets rejected, wire-bit-budget decomposes into the two factors,
  Display strings, `raw_index` helpers, 5-bit field covers `0..=31`,
  batch matches per-position helper, `is_absent()` tracks frame factor
  only (not the correction). Plus 2 new integration tests in
  `tests/narrowband_body_fixture.rs`: every audio sub-frame of the
  mode-5 fixture composes as `(Indexed(0..=31), ThreeBit(0..=7))` with
  varying frame-level indices and at least one non-zero correction;
  silence-mode body composes as `(Silence, Absent)` for every
  sub-frame.

- Round r244: narrowband **raw excitation composition primitive**
  composing r241 + r220 into the per-sub-frame `[i32; 40]` raw-integer
  evaluation of Speex Codec Manual §8.4 / CELP companion §2.3
  `e[n] = p[n] + c[n]` where `p[n] = ea[n]` is the r241 adaptive-codebook
  contribution and `c[n]` is the r220 fixed-codebook (innovation)
  sub-vector. New `excitation` module exposes
  `raw_excitation_subframe(ea, c)` (whole sub-frame batch over
  `[i32; 40] + [i16; 40]`) and `raw_excitation_sample(n, ea_n, c_n)`
  (per-sample helper) returning the per-sample widening sum
  `ea[n] + i32::from(c[n])`. The output is **Q-format-agnostic** by
  design — both inputs are raw integer values (post-bias gain integers
  times `i16` historical samples on the r241 side; raw `i16` codebook
  entries on the r220 side), so the per-sample sum stays in the same
  raw integer units. Headroom argument: `|ea[n]| + |c[n]|` is bounded
  by `1.6 × 10⁷ + 3.3 × 10⁴ ≈ 1.6 × 10⁷`, well below `i32::MAX ≈
  2.1 × 10⁹`, so the `i32 + i32` accumulator stays in range across
  the entire `[17, 144]` pitch range and the full post-bias gain
  envelope. Stream-start behaviour is inherited from r241: with the
  all-zero default `ExcitationBuffer` from r234 the r241 `ea` term is
  identically zero, so the composed `e_raw` follows the first-frame
  innovation sub-vector verbatim (the documented "no spurious
  transient" envelope). New public constant `RAW_EXCITATION_SAMPLES`
  restating `40` at the composition layer. Public re-exports
  `raw_excitation_subframe`, `raw_excitation_sample`,
  `RAW_EXCITATION_SAMPLES`. 12 new unit tests in `excitation::tests`
  (290 lib tests total, up from 278 in r241): both-zero → zero,
  zero-`ea` → widened-`c`, zero-`c` → `ea` exact, linearity in `ea`,
  pointwise pin matching the documented formula, per-sample vs batch
  agreement, stream-start envelope follows mode-6 documented dispatch
  innovation only (eight-sub-vector `Sv5_256` walk), silence
  sub-mode → all-zero envelope, headroom argument at the analytic
  bounds with `checked_add` proofs, hand-summed worked example
  (`out[0] = 1_007`, `out[1] = -503`, `out[39] = 12_300`),
  negation-commutation algebra invariant, and the mode-8 `Sv20_32`
  documented dispatch + non-trivial 150-sample buffer → element-wise
  composition check. The fixed-codebook gain composition (CELP
  companion §9 open-loop scalar quantiser) + saturating
  `i32 → i16` buffer-push step + final Q-format pin stay deferred
  behind the documented pitch-gain Q-format gap (see the
  `adaptive_contribution` module docs).

- Round r241: narrowband **adaptive-codebook contribution sum**
  composing r208 + r234 into the closed-form per-sub-frame
  `[i32; 40]` evaluation of Speex Codec Manual §9.2 Eq. 9.1
  (`ea[n] = g0·e[n − T − 1] + g1·e[n − T] + g2·e[n − T + 1]`). New
  `adaptive_contribution` module exposes
  `adaptive_contribution_subframe(pitch_period, taps, &buffer)` (whole
  sub-frame batch) and `adaptive_contribution_sample(n, pitch_period,
  taps, &buffer)` (per-sample helper); for every output position the
  three substituted lookbacks are resolved via
  `sample_lookback_indices` (r234) and three historical samples are
  read off the `ExcitationBuffer` (r234), then accumulated as the raw
  integer dot product `Σ taps[j] · e[lookbacks[j]]` into an `i32`. The
  output is **Q-format-agnostic**: each `g · e` is an integer ×
  integer product so the sum is well-defined without committing to a
  Q-format choice (any downstream scaling is a single arithmetic shift
  over the whole `[i32; 40]` vector). Stream-start behaviour is
  inherited from the all-zero default buffer (the documented
  "no spurious transient" envelope drops out for free). New
  `AdaptiveContributionError` variants `PitchOutOfRange { period }`
  (refuses an out-of-spec `[17, 144]` pitch) and `Buffer(ExcitationError)`
  (unreachable for an in-spec pitch but surfaced for diagnostics).
  Public re-exports `adaptive_contribution_subframe`,
  `adaptive_contribution_sample`, `AdaptiveContributionError`. 12 new
  unit tests in `adaptive_contribution::tests` (278 unit total, up
  from 266 in r234): empty-buffer-yields-zero, silence-taps-yield-zero,
  pitch-range rejection, constant-buffer pin, the hand-computed worked
  example `ea[0] = -499` / `ea[1] = -489` / `ea[39] = -109` for
  `T = 50, taps = (2, 5, 3)`, agreement between batch and per-sample
  paths across the full pitch range, short-pitch repeat-rule pin
  `ea[0] = 698` for `T = 17`, analytic `i32` headroom argument, the
  stream-start zero envelope for the extreme tap triples
  `(-96, -96, -96)` / `(159, 159, 159)` / `(0, 159, -96)`, linearity
  in the gain triple and in the buffer pointwise, and an integration
  smoke test composing `reconstruct_pitch_gain` + a non-trivial
  buffer to pin `ea[0] = 300` for codebook index 1 of the 5-bit
  table. The downstream `e[n] = p[n] + c[n]` final-excitation
  composition stays deferred behind the `pitch_gain` Q-format gap
  and the fixed-codebook gain scalar-quantiser gap recorded in CELP
  companion §9.
- Round r234: narrowband **adaptive-codebook (long-term predictor)
  index resolution + excitation history buffer** per Speex Codec
  Manual §9.2 Eq. 9.1 (`ea[n] = g0·e[n − T − 1] + g1·e[n − T] +
  g2·e[n − T + 1]`) and the documented excitation-repeat rule for
  short pitches (*"when the pitch is smaller than the sub-frame
  size, we repeat the excitation at a period T. For example, when
  `n − T + 1 ≥ 0`, we use `n − 2T + 1` instead"*). New
  `adaptive_codebook` module exposes the typed index-arithmetic
  helpers `resolve_lookback(k, t)` (iterates the documented
  `k ← k − T` substitution while `k ≥ 0`), `sample_lookback_indices(n, t)`
  (per-sample three-tap offsets, strictly negative), and
  `subframe_lookback_indices(t)` (a `[[i32; 3]; 40]` matrix per
  pitch period), plus the typed `ExcitationBuffer` rolling buffer
  of the last `EXCITATION_HISTORY_LEN = 145` samples of the
  emitted excitation `e[·]`, with `push` / `extend_from_slice` /
  `lookup(k)` (negative-offset addressing matching the manual's
  `e[n − k]` notation) + typed `ExcitationError` variants for
  non-historical and out-of-history offsets. Public constants
  `EXCITATION_HISTORY_LEN`, `ADAPTIVE_CODEBOOK_TAPS = 3`, and
  `TAP_PITCH_OFFSETS = [-1, 0, 1]` pin the documented derivation.
  Public re-exports `resolve_lookback`, `sample_lookback_indices`,
  `subframe_lookback_indices`, `ExcitationBuffer`,
  `ExcitationError`, `ADAPTIVE_CODEBOOK_TAPS`,
  `EXCITATION_HISTORY_LEN`, `TAP_PITCH_OFFSETS`. 22 new unit tests
  in `adaptive_codebook::tests` (266 unit total, up from 244 in
  r230). Module is **Q-format-agnostic** by design — the gain
  multiplication `gj · e[kj]` is deferred until the documented
  β Q-format pin (see `pitch_gain` module docs for the recorded
  gap), so the index resolution + buffer state machine land
  independently and the eventual long-term-predictor sum can pin
  its scaling in a single follow-up round.

- Round r230: wideband **high-band innovation sub-vector lookup
  primitive + per-mode dispatcher** mirroring r220's narrowband path
  for the sub-band-CELP high band (Speex Codec Manual §10.3 + Table
  10.1 + CELP companion §9 / `tables/README.md`). New `hb_innovation`
  module exposes `HbInnovationCodebook` selecting between the two
  documented high-band codebook shapes (`HbSv8_128` — 8-sample × 7-bit
  index + 1-bit sign, 128 entries; `HbSv10_32` — 10-sample × 5-bit,
  32 entries), `hb_innovation_sub_vector(codebook, index)` returning
  the `&'static [i16]` slice for one row, `HbInnovationMapping::for_mode`
  dispatching modes 0 and 1 to `Silence`, mode 2 to `Documented` (4 ×
  `HbSv10_32`), mode 3 to `Documented` (5 × `HbSv8_128` with sign bit),
  and mode 4 to `Undocumented`, plus `decode_hb_subframe(submode,
  excitation_vq_index)` decoding the 40-sample fixed-codebook
  excitation sub-vector for one high-band CELP sub-frame by
  concatenating `count` MSB-first slot lookups (each slot being a 7-bit
  index + 1-bit sign for `HbSv8_128` or a 5-bit index for `HbSv10_32`)
  off the raw `excitation_vq_index` field, with the sign bit (when
  present) negating the looked-up sub-vector element-wise. New
  `WidebandHighBandBody::hb_innovation_sub_vector(submode, sub_idx)`
  convenience method wires the dispatcher off the existing
  per-sub-frame `excitation_vq_index`. Public re-exports
  `HbInnovationCodebook`, `HbInnovationError`, `HbInnovationMapping`,
  `decode_hb_subframe`, `hb_innovation_sub_vector`,
  `HB_SUBFRAME_SAMPLES`. 20 new unit tests in `hb_innovation::tests`
  (244 unit total, up from 224 in r220) + 6 new integration tests in
  `tests/hb_innovation_dispatch.rs`
  (`parse_then_dispatch_matches_direct_path_for_mode_2_body`,
  `parse_then_dispatch_matches_direct_path_for_mode_3_body`,
  `silence_modes_return_all_zero_sub_vector`,
  `mode_4_dispatcher_is_undocumented_with_full_excitation_vq_field`,
  `dispatcher_for_every_documented_mode_satisfies_bit_budget`,
  `codebook_row_zero_for_each_shape_is_accessible_through_public_api`).

- Round r220: narrowband **innovation sub-vector lookup primitive +
  per-mode dispatcher** for the two modes whose codebook binding is
  grounded by the staged material (Speex Codec Manual §9.2 + CELP
  companion §2.3). New `innovation` module exposes
  `InnovationCodebook` selecting one of the six documented
  sub-vector codebook shapes (`Sv5_64` / `Sv5_256` / `Sv8_128` /
  `Sv10_16` / `Sv10_32` / `Sv20_32`), `sub_vector(codebook, index)`
  returning the `&'static [i16]` slice for one row,
  `InnovationMapping::for_mode(submode)` dispatching mode 0 to
  `Silence`, modes 6 (8 × `Sv5_256`) and 8 (2 × `Sv20_32`) to
  `Documented`, and modes 1 / 2 / 3 / 4 / 5 / 7 to `Undocumented`,
  plus `decode_subframe(submode, innovation_vq_index)` decoding the
  40-sample fixed-codebook `c[n]` sub-vector for one CELP sub-frame
  by concatenating `count` MSB-first sub-vector lookups off the
  raw `innovation_vq_index` field. New
  `NarrowbandSubFrameIndices::innovation_sub_vector(submode)`
  convenience method wires the dispatcher off the existing
  per-sub-frame `innovation_vq_index`. Public re-exports
  `InnovationCodebook`, `InnovationError`, `InnovationMapping`,
  `decode_innovation_subframe`, `innovation_sub_vector`,
  `SUBFRAME_SAMPLES`. 17 new unit tests in `innovation::tests`
  (224 unit total, up from 207 in r214) + 2 new integration tests
  in `tests/narrowband_body_fixture.rs`
  (`innovation_dispatcher_is_undocumented_for_mode_5_fixture`,
  `innovation_subvector_for_silence_mode_is_all_zero`).
- Round r214: wideband **high-band LSP MSVQ reconstruction** per
  Speex Manual §10.1 / CELP companion §9. New `hb_lsp` module
  exposes `HbLspStages::from_packed(lsp_index, submode)` splitting
  the 12-bit packed `lsp_index` already surfaced by
  `WidebandHighBandBody::lsp_index` into per-stage 6-bit indices
  (top 6 bits → stage 1 / level-1 codebook, bottom 6 bits → stage 2
  / residual codebook), plus
  `reconstruct_q10(stages) -> Option<[i32; 8]>` summing the two
  staged codebook rows with the `.meta`-documented per-stage
  scaling (`hb-lsp-cdbk-stage1` 1/256 → ×4, `hb-lsp-cdbk-stage2`
  1/512 → ×2) into a common Q10 fixed-point eight-coefficient LSP
  vector (matching r194's narrowband Q-format so both bands speak
  the same downstream format). Silence mode (high-band mode 0 —
  `submode.lsp_bits == 0`) returns `None`. New
  `WidebandHighBandBody::lsp_stages(submode)` and
  `WidebandHighBandBody::reconstructed_lsp_q10(submode)` convenience
  methods wire the new module off the existing parsed body,
  mirroring r194's `NarrowbandFrameBody::reconstructed_lsp_q10`
  for the high band.
- Round r214: public re-exports `reconstruct_hb_lsp_q10`,
  `HbLspStages`, `HB_LSP_INDEX_MASK`, `HB_LSP_OUTPUT_Q`,
  `HB_LSP_PACKED_BITS`, `HB_LSP_STAGE_BITS` (the existing
  `HB_LSP_STAGE_ENTRIES` + `HB_LPC_ORDER` from r191 cover the
  underlying dimensions). 12 new unit tests in `hb_lsp::tests`
  (silence-mode rejection; MSB-first packing round-trip over the
  full 64×64 index space; 12-bit index-mask saturation; eight-
  coefficient output length matches `HB_LPC_ORDER`; stage 1 +
  stage 2 contributions isolated via difference tests; full 4096-
  point exhaustive scan never panics and stays bounded by 762 in
  Q10; out-of-range stage 1 / stage 2 indices return `None`;
  from_packed → reconstruct matches direct path; `HB_LSP_OUTPUT_Q`
  equals `NB_LSP_OUTPUT_Q`; `HB_LSP_PACKED_BITS` matches every
  documented submode's `lsp_bits`).
- Round r214: 4 new integration tests in
  `tests/hb_lsp_reconstruction.rs` build synthetic high-band
  bodies via the public `BitWriter` (with a 32-bit-chunked
  zero-bit helper for the 80-bit mode-4 excitation VQ), parse
  them through `WidebandHighBandBody::parse`, and verify the new
  accessor matches the direct path for a synthesised mode-2
  packet; silence-mode 0 yields `None`; round-trip succeeds for
  every documented mode 1..=4 (covering the 20 / 40 / 80-bit
  excitation-VQ fields); and the Q10 dynamic range is bounded by
  762 at maximum-index reconstruction. 223 tests total
  (207 unit + 16 integration), up from 207 in r208.

- Round r208: narrowband **3-tap pitch-gain VQ reconstruction** per
  Speex Manual Eq. 9.1 / CELP companion §2.2. New `pitch_gain` module
  exposes `reconstruct(index, quant) -> Option<PitchGainTaps>` and
  the typed `PitchGainTaps { taps: [i16; 3] }` carrying the three
  β tap coefficients `(g0, g1, g2)` of the long-term predictor
  equation `ea[n] = g0·e[n−T−1] + g1·e[n−T] + g2·e[n−T+1]` with the
  documented `+32` codebook bias applied. `PitchGainQuant::None`
  (mode 0, silence) returns the all-zero `PitchGainTaps::SILENCE`
  constant; the 5-bit codebook (32 entries) handles the low-bit-rate
  modes, the 7-bit codebook (128 entries) handles the higher-rate
  modes. Column 3 (`search_aid`) is an encoder-only term and is
  dropped. New `NarrowbandSubFrameIndices::pitch_gain_taps(submode)`
  convenience method wires the lookup off the existing per-sub-frame
  raw `pitch_gain_index`.
- Round r208: 12 new unit tests in `pitch_gain::tests` (silence-quant
  ignores-index; 5-bit row 0 yields the documented `(0, 0, 0)`
  silence taps after bias; 5-bit row 1 and 7-bit row 0 match the
  staged CSV with bias applied; 5-bit max-index = 31; 7-bit
  max-index = 127; out-of-range indices return `None`; both
  codebooks accept their full valid range; search-aid column is
  dropped from output; +32 bias applied consistently across every
  row of both codebooks; SILENCE constant equals 5-bit row 0; every
  post-bias value falls in `-96..=159`). 2 new integration tests in
  `tests/narrowband_body_fixture.rs` walk every audio packet of the
  `speexenc`-encoded fixture and confirm every sub-frame's β taps
  resolve, at least one sub-frame produces non-zero β coefficients,
  and the silence-mode wrapper returns SILENCE for a hand-built
  mode-0 frame. Total tests: 193 unit + 12 integration = 205, up
  from 193 unit + 8 integration = 201 after r200.
- Round r208 docs: README "Spec gaps noted" entry on the
  **pitch-gain β Q-format**. The staged 3-tap pitch-gain VQ tables
  carry the gain bytes biased (`+32`) but neither the in-repo
  manual nor the CELP companion documents a fixed-point Q-format
  for the post-bias β values themselves (Q6 is widely used in CELP
  literature but in-repo material does not commit to it).
  `pitch_gain::reconstruct` surfaces post-bias raw integers,
  leaving the scaling choice to the downstream long-term predictor.

- Round r200: narrowband **sub-frame LSP interpolation** per Speex
  manual §9.1 ("The LSP's are considered to be associated to the 4th
  sub-frames and the LSP's associated to the first 3 sub-frames are
  linearly interpolated using the current and previous LSP
  coefficients"). New `lsp_interp` module exposes
  `NbSubFrameLsp::new(prev_q10, curr_q10)` returning a
  `[[i32; 10]; 4]` matrix of per-sub-frame LSP vectors in Q12
  fixed-point — the unique linear-interpolation weight set
  `(3·prev + 1·curr)/4`, `(2·prev + 2·curr)/4`,
  `(1·prev + 3·curr)/4`, `(0·prev + 4·curr)/4 = curr`. Output is
  emitted in Q12 (Q10 + 2 extra bits from the un-divided weight
  multiplication) so every interpolation operation is exact integer
  arithmetic with no rounding direction question for the spec to be
  silent about. A `first_frame(curr_q10)` constructor handles the
  stream-start case (prev = curr → flat envelope, no spurious
  transient). New `NarrowbandFrameBody::interpolated_lsp_q12(submode,
  prev_q10)` convenience method composes the r194 reconstruction
  with the r200 sub-frame interpolation; silence mode (mode 0)
  propagates `None`.
- Round r200: 15 new tests — 10 unit tests in `lsp_interp::tests`
  (per-weight verification, Q-format self-check, per-coefficient
  independence, monotone-envelope-on-monotone-input,
  first-frame-flatness, negative-coefficient handling, out-of-range
  subframe accessor) and 3 new integration tests in
  `tests/narrowband_body_fixture.rs` walking every audio packet of
  the `speexenc`-encoded fixture frame-by-frame, threading the
  previous LSP state, and asserting (a) sub-frame 4 equals 4·curr in
  Q12 for every frame, (b) first-frame envelope is flat, (c) a
  non-zero number of steady-state frames produce a non-flat envelope
  (the previous-frame state is actually being used).
- Round r200 docs: README "Spec gaps noted" entry on the
  **LSP → LPC conversion algorithm** — manual §9.1 only states the
  interpolated LSPs are "converted back to the LPC filter Â(z)"
  without giving the procedure, and the staged companion is silent
  (its §9 covers raw codebook data; the conversion is algorithmic).
  Also a docs-gap entry on the first-frame-LSP-initialisation
  convention used by `NbSubFrameLsp::first_frame`.

- Round r194: first companion-table → decoder pipeline wiring. The
  5-stage narrowband LSP-VQ codebooks staged in `codebooks` now drive
  a reconstructed ten-coefficient LSP frequency vector in a common
  Q10 fixed-point format. New `lsp` module exposes
  `NbLspStages::from_packed(packed, quant)` to split an 18-bit or
  30-bit packed LSP field into per-stage 6-bit indices, and
  `reconstruct_q10(stages)` to sum the per-stage codebook
  contributions with the `.meta`-documented per-stage scaling
  factors (1/256 → ×4, 1/512 → ×2, 1/1024 → ×1 — common Q10 output
  unit means each stage contributes by integer multiplication, no
  rounding direction question). Wired from `NarrowbandFrameBody` as
  two new methods: `lsp_stages(submode)` and
  `reconstructed_lsp_q10(submode)`, both returning `None` for the
  silence mode (mode 0 — no LSP field transmitted).
- Round r194: 11 new unit tests in `lsp::tests` covering the splitter
  (silence regime, 18-bit MSB-first split, 30-bit MSB-first split,
  defensive masking of stray high bits) and the reconstruction
  (per-stage cross-checks against the staged CSV row 0 values for
  both 18-bit and 30-bit regimes, linearity-per-stage probe,
  in-range max-index handling, out-of-range index rejection,
  end-to-end packed-field round-trip, and a stage-shift-factor
  self-check against the module-doc table). Three new integration
  tests in `tests/narrowband_body_fixture.rs` exercise the wiring
  against the real `speexenc`-encoded fixture: every audio packet
  (mode 5, 30-bit LSP) splits and reconstructs without panic, the
  per-stage indices land in 0..64, and ≥ 90 % of frames produce a
  non-zero coefficient vector (confirming the staged codebooks are
  contributing actual signal, not silently returning zeros).
  Silence-mode body returns `None` for both `lsp_stages` and
  `reconstructed_lsp_q10`.

### Notes

- The bit-stream stage ordering convention used by
  `NbLspStages::from_packed` (stage 0 in the most-significant 6 bits,
  followed by `low1`, `low2`, `high1`, `high2` for the 30-bit regime)
  is not explicitly documented in the in-repo manual / RFC / staged
  companion. The convention is supported by three signals (coarse
  stage first matches every multi-stage VQ in the staged CELP
  family + matches the in-crate wideband 12-bit MSVQ split; the
  companion table inventory lists stages in this order; the
  per-stage widths sum exactly to the 18-bit / 30-bit field widths)
  and is the only place the assumption is encoded — reconstruction
  consumes the resolved per-stage indices and is independent of the
  unpack order. See `src/lsp.rs` module docs for the full
  justification.

## [0.0.6](https://github.com/OxideAV/oxideav-speex/releases/tag/v0.0.6) - 2026-05-30

### Other

- r191 fixup: vendor table CSVs into crate, switch include_str paths
- CELP companion-table accessors (codebooks module)
- scrub libspeex *_table.c citations in clean-room comments
- structured write methods symmetric to parse for framing-level types
- add MSB-first BitWriter, retire test-only BitPacker
- round r165: typed packet → frame iterator composing rounds 2..r160
- round 5: wideband high-band sub-mode table + body bit-reader (manual §10.4 / Table 10.1)
- round 4: §5.5 in-band signalling body parsers (modes 13 + 14)
- round 3: narrowband CELP frame-body bit-reader (manual §9.1-§9.3 / Table 9.1)
- round 2: narrowband frame-header + sub-mode table (manual §9.3 / Table 9.1)
- round 1: Ogg/Speex stream-header packet parser (manual §7.3 Table 7.1)
- orphan rebuild: clean-room scaffold post 2026-05-19 audit

### Added

- Round 191: CELP companion-table accessors (`codebooks` module,
  re-exported at the crate root). Embeds the clean-room
  `docs/audio/speex/tables/` CSVs via `include_str!` and parses each
  on first use into typed `&'static [Row]` slices: narrowband LSP VQ
  stage 0 (64 × 10) + four split-band stages (64 × 5), 5-bit and
  7-bit 3-tap pitch-gain VQ (32 × 4 / 128 × 4), six narrowband
  innovation codebooks covering the Table 9.1 shapes
  (5×64, 5×256, 8×128, 10×16, 10×32, 20×32), wideband high-band LSP
  MSVQ (2 × 64 × 8 — the 12-bit total matches `wideband.rs`'s raw
  `lsp_msvq_index`), two high-band innovation codebooks (8×128 +
  10×32), the 200-sample Q15 LPC analysis window, 11-tap Q15
  autocorrelation lag window, and 64-tap Q15 QMF analysis filter
  `h0` used by the wideband split (§10.1). Scaling regimes from the
  `.meta` sidecars (`LSP_DIV_256` / `Div512` / `Div1024`) are
  exposed as `NbLspScale` + `nb_lsp_scale(stage)` / `hb_lsp_scale`.
- Round 191: 16 new dimension self-checks and submode cross-checks
  in `codebooks::tests` (every accessor is hit so first-use
  `OnceLock` parsers and assertions run under `cargo test`; codebook
  row counts are confirmed to equal `1 << bits` for every shape;
  the 12-bit high-band LSP MSVQ space is asserted to equal
  64 × 64 = 1 << 12; the 5-bit pitch-gain row 0 is verified to
  decode to the all-zero "silence" tap after the documented +32
  bias). Total unit test count rises from 140 to 156.
- Ogg/Speex stream-header packet parser per *The Speex Codec Manual*
  §7.3 (Table 7.1): `SpeexHeader::parse(buf)` validates the
  `b"Speex   "` magic and decodes all 13 little-endian `int32` fields
  plus the `speex_version` ASCII string. Surfaces typed
  `HeaderError::TooShort` / `HeaderError::BadMagic` for malformed
  inputs.
- Public constants `SPEEX_MAGIC`, `SPEEX_HEADER_LEN` (80), and the
  three documented mode IDs `SPEEX_MODE_NARROWBAND` /
  `SPEEX_MODE_WIDEBAND` / `SPEEX_MODE_ULTRAWIDEBAND` for downstream
  consumers.
- Seven unit tests covering NB / WB / UWB synthetic headers, bad
  magic, short buffers, trailing-byte tolerance, and exhaustive
  field-order mapping against Table 7.1.
- Round 2: minimal MSB-first `BitReader` matching the Speex
  bit-packing convention (next-read bit is immediately to the right
  of the previous bit within the same byte, MSB-first); surfaces
  `BitError::Underflow { requested, remaining }` and
  `BitError::TooWide` diagnostics.
- Round 2: narrowband frame-header parser per
  *The Speex Codec Manual* §9.3 — `NarrowbandFrameHeader::parse`
  consumes the leading 5 bits of every Speex frame (1-bit wideband
  flag from §10.4 + 4-bit mode ID), dispatches the mode ID through
  the new `Submode` enum (regular CELP, custom in-band mode 13,
  in-band signalling mode 14, terminator mode 15 — per §5.5 +
  Table 5.1), and rejects the spec's "reserved" range 9..=12 as
  `FrameError::ReservedMode`.
- Round 2: typed `NarrowbandSubmode` records for modes 0..=8 covering
  every column of Table 9.1 — LSP-VQ width (`LspQuant`), frame-level
  open-loop pitch / pitch-gain / excitation-gain bit counts,
  sub-frame fine-pitch / pitch-gain (`PitchGainQuant`) / innovation
  gain / innovation VQ bit counts, plus the `Total` row — exported as
  the public `NARROWBAND_SUBMODES: [NarrowbandSubmode; 9]` table. A
  self-consistency test re-derives `Total` from the field breakdown
  for every column.
- Round 2: 24 new unit tests (31 total), covering MSB-first bit
  reading + boundary straddling + underflow + zero-width reads,
  frame-header parsing for every documented mode ID, reserved-mode
  rejection, and Table 9.1 row totals.
- Round 3: narrowband CELP frame-body bit-reader per
  *The Speex Codec Manual* §9.3 — `NarrowbandFrameBody::parse` walks
  Table 9.1's columns in bit-stream order ("all frame-based parameters
  packed before sub-frame parameters; the parameters for a sub-frame
  packed before the next sub-frame") and surfaces every field as a
  raw bit-index. Codebook lookup, LSP→LPC conversion, and the
  synthesis filter remain deferred until the Speex CELP companion
  tables are staged under `docs/audio/speex/`.
- Round 3: public `NarrowbandFrameBody` + `NarrowbandSubFrameIndices`
  structs carrying the LSP VQ index, open-loop pitch / pitch-gain /
  excitation-gain indices (frame-level), and the per-sub-frame fine
  pitch index, pitch-gain VQ index, innovation-gain index, and raw
  innovation VQ field (`u128`, sized for mode 7's 96-bit
  per-sub-frame innovation).
- Round 3: public `PITCH_PERIOD_MIN` / `PITCH_PERIOD_MAX` constants
  (17 / 144) per §9.2's "[17, 144] range" wording, plus
  `NarrowbandSubFrameIndices::pitch_period` de-biasing helper that
  returns `None` when the sub-mode's `fine_pitch_bits` is zero.
- Round 3: typed `NarrowbandBodyError::Underflow(BitError)` with a
  `From<BitError>` conversion and a top-level `Error::NarrowbandBody`
  wrapping.
- Round 3: 11 new unit tests in `src/narrowband_body.rs` (42 unit
  tests total) covering silence-mode empty body, body-bit-count
  derivation for every documented sub-mode, exact field widths for
  mode 5 + mode 7 (the widest fields), pitch-period de-biasing, and
  truncated-packet underflow diagnosis.
- Round 3: integration test
  `tests/narrowband_body_fixture.rs` (2 tests) parses every audio
  packet of a real `speexenc`-encoded narrowband fixture
  (`tests/fixtures/nb_440hz_q8.spx`, 1 s of 8 kHz 440 Hz tone at
  quality 8 → narrowband sub-mode 5) end-to-end through the round-2
  header parser + round-3 body parser. A minimal inline Ogg
  page-walker (≈30 lines, not a general-purpose Ogg implementation)
  lifts Speex packets out of the `.spx` file to keep the test free of
  cross-crate dev-dependencies per workspace policy. The fixture
  regeneration command is recorded in `tests/fixtures/Makefile`;
  `speexenc` is invoked as an opaque binary.
- Round 3: 44 tests total (42 unit + 2 integration), up from 31 in
  round 2.
- Round 4: §5.5 in-band signalling body parsers — `InbandMessage`
  (mode 14: 4-bit Table 5.1 code + `1 / 4 / 8 / 16 / 32 / 64` bits of
  payload) and `CustomInbandMessage` (mode 13: 5-bit byte-count +
  opaque payload skipped per §5.5's "decoder can skip it if it
  doesn't know how to interpret it" rule). Both consume the body
  that the round-2 dispatcher leaves immediately after the 5-bit
  frame prefix, so the cursor lands on the first bit of the next
  frame in the same packet without a manual re-sync.
- Round 4: public `INBAND_TABLE_5_1: [InbandCodeSpec; 16]` table
  staging every row of *The Speex Codec Manual* Table 5.1 verbatim
  (`PerceptualEnhancement` / `LessAggressive` / `SwitchMode` /
  `SwitchModeLowBand` / `SwitchModeHighBand` / `SwitchQualityVbr` /
  `RequestAcknowledge` / `SetRateMode` / `TransmitCharacter` /
  `IntensityStereo` / `AnnounceMaxBitrate` / `AcknowledgePacket` +
  reserved-row category). Lookup helper `inband_code_spec(code)`
  returns the row for any of the sixteen 4-bit codes.
- Round 4: public constants `CUSTOM_INBAND_SIZE_BITS` (5),
  `CUSTOM_INBAND_MAX_BYTES` (31), and `INBAND_CODE_BITS` (4) for
  downstream consumers' bit-budget math.
- Round 4: typed `Error::Signalling(SignallingError)` top-level
  variant + `From<SignallingError>` conversion matching the
  round-2 / round-3 error envelopes.
- Round 4: 22 new unit tests (66 total: 64 unit + 2 integration)
  covering Table 5.1 structural sanity (sixteen rows, codes match
  index, payload widths match the manual, reserved-row taxonomy),
  parsing every documented code category (perceptual enhancement /
  switch mode / transmit character / max bit-rate / packet ack /
  reserved 64-bit), the 32-bit + 64-bit width split paths,
  truncated-code + truncated-payload underflow diagnosis, mode-13
  size-zero / size-one / max-31-byte payload skip, and end-to-end
  round-trip parsing through `NarrowbandFrameHeader::parse` for
  mode 14 (transmit `'B'`) + mode 13 (zero-size custom message).
- Round 5: wideband high-band sub-mode table from *The Speex Codec
  Manual* §10.4 / Table 10.1 (modes 0..=4, 5 columns). Public
  `WIDEBAND_HIGH_BAND_SUBMODES: [WidebandHighBandSubmode; 5]` stages
  every column verbatim with rows `Wideband bit` / `Mode ID` (3 bits,
  not 4) / `LSP` (12-bit MSVQ for modes 1..=4, 0 for silence mode 0) /
  `Excitation gain` (sub-frame, 0/5/4/4/4 bits) / `Excitation VQ`
  (sub-frame, 0/0/20/40/80 bits) / `Total` (4/36/112/192/352 bits per
  20 ms high-band frame).
- Round 5: `WidebandHighBandFrameHeader::parse` consumes the 4-bit
  high-band prefix (1-bit wideband flag + 3-bit mode ID, distinct
  from the narrowband 5-bit prefix); `WidebandHighBandBody::parse`
  walks Table 10.1's columns in `frame-LSP || (gain || VQ) × 4
  sub-frames` order per §10.4 ("the entire narrowband frame is packed
  before the high-band is encoded. The narrowband part of the
  bit-stream is as defined in table 9.1. The high-band follows, as
  described in table 10.1.").
- Round 5: `WidebandSubmode::for_id` dispatches the 3-bit mode ID
  into `Documented(WidebandHighBandSubmode)` for modes 0..=4 and
  `ReservedHighRate(u8)` for modes 5..=7 (encodable in the 3-bit
  field but the staged Table 10.1 stops at column 4; Table 10.2 lists
  modes 5..=10 with composite bit-rates but does not detail the
  per-field budgets, so these IDs surface without a bit-budget
  contract until a follow-up docs round stages the missing columns).
- Round 5: public constants `HIGH_BAND_FRAME_PREFIX_BITS` (4) and
  `HIGH_BAND_SUBFRAMES_PER_FRAME` (4) for downstream consumers; new
  `HighBandSubFrameIndices` struct carrying per-sub-frame
  `excitation_gain_index` (u8) and `excitation_vq_index` (u128,
  sized for mode 4's 80-bit max).
- Round 5: typed `WidebandBodyError::Underflow(BitError)` +
  `ReservedHighRate(u8)` errors with `From<BitError>` conversion;
  new `Error::Wideband(WidebandBodyError)` top-level envelope
  variant + `From<WidebandBodyError>` plumbing.
- Round 5: 21 new unit tests in `src/wideband.rs` (87 unit tests
  total, up from 66) covering Table 10.1 structural sanity (five
  columns, mode IDs match index, totals match field breakdown, totals
  match the manual's verbatim `Total` row), per-column field-width
  assertions (LSP / excitation gain / excitation VQ), mode 0 silent
  high-band with empty body, mode 4 widest-fields round-trip, mode 1
  no-innovation-VQ assertion, truncated-frame underflow diagnosis,
  reserved-high-rate dispatch (modes 5..=7), out-of-3-bit-range
  rejection, and the wideband-flag-cleared surfacing.
- Round 5: 89 tests total (87 unit + 2 integration), up from 68 in
  round 4.
- Round r165: typed packet → frame iterator composing the
  round-2 / 3 / 4 / r160 primitives end-to-end per *The Speex Codec
  Manual* §5.5 — `PacketFrames::new(buf)` yields one
  `PacketFrame` per frame in the body and halts cleanly on a mode-15
  terminator or on `< NARROWBAND_FRAME_PREFIX_BITS` of remaining
  padding. New `parse_packet(buf) -> Vec<PacketFrame>` convenience.
- Round r165: `PacketFrame` enum with `Narrowband` / `Wideband` /
  `InbandSignalling` / `CustomInband` variants — each carrying the
  5-bit prefix alongside the typed body. Wideband variant captures
  both halves (narrowband body + high-band header + high-band body)
  per §10.4 *"the entire narrowband frame is packed before the
  high-band is encoded"*.
- Round r165: typed `PacketError` envelope (`Frame` /
  `NarrowbandBody` / `Wideband` / `Signalling` / `UnexpectedEnd`)
  with `From` conversions from every underlying error type; new
  top-level `Error::Packet(PacketError)` variant + `From<PacketError>`
  plumbing.
- Round r165: iterator surfaces a clean halt for the §5.5
  terminator + padding tail, and a `PacketError::Wideband(
  ReservedHighRate(id))` when the high-band 3-bit mode field lands
  in `5..=7` (the Table 10.1 docs gap).
- Round r165: 19 new unit tests in `src/packet.rs` (108 unit tests
  total, up from 89) covering empty buffer, terminator-only packet,
  single silence frame + padding, multi-frame packets, in-band
  signalling intermixed with CELP, custom in-band size-0 + silence,
  reserved-mode rejection, truncated-body underflow, error-then-None
  iterator invariant, wideband-silence round-trip, reserved-high-rate
  dispatch, and `Iterator` combinator usage (`filter` / `count`).
- Round r165: new integration test `tests/packet_iterator_fixture.rs`
  (2 tests) walks every audio packet of the existing
  `speexenc`-encoded narrowband fixture through `PacketFrames` and
  asserts every packet yields the expected mode-5 narrowband frame,
  the iterator halts cleanly, and `< 5` bits of trailing padding
  remain after the terminator.
- Round r165: 112 tests total (108 unit + 4 integration), up from
  91 in round 5.
- Round r179: public `BitWriter` MSB-first bit sink — the symmetric
  companion to `BitReader`, exposing `new()` / `with_capacity()` /
  `write_bit()` / `write(value, n)` / `pad_to_byte()` /
  `bits_written()` / `bits_left_in_last_byte()` / `is_byte_aligned()`
  / `as_bytes()` / `into_bytes()`. Defining contract: the writer
  produces a buffer the `BitReader` round-trips back to the original
  `(value, n)` pairs in the same order — proven by three round-trip
  tests (short curated sequence, per-bit, and 256-step LCG-driven
  random pattern). `BitError::TooWide(n)` is reused for `n > 32` so
  the writer's error envelope matches the reader's.
- Round r179: cfg-test-only `BitPacker` helper that the `packet`
  module had been using to assemble synthetic Speex packets has been
  retired in favour of the public `BitWriter`. The conversion is
  behaviour-preserving — every existing `packet::tests` call site now
  exercises the same public bit-packing routine an encoder would call.
- Round r179: 15 new unit tests on `BitWriter` (123 unit tests
  total, up from 108 in round r165) covering empty-writer state,
  single-bit MSB-first writes, multi-bit writes, byte-boundary
  straddling, zero-width no-op, `TooWide` diagnosis, full-`u32`
  payload, byte-pad-with-zeros, high-bits-above-`n` ignored,
  `bits_left_in_last_byte()` cursor tracking, `with_capacity()`
  pre-allocation, and the three reader+writer round-trip invariants.
- Round r179: 127 tests total (123 unit + 4 integration), up from
  112 in round r165.
- Round r187: encoder-side `write` methods symmetric to the existing
  `parse` paths for the framing-level types whose layout is fully
  defined by the manual without any CELP companion tables:
  `NarrowbandFrameHeader::write` (5-bit wideband flag + mode ID per
  §9.3), `InbandMessage::write` (4-bit Table 5.1 code + 1/4/8/16/32/64
  payload bits, with the same >32-bit split path the parser uses for
  reserved codes 14/15), and `CustomInbandMessage::write` (5-bit
  `size_bytes` per §5.5 + opaque payload bytes taken from a
  caller-supplied slice). All three writers are the inverse operations
  of the corresponding parsers and pass round-trip tests.
- Round r187: new `NarrowbandFrameHeader::new(wideband, mode_id)`
  constructor that dispatches the mode ID through `Submode::for_id`
  and rejects the reserved range 9..=12 — the encoder-side counterpart
  of the round-2 parser's reserved-mode rejection.
- Round r187: 17 new unit tests (140 unit tests total, up from 123
  in round r179) covering header round-trip for every documented mode
  ID with both wideband-flag values, the three §5.5 signalling-slot
  dispatches, exact-5-bit emission, high-bit masking of mode-ID,
  round-trip for every Table 5.1 in-band code (including the >32-bit
  split path for reserved codes), payload-bit truncation above the
  declared width, custom in-band size-zero and size-31 boundary
  cases, payload-slice over-supply tolerance, and an end-to-end
  "write header + write inband message → parse back" path that
  exercises the round-2 dispatcher's reader-side accept against the
  writer-side emit.
- Round r187: 144 tests total (140 unit + 4 integration), up from
  127 in round r179.

### Erased

- Prior master history was force-erased on **2026-05-19** under
  Hat-3 cold enforcement of the workspace clean-room policy
  (`docs/IMPLEMENTOR_ROUND.md`).

### Reset

- Crate reduced to a minimal `oxideav_core::register!` stub. Every
  public API returns `Error::NotImplemented`. The crates.io version
  (`0.0.6`) is preserved on the new master to avoid breaking
  downstream version pins; the published versions on crates.io will
  be yanked by the maintainer.
- The `oxideav-ogg` dev-dependency (used by the prior integration
  test) is dropped from the scaffold and will be re-introduced in a
  future round if needed.

### Next

- Clean-room re-implementation of the CELP frame decoder against the
  published Speex specifications + RFC 5574 in subsequent rounds.
