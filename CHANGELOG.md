# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
