# oxideav-speex

[![CI](https://github.com/OxideAV/oxideav-speex/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-speex/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-speex.svg)](https://crates.io/crates/oxideav-speex) [![docs.rs](https://docs.rs/oxideav-speex/badge.svg)](https://docs.rs/oxideav-speex) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure-Rust Speex (CELP speech codec) decoder — plus functional
narrowband, wideband and ultra-wideband encoders — for the
[oxideav](https://github.com/OxideAV/oxideav) framework. Implemented
from *The Speex Codec Manual*, RFC 5574, and the clean-room codebook
material staged at
[`docs/audio/speex/`](../../docs/audio/speex/).

## Status

**Clean-room rebuild — reference-tracking decoder across all three
rate classes, all quality ladders encoding (r450).** The r450
crafted-bitstream probe campaign (the crate's own frame writers emit
streams with every transmitted field chosen directly; the reference
decoder is invoked as an opaque binary — `tests/fixtures/
hb-gain-probes/NOTES.md`) replaced every fitted high-band gain law
with **measured closed forms**, all instances of one architecture —
*spectral continuity at the band joins*: each layer's transmitted gain
codes the amplitude ratio of the two adjacent bands at their crossover
(4 kHz: `gc_recon·|A_hb(π)|·rms(e_lb)/|A_lb(π)|` for HB modes 2/3/4
and the analogous 5-bit law over the innovation-only excitation for
mode 1; 8 kHz: `0.664·fold[g5]·|A_l2(π)|/|A_hb1(0)|` over the
zero-stuffed first-band excitation for the UWB layer). The same
campaign measured the narrowband short-lag pitch conventions, the
mode-7 stage-2 weight (0.455), the forced-pitch float law with its
0.99 cap, and the reference's default output high-pass transfers.
All CI-gated, absolute — no fitted gain:

* **Narrowband** (`tests/nb_conformance_fixture.rs`): the ten
  black-box reference (`--no-enh`) fixtures — Table 9.2 sub-modes
  8/2/3/4/5/6 on a tone mix plus a pitch-gliding speech-like source —
  decode at **11.4–14.4 dB** raw and **25.6–33.3 dB / corr
  0.9986–0.9998 through the measured output high-pass** (the raw rows
  compare an un-high-passed decode against the reference's default
  high-passed output, so the hp rows are the like-for-like metric;
  r449 stood at 13.1–19.5 dB with a fitted 30 Hz filter).
* **Wideband** (`tests/wb_conformance_fixture.rs` + tone/mode-4
  gates): speech fixtures at qualities 4/6/8 decode at **20.5 / 20.1
  / 20.8 dB** full-signal (energy ratios 0.99; r449: 15.6/18.3/18.3),
  the q4 folded high band at **+16.5 dB / corr 0.989** (was −6.9 dB);
  the mode-1 tone oracle's folded band reaches **44.1 dB / corr
  1.00000**; the quality-10 (HB mode 4) oracle decodes with its low
  band at **25.6 dB / corr 0.9986** and its isolated 4–8 kHz sub-band
  at **20.7 dB / corr 0.998** at the parity-corrected alignment
  (r449: ≈6 dB band error at corr 0.44 — the r450 probes exposed a
  one-sample parity error that had inverted the r440 polarity pin).
* **Ultra-wideband** (`tests/uwb_*`): the 3-layer tone fixture at
  **22.6 dB / corr 0.9973** with the outer folded band at **46.8 dB /
  corr 1.00000** (the r403-era tone-vs-speech oracle conflict
  dissolved — the "independent" outer tones are the zero-stuffed fold
  source's spectral images); the 3-layer speech oracle per-band mean
  error **0.85 / 0.84 / 1.12 dB** (campaign A closed); the stacked
  quality-10 oracle at **20.2 dB** full-signal with per-band means
  **0.10 / 0.49 / 0.70 dB** and bit-exact framing.

What is implemented and tested:

* **Framework integration** (round r438) — `register()` installs real
  `oxideav-core` decoder + encoder factories under the codec id
  `speex` (claiming the Ogg payload magic `Speex   `), alongside the
  dual-API `make_decoder` / `make_encoder` free functions. The
  framework decoder drives `SpeexStreamDecoder` (header from
  `extradata`, from `sample_rate`, or from the in-band header packet
  with comment/extra-header skip) and emits interleaved-S16 audio
  frames; the framework encoder re-blocks S16 input into 20 ms frames,
  emits one self-contained packet per frame, honours a `quality`
  (0..=10) option, and carries the 80-byte stream header
  (`SpeexHeader::write_bytes`) in its output parameters.
* **Intensity stereo — true in-band decode and encode** (campaign B,
  `src/stereo.rs`, staged `intensity-stereo.md`). The 8-bit code-9
  payload (`[sign][5-bal][2-e_ratio]`) reconstructs an `(gL,gR)` gain
  pair (`b=exp(bal/8)`, `F=√(0.5/e_ratio)`,
  `e_ratio={0.25,0.315,0.397,0.5}`) with the §4 intra-frame
  interpolation (`a=0.980`), producing interleaved L/R from the single
  mono decode; the encoder emits the `(L+R)/2` downmix with the
  per-frame code-9 message prefixed and declares 2 channels. On the
  staged `stereo-nb-ladder-q4` oracle the interleaved SNR tracks the
  mono decode within ≈2.3 dB (18.4 vs 20.7 dB — the r450 narrowband
  fixes lifted the mono floor so far that the §4.1 block-phase
  approximation is now the visible term;
  `tests/intensity_stereo_fixture.rs`). The §4.1
  sub-frame block-phase offset is a `speexdec`-pipeline detail this
  decoder does not reproduce (bounds byte-exactness). The raw payload
  also stays available via `InbandRequest::IntensityStereo`.
* **All nine Table 9.1 narrowband modes decode and encode** (round
  r438, staged `nb-innovation-binding.md`): mode 1 (2.15 kbps vocoder,
  quality 0) carries no innovation codebook — its four 1-bit
  innovation-gain fields are read and discarded (inert in the
  reference decoder, §4 of the binding doc) and its excitation is the
  frame-level forced pitch path (r450: an **unbounded float
  centre-tap recursion**, coefficients `0.066667·quant` capped at
  0.99 — probe grids at `T = 50` recover the reference's taps to
  ≤ 0.04 %); mode 7 (24.6 kbps, quality 10) is two 48-bit innovation
  stages of eight 6-bit `sv5-64` lookups, **stage 2 at the measured
  0.455 weight** (`NB_MODE7_STAGE2_WEIGHT`, r450 — the unweighted sum
  ran every quality-10 low band ≈ +1.6 dB hot), stage 2 searched on
  stage 1's residual by the encoder at that weight. Narrowband, wideband and
  ultra-wideband qualities **0..=10 all encode** (r450 — the mode-4
  two-stage search + the measured gain laws unlocked the 44 kbit/s
  ladder tops).
* **High-band modes 2/3/4 — the exact crossover-anchored absolute
  gain law** (r450 crafted-bitstream probes, superseding the r440
  fitted `(gc·lb_rms)²` reading and the r446 backward-adaptive
  interpretation). Streams whose per-sub-frame gain index, innovation
  content, low-band level / envelope / innovation / pitch and
  high-band envelope were varied one at a time measure, with **no
  fitted constant**:

  ```text
  g = gc_recon · |A_hb(π)| · rms(e_lb) / |A_lb(π)|
  gc_recon = 0.87360 · gc_quant_bound[q]      (staged table + multiplier)
  ```

  — the transmitted 4-bit index codes the ratio of the two bands'
  spectral amplitudes at the 4 kHz QMF crossover (linear in the
  correction to 0.3 %, no innovation-energy term, no memory, one law
  for modes 2, 3 and 4; `hb_gc_crossover_gain`, wired per-sub-frame
  through `NarrowbandDecoder::last_crossover_response`). The mode-4
  polarity returns to direct: the r440 flip compensated a one-sample
  **parity** error in the sub-band gate's even-only delay sweep (an
  odd full-rate offset negates a QMF-recovered high band). The 80-bit
  two-stage binding (`hb-innovation-binding.md`) stands, its stage-2
  weight re-measured **exactly 0.4** by stage-isolation probes.
  Measured on the staged q10 oracle (`tests/hb_mode4_fixture.rs`):
  isolated 4–8 kHz sub-band **20.7 dB / corr 0.998** (r449: corr
  0.44), low band 25.6 dB / 0.9986, band-magnitude means 1.5 / 3.1 dB.
* **The provenance/08 measurement replicates through crate machinery —
  on BOTH mode-4 fixtures** (round r446). Two oracle-free gates re-run
  the doc's whole QMF-recovered-excitation route with the crate's own
  `QmfAnalysis` / `decode_hb_subframe_mode4_f32` /
  `reconstruct_hb_exc_gain` chain:
  - `tests/hb_mode4_recovered_gain_table.rs` (WB fixture) reproduces
    the staged 299-row `tables/hb-mode4-recovered-gain.csv` **row for
    row** — the filter-free `lb_frame_rms` column to < 3 × 10⁻⁵ log
    (the crate's bank *is* the staged instrument, window convention
    included), mean |ρ| **0.8839** at the uniquely-peaked −40
    alignment, and the doc's gain-direction regression **exactly**
    (fixed-2 R² 0.791 / rms 8.89 dB, correction-only 0.005).
  - `tests/hb_mode4_uwb_recovered_excitation.rs` (UWB q10 fixture —
    the replication provenance/08 itself asks for) recovers the inner
    4–8 kHz band through a **two-stage** split with a per-frame LPC
    fit and confirms the binding on a stream whose transmitted LSP
    pair **varies**: 298 sub-frames, mean |ρ| **0.9316**, positive on
    99.7 %, same −40 peak at a 3.8× margin.
  Local black-box check: an independent reference-decoder build
  re-decodes both mirrored fixtures to within ±1 / ±2 LSB of the
  staged `expected.pcm` (the notes' stated cross-platform bound).
* **UWB quality-10 (stacked mode-4) decodes and is gated** (r446
  framing + r450 laws, `tests/hb_mode4_uwb_fixture.rs` on the staged
  `hb-mode4-uwb-q10` oracle): **bit-exact framing across all 76
  frames** against the reference's own trace — including the packed
  80-bit mode-4 excitation fields verbatim — and, through the r450
  crossover-anchored laws in all three layers, full-signal **20.2 dB**
  (alignment corr 0.9967) with per-band mean errors **0.10 / 0.49 /
  0.70 dB** (r449: 1.29 / 5.55 / 7.98 dB).
* **Ogg/Speex stream-header parse** (`SpeexHeader`) — the `Speex   `
  magic plus all 13 little-endian fields and the narrowband / wideband
  / ultra-wideband mode cross-check (manual §7.3, RFC 5574 §3), plus
  the exact Table 7.1 serialiser `write_bytes` (r438).
* **Frame framing** — the per-frame 5-bit prefix
  (`NarrowbandFrameHeader`), the typed narrowband sub-mode table
  (modes 0..=8, manual Table 9.1), the §5.5 in-band signalling and
  custom-message bodies, and the wideband high-band sub-mode table
  (modes 0..=4, manual Table 10.1).
* **Packet iterator** (`PacketFrames`) — dispatches a Speex packet
  into narrowband, wideband, in-band-signalling, custom-in-band, and
  terminator frames end-to-end.
* **Bit primitives** — MSB-first `BitReader` / `BitWriter` with
  round-trip `write`/`parse` methods for the framing-level types.
* **Codebook tables** — the narrowband and high-band LSP-VQ,
  pitch-gain VQ, and innovation codebooks, plus the Q15 LPC analysis
  window / lag window / QMF analysis filter, embedded as typed
  pure-data accessors.
* **Reconstruction primitives** — narrowband and high-band LSP
  reconstruction and sub-frame interpolation, LSP→LPC conversion, the
  LPC synthesis filter, innovation sub-vector lookup, 3-tap pitch-gain
  reconstruction, the adaptive-codebook (long-term predictor)
  contribution sum with its excitation history buffer, and the
  **exact scalar gain reconstruction tables** for the narrowband
  frame-level OL excitation gain (32-level `ol_gain_table`, float law
  `exp(qe/3.5)`), the narrowband per-sub-frame innovation-gain
  correction (8-/2-level `g_subf`, composing the fixed-codebook gain
  `g_frame · g_subf`), and the high-band excitation gain (32-level
  5-bit folded gain + 16-level 4-bit gain-correction with its
  `0.87360` reconstruction multiplier).
* **Gain-scaled fixed-codebook contribution** — folds the reconstructed
  fixed-codebook gain `g = g_frame · g_subf` into the raw innovation
  sub-vector, producing the magnitude-correct `c[n]` that enters the
  §8.4 excitation composition `e[n] = p[n] + c[n]`
  (`gain_scaled_innovation_subframe` / `_from_indices` / `_sample`). A
  silent frame drives the gain to `0.0`, vanishing the contribution.
* **Gain-scaled adaptive-codebook (pitch) contribution** — divides the
  §9.2 long-term-predictor dot product by the now-staged **Q6**
  pitch-gain scaling (`GAIN_SCALING = 64`, `GAIN_SHIFT = 6`, from
  `provenance/02`), producing the pitch contribution `p[n]` as `[f32;
  40]` in the **same normalised float signal domain** as the
  gain-scaled `c[n]` (`gain_scaled_pitch_subframe` / `_sample`). Both
  contributions now share one domain, so the §8.4 sum `e[n] = p[n] +
  c[n]` is well-posed. Stream-start and silence-tap cases vanish to
  `0.0`.
* **Forced (open-loop) pitch gain** — narrowband modes 1 and 8 carry a
  frame-level 4-bit *forced* pitch-gain field (Table 9.1 `OL pitch gain`
  row) instead of the per-sub-frame 3-tap VQ. `forced_pitch_gain`
  reconstructs the coefficient from the `provenance/02` decode law
  `pitch_coef = 0.066667 · quant` (`quant ∈ 0..=15`, unit gain at 15) and
  applies it as a single Q6 centre tap `round(0.066667 · quant · 64)`, so
  the forced path reuses the same §9.2 convolution as the VQ path. The
  narrowband decoder now dispatches the two mutually-exclusive Table 9.1
  pitch-gain rows (per-sub-frame VQ for modes 2..=7, forced gain for
  modes 1/8, silence for mode 0) — modes 1 and 8 previously fell back to
  silence and produced no pitch contribution.
* **Float-domain excitation composition** — joins the two gain-scaled
  contributions into the final per-sub-frame excitation `e[n] = p[n] +
  c[n]` (§8.4 / companion §2.3). Because the gain-scaled pitch `p[n]` and
  innovation `c[n]` are now both `[f32; 40]` in the same normalised float
  signal domain, the composition is a plain elementwise `f32` sum
  (`gain_scaled_excitation_subframe` / `_sample`) — the
  magnitude-correct float analogue of the raw-integer `e[n]` sum. At
  stream start the pitch term vanishes so `e[n] = c[n]`; a silent frame
  drives both terms to `0.0`.
* **Open-loop / scalar gain quantiser** — the encode-direction inverse
  of the gain reconstruction tables. The `scal_quant` sorted-threshold
  search maps a target gain magnitude to its field index (count of
  decision boundaries met-or-exceeded, saturated to the field width):
  `quantise_frame_ol_exc_gain` (NB 5-bit OL gain),
  `quantise_subframe_gain_correction` (NB 1-/3-bit innovation-gain
  correction), and `quantise_hb_exc_gain` (HB 5-bit folded / 4-bit
  gain-correction, the latter dividing out the `0.87360` multiplier).
  Each returns the same typed index enum the parser produces, so it is
  the exact inverse of the matching reconstruction function at every
  cell.

* **Wideband high-band synthesis** — the complete high-band branch of
  the wideband (sub-band CELP) decode path. The high-band excitation is
  the gain-scaled innovation `e_hb[n] = g · c_hb[n]`
  (`gain_scaled_hb_innovation`) — per manual §10.2 there is **no pitch
  prediction in the high band**, so the §8.4 sum collapses to the
  innovation alone. That excitation runs through the order-8 high-band
  LPC synthesis filter `1/A_hb(z)` (`HbSynthesisFilter`, the high-band
  analogue of the narrowband `SynthesisFilter` at `HB_LPC_ORDER = 8`),
  and `synthesise_high_band_frame` assembles the per-sub-frame
  LSP→LPC + excitation + synthesis into the 160-sample high-band 8 kHz
  half-band signal `x_hb[n]` — the second of the two 8 kHz signals the
  QMF synthesis filterbank recombines into 16 kHz wideband PCM.
* **High-band per-sub-frame LSP interpolation** — the high-band
  synthesis reconstructs a **separate order-8 LPC set per sub-frame**
  from the §9.1 four-way linear interpolation of the previous and
  current frame's high-band LSP vectors (`HbSubFrameLsp`, the high-band
  analogue of the narrowband `NbSubFrameLsp`; Q10→Q12 exact integer),
  reconstructed through the base-aware `hb_subframe_lpc_set_with_base`.
  Spec basis: manual §10.1 states the high-band linear prediction is
  *"very similar to narrowband. The only difference is that we use only
  12 bits"* — the manual names exactly one difference, so the §9.1
  interpolation applies verbatim to the high band. The interpolating
  entry `synthesise_high_band_frame_interp` threads the previous frame's
  high-band LSP for cross-frame continuity; `WidebandDecoder` /
  `SpeexDecoder` carry that state (silence frames leave the previous
  envelope untouched). The stateless `synthesise_high_band_frame`
  delegates with the `prev = curr` first-frame convention, reproducing
  the prior frame-level-LPC behaviour exactly.
* **Ultra-wideband framing recursion** — the `UwbFrameLayout` /
  `SubBandLayer` descriptor captures the embedded, scalable bit-stream
  structure (manual §2.2 "Embedded wideband structure"): narrowband →
  wideband → ultra-wideband adds one high-band layer (one 1-bit
  wideband-flag recursion marker) per step, doubling the reconstructed
  sample rate (8 / 16 / 32 kHz).

* **Closed narrowband decode loop** (`NarrowbandDecoder`) — the full
  §8–§9 per-sub-frame recurrence with the **excitation feedback wired**:
  LSP reconstruct → interpolate → LSP→LPC, then per sub-frame
  `p[n]` = gain-scaled adaptive-codebook tap sum (reading the excitation
  history), `c[n]` = gain-scaled innovation, `e[n] = p[n] + c[n]` pushed
  back into the [`ExcitationBuffer`] (so the next sub-frame's pitch term
  is live), and `x[n] = 1/A(z)·e[n]`. Pitch period resolves from
  whichever Table 9.1 row the mode carries (per-sub-frame fine pitch
  modes 3..7, frame-level OL pitch modes 1/2/8); short lags follow the
  r450-measured conventions — the VQ path applies the §9.2
  substitution **once** with common-`T` folding
  (`gain_scaled_pitch_subframe_repeat`; probe tap-fit grids recover
  the staged tables exactly), the forced OL path recurses unbounded
  over its float centre tap (`gain_scaled_pitch_subframe_forced`) —
  and silence rings the IIR out on zero excitation. `decode_frame` / `decode_frame_i16` emit the
  full 160-sample frame with the live pitch path — earlier rounds fed the
  synthesis filter the innovation alone because the feedback was unwired.
* **QMF analysis bank pinned as the provenance/08 instrument** (round
  r440, `tests/qmf_band_isolation.rs`). Provenance/08 validates the
  staged 64-tap prototype as a measurement instrument (its 88–95 dB
  sub-band isolation is what makes fixture sub-bands recoverable from
  decoded PCM without an oracle); the crate's `QmfAnalysis` reproduces
  all six documented tone-isolation figures **to their 0.1 dB print
  precision** (−95.10 / −95.61 / −87.94 / +87.94 / +95.61 / +95.10 dB),
  gated at ±0.5 dB plus mirror-pair symmetry. The mode-4 fixture gains
  a QMF-route sub-band conformance gate built on it (true 8 kHz
  sub-band scoring + a pin on the ≈142-sample decoder delay against
  the source-length-trimmed reference).
* **§7.6 crossover-response parity closed form** (round r440,
  `hb-folded-gain.md` §7.6): `hb_crossover_response_from_lsp` is the
  pinned odd-parity product `|Â(π)| = Π 4·cos²(ωᵢ/2)` over the
  odd-indexed LSP class, verified equal to the direct polynomial
  evaluation on every swept envelope (≤0.008 log10);
  `hb_crossover_response_bwexp` + `HB_FOLD_ENVELOPE_COMPRESSION_GAMMA
  = 0.944` surface §7.6's compression model (inferred, not wired into
  the default fold scale). `tests/fold_envelope_sweep.rs` cross-checks
  the whole chain against the staged
  `hb-fold-envelope-vs-transmitted.csv` columns — reproduced to
  ≤0.007 log10 on all six settings.
* **QMF synthesis filterbank** (`QmfSynthesis`, round r365) — the final
  wideband recombination of the two reconstructed 8 kHz half-band signals
  (low band 0–4 kHz + high band 4–8 kHz folded) into a single 16 kHz
  wideband PCM stream, the §10 stage earlier rounds stopped short of. The
  reconstruction is the **classical two-band quadrature-mirror filterbank**
  (Croisier–Esteban–Galand) driven by the staged 64-tap prototype `h0`
  (`qmf-filter-h0-float.csv`, surfaced via the new `qmf_h0_float()`
  accessor) — a textbook multirate-DSP construction, the same clean-room
  category the staged LSP→LPC trace grants for the LSP polynomial step.
  Synthesis relations `g0 = 2·h0`, `g1 = -2·(-1)ⁿ·h0`, implemented in
  **polyphase** form and unit-pinned identical to the direct
  upsample-filter-sum reference. The prototype is normalised `Σh0 ≈ 1`
  (`Σh0_even = Σh0_odd ≈ 0.5`), so the factor-2 synthesis gain gives a
  **unity passband** — a constant low band reconstructs to the same
  constant. The §10.2 4–8 kHz → 4–0 kHz frequency fold is intrinsic to the
  `(-1)ⁿ` synthesis modulation. FIR band histories persist across frames
  for seamless streaming.
* **Wideband sub-band decode loop** (`WidebandDecoder`) — walks an
  embedded wideband packet (manual §10.4: narrowband frame packed first,
  then the high band) and returns both reconstructed 8 kHz half-band
  signals **plus the QMF-recombined 16 kHz wideband PCM**
  (`WidebandFrame { low_band, high_band, wideband_pcm }`), each band
  carrying its persistent IIR state — and the QMF its FIR history — across
  packets.
* **Top-level packet decoder** (`SpeexDecoder`) — drives the
  `PacketFrames` iterator through the per-frame decode loops, decoding a
  whole multi-frame Speex packet to a `Vec<DecodedFrame>` (narrowband
  PCM, wideband half-band pair **+ 16 kHz `wideband_pcm`**, or a control
  pseudo-frame). One decoder instance handles a stream mixing NB / WB
  frames with continuous state (a wideband frame's low band *is* an
  embedded narrowband frame, so the shared narrowband state stays
  continuous; RFC 5574 §3.1).
* **`i16` PCM convenience surface** (round r369) — a single
  round-to-nearest-and-clamp quantiser (`saturate_i16`) shared by every
  PCM-out path so the float reconstruction reduces identically wherever
  the same sample appears. `NarrowbandDecoder::decode_frame_i16` (8 kHz),
  `WidebandFrame::wideband_pcm_i16` / `WidebandDecoder::decode_packet_i16`
  (16 kHz), `DecodedFrame::pcm_i16` (band-correct full-rate, `None` for a
  control frame) with a matching `DecodedFrame::sample_rate_hz`, and
  `SpeexDecoder::decode_packet_pcm_i16` — the whole-packet flat `Vec<i16>`
  that concatenates the audio frames and drops the control pseudo-frames.
* **Header rate accessors** (round r369) — `SpeexHeader::is_narrowband` /
  `is_wideband` / `is_ultrawideband`, the canonical mode-class rate
  `mode_sampling_rate_hz` (NB 8 k / WB 16 k / UWB 32 k, §2.2 / §7.3) and
  `rate_matches_mode` (flags a header whose declared `rate` contradicts
  its mode class), plus `UwbFrameLayout::for_header_mode` linking the
  header's mode field to the embedded sub-band recursion descriptor.

* **WB mode-1 folded high band — exact crossover-anchored law over
  the innovation-only source** (r450, superseding the r393/r410
  fitted ceiling law). Crafted mode-1 streams sweeping the 5-bit gain
  grid, both envelopes and the low-band level one at a time measure

  ```text
  e_hb[n] = 0.8518 · fold_quant_bound[g5] · |A_hb(π)|/|A_lb(π)| · (−1)ⁿ · c_lb[n]
  ```

  — **linear over the whole 32-level staged table** (flat to 0.2 %;
  the old `min(C·|Â|, 1/(2√2))` ceiling was an artefact of fitting
  without the `1/|A_lb(π)|` term), and the fold **source is the
  innovation-only excitation** `c_lb` — a two-source fit on a
  pitch-plus-random-innovation stream reads pitch weight −0.0001
  (the r393/r403 composed-excitation source folded the pitch
  contribution too, up to ≈10× hot in energy on pitchy speech).
  `NarrowbandDecoder::last_frame_innovation` carries the source;
  the encoder's mode-1 gain selection mirrors the law. Measured:
  the tone oracle's folded band 38.9 → **44.1 dB / corr 1.00000**,
  the q4 speech high band −6.9 → **+16.5 dB / corr 0.989**
  (`crate::hb_fold`).
* **Absolute signal-domain calibration** (round r393,
  `INNOVATION_CODEBOOK_SCALE`). The same fixture calibrates the
  `signed char` innovation codebook rows as **Q5 fractions**
  (`c[n] = g·c_raw[n]/32`), landing decoded PCM at the reference's
  absolute level (fixture full-signal energy ratio 0.97, previously
  32× hot). Mirrored into the encoder gain selection so transmitted
  indices live in the reference quantiser range.
* **Opt-in output high-pass — measured transfer** (r450,
  `OutputHighpass`). The manual documents the codec's default-on
  output high-pass (`SPEEX_SET_HIGHPASS`) without its transfer; with
  the narrowband innovation path verified reference-exact to 0.1 %,
  the r450 cross-spectral measurement isolates it: 8 kHz — bilinear
  biquad `fc ≈ 80.7 Hz, Q ≈ 0.87` (5.7 % peaking near 150 Hz);
  16/32 kHz — third-order (biquad `fc ≈ 41.75 Hz, Q ≈ 1.38` ×
  first-order ≈ 33 Hz; the 32 kHz response matches 16 kHz in absolute
  Hz). Supersedes the r393 fitted 30 Hz Butterworth. Applying it is
  what turns the NB matrix's 11–14 dB raw rows into 25.6–33.3 dB
  like-for-like conformance.
* **On-wire layer-prefix grammar** (round r393, fixture-arbitrated) —
  every layer is introduced by the 1-bit wideband flag: `0` narrowband
  layer (a wideband frame's leading prefix starts `0`), `1` each
  high-band layer; the packet walker detects a high-band extension by
  the bit following the narrowband body (terminator / next prefix /
  padding all start `0`). The top-level `SpeexDecoder` now delegates
  its wideband assembly to `WidebandDecoder` (bit-identical paths,
  pinned via `SpeexStreamDecoder` on the fixture).
* **Fixture conformance gate** (`tests/wb_mode1_folded_fixture.rs`) —
  the 101-frame fixture decode is scored absolutely (no fitted gain)
  at the fixed 80-sample reference lead: r450 full-signal 16.7 dB SNR
  / 0.989 correlation with the folded high band at **44.1 dB /
  corr 1.00000**; CI floors 14 dB / 40 dB with pinned energy ratios.

The narrowband decode loop and the **full wideband decode-to-16 kHz-PCM
path** (NB low band + HB synthesis + QMF recombination) are wired
end-to-end and produce finite, input-responsive, deterministic PCM from a
real `speexenc` stream through the top-level `SpeexDecoder` —
externally validated against the reference decoder on the staged WB
mode-1 fixture.

* **In-band signalling — semantic interpretation** (round r372). The §5.5
  mode-14 in-band messages parse to a raw `(code, payload)` `InbandMessage`
  *and* now decode to a typed `InbandRequest` via `InbandMessage::interpret`
  (Table 5.1 "Content" column): perceptual-enhancement / mode-switch /
  rate-mode (the `CBR/VAD/DTX/VBR` bitmask decoded into independent flags
  by `RateModeConfig`) / acknowledge-policy (`AcknowledgePolicy`) /
  intensity-stereo balance / max-bitrate / packet-ack, with reserved-code
  passthrough. The top-level `SpeexDecoder` surfaces this end-to-end:
  `DecodedFrame::Control` carries a typed `ControlMessage`
  (`Inband { message, request }` / `Custom { size_bytes }`), so a consumer
  can act on a mode-switch / DTX-rate / stereo request without re-parsing
  the bit-stream.
* **Packet structural inspection** (round r372). `PacketFrame::kind` /
  `is_audio` / `is_control` / `mode_id` classify a frame by its mode-class
  (`FrameKind`); `PacketSummary::walk` walks a packet once (no audio
  decode) and tallies its per-kind frame counts — `audio_frames` /
  `control_frames` / `total_frames` / `is_wideband` — for a header-vs-payload
  cross-check against `SpeexHeader::frames_per_packet` (§7.3) and rate-class
  routing.
* **Encoder front-end — LPC analysis** (round r372). The first encoder
  stage (`lpc_analysis` module): window (the staged 200-sample asymmetric
  analysis window) → autocorrelate (`R(m) = Σ x[i]·x[i−m]`, order 10) →
  stabilise (`R(0) *= 1.0001` white-noise floor + the staged lag window) →
  Levinson-Durbin (`R·a = r` → order-10 LPC). Output `a[0..10]` uses the
  decoder's `A(z) = 1 − Σ aᵢ z⁻ⁱ` convention, so it is round-trippable
  against the existing synthesis path. Grounds the encode direction
  (manual §8.2 / §9.1) on textbook linear-prediction primitives + the
  staged window data. `lpc_analyse` / `apply_analysis_window` /
  `autocorrelate` / `stabilise_autocorrelation` / `levinson_durbin` /
  `LpcCoefficients`.
* **Encoder front-end — LPC→LSP conversion** (round r372). The encode
  inverse of the decoder's `lsp_to_lpc` (`lpc_to_lsp` module): form the
  auxiliary polynomials `P/Q`, deflate the `(1 ± z⁻¹)` boundary roots, and
  root-find their unit-circle roots (grid sign-change scan + bisection) to
  recover the order-10 LSP frequencies for quantisation (§9.1). Round-trip
  validated against `lsp_to_lpc` (`lsp_to_lpc(lpc_to_lsp(a)) ≈ a`), so the
  encoder envelope path analyse → LPC → LSP is closed end-to-end.
* **Encoder front-end — LSP-VQ quantiser** (round r372). The encode
  inverse of the r194 decoder LSP reconstruction (`lsp_quant` module):
  `quantise_lsp_q10` runs the 5-stage multi-stage-VQ greedy search (stage
  0 full vector → split low/high refinements) to pick the per-stage 6-bit
  indices best representing a Q10 LSP vector against the staged `nb_lsp_*`
  codebooks; `pack_lsp_index` packs them into the on-wire `lsp_index`
  field. Closes the LSP encode→pack→decode round-trip through the existing
  decoder path. So the envelope path is now analyse → LPC → LSP →
  quantise → pack.
* **Encoder front-end — full envelope chain wired** (round r372). The
  `radians_to_lsp_q10` bridge (encode inverse of `lsp_q10_to_radians`)
  joins the pieces so the complete short-term envelope encode path runs
  end-to-end: `signal → lpc_analyse → lpc_to_lsp → radians→Q10 → (−base) →
  quantise_lsp_q10 → pack_lsp_index → [wire] → from_packed →
  reconstruct_q10 → (+base) → lsp_to_lpc`, validated by
  `tests/encoder_envelope_chain.rs` (the reconstructed envelope faithfully
  matches the analysed one through both LSP regimes).
* **Encoder front-end — high-band LSP-VQ quantiser** (round r372). The
  wideband counterpart (`hb_lsp::quantise_q10` / `pack_hb_lsp_index`):
  inverts the r214 high-band 2-stage MSVQ reconstruction (stage 1 full
  8-coeff VQ, stage 2 residual) via the same sequential greedy search,
  packing into the on-wire 12-bit `lsp_index`.
* **Ultra-wideband (32 kHz) subsystem** (round r389). The §2.2 embedded
  recursion walked one level above wideband, both directions:
  - `UltraWidebandDecoder` — the three-layer UWB frame (embedded
    narrowband + first high band via the wideband decoder's new
    cursor-level entries, then a second Table 10.1 high-band layer)
    with the full §5.5 packet walk, recombined to 640-sample 32 kHz
    PCM by an **outer** QMF synthesis bank (the QMF filterbanks are
    now slice-generic — the same mirror structure serves both the
    320→2×160 inner and 640→2×320 outer geometries, pinned
    perfect-reconstructing at both).
  - `UltraWidebandEncoder` — outer QMF analysis split, embedded
    `WidebandEncoder` low half, and the second layer's 12-bit LSP MSVQ
    envelope + four 5-bit folded-gain fields; `encode_packet` /
    `encode_packet_quality`; full encode→decode round trips with
    per-quality packet sizes pinned to the staged rate tables.
  - **Quality→sub-mode ladders** (`quality`): NB from Table 9.2; WB
    derived arithmetically from Table 10.2 bit totals; UWB from
    RFC 5574 Table 2, whose constant +1,800 bit/s (= +36 bits/frame)
    over the WB column pins the conformant second sub-band layer to
    the gain-only **mode-1** frame at every quality.
  - `SpeexStreamDecoder` — header-mode-driven dispatch across the
    three rate classes (the UWB layer walk needs the §7.3 out-of-band
    rate-class context), flat `i16` PCM at 8/16/32 kHz.
  - **VAD/DTX** (`vad`): §2.1's pinned 5-bit mode-0 DTX frame (250 bps;
    9/13 bits for the WB/UWB all-mode-0 frames) behind
    `encode_packet_dtx` on all three encoders, driven by the
    `EnergyVad` RMS-threshold + hangover detector (the VAD decision
    algorithm is unpinned by the manual — documented encoder freedom).

* **Ultra-wideband second-layer fold — exact outer crossover law**
  (r450, superseding the r403 linear-interpolated source and its
  `1/16` flat constant). Crafted 3-layer streams measure

  ```text
  e_l2[n] = 0.664 · fold_quant_bound[g5] · |A_l2(π)|/|A_hb1(0)| · (−1)ⁿ · zerostuff(e_hb1)[n]
  ```

  — the source is the first high band's excitation **zero-stuffed**
  to the 16 kHz half-band rate, both spectral images kept (fit
  0.98–0.997 vs 0.61–0.67 for linear interpolation), and the law is
  spectral continuity at the **8 kHz join** (after the two QMF folds
  the second layer's `π` edge and the first band's `0` edge both land
  at 8 kHz). The r403-era tone-vs-speech oracle conflict dissolves:
  the tone fixture's "independent" outer tones are the zero-stuffed
  source's images. Measured: tone fixture **22.6 dB / 0.9973** full
  with the outer band at **46.8 dB / corr 1.00000**
  (`tests/uwb_fold_geometry_fixture.rs`); speech oracle per-band
  **0.85 / 0.84 / 1.12 dB** (`tests/uwb_speech_3layer_fixture.rs`);
  the q4 tracking gate 2 → **21.9 dB / 0.9968**
  (`tests/uwb_conformance_fixture.rs`). Fuzzed by
  `tests/uwb_robustness.rs`. Second-layer excitation-VQ modes are
  **rejected by the reference decoder itself** (r450 crafted-stream
  probe: Table-10.1-framed VQ-mode second layers error as corrupted
  where the identical mode-1 construction decodes; the RFC ladder
  never emits them), so this crate's typed `UwbLayerUndocumented`
  rejection is the conformant surface — that gap is closed as a
  negative result.

* **Encoders — narrowband, wideband and ultra-wideband, all
  qualities 0..=10** (rounds r372–r450). The full encode chains ship
  as composable modules: LPC analysis → LPC→LSP → multi-stage LSP-VQ
  (`lsp_quant` / `hb_lsp`), perceptual weighting + open-loop and
  analysis-by-synthesis pitch search, innovation VQ search (single-
  and two-stage, both bands), exact scalar gain quantisers, and the
  Table 9.1 / §10.4 frame writers (`parse(write(body)) == body`).
  r450 aligned every high-band gain selection with the measured
  decode laws (mode-1 fold target over the innovation-only source;
  modes 2/3/4 searched at the crossover-anchored base; mode-4
  two-stage search at the 0.4 stage weight; the UWB layer-2 target on
  the outer law) and the mode-7 stage-2 search at 0.455. Functional,
  not bit-exact: conformant-stream producers whose packets round-trip
  through the decoder at the staged Table-10.2 / RFC-Table-2 wire
  budgets, with `encode_packet_dtx` (§2.1 DTX) behind an RMS VAD.

## Not yet supported

* **Bit-exact decode.** The r450 laws close every *magnitude* law the
  probes could reach; the decode is reference-tracking (NB hp rows at
  corr 0.9986–0.9998), not bit-exact. What remains is float-rounding
  territory: the exact evaluation-order of the reference float build's
  arithmetic, the fixed-point build's `lsp_cos` table (recorded docs
  gap, gates only fixed-point interop), and the reference's exact QMF
  buffering (the r450 probes measure our decode one sub-frame *ahead*
  of the reference on both bands equally — a pure global-delay
  convention, alignment-absorbed by every gate).
* **The perceptual enhancer.** All conformance material is `--no-enh`;
  the default-on enhancer's transfer is unstaged and unmeasured — a
  future crafted-stream campaign (enh-on vs enh-off reference decodes
  of the same streams) could measure it the same way the output
  high-pass fell.
* **Mode-1 comfort-noise excitation.** The vocoder mode's
  noise-generation rule (what the OL excitation gain scales when the
  pitch history is cold) is not staged; this crate renders the forced
  pitch path over a zero innovation — exact in structure, noise-free.
  A PRNG's output cannot be recovered black-box; only a docs-side
  trace of the noise rule would close it.
* **Intensity stereo §4.1 block phase.** The interleaved decode tracks
  the (now 20.7 dB) mono decode within ≈2.3 dB; the sub-frame
  block-phase offset of the stereo reconstruction is the visible
  residual — a crafted-stereo-stream probe campaign is the natural
  next step.
* **Encoder reference-equivalence.** All three encoders now mirror the
  measured decode laws in their gain selection (including the mode-4
  two-stage search), but the reference's *search* strategies (joint
  perceptual pitch+innovation ordering, gain-target normalisation)
  remain undocumented — the encoders are conformant-stream producers,
  not bit-stream twins.
* **Sub-1 % constants.** The r450 constants are measured to
  0.2–2.5 %: `0.8518` (mode-1 fold), `0.664` (outer fold), the mode-7
  `0.455` stage weight, the forced-pitch `0.99` cap, and the two
  output-high-pass transfers. The modes-2/3/4 law needs no constant at
  all (its `0.87360` is the staged multiplier). The two deepest swept
  envelopes read 7–45 % off through this crate's `|A(π)|` — the same
  near-degenerate-envelope divergence `hb-folded-gain.md` §7.6
  records; whether that is this crate's LSP-margin convention or the
  reference's own clamping needs a bit-exact envelope trace.

## Usage

```toml
[dependencies]
oxideav-speex = "0.0"
```

Register into an `oxideav_core::RuntimeContext` via `register()`, build
codecs directly with `make_decoder` / `make_encoder`, or use the direct
types (`SpeexDecoder`, `SpeexStreamDecoder`, `NarrowbandEncoder`,
`WidebandEncoder`, `UltraWidebandEncoder`) without the registry.

## License

MIT — see [LICENSE](./LICENSE).
