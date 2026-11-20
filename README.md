# oxideav-speex

[![CI](https://github.com/OxideAV/oxideav-speex/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-speex/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-speex.svg)](https://crates.io/crates/oxideav-speex) [![docs.rs](https://docs.rs/oxideav-speex/badge.svg)](https://docs.rs/oxideav-speex) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure-Rust Speex (CELP speech codec) decoder and encoder — narrowband,
wideband and ultra-wideband, constant / variable / average bit-rate,
VAD and DTX, intensity stereo — for the
[oxideav](https://github.com/OxideAV/oxideav) framework. Implemented
from *The Speex Codec Manual*, RFC 5574, and the clean-room codebook
material staged at
[`docs/audio/speex/`](../../docs/audio/speex/).

## Status

**Reference-exact float decode (±1 LSB) in every rate class, with the
perceptual enhancer; working encoders across qualities 0..=10 (r483).**
The decoder reproduces the reference float decoder's PCM within ±1 LSB
on every staged fixture — 93–99 % of samples identical, the rest one
LSB off, aligned sample for sample (`tests/decode_exactness.rs`):

| fixture | mode | identical | within ±1 |
| --- | --- | --: | --: |
| `nb-enhancer-q4` | NB submode 3 | 96.3 % | 100 % |
| same, enhancer on | NB 3 + comb | 96.5 % | 100 % |
| `nb-mode1-vocoder-q0` | NB 1 (vocoder) | 98.2 % | 100 % |
| `nb-mode1-dtx-q0` | NB 1 + null frames | 96.4 % | 100 % |
| `hb-mode4-wb-q10` | WB NB 7 + HB 4 | 93.6 % | 100 % |
| `hb-mode2-wb-q6` / `hb-mode3-wb-q8` | WB HB 2 / 3 | 95.3 / 95.2 % | 100 % |
| `wb-mode1-folded` | WB HB 1 (fold) | 74.6 % | 99.99 % |
| `hb-mode4-uwb-q10` | UWB q10 | 91.4 % | 99.7 % |
| `uwb-speech-3layer` | UWB q1 | 84.2 % | 97.6 % |
| `stereo-*` (4 fixtures) | intensity stereo, NB/WB/UWB | 97–98.5 % L/R | ±2 (= mono ±1 × gain) |

The float reference itself is reproducible only to ±2 LSB across hosts,
so this is as exact as a float decode gets. The pieces that got it
there (r483), each measured against the reference: the **transposed**
synthesis form, the float excitation memory, the float path's decimal
LSP stage scalings and polynomial cosine, the **output high-pass**
(measured biquads, on by default), the one-sub-frame **synthesis lag**
in every layer, the exact vocoder / comfort-noise generator, mode 7's
`1/2.2` stage weight, and the backwards one-pole intensity-stereo
smoother. The narrowband conformance matrix (tone + pitch-glide speech,
qualities 1–9) sits at 77–82 dB, ≥ 98.4 % within ±1 LSB.

The **perceptual enhancer** (tracer §2 pitch comb) is on by default, as
in the reference, and switchable (`set_enhancer`, the framework option
`enhancer`, in-band code 0). All 152 enhancer blocks of the staged trace
reproduce; the enhanced decode is within ±1 LSB everywhere.

The **encoder** covers the bit-stream the decoder accepts: every
narrowband sub-mode with every high-band sub-mode, qualities 0..=10 in
all three rate classes, variable and average bit-rate, voice-activity
detection with comfort noise, discontinuous transmission, intensity
stereo, in-band messages and a 1..=10 complexity setting. Its searches
are analysis-by-synthesis through a mirror of the decoder, in the
manual's noise-weighted domain; a black-box reference decoder turns
its streams into the same PCM as this crate's own decoder to a few LSB.
At the default complexity, in the mean over qualities 1..=10 on
speech-like material, it leads a black-box encoder at the same quality
on SNR, segmental SNR and noise-weighted SNR at 8 kHz and is level with
it at 16 kHz; at 32 kHz segmental and weighted SNR are level and
whole-signal SNR trails — see [Encoder](#encoder) for the measurements.

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
  `e_ratio={0.25,0.315,0.397,0.5}`) through the reference's gain
  smoother — `g ← 0.98·g + 0.02·g_target` run **backwards** over each
  frame (the in-place mono→stereo expansion order, which is the §4
  "weight on the current gains largest at the block's first sample"),
  state carried across frames from a neutral 1.0, each sample
  **truncated** to 16 bits (r483: a crafted 128-payload grid through
  the black-box reference confirms the steady law to 5 × 10⁻⁵ once
  the truncation is accounted for). The four staged two-channel decodes
  reproduce to 97–98.5 % identical samples, ±2 LSB (the mono decode's
  ±1 times gains up to 2). The encoder emits the `(L+R)/2` downmix with
  the per-frame code-9 message prefixed and declares 2 channels. The
  raw payload also stays available via `InbandRequest::IntensityStereo`.
* **All nine Table 9.1 narrowband modes decode and encode** (round
  r438, staged `nb-innovation-binding.md`): mode 1 (2.15 kbps vocoder,
  quality 0) carries no innovation codebook — its four 1-bit
  innovation-gain fields are read and discarded (inert in the
  reference decoder, §4 of the binding doc) and its excitation is the
  tracer §4 vocoder (r483): the seed-1000 noise generator times the
  frame gain mixed with a pulse train at the open-loop period, the
  decoded adaptive contribution discarded; mode 8's forced gain is an
  **unbounded float centre-tap recursion** (coefficients
  `0.066667·quant` capped at 0.99); mode 7 (24.6 kbps, quality 10) is
  two 48-bit innovation stages of eight 6-bit `sv5-64` lookups,
  **stage 2 at weight `1/2.2`** (`NB_MODE7_STAGE2_WEIGHT`, r483 — the
  r450 probe's 0.455 cost 5 dB of exactness), stage 2 searched on
  stage 1's remainder by the encoder at that weight. Narrowband, wideband and
  ultra-wideband qualities **0..=10 all encode** (r450 — the mode-4
  two-stage search + the measured gain laws unlocked the 44 kbit/s
  ladder tops).
* **High-band modes 2/3/4 — the exact absolute gain law**
  (`tracer-2026-09.md` §1, r480; supersedes the r450 probe reading
  `gc·|A_hb(π)|·rms(e_lb)/|A_lb(π)|`, which lacked the two
  regularisers and drifted up to 22 dB on quiet sub-frames):

  ```text
  S   = 0.87360·bound[q] · e_L / R          (stage 2 of mode 4 at 0.4·S)
  e_L = sqrt(0.1 + mean e_nb²)              (same sub-frame's NB excitation)
  R   = (A_nb(−1)+0.01) / (A_hb(−1)+0.01)   (interpolated quantised LPC)
  ```

  — no gain memory, one law for modes 2, 3 and 4
  (`hb_gc_crossover_gain` / `hb_gain_lb_base`). The 80-bit two-stage
  mode-4 binding (`hb-innovation-binding.md`: 2 × 5 × [sign + 7-bit
  `sv8-128` index], stage 2 at exactly 0.4) decodes and encodes; the
  encoder's 4-bit field is the reference rule
  `min{q : R·(1+e_H)/(1+e_L) ≤ bound[q]}` (`hb_encoder_gain_index`, §1.3).
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
  80-bit mode-4 excitation fields verbatim — and, through the exact
  laws in all three layers, full-signal **25.0 dB** with per-band mean
  errors **0.09 / 0.30 / 0.45 dB** (r450: 20.2 dB, 0.10 / 0.49 /
  0.70 dB; r449: 1.29 / 5.55 / 7.98 dB).
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

* **Closed narrowband decode loop** (`NarrowbandDecoder`; rebuilt r483
  on the reference float structure, see Status) — the full
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

* **Folded (mode-1) layers — exact laws** (`tracer-2026-09.md` §5–§6,
  r480). WB: `x = 0.9·exp((q−10)/8)/R·(−1)ⁿ·c_lb` over the narrowband
  fixed-codebook innovation (`NarrowbandDecoder::last_frame_innovation`).
  UWB: `x = 0.7·exp((q−10)/8)·(A_l2(−1)+0.01)/(A_hb(+1)+0.01)·src` with
  `src` the inner high band's excitation zero-stuffed to 16 kHz. The
  staged `fold_quant_bound` table is exactly `1.0645·exp((q−10)/8)` —
  it is the encoder's boundary set; the r450 constants 0.8518 / 0.664
  were `0.9 / 1.0645` and `0.7 / 1.0645` read through it. Both encoders
  invert the same laws in the log domain.
* **Absolute signal-domain calibration** (round r393,
  `INNOVATION_CODEBOOK_SCALE`). The same fixture calibrates the
  `signed char` innovation codebook rows as **Q5 fractions**
  (`c[n] = g·c_raw[n]/32`), landing decoded PCM at the reference's
  absolute level (fixture full-signal energy ratio 0.97, previously
  32× hot). Mirrored into the encoder gain selection so transmitted
  indices live in the reference quantiser range.
* **Output high-pass — built in** (r483). The manual documents the
  codec's default-on output high-pass without its transfer. With the
  rest of the path exact, output-error identification of crafted
  constant-envelope streams against the reference pins it: at 8 kHz a
  biquad with poles at 66.6 Hz (`NB_OUTPUT_HIGHPASS`); a wideband or
  ultra-wideband stream instead runs a different biquad (poles at
  38.2 Hz, `WB_LOW_BAND_OUTPUT_HIGHPASS`) on its **narrowband layer
  before the QMF**, and nothing at 16 / 32 kHz. The decoders apply it
  by default. The earlier fitted post filter (`OutputHighpass`, r450)
  remains for API compatibility but must not be applied on top.
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
  at the fixed 80-sample reference lead: full-signal 16.7 dB SNR
  / 0.989 correlation with the folded high band at **57.4 dB /
  corr 1.00000** (r480 exact fold law; r450: 44.1 dB); CI floors
  14 dB / 40 dB with pinned energy ratios.

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

* **Ultra-wideband second-layer fold — exact outer law** (r480,
  `tracer-2026-09.md` §5–§6; the r450 probe reading
  `0.664·fold_quant_bound[q]·|A_l2(π)|/|A_hb1(0)|` was the same law
  through the boundary table without the regulariser):

  ```text
  e_l2[n] = 0.7 · exp((q−10)/8) · (A_l2(−1)+0.01)/(A_hb1(+1)+0.01) · zerostuff(e_hb1)[n]
  ```

  — the source is the first high band's excitation **zero-stuffed**
  to the 16 kHz half-band rate, both spectral images kept (fit
  0.98–0.997 vs 0.61–0.67 for linear interpolation), and the law is
  spectral continuity at the **8 kHz join** (after the two QMF folds
  the second layer's `π` edge and the first band's `0` edge both land
  at 8 kHz). The r403-era tone-vs-speech oracle conflict dissolves:
  the tone fixture's "independent" outer tones are the zero-stuffed
  source's images. Measured: tone fixture **22.6 dB / 0.9973** full
  with the outer band at **73.4 dB / corr 1.00000** (r450: 46.8 dB)
  (`tests/uwb_fold_geometry_fixture.rs`); speech oracle per-band
  **0.85 / 0.75 / 1.02 dB** (`tests/uwb_speech_3layer_fixture.rs`);
  the q4 tracking gate 2 → **21.9 dB / 0.9968**
  (`tests/uwb_conformance_fixture.rs`). Fuzzed by
  `tests/uwb_robustness.rs`. Second-layer excitation-VQ modes are
  **rejected by the reference decoder itself** (r450 crafted-stream
  probe: Table-10.1-framed VQ-mode second layers error as corrupted
  where the identical mode-1 construction decodes; the RFC ladder
  never emits them), so this crate's typed `UwbLayerUndocumented`
  rejection is the conformant surface — that gap is closed as a
  negative result.

* **Encoders — narrowband, wideband and ultra-wideband** (rounds
  r372–r491). The encode chains ship as composable modules: LPC
  analysis → LPC→LSP → multi-stage LSP-VQ (`lsp_quant` / `hb_lsp`),
  open-loop and analysis-by-synthesis pitch search, innovation VQ
  search (single- and two-stage, both bands), exact scalar gain
  quantisers, and the Table 9.1 / §10.4 frame writers
  (`parse(write(body)) == body`). r480 fixed the LPC analysis sign
  convention, the mode-7 frame-gain estimate and the mode-1 vocoder
  gain, and moved the high-band gain decisions onto the Tracer round's
  rules; r491 added the noise-weighted search, the probe-centred frame
  gain, the rate modes and the complexity ladder described under
  [Encoder](#encoder).

## Encoder

`SpeexStreamEncoder` (and the framework encoder built on it) takes
20 ms frames at 8, 16 or 32 kHz and writes one packet per frame.

| feature | API | what goes on the wire |
| --- | --- | --- |
| constant quality 0..=10 | `new(rate, q)`, `set_quality`, `set_bitrate` | the Table 9.2 / 10.2 / RFC 5574 Table 2 sub-mode ladder |
| explicit sub-modes | `set_submodes(Some((nb, hb)))` | any narrowband mode 1..=8 with any high-band mode 0..=4 (1..=4 at 32 kHz; the 8–16 kHz layer is always the gain-only mode 1) |
| variable bit-rate | `set_vbr(Some(q))`, `set_vbr_max_bitrate` | each frame picks its sub-modes around the float quality `q`: vowels keep it, onsets go above, unvoiced and quiet frames below, background becomes comfort noise |
| average bit-rate | `set_abr(Some(bps))` | VBR with its quality steered by the accumulated bit surplus |
| voice-activity detection | `set_vad(true)` | non-speech frames as comfort noise (narrowband mode 1) |
| discontinuous transmission | `set_dtx(true)` | null frames (5 / 9 / 13 bits) while the background holds still |
| intensity stereo | `new_stereo(rate, q)` | the `(L+R)/2` fold plus a code-9 message per frame |
| in-band messages | `queue_inband`, `queue_user_data` | Table 5.1 requests (mode 14) and application payloads (mode 13) at the head of the next packet |
| complexity 1..=10 | `set_complexity` | nothing — the same bit-rate, deeper searches |
| search domain | `set_noise_weighting` | nothing — noise-weighted (default) or plain waveform error |

The manual defines what VBR, ABR, VAD and DTX are and leaves every
decision rule to the encoder; the rules used here are written out in
`src/rate_control.rs`. Packets end with the RFC 5574 §3.3 padding, so
a frame takes exactly `ceil(bits / 8)` octets.

### How it searches

Per frame: LPC analysis → LSP quantisation (error weighted by formant
proximity) → per sub-frame, against the decoder's own synthesis filter
and state, through the noise-weighting filter `W(z) = A(z/0.9)/A(z/0.6)`
(manual §8.5): the adaptive codebook (every lag pre-scored by its
centre tap, the best few searched over the whole 3-tap gain codebook),
then the innovation sub-vectors (a few partial paths kept per
sub-vector), then the sub-frame gain correction. The 5-bit frame gain
is centred by a **probe pass** that codes the frame with unquantised
innovation gains and picks the level whose correction grid quantises
them best; its neighbours and the open-loop estimate are then coded
for real and the lowest noise-weighted error wins. The high band picks
each sub-frame's gain field and shape together the same way. Vocoder
and null frames — whose excitation the decoder generates — are decoded
on the spot and the encoder continues from the decoder's state, so the
sub-mode may change on any frame.

### Encoder quality

`cargo run --release --example encoder_report -- --ref` reproduces
these tables. Each figure is encode → decode against the input, with
the enhancer off; the 8 kHz input is taken through the decoder's own
output high-pass first, the one thing no conforming decode can undo.
*Before* is the encoder of release 0.0.12, *after* this one at its
default complexity, and the last column a black-box encoder at the
same quality and its own default complexity, decoded by its black-box
decoder (its input high-pass switched off, as this encoder has none —
left on, it costs that encoder up to 12 dB on these waveform measures
at 8 kHz, which is what earlier figures in this README compared
against).
Quality 0 is the noise-excited vocoder mode; waveform measures say
little about it.

**8 kHz, speech-like source** (SNR / segmental SNR / noise-weighted segmental SNR, dB)

| q | kbit/s | before | after | black-box encoder |
| --: | --: | --: | --: | --: |
| 0 | 2.40 | -1.0 / -1.2 / -1.6 | -1.0 / -1.1 / -1.4 | -1.6 / -1.2 / -1.8 |
| 1 | 4.00 | 6.1 / 6.4 / 3.5 | 7.0 / 7.5 / 5.2 | 7.0 / 7.3 / 5.0 |
| 2 | 6.00 | 8.9 / 9.8 / 5.9 | 10.0 / 11.8 / 8.2 | 9.7 / 11.0 / 7.8 |
| 3 | 8.00 | 11.9 / 13.5 / 8.7 | 13.1 / 15.9 / 12.3 | 12.4 / 14.2 / 11.1 |
| 4 | 8.00 | 11.9 / 13.5 / 8.7 | 13.1 / 15.9 / 12.3 | 12.4 / 14.2 / 11.1 |
| 5 | 11.20 | 14.2 / 15.9 / 10.7 | 15.5 / 18.7 / 14.7 | 15.3 / 18.2 / 14.4 |
| 6 | 11.20 | 14.2 / 15.9 / 10.7 | 15.5 / 18.7 / 14.7 | 15.3 / 18.2 / 14.4 |
| 7 | 15.20 | 16.1 / 18.3 / 13.0 | 17.5 / 21.7 / 17.8 | 17.6 / 21.6 / 17.5 |
| 8 | 15.20 | 16.1 / 18.3 / 13.0 | 17.5 / 21.7 / 17.8 | 17.6 / 21.6 / 17.5 |
| 9 | 18.40 | 17.5 / 19.7 / 14.3 | 19.4 / 23.3 / 19.2 | 19.7 / 24.3 / 19.9 |
| 10 | 24.80 | 19.9 / 21.9 / 15.8 | 22.1 / 25.6 / 21.3 | 22.7 / 27.1 / 22.5 |

**16 kHz, speech-like source** (SNR / segmental SNR / noise-weighted segmental SNR, dB)

| q | kbit/s | before | after | black-box encoder |
| --: | --: | --: | --: | --: |
| 0 | 4.00 | -0.5 / -0.4 / -0.8 | -1.0 / -0.9 / -1.2 | -1.1 / -1.0 / -1.5 |
| 1 | 6.00 | 5.0 / 6.3 / 4.1 | 5.5 / 7.2 / 5.5 | 5.8 / 7.7 / 5.5 |
| 2 | 8.00 | 6.5 / 9.6 / 6.5 | 6.7 / 10.9 / 8.8 | 6.8 / 10.3 / 8.3 |
| 3 | 10.00 | 7.0 / 11.1 / 8.2 | 7.0 / 11.9 / 10.2 | 7.1 / 11.1 / 9.2 |
| 4 | 12.80 | 7.0 / 12.1 / 9.2 | 7.1 / 12.7 / 11.4 | 7.5 / 12.6 / 11.1 |
| 5 | 16.80 | 7.1 / 12.7 / 10.3 | 7.3 / 13.1 / 12.2 | 7.7 / 13.2 / 12.1 |
| 6 | 20.80 | 11.0 / 13.5 / 11.2 | 11.2 / 13.9 / 13.0 | 11.2 / 13.9 / 13.0 |
| 7 | 24.00 | 11.2 / 13.7 / 11.6 | 11.3 / 14.1 / 13.3 | 11.4 / 14.2 / 13.6 |
| 8 | 28.00 | 12.7 / 14.0 / 11.9 | 12.9 / 14.5 / 13.7 | 13.0 / 14.6 / 14.0 |
| 9 | 34.40 | 12.8 / 14.4 / 12.2 | 13.1 / 14.7 / 14.1 | 13.1 / 14.8 / 14.3 |
| 10 | 42.40 | 14.4 / 14.8 / 12.6 | 14.8 / 15.1 / 14.5 | 14.8 / 15.2 / 14.7 |

**32 kHz, speech-like source** (SNR / segmental SNR / noise-weighted segmental SNR, dB)

| q | kbit/s | before | after | black-box encoder |
| --: | --: | --: | --: | --: |
| 0 | 6.00 | -0.5 / -0.6 / -0.8 | -0.9 / -1.2 / -1.4 | -0.6 / -0.6 / -1.1 |
| 1 | 7.60 | 4.8 / 6.3 / 4.3 | 4.9 / 6.8 / 5.2 | 6.0 / 7.5 / 5.1 |
| 2 | 9.60 | 5.9 / 9.6 / 6.6 | 6.2 / 10.9 / 8.5 | 7.4 / 10.2 / 7.8 |
| 3 | 11.60 | 6.2 / 11.0 / 7.6 | 6.3 / 12.0 / 9.5 | 7.7 / 11.1 / 8.7 |
| 4 | 14.80 | 6.3 / 12.1 / 8.5 | 6.5 / 12.6 / 10.3 | 8.0 / 12.8 / 10.6 |
| 5 | 18.80 | 6.4 / 12.4 / 9.2 | 6.4 / 13.0 / 11.0 | 8.1 / 13.3 / 11.7 |
| 6 | 22.40 | 6.8 / 12.6 / 9.8 | 6.8 / 13.1 / 12.1 | 7.8 / 13.4 / 12.0 |
| 7 | 25.60 | 6.8 / 12.8 / 10.2 | 6.8 / 13.3 / 12.4 | 7.9 / 13.6 / 12.8 |
| 8 | 29.60 | 6.9 / 12.9 / 10.2 | 7.0 / 13.3 / 12.4 | 8.1 / 13.7 / 12.8 |
| 9 | 36.00 | 7.0 / 13.1 / 10.3 | 7.0 / 13.5 / 12.7 | 8.1 / 13.7 / 13.1 |
| 10 | 44.00 | 7.1 / 13.2 / 10.4 | 7.0 / 13.5 / 12.7 | 8.2 / 13.8 / 13.2 |

**Mean over qualities 1..=10, every source**

| source | before | after | black-box encoder |
| --- | --: | --: | --: |
| 8 kHz speech-like (6 s) | 13.70 / 15.32 / 10.43 | 15.07 / 18.08 / 14.38 | 14.96 / 17.78 / 14.12 |
| 8 kHz pitch-glide fixture | 17.34 / 18.72 / 12.21 | 19.68 / 22.16 / 17.77 | 16.51 / 20.12 / 17.28 |
| 16 kHz speech-like (6 s) | 9.47 / 12.22 / 9.79 | 9.69 / 12.80 / 11.68 | 9.84 / 12.77 / 11.57 |
| 16 kHz pitch-glide fixture | 11.68 / 13.53 / 10.91 | 12.80 / 14.54 / 12.51 | 10.88 / 13.28 / 11.65 |
| 16 kHz staged `hb-mode4-wb-q10` source | 1.61 / 5.70 / 3.93 | 1.84 / 5.95 / 4.86 | 2.13 / 6.09 / 4.93 |
| 32 kHz speech-like (6 s) | 6.40 / 11.60 / 8.71 | 6.50 / 12.20 / 10.67 | 7.72 / 12.30 / 10.77 |
| 32 kHz pitch-glide fixture | 6.95 / 7.38 / 5.76 | 7.48 / 7.89 / 6.32 | 6.63 / 7.30 / 6.01 |
| 32 kHz staged `hb-mode4-uwb-q10` source | -1.45 / 4.63 / 3.46 | -1.42 / 4.78 / 3.97 | -0.15 / 5.28 / 4.72 |

**Encode speed, × real time** (one core, median of three runs, speech-like source; before → after)

| q | 8 kHz | 16 kHz | 32 kHz |
| --: | --: | --: | --: |
| 0 | 272 → 510 | 180 → 320 | 118 → 220 |
| 1 | 36 → 134 | 35 → 118 | 31 → 97 |
| 2 | 62 → 139 | 57 → 119 | 48 → 96 |
| 3 | 20 → 75 | 20 → 71 | 19 → 62 |
| 4 | 20 → 74 | 10 → 40 | 9 → 37 |
| 5 | 9 → 41 | 14 → 41 | 14 → 38 |
| 6 | 10 → 41 | 14 → 36 | 13 → 35 |
| 7 | 14 → 40 | 5 → 17 | 5 → 17 |
| 8 | 14 → 41 | 5 → 16 | 5 → 15 |
| 9 | 6 → 18 | 8 → 22 | 8 → 21 |
| 10 | 9 → 27 | 8 → 20 | 7 → 19 |

Decode speed, × real time (median over qualities; the decoder is unchanged, the spread is the shared machine): 8 kHz 11412 → 13632, 16 kHz 2698 → 3126, 32 kHz 1073 → 1249.

**Complexity** (speech-like source, mean over qualities 1..=10: SNR / segmental / weighted, and median encode speed)

| complexity | 8 kHz | × | 16 kHz | × | 32 kHz | × |
| --: | --: | --: | --: | --: | --: | --: |
| 1 | 13.04 / 14.48 / 11.46 | 172 | 9.46 / 12.14 / 10.77 | 105 | 6.44 / 11.60 / 9.89 | 92 |
| 2 | 14.58 / 17.19 / 13.62 | 92 | 9.59 / 12.66 / 11.43 | 71 | 6.50 / 12.05 / 10.42 | 63 |
| 3 (default) | 15.07 / 18.08 / 14.38 | 41 | 9.69 / 12.80 / 11.68 | 38 | 6.50 / 12.20 / 10.67 | 36 |
| 5 | 15.32 / 18.55 / 14.87 | 23 | 9.73 / 12.88 / 11.83 | 21 | 6.50 / 12.28 / 10.89 | 19 |
| 8 | 15.54 / 18.96 / 15.26 | 9 | 9.77 / 12.94 / 11.95 | 8 | 6.53 / 12.34 / 10.96 | 8 |
| 10 | 15.62 / 19.14 / 15.42 | 5 | 9.76 / 12.96 / 12.01 | 4 | 6.52 / 12.35 / 11.00 | 4 |

At 8 kHz the default leads the black-box encoder on all three measures
in the mean and through quality 6, is level with it at 7 and 8, and
trails at 9 and 10 (0.3–0.6 dB SNR, 1.0–1.5 dB segmental) — complexity
8 closes most of that. At 16 kHz the two are level. At 32 kHz
segmental and weighted SNR are level within 0.1 dB while whole-signal
SNR trails by 1.2 dB on the speech-like source: the 8–16 kHz layer is gain-only, this encoder sets its gain to
match the band's level, and a level-matched but uncorrelated band
costs more waveform SNR than a quieter one would. On the two staged
`hb-mode4` sources (wide-band test material with a strong high band)
the same effect leaves the black-box encoder ahead in the mean.

Interoperability, measured by `tests/encoder_blackbox.rs` and the
report's cross-decode: a black-box reference decoder decodes every
quality in every rate class, every 16 kHz sub-mode pair, VBR, ABR, VAD,
DTX, stereo and in-band-message streams without a diagnostic, and its
PCM equals this crate's own decode within ±33 LSB on everything
measured (±8 on the speech-like source) — so the encoder's decoder
mirror is the decoder. A second black-box decoder decodes all of them
without a diagnostic.

**Rate modes** (8 s of speech-like material with pauses, quality 8
where one is set; coded bits only)

| mode | 8 kHz kbit/s | 16 kHz kbit/s | 32 kHz kbit/s |
| --- | --: | --: | --: |
| constant quality 8 | 15.00 | 27.80 | 29.60 |
| + VAD | 13.88 | 25.89 | 27.51 |
| + VAD + DTX | 13.75 | 25.69 | 27.18 |
| VBR 4 | 6.85 | 10.76 | 12.63 |
| VBR 6 | 9.38 | 16.91 | 18.74 |
| VBR 8 | 12.78 | 23.53 | 25.42 |
| VBR 10, 12 kbit/s frame ceiling | 10.23 | 9.33 | 11.09 |
| ABR 8 kbit/s | 7.89 | 8.07 | 8.09 |
| ABR 12 kbit/s | 12.31 | 11.86 | 12.12 |
| ABR 20 kbit/s | 19.61 | 19.99 | 20.30 |

VBR 8 codes this material's speech at a segmental SNR within 0.5 dB of
constant quality 8 for 15 % fewer bits.

### Open questions

Measured on this crate's streams; each is a point where the staged
material stops.

* **Application payloads (mode 13).** `queue_user_data` writes the
  layout manual §5.5 states (a 5-bit size in bytes, then the bytes),
  and this crate's decoder reads it. A black-box reference decoder
  does not skip such a message correctly — the frames behind it are
  lost — so the layout it expects differs from the manual's sentence.
  Table 5.1 in-band messages are unaffected.
* **A silent first high band under a coded second layer** (32 kHz,
  high-band mode 0 with a narrowband frame): a black-box reference
  decoder outputs silence for such frames, this crate's decoder does
  not. `set_submodes` refuses the combination; DTX's all-null frames
  are fine.
* **Excitation memory after null frames.** After a run of null frames
  the two decoders' following frames differ by up to a few hundred LSB
  for about a second, then converge: the adaptive-codebook
  memory a null frame leaves behind is not pinned sample-exactly.
* **Overload.** Where the decoded signal exceeds 16 bits the two
  decoders saturate at different points of the chain and diverge
  around the clipped peaks.
* **Analysis look-ahead.** The staged 200-sample analysis window peaks
  at its sample 120; laid over the coded frame plus the 40 samples
  after it, that is the manual's window "centered on the fourth
  sub-frame" and its 30 ms delay. This encoder lays it over the 40
  samples before the frame and the frame, i.e. without look-ahead.
  Moving it measured +0.15…0.3 dB SNR on the speech-like source and
  −0.2 dB segmental on the pitch-glide fixture, for 5 ms more delay;
  it is not adopted.

## Not yet supported

* **The fixed-point decode path.** The staged material covers the
  fixed-point LSP cosine and LSP→LPC (tracer §3), the high-band gain
  schedule (§1.2), the enhancer's fixed-point differences (§2) and the
  noise generator (§4); the narrowband excitation, pitch, gain, synthesis,
  QMF and high-pass arithmetic of the fixed-point build are not staged,
  so the host-independent `expected-fixed.pcm` outputs are not
  reproduced (the float decode is within a few LSB of them).
* **Short pitch lags T ∈ {17, 18, 19}**: the sub-frame tail past `2T`
  carries unpinned reference content (`nb-pitch-short-lag.md` §2.4).
* **Encoder reference-equivalence.** The searches and the rate
  decisions are this crate's own (see [Encoder](#encoder)); streams
  are conformant, not bit-stream twins of the reference encoder's.
* **Second-layer excitation VQ at 32 kHz.** The 8–16 kHz layer is the
  gain-only mode 1 of the RFC 5574 rate table; the excitation-VQ
  columns there are rejected by the reference decoder itself.
* **Several frames per packet** from `SpeexStreamEncoder` (the
  per-band encoders' `encode_packet` / `encode_frame_into` do it).

## Usage

```toml
[dependencies]
oxideav-speex = "0.0"
```

Three entry points; everything else the crate exports is the lower
layer they are built from.

| you have | use |
| --- | --- |
| raw 20 ms PCM frames, want packets (8 / 16 / 32 kHz, mono or stereo) | `SpeexStreamEncoder` |
| Speex packets + an Ogg/Speex header, or a known rate | `SpeexStreamDecoder` |
| the `oxideav-core` `Decoder` / `Encoder` traits | `register(ctx)`, or `make_decoder` / `make_encoder` |

### Direct API (no registry)

One packet per 20 ms frame: 160 / 320 / 640 samples per channel at
8 / 16 / 32 kHz. Quality is 0..=10 in every rate class.

```rust
use oxideav_speex::{SpeexHeader, SpeexStreamDecoder, SpeexStreamEncoder};

let mut enc = SpeexStreamEncoder::new(16_000, 8)?;     // wideband, quality 8
let header: [u8; 80] = enc.header().write_bytes();     // Ogg packet 0 / extradata

let mut dec = SpeexStreamDecoder::for_header(&SpeexHeader::parse(&header)?)?;
// No header on the wire (RTP, RFC 5574)? SpeexStreamDecoder::for_sample_rate(16_000)?

let frame = vec![0i16; enc.frame_samples()];          // 320 samples = 20 ms
let packet: Vec<u8> = enc.encode_frame(&frame)?;
let pcm: Vec<i16> = dec.decode_packet_pcm_i16(&packet)?;
```

Stereo is Speex's in-band intensity stereo:
`SpeexStreamEncoder::new_stereo(rate, quality)` takes interleaved L/R
frames (`2 × frame_samples()` samples) and
`SpeexStreamDecoder::decode_packet_stereo_i16` returns interleaved L/R.
`SpeexStreamDecoder::set_enhancer(false)` turns the perceptual enhancer
off (on by default).

Rate modes and complexity are setters on the same encoder; take the
header after configuring them (it records the nominal bit-rate and the
VBR flag):

```rust
let mut enc = SpeexStreamEncoder::new(16_000, 8)?;
enc.set_vbr(Some(7.5));            // variable bit-rate around quality 7.5
// enc.set_abr(Some(18_000));      // …or an 18 kbit/s average
enc.set_dtx(true);                 // null frames over a steady background
enc.set_complexity(5);             // 1..=10, default 3
let header = enc.header().write_bytes();
let packet = enc.encode_frame(&frame)?;
let spent = enc.last_frame_bits(); // and enc.last_submodes()
```

### Framework API

```rust
use oxideav_core::{CodecId, CodecParameters, RuntimeContext, SampleFormat};

let mut ctx = RuntimeContext::new();
oxideav_speex::register(&mut ctx);

let mut params = CodecParameters::audio(CodecId::new("speex"));
params.sample_rate = Some(32_000);
params.channels = Some(1);                  // 2 = intensity stereo
params.sample_format = Some(SampleFormat::S16);
params.options.insert("quality", "10");
let mut enc = ctx.codecs.first_encoder(&params)?;
let mut dec = ctx.codecs.first_decoder(enc.output_params())?;
```

The encoder takes interleaved S16 frames of any length, re-blocks them
into 20 ms packets and publishes the 80-byte stream header as
`output_params().extradata`. The decoder takes that header as
`extradata`, a bare `sample_rate`, or in-band (the Ogg header + comment
packets are consumed), and emits interleaved S16, one audio frame per
packet. Encoder options: `quality` (0..=10, default 8), `complexity`
(1..=10, default 3), `bitrate` (highest quality at or below it), `vbr`,
`vbr_quality`, `abr`, `vbr_max_bitrate`, `vad`, `dtx`,
`noise_weighting` (default on). Decoder option: `enhancer` (default
on). The codec claims the Ogg payload magic
`Speex   `, so an `oxideav-ogg` demuxer routes `.spx` streams to it.

### Examples

```sh
cargo run --example roundtrip -- 32000 10 1   # RATE QUALITY CHANNELS, direct API
cargo run --example registry                  # Encoder/Decoder traits via the registry
cargo run --release --example encoder_report -- --ref   # quality / speed tables
```

## License

MIT — see [LICENSE](./LICENSE).
