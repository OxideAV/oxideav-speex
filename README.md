# oxideav-speex

[![CI](https://github.com/OxideAV/oxideav-speex/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-speex/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-speex.svg)](https://crates.io/crates/oxideav-speex) [![docs.rs](https://docs.rs/oxideav-speex/badge.svg)](https://docs.rs/oxideav-speex) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure-Rust Speex (CELP speech codec) decoder — plus functional
narrowband and wideband encoders — for the
[oxideav](https://github.com/OxideAV/oxideav) framework. Implemented
from *The Speex Codec Manual*, RFC 5574, and the clean-room codebook
material staged at
[`docs/audio/speex/`](../../docs/audio/speex/).

## Status

**Clean-room rebuild in progress — externally validated decoder,
framework-integrated (r438), intensity-stereo + HB-mode-4 decode
(campaign B).** The full decode path is not yet bit-exact; the framework
codec entry points are **wired**, every Table 9.1 narrowband mode
decodes and encodes, **intensity stereo** decodes and encodes (true
in-band L/R, `src/stereo.rs`), **high-band mode 4 / quality 10**
decodes, and the decoder is **externally validated against the
reference decoder** across all three rate classes (all CI-gated,
absolute — no fitted gain):

* **Narrowband** (round r410, `tests/nb_conformance_fixture.rs`):
  ten black-box reference-decoder (`--no-enh`) fixtures — Table 9.2 sub-modes
  8/2/3/4/5/6 on a tone mix plus a pitch-gliding speech-like source —
  decode at **10.5–14.4 dB** absolute SNR (energy ratio 0.95–1.08),
  **13.1–19.5 dB** through the fitted output high-pass; up from
  2.8–6.8 dB (half the reference energy) before the round's two
  pitch-path fixes (VQ column↔lag reversal + in-sub-frame recursion).
* **Wideband** (r393 tone gate + r410 speech matrix,
  `tests/wb_mode1_folded_fixture.rs` / `tests/wb_conformance_fixture.rs`):
  the tone fixture at 16.7 dB full-signal / **38.9 dB, corr 0.99994**
  on the folded high band; speech-like fixtures at qualities 4/6/8 at
  **15.6 / 18.3 / 18.3 dB** full-signal — the first reference
  comparison of the high-band excitation-VQ sub-modes 2/3, and the
  fixture set that arbitrated the r410 **crossover-shaped folded
  high-band law** (see `hb_fold`). Round r440 fixed the mode-3
  sign/index wire order to the measured binding (`[sign][7-bit index]`,
  leading sign bit — `hb-innovation-binding.md` §1/§2.2).
* **Ultra-wideband** (round r403, `tests/uwb_fold_geometry_fixture.rs`):
  the 3-layer tone fixture at 19.1 dB / 0.994 full-signal 32 kHz,
  embedded wideband layers 21.7 dB / 0.997, folded second layer
  correlation 0.93. On the staged **3-layer speech** oracle
  (`tests/uwb_speech_3layer_fixture.rs`, campaign A) the decode measures
  **16.33 dB** full-signal 32 kHz, and its framing is **bit-exact
  against the reference decoder's own per-frame trace** — all 126
  frames' NB / HB / UWB sub-modes, both high-band layers' 12-bit LSP
  MSVQ stage indices `(i1,i2)`, and all eight 5-bit folded-gain indices
  match, so the whole three-layer parse + index-extraction path is
  reference-exact. Per-band mean |error|: 0–4 kHz ≈ 1.1 dB (accurate),
  4–8 kHz ≈ 4.8 dB, 8–16 kHz ≈ 7.1 dB — the residual is in the folded
  high-band *reconstruction*, not the framing (see "Not yet supported").

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
  staged `stereo-nb-ladder-q4` oracle the interleaved SNR **tracks the
  mono decode within ~0.4 dB** (13.40 vs 13.83 dB) — the L/R law adds
  no material error (`tests/intensity_stereo_fixture.rs`). The §4.1
  sub-frame block-phase offset is a `speexdec`-pipeline detail this
  decoder does not reproduce (bounds byte-exactness). The raw payload
  also stays available via `InbandRequest::IntensityStereo`.
* **All nine Table 9.1 narrowband modes decode and encode** (round
  r438, staged `nb-innovation-binding.md`): mode 1 (2.15 kbps vocoder,
  quality 0) carries no innovation codebook — its four 1-bit
  innovation-gain fields are read and discarded (inert in the
  reference decoder, §4 of the binding doc) and its excitation is the
  frame-level forced pitch path, encoder-driven by a real open-loop
  pitch estimate + the staged `provenance/02` forced-gain law; mode 7
  (24.6 kbps, quality 10) is two independent 48-bit innovation stages
  of eight 6-bit `sv5-64` lookups, summed by the decoder, stage 2
  searched on stage 1's residual by the encoder. Narrowband qualities
  0..=10 and wideband/ultra-wideband qualities 0..=9 all encode; WB/UWB
  quality 10 *decodes* (see next) but encoding it stays declined.
* **High-band mode 4 (WB/UWB quality 10) decodes — state-derived gain
  base + pinned polarity** (campaign B + r440, staged
  `hb-innovation-binding.md` + `provenance/08`). The 80-bit two-stage
  innovation — 2 × 5 × 8-bit groups over the same five `sv8-128`
  8-sample slots, leading sign bit, stage 2 at weight 0.4
  (`decode_hb_subframe_mode4_f32`) — decodes with the r440 absolute
  gain `g = HB_GC_STATE_SCALE·(gc_recon·lb_frame_rms)²`: provenance/08
  measures that the transmitted 4-bit correction alone explains
  essentially nothing (R² = 0.005) and that the base tracks the **same
  frame's reconstructed low band**; the exponents take the doc's
  nearly-free fixed-2 reading and the scale is fixture-calibrated
  (documented fitted — the exact law is the recorded docs gap). The
  innovation **polarity** through this crate's QMF conventions is
  pinned (`HB_INNOVATION_POLARITY = −1`, the binding doc §4's one-bit
  trial against `expected.pcm`). Measured on the staged q10 oracle
  (`tests/hb_mode4_fixture.rs`): 4–8 kHz band mean |err| **13.1 →
  6.1 dB**, isolated high sub-band correlation **≈0 → +0.44**, energy
  ratio 0.086 → 0.74; low band 0–4 kHz ≈ 2.2 dB throughout. Encoding
  q10 stays declined (the encoder mode-4 search is unpinned).
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
  modes 3..7, frame-level OL pitch modes 1/2/8); silence rings the IIR
  out on zero excitation. `decode_frame` / `decode_frame_i16` emit the
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

* **WB mode-1 folded high-band excitation — externally arbitrated**
  (rounds r393 + r410, `hb_fold`). The gain-only high-band sub-mode's
  reconstruction law is pinned against the staged reference decodes:
  `e_hb[n] = min(C·|A_hb(π)|, K)·g·(−1)ⁿ·e_lb[n]`, where `e_lb` is the
  embedded narrowband frame's composed excitation
  (`NarrowbandDecoder::last_frame_excitation`), `(−1)ⁿ` is the
  sample-level spectral fold (manual §10.2's QMF axis reversal;
  candidate conventions without it score ≤ 0.31 high-band correlation
  against the reference vs **0.9999** for this law), `g` the staged
  32-level `fold_quant_bound` level, and the scale is the r410
  **crossover-shaped** factor: proportional (slope `C = 0.17`,
  oracle-measured 0.171…0.189) to the high-band envelope's magnitude
  response at the 4 kHz QMF crossover, saturating at the r393 flat
  constant `K = 1/(2·√2)` where both real-stream anchors sit. The flat
  law overshot speech troughs by up to 130× per frame (`wb_q4` fixture
  −12.9 dB → 15.6 dB with the shaping). Wired through
  `synthesise_high_band_frame_folded` into every wideband decode path
  and mirrored in the WB encoder's mode-1 gain selection.
* **Absolute signal-domain calibration** (round r393,
  `INNOVATION_CODEBOOK_SCALE`). The same fixture calibrates the
  `signed char` innovation codebook rows as **Q5 fractions**
  (`c[n] = g·c_raw[n]/32`), landing decoded PCM at the reference's
  absolute level (fixture full-signal energy ratio 0.97, previously
  32× hot). Mirrored into the encoder gain selection so transmitted
  indices live in the reference quantiser range.
* **Opt-in output high-pass** (round r393, `OutputHighpass`) — the
  manual documents the codec's default-on output high-pass
  (`SPEEX_SET_HIGHPASS`) without its transfer; the fixture's
  behavioural trace fits a 2nd-order Butterworth at 30 Hz (flat
  optimum — documented as fitted, not reference-pinned). Raises the
  fixture match from 16.7 dB to 18.3 dB when applied.
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
  at the fixed 80-sample reference lead: measured r393 full-signal
  16.7 dB SNR / 0.989 correlation, folded high band **38.9 dB / 0.99994**;
  CI floors 14 dB / 30 dB with pinned energy ratios.

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

* **Ultra-wideband 3-layer decode — externally validated** (round
  r403, `docs/audio/speex/fixtures/uwb-fold-geometry/`). The staged
  32 kHz 3-layer fixture (101 frames, NB mode 8 + both high-band layers
  sub-mode 1) pins the **second-layer (8–16 kHz) fold-source geometry**
  that a wideband-only stream cannot show. Black-box arbitration
  against the reference `--no-enh` decode (the r393 method) pins the
  fold source as the embedded wideband layer's **first-high-band
  excitation** (`WidebandDecoder::last_hb_excitation`),
  **linear-interpolated** to the 16 kHz second-layer geometry
  (`upsample_hb_excitation_linear`) and re-folded by the same `(−1)ⁿ`
  law scaled by the outer `UWB_FOLD_RECONSTRUCTION_MULT = 1/16`
  (`folded_uwb_excitation_slice`). This **replaces** the earlier
  QMF-recombined-excitation generalisation, which over-scaled the band
  25× and decorrelated it. Measured delta on the fixture: second-layer
  correlation **0.04 → 0.93**, high-band energy ratio **25× → 0.95**,
  full 32 kHz decode **≈0 dB / 0.65 → 19.1 dB / 0.994** absolute SNR;
  the embedded wideband layers (outer-QMF low half) measure **21.6 dB /
  0.997** — the first end-to-end external validation of the UWB path's
  first two layers. The encoder mirrors the pinned source/scale in its
  local analysis-by-synthesis gain selection. CI-gated by
  `tests/uwb_fold_geometry_fixture.rs` (full-signal, per-band,
  output-high-pass, header-path, determinism, framing) and fuzzed by
  `tests/uwb_robustness.rs`.

## Not yet supported

* Bit-exact full decode. The scalar excitation-gain quantiser levels are
  exact, the fixed-codebook gain `g = g_frame·g_subf` is folded into the
  innovation, the Q6 pitch scaling lands `p[n]` in the same float signal
  domain, and the composed excitation `e[n] = p[n] + c[n]` is now
  **threaded end-to-end** through the closed `NarrowbandDecoder` loop
  (the float → `i16` excitation-buffer feedback for the next sub-frame's
  pitch lookup is wired). The **LSP base vector + Q-format** is now
  **pinned** (round r347, `lsp_base` module): the documented linear-init
  base vector `LSP_LINEAR(i) = .25·i + .25` rad (NB) / `LSP_LINEAR_HIGH(i)
  = .3125·i + .75` rad (HB) — recorded as numeric facts in
  `docs/audio/speex/provenance/02-speex-gain-quant.md` — is added to the
  r194 codebook-delta reconstruction, so every well-formed frame's LSP
  angles land **strictly inside the conformant `(0, π)` band by
  construction** (base `0.25 … 2.5` rad NB / `0.75 … 2.94` rad HB plus a
  small signed codebook delta), no longer relying on the radian-clamp
  fallback. The narrowband decoder loop and the wideband `hb_lpc`
  accessor reconstruct LPC through the bounded base-aware path
  (`subframe_lpc_set_with_base` / `lpc_from_hb_lsp_delta_q10`), which
  also apply the pinned `LSP_MARGIN` minimum-spacing safeguard (`.002`
  rad NB / `.05` rad HB, `enforce_lsp_margin_radians`) so the LSP set
  stays strictly interlaced and the resulting filter is always stable.
  The Q10-radian base vector is exact — both the `LSP_PI = 25736`
  Q15-domain path and the `M_PI` float path pin the same integers
  (cross-checked in `lsp_base` tests).

  The **LSP angular interpretation is now a provenance-confirmed fact**
  (round r359, `lsp_pi_domain` module), no longer a documented assumption:
  the codec stores LSP frequencies in an angular domain where the staged
  constant `LSP_PI = 25736`
  (`docs/audio/speex/provenance/02-speex-gain-quant.md`) measures `π`, so
  `ω = v_storage · π / LSP_PI` radians exactly. The new module pins
  `LSP_PI`, the staged Q15-storage `LSP_LINEAR(i) = (i+1)·2048` /
  `LSP_LINEAR_HIGH(i) = i·2560 + 6144` base vectors, and the
  storage↔radian / storage→Q10 conversions, and **cross-checks** that the
  `LSP_PI`-domain conversion of the storage base vector lands on the same
  Q10-radian base vector `lsp_base` derived independently from the float
  `LSP_LINEAR` form (two staged numeric facts pinning one angle). The
  internal `lsp_q*_to_radians` path's `ω = value / 2^Q rad` is therefore
  the same angle re-expressed at `2^10` LSB/rad, confirmed rather than
  assumed.

  What remains for *reference-equivalence* (the bit-exactness half) is
  the exact **cosine-series fixed-point evaluation order** the
  fixed-point reference build uses for `cos(ω)` inside the LSP→LPC
  conversion (`lsp_cos` lookup table + interpolation, not staged) —
  recorded docs gap. **r393 external-validation finding:** the staged
  fixture's reference decoder is the default *float* build, whose
  LSP→LPC path evaluates `cos` in floating point — this crate's float
  path is therefore directly comparable, and the fixture measures the
  whole decode at 16.7 dB absolute SNR (0.989 correlation; high band
  38.9 dB) with the remaining low-band deltas attributed to the
  unpinned output high-pass transfer and a frame-rate AM sideband
  difference around strong tones, **not** to `lsp_cos` (which gates
  only fixed-point-build interop). Both API layers ship: the framework
  `Decoder` / `Encoder` factories (r438) and the free-function
  `SpeexDecoder` / `SpeexStreamDecoder` / `NarrowbandDecoder` /
  `WidebandDecoder` decode paths.
* **Narrowband encoder — end-to-end (functional).** Round r382 drove the
  encoder from the r372 envelope chain to a full narrowband encode
  (`NarrowbandEncoder`): LPC analysis → multi-stage LSP-VQ → per-sub-frame
  residual `A(z)·input` → open/closed-loop pitch search + 3-tap gain VQ →
  innovation (fixed-codebook) sub-vector search → frame OL exc gain +
  per-sub-frame correction → Table 9.1 frame packing, with the
  reconstructed excitation `e = p + g·c` pushed into the live pitch
  history. `encode_frame` emits a decodable narrowband frame;
  `encode_frame_body` exposes the quantised indices. The pieces landed as
  composable modules: `weighting` (perceptual `W(z) = A(z/γ1)/A(z/γ2)`,
  §8.5), `ol_pitch` (§9.2 normalised-correlation open-loop pitch),
  `abs_search` (analysis-by-synthesis weighted-domain pitch search),
  `innovation_search` (§9.2 sub-vector VQ), and `nb_encode` (the exact
  Table 9.1 writer, `parse(write(body)) == body`). An
  `encoder_nb_roundtrip` integration test drives encode → wire → parse →
  `NarrowbandDecoder` and confirms finite PCM + input-energy tracking.
  **Functional, not bit-exact**: the OL-gain *quantiser law* itself is
  now exact (r440, `quantise_frame_ol_exc_gain_exact` — the staged
  `qe = floor(0.5 + 3.5·ln g)` float-build expression from
  provenance/02), but the reference's normalisation (which residual
  magnitude feeds that law — the analysis window / energy definition)
  remains undocumented, so this encoder chooses its gain target by
  direct magnitude matching and refines closed-loop. All nine Table 9.1 modes are
  supported (r438: mode 1 via the forced-pitch vocoder path, mode 7 via
  the two-stage innovation search). Still missing for a
  *reference-equivalent* encoder: the exact perceptual-domain joint
  pitch+innovation search ordering and the exact gain normalisation.
* **Wideband (sub-band CELP) encoder — end-to-end (functional).** Round
  r385 mirrored the §10 wideband decode path in the encode direction
  (`WidebandEncoder`): the QMF **analysis** filterbank (`QmfAnalysis`,
  streaming two-band split of the 320-sample 16 kHz frame, pinned
  against the r365 synthesis bank's perfect-reconstruction property) →
  embedded narrowband encode of the low band (shared NB state) →
  high-band envelope (order-8 LPC analysis `analyse_hb` + `hb_lpc_to_lsp`
  → Q10 − pinned HB base → 2-stage MSVQ → 12-bit `lsp_index`, with the
  quantised LSPs driving per-sub-frame LPC through the §9.1
  interpolation exactly as the decoder does) → per-sub-frame high-band
  excitation with **closed-loop gain selection** (the 4/5-bit gain grid
  is searched exhaustively; each level's greedy innovation search —
  both codebook shapes, including the `HbSv8_128` polarity sign — is
  scored by its *decoded* error; the gain-only mode 1 transmits the
  r393 **fold-consistent** target `g = rms(residual)/(K·rms(e_lb))`
  against the embedded encoder's local excitation) → the §10.4 embedded
  packing
  (`hb_encode`, `parse(write(body)) == body` for every documented HB
  mode). Packet-level entry points (`encode_packet` on both encoders,
  closing with the §5.5 mode-15 terminator) round-trip through the
  top-level `SpeexDecoder`, with packetisation pinned
  decode-transparent. HB modes 0/1/2/3 encode; mode 4 now **decodes**
  (campaign B) but is not encoded (the mode-4 codebook search + the
  absolute HB-innovation gain law are unpinned). Functional, not
  bit-exact — same gain-normalisation posture as the narrowband encoder.
* **Folded high-band reconciliation — measured, default unchanged**
  (campaign B). The newly-staged `fold-envelope-sweep` material lets the
  crate's own `|Â(π)|` be checked against the reference decoder's
  measured per-band scale ratios (`tests/fold_envelope_sweep.rs`): it
  matches to ~0.5 dB in the shallow/mid envelope range, confirming the
  kneeless `s=C·|Â(π)|` law of `hb-folded-gain.md` §7.3/§7.5, and
  diverges only at the near-degenerate deep envelopes the doc itself
  flags. But adopting the kneeless law (removing the crate's ceiling)
  **regresses** the `wb-mode1-folded` tone oracle (best 32.6 vs 38.9 dB
  flat), and the §7.4 synthesized-WB-HB-signal outer source regresses
  **both** the tone oracle (19→6 dB) and the speech 8–16 kHz band-mean —
  because the crate's `|Â(π)|` is the reference's normalising response
  only in the shallow/mid range, not at the high `|Â(π)|` the anchor
  fixtures operate at. The default decode path is therefore unchanged
  (the ceiling law is the best the crate realises). **Precise remaining
  gaps:** the high-`|Â(π)|` crossover-response normalisation, and the
  outer fold's exact image weighting (`hb-folded-gain.md` §7.5 residual
  2).
* **High-band innovation-mode gain law (modes 2/3/4)** — provenance/08
  measured the mode-4 base as state-derived from the same frame's low
  band, and the r440 fixture-calibrated squared law drops the 4–8 kHz
  band error to ~6.1 dB (from ~13 dB doc-faithful) with the isolated
  sub-band correlating at +0.44. What remains open: the **exact** gain
  law (provenance/08 deliberately asserts none — R² = 0.80 at 8.7 dB
  rms is a direction, not a formula; its ask is a fixture pair
  separating the low-band-level and high-band-content drivers, or a
  second LSP setting, or the `hb-mode4-uwb-q10` trace re-emitted in
  the WB format), and the **modes 2/3 base**: the mode-4-calibrated
  law measurably regresses the staged `wb_q6` oracle when applied to
  mode 2 (it is level-inhomogeneous and single-fixture-scaled), so
  modes 2/3 keep the correction-only gain until a sub-band
  measurement covers them — a precise docs ask. The WB/UWB
  intensity-stereo ladders' mono decode is bounded by the same
  residual.
* **Bit-exact QMF delay convention** — the QMF synthesis filterbank now
  **lands** (`QmfSynthesis`, round r365): the two half-bands recombine
  into 16 kHz wideband PCM via the textbook two-band quadrature-mirror
  reconstruction from the staged prototype `h0` (see the "QMF synthesis
  filterbank" entry above). What is *not* pinned by the staged manual is
  the exact polyphase **delay / phase** convention the reference decoder
  uses (the inter-band group-delay alignment and the absolute output
  offset) — these are bit-exactness details, not sample-correctness ones;
  the sample-correct textbook reconstruction is what ships. Recorded docs
  gap, isolated to the delay convention.
* **Ultra-wideband 3-layer speech fold — outer-layer reconstruction
  divergence (campaign A).** The framing is bit-exact against the
  staged speech oracle's per-frame trace (above), so the residual is
  purely in the folded high-band *reconstruction*: 4–8 kHz ≈ 4.8 dB and
  8–16 kHz ≈ 7.1 dB mean band |error|, concentrated on high-formant
  frames. `docs/audio/speex/hb-folded-gain.md` §7.4 pins the outer
  layer's fold source as the wideband layer's **synthesized** high-band
  signal (post HB synthesis filter) 2×-upsampled with the zero-stuff
  image pair, scaled by the same kneeless `C·|Â_uwb(π)|` law as the WB
  layer — and on the frames whose fold source is itself accurate that
  law drives the 8–16 kHz band from ≈7 dB to **<1 dB**. It is not
  adopted because it **conflicts with the `uwb-fold-geometry` tone
  oracle**: that fixture's outer band carries independent 10/13.5 kHz
  source tones a gain-only mode-1 fold cannot reproduce, and the
  synthesized-signal source over-amplifies its resonant envelope
  (outer-band 6 → −7 dB, full 19 → 6 dB). No single scale reconciles
  the two staged oracles. **Docs gap:** the exact outer source
  normalisation across flat vs resonant transmitted envelopes (a
  reference behavioural trace of a *speech* mode-1 UWB stream sweeping
  the transmitted high-band envelope, which §7.3's synthetic sweeps did
  not cover). The current default keeps the r403 excitation-source /
  `UWB_FOLD_RECONSTRUCTION_MULT = 1/16` law (tone-fixture-green,
  speech outer ≈7 dB low), fixture-calibrated to ≈±5 %.
* **Ultra-wideband excitation-VQ modes (2..=4) — sub-frame geometry
  gap.** The second-layer fold *source* is externally pinned (r403);
  what remains gapped for the ultra-wideband band is the sub-frame
  geometry of the excitation-VQ modes 2..=4 at the 16 kHz half-band
  rate (Table 10.1's VQ budgets are stated for the 8 kHz half-band),
  which the decoder surfaces as `UwbLayerUndocumented` rather than
  guessing.
* **Wideband / ultra-wideband quality-10 (44 kbit/s) encoding.** The
  mode-4 *decode* binding is pinned (staged `hb-innovation-binding.md`,
  re-confirmed from staged bytes by provenance/08) and the r440
  state-derived gain base decodes it to ~6 dB band error — but the
  encoder side stays declined: the reference's mode-4 codebook search
  and the exact absolute gain law (needed to *choose* conformant gain
  corrections) are unpinned. Gated on the same provenance/08
  discriminating-fixture ask as the decode law.
* **Mode-1 comfort-noise excitation.** The binding doc pins that mode 1
  transmits no innovation and that its excitation is set by the
  frame-level fields, but the decoder-side noise-generation rule the
  reference uses for the vocoder mode (what the OL excitation gain
  scales when the pitch history is cold) is not staged — this crate's
  mode-1 decode renders the forced pitch path over a zero innovation,
  which is exact in structure but noise-free. Recorded docs gap
  (behavioural trace of a quality-0 stream would close it).
* **Sub-1 % constants pending a bit-exact low band** — the fixture
  arbitrations leave constants pinned only to a few %: the fold
  ceiling `K` (adopted `1/(2·√2)` inside the measured `0.3516…0.3549`
  window), the crossover slope `C = 0.17` (oracle-measured
  `0.171…0.189`), and the exact-vs-adopted `1/32` innovation row scale
  (measured `0.03154`). Residuals remaining unattributed: the exact
  shape of the fold law's transition between the oracle-probed linear
  region (`|A_hb(π)| ≤ 1.4`) and the fixture-pinned saturated region
  (`≥ 2.4`); a per-frame folded-band factor of up to ≈6× on some
  speech frames (`wb_q4`'s high band scores −6.9 dB at the right
  energy); the reference's default **output high-pass** (manual
  §Codec-control `SPEEX_SET_HIGHPASS`, default on — transfer not
  staged; measured ≈ 1st/2nd-order, cutoff ≈ 30 Hz; +3…5 dB across the
  NB matrix when fitted) and a frame-rate AM sideband difference
  around strong tones (≈ 1.25 kHz on the tone fixture). The
  in-sub-frame pitch recursion's fine-pitch reads sit within ±0.4 dB
  of the manual's repeat rule on the staged fixtures — the exact
  reference behaviour there needs a behavioural trace (docs ask).

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
