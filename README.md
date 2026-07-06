# oxideav-speex

[![CI](https://github.com/OxideAV/oxideav-speex/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-speex/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-speex.svg)](https://crates.io/crates/oxideav-speex) [![docs.rs](https://docs.rs/oxideav-speex/badge.svg)](https://docs.rs/oxideav-speex) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure-Rust Speex (CELP speech codec) decoder — plus functional
narrowband and wideband encoders — for the
[oxideav](https://github.com/OxideAV/oxideav) framework. Implemented
from *The Speex Codec Manual*, RFC 5574, and the clean-room codebook
material staged at
[`docs/audio/speex/`](../../docs/audio/speex/).

## Status

**Clean-room rebuild in progress — externally validated decoder.** The
full decode path is not yet bit-exact, and the framework codec entry
points still return `Error::NotImplemented`; as of round r393 the
wideband decode is **externally validated against the reference
decoder** on the staged `wb-mode1-folded` fixture (16.7 dB absolute
SNR / 0.989 correlation full-signal, 38.9 dB / 0.99994 on the folded
high band, absolute level calibrated to 0.97× — CI-gated). What is
implemented and tested:

* **Ogg/Speex stream-header parse** (`SpeexHeader`) — the `Speex   `
  magic plus all 13 little-endian fields and the narrowband / wideband
  / ultra-wideband mode cross-check (manual §7.3, RFC 5574 §3).
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
  (round r393, `hb_fold`). The gain-only high-band sub-mode's
  reconstruction law is now pinned against the staged
  `docs/audio/speex/fixtures/wb-mode1-folded/` reference decode:
  `e_hb[n] = K·g·(−1)ⁿ·e_lb[n]`, where `e_lb` is the embedded
  narrowband frame's composed excitation
  (`NarrowbandDecoder::last_frame_excitation`), `(−1)ⁿ` is the
  sample-level spectral fold (manual §10.2's QMF axis reversal;
  candidate conventions without it score ≤ 0.31 high-band correlation
  against the reference vs **0.9999** for this law), `g` the staged
  32-level `fold_quant_bound` level and `K =
  HB_FOLD_RECONSTRUCTION_MULT` (adopted `1/(2·√2)`, inside the
  measured `0.3516…0.3549` window). Wired through
  `synthesise_high_band_frame_folded` into every wideband decode path.
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
  only fixed-point-build interop). The framework `Decoder` endpoints
  return `Error::NotImplemented` until reference-equivalence closes;
  the free-function `SpeexDecoder` / `SpeexStreamDecoder` /
  `NarrowbandDecoder` / `WidebandDecoder` decode paths are the public
  surface in the meantime.
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
  **Functional, not bit-exact**: the reference gain normalisation (the
  mapping between residual magnitude and the `exp(qe/3.5)` OL-gain domain)
  is part of the documented gain-Q-format gap, so this encoder chooses
  gains by direct magnitude matching. Modes 2/3/4/5/6/8 are supported;
  modes 1/7 (undocumented innovation) are rejected. Still missing for a
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
  decode-transparent. HB modes 0/1/2/3 supported; mode 4 stays the
  recorded innovation-binding docs gap. Functional, not bit-exact —
  same gain-normalisation posture as the narrowband encoder.
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
* **Ultra-wideband second-layer fold source — crate convention, not
  reference-pinned.** The second (8–16 kHz) layer now reconstructs a
  real half-band with the fixture-pinned fold law (r393): the fold
  *source* is the crate's **recursion-consistent generalisation** —
  the embedded wideband layer's two 8 kHz excitation tracks
  (narrowband composed excitation + first-high-band excitation)
  recombined to 16 kHz through the QMF synthesis bank — and the
  encoder chooses its 5-bit gains against that exact source via a
  local (analysis-by-synthesis) wideband decode, so UWB round-trips
  reconstruct the 8–16 kHz energy envelope. The staged fixture is
  wideband-only, so the reference's actual 16 kHz fold-source geometry
  (80-sample sub-frames; the provenance-02 "80-sample-subframe kludge"
  scalars confirm the reference treats this rate specially) remains
  the recorded residue of #170 pending an ultra-wideband fixture. Also
  still gapped: the sub-frame geometry of the excitation-VQ modes
  2..=4 at the 16 kHz half-band rate.
* Per-mode innovation handling for narrowband modes 1 and 7 and
  high-band mode 4, whose decomposition the staged inventory does not
  yet uniquely fix. (Mode 4 = 80 bits / 40-sample sub-frame: neither
  staged high-band codebook shape — `HbSv8_128` (8 samples, 8-bit
  composite) nor `HbSv10_32` (10 samples, 5-bit) — yields a split
  matching both the 80-bit budget *and* the 40-sample count, so the
  binding stays a recorded docs gap.)
* **Sub-1 % constants pending a bit-exact low band** — the r393 fixture
  arbitration (below) leaves two constants pinned only to ≈ ±1 %: the
  exact fold constant `K` (adopted `1/(2·√2)` inside the measured
  `0.3516…0.3549` window) and the exact-vs-adopted `1/32` innovation
  row scale (measured `0.03154`). Both windows collapse once the
  remaining low-band envelope deltas close. Two measured residuals
  remain unattributed: the reference's default **output high-pass**
  (manual §Codec-control `SPEEX_SET_HIGHPASS`, default on — transfer
  not staged; measured ≈ 1st/2nd-order, cutoff ≈ 30 Hz, +1.6 dB if
  fitted) and a frame-rate AM sideband difference around strong tones
  (≈ 1.25 kHz on the fixture).

## Usage

```toml
[dependencies]
oxideav-speex = "0.1"
```

Disable default features for the framing / parse surface without the
framework dependency.

## License

MIT — see [LICENSE](./LICENSE).
