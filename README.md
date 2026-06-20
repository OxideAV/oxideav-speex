# oxideav-speex

A pure-Rust Speex (CELP speech codec) decoder for the
[oxideav](https://github.com/OxideAV/oxideav) framework. Implemented
from *The Speex Codec Manual*, RFC 5574, and the clean-room codebook
material staged at
[`docs/audio/speex/`](../../docs/audio/speex/).

## Status

**Clean-room rebuild in progress — partial decoder.** The full decode
path is not yet bit-exact, and the framework codec entry points still
return `Error::NotImplemented`. What is implemented and tested:

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
* **Wideband sub-band decode loop** (`WidebandDecoder`) — walks an
  embedded wideband packet (manual §10.4: narrowband frame packed first,
  then the high band) and returns both reconstructed 8 kHz half-band
  signals (`WidebandFrame { low_band, high_band }`), each band carrying
  its persistent IIR state across packets.
* **Top-level packet decoder** (`SpeexDecoder`) — drives the
  `PacketFrames` iterator through the per-frame decode loops, decoding a
  whole multi-frame Speex packet to a `Vec<DecodedFrame>` (narrowband
  PCM, wideband half-band pair, or a control pseudo-frame). One decoder
  instance handles a stream mixing NB / WB frames with continuous state
  (a wideband frame's low band *is* an embedded narrowband frame, so the
  shared narrowband state stays continuous; RFC 5574 §3.1).

The narrowband + wideband decode loops are wired end-to-end and produce
finite, input-responsive, deterministic PCM from a real `speexenc`
stream through the top-level `SpeexDecoder`.

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
  (cross-checked in `lsp_base` tests). What remains for
  *reference-equivalence* (the
  bit-exactness half) is the exact **cosine-series fixed-point evaluation
  order** the reference decoder uses for `cos(ω)` (Q-precision +
  rounding), which the staged manual prose does not pin — recorded docs
  gap, isolated to the `lsp_q*_to_radians` / LSP→LPC core. The framework
  `Decoder` endpoints return `Error::NotImplemented` until that closes;
  the free-function `SpeexDecoder` / `NarrowbandDecoder` /
  `WidebandDecoder` decode paths are the public surface in the meantime.
* Encoder.
* **QMF synthesis filterbank** — the final recombination of the low-band
  (narrowband) + high-band 8 kHz half-band signals into 16 kHz wideband
  PCM. The staged material provides the 64-tap QMF prototype `h0` as
  pure data and states structurally that a QMF splits / recombines the
  bands, but does **not** specify the synthesis filterbank algorithm
  (polyphase recombination structure, the `h0 → {h0, h1}` analysis /
  synthesis pair derivation, the 2× interpolation + decimation factors,
  or the inter-band delay alignment). Recorded docs gap; the high-band
  branch stops at the reconstructed half-band signal.
* **Per-sub-frame high-band LSP interpolation** — the high-band
  synthesis currently uses the frame-level high-band LPC set for all
  four sub-frames; the per-sub-frame interpolation (the high-band
  analogue of the narrowband §9.1 path) is a follow-up.
* **Ultra-wideband high-band bit allocation** — the UWB framing
  *recursion* is surfaced (`UwbFrameLayout`), but the per-mode UWB
  high-band bit budget (a "Table 11.x" analogue of Table 10.1 for the
  8–16 kHz band) is not in the staged manual. Recorded docs gap.
* Per-mode innovation handling for narrowband modes 1 and 7 and
  high-band mode 4, whose decomposition the staged inventory does not
  yet uniquely fix. (Mode 4 = 80 bits / 40-sample sub-frame: neither
  staged high-band codebook shape — `HbSv8_128` (8 samples, 8-bit
  composite) nor `HbSv10_32` (10 samples, 5-bit) — yields a split
  matching both the 80-bit budget *and* the 40-sample count, so the
  binding stays a recorded docs gap.)

## Usage

```toml
[dependencies]
oxideav-speex = "0.1"
```

Disable default features for the framing / parse surface without the
framework dependency.

## License

MIT — see [LICENSE](./LICENSE).
