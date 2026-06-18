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

The end-to-end synthesis path is wired (LSP reconstruction →
interpolation → LSP→LPC → innovation → synthesis filter) and produces
stable, input-responsive PCM from a real stream.

## Not yet supported

* Bit-exact full decode. The scalar excitation-gain quantiser levels
  are now exact (staged `ol_gain_table` / `exc_gain_quant_scal{1,3}` /
  `gc_quant_bound` / `fold_quant_bound`), and the reconstructed
  fixed-codebook gain is now folded into the innovation (the gain ×
  innovation scaling lands `c[n]` in magnitude-correct float units), and
  the adaptive-codebook (pitch) contribution `p[n]` is now scaled into
  the same normalised float signal domain by the staged **Q6**
  pitch-gain factor (`GAIN_SCALING = 64`), so the two §8.4 contributions
  finally share one domain. What remains: the float-domain composition
  step `e[n] = p[n] + c[n]` joining the two scaled contributions is not
  yet wired into the end-to-end synthesis path, and the LSP angular-unit
  / fixed-point domain is not yet pinned by the staged material, so the
  output is not yet reference-equivalent. The framework `Decoder`
  endpoints return `Error::NotImplemented` until that closes.
* Encoder.
* Ultra-wideband framing (no dedicated chapter in the staged manual).
* Per-mode innovation handling for narrowband modes 1 and 7 and
  high-band mode 4, whose decomposition the staged inventory does not
  yet uniquely fix.

## Usage

```toml
[dependencies]
oxideav-speex = "0.1"
```

Disable default features for the framing / parse surface without the
framework dependency.

## License

MIT — see [LICENSE](./LICENSE).
