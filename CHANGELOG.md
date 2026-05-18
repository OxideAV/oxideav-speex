# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.7](https://github.com/OxideAV/oxideav-speex/compare/v0.0.6...v0.0.7) - 2026-05-18

### Other

- drop two stray libspeex references in stereo encoder docs
- intensity-stereo encode (Speex manual §5.5 Table 5.1 code 9)

### Added

- **Intensity-stereo encode.** The top-level encoder factory now
  accepts `channels = Some(2)` for the NB, WB, and UWB rates. Each
  output packet is prefixed by the 17-bit Speex manual §5.5
  Table 5.1 code-9 side-channel request (`wb=0 || m=14 || id=9 ||
  sign || dexp || e_ratio_idx`) computed from the per-frame
  `(eL, eR, eM)` energies, followed by the standard mono CELP frame
  produced from the `(L+R)/2` downmix. NB sub-mode 5 stereo packets
  are 40 bytes (317 bits before pad). Silent frames emit the
  neutral payload `(0, 0, 3)` matching the decoder's
  `StereoState::new`. New `StereoSideChannel` type carries the
  pre-quantised triple between the energy analysis and the bit
  writer; `mix_to_mono`, `energies`, and `write_inband` helpers
  expose the building blocks for callers driving the bitstream
  directly. The Speex-in-Ogg header byte 48..52 now reflects the
  caller-supplied channel count.
- `tests/encode_stereo.rs` — 9 integration tests covering NB / WB /
  UWB factory acceptance, the 38-vs-40-byte size delta between mono
  and stereo NB-5 packets, left-loud / right-loud / balanced
  round-trips through the encoder + decoder, and the quad-channel
  rejection path.
- `StereoSideChannel::{from_lr, to_state, mix_to_mono, energies,
  write_inband}` plus the `E_RATIO_QUANT_VALUES` constant
  (encoder-side equivalent of the decoder's existing quantisation
  table).

## [0.0.6](https://github.com/OxideAV/oxideav-speex/compare/v0.0.5...v0.0.6) - 2026-05-06

### Other

- drop dead `linkme` dep
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-speex/pull/502))

## [0.0.5](https://github.com/OxideAV/oxideav-speex/compare/v0.0.4...v0.0.5) - 2026-05-03

### Other

- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- typed Speex in-band signalling per RFC 5574 / manual §5.5
- adopt slim VideoFrame/AudioFrame shape
- pin release-plz to patch-only bumps

### Added

- Typed in-band signalling per Speex manual §5.5 / RFC 5574 — new
  `inband` module exposes `InbandMessage` covering all 16 `m=14`
  request codes plus the `m=13` user payload and the `m=15` frame
  terminator. `encode_inband` / `decode_inband` round-trip every
  variant; `pad_to_octet_boundary` writes the RFC 5574 §3.3 padding
  pattern (`0` followed by all-ones, LSB-aligned). The CELP decoders
  still skip unrecognised requests opaquely; typed parsing is opt-in.
- `tests/inband.rs` integration tests: real CELP packets prefixed with
  a chain of mixed-width in-band requests still decode to coherent
  audio; round-trip every typed message; padding pattern matches the
  RFC 5574 §3.3 example byte-for-byte.

### Fixed

- README: WB sub-modes 2 and 4 are emitted by the encoder (table was
  out of date with the implementation). WB selection thresholds now
  match `wb_submode_for_rate`.

## [0.0.4](https://github.com/OxideAV/oxideav-speex/compare/v0.0.3...v0.0.4) - 2026-04-25

### Other

- bump oxideav-ogg dep-dev range to 0.1
- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- plumb WB hb_innov into UWB folding encoder
- implement WB encoder sub-modes 2 and 4
- add BSD-3-Clause attribution for libspeex-derived code
- Merge remote-tracking branch 'origin/master'
- drop unused interp_qlpc field from NbEncoder
- add stereo integration tests + refresh README coverage tables
- encode full NB sub-mode ladder + intensity-stereo decode
