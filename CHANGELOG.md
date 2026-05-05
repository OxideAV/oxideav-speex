# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.6](https://github.com/OxideAV/oxideav-speex/compare/v0.0.5...v0.0.6) - 2026-05-05

### Other

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
