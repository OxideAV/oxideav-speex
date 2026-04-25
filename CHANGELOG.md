# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
