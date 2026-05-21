# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
