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
- Round 3: narrowband CELP frame-body bit-reader per
  *The Speex Codec Manual* §9.3 — `NarrowbandFrameBody::parse` walks
  Table 9.1's columns in bit-stream order ("all frame-based parameters
  packed before sub-frame parameters; the parameters for a sub-frame
  packed before the next sub-frame") and surfaces every field as a
  raw bit-index. Codebook lookup, LSP→LPC conversion, and the
  synthesis filter remain deferred until the Speex CELP companion
  tables are staged under `docs/audio/speex/`.
- Round 3: public `NarrowbandFrameBody` + `NarrowbandSubFrameIndices`
  structs carrying the LSP VQ index, open-loop pitch / pitch-gain /
  excitation-gain indices (frame-level), and the per-sub-frame fine
  pitch index, pitch-gain VQ index, innovation-gain index, and raw
  innovation VQ field (`u128`, sized for mode 7's 96-bit
  per-sub-frame innovation).
- Round 3: public `PITCH_PERIOD_MIN` / `PITCH_PERIOD_MAX` constants
  (17 / 144) per §9.2's "[17, 144] range" wording, plus
  `NarrowbandSubFrameIndices::pitch_period` de-biasing helper that
  returns `None` when the sub-mode's `fine_pitch_bits` is zero.
- Round 3: typed `NarrowbandBodyError::Underflow(BitError)` with a
  `From<BitError>` conversion and a top-level `Error::NarrowbandBody`
  wrapping.
- Round 3: 11 new unit tests in `src/narrowband_body.rs` (42 unit
  tests total) covering silence-mode empty body, body-bit-count
  derivation for every documented sub-mode, exact field widths for
  mode 5 + mode 7 (the widest fields), pitch-period de-biasing, and
  truncated-packet underflow diagnosis.
- Round 3: integration test
  `tests/narrowband_body_fixture.rs` (2 tests) parses every audio
  packet of a real `speexenc`-encoded narrowband fixture
  (`tests/fixtures/nb_440hz_q8.spx`, 1 s of 8 kHz 440 Hz tone at
  quality 8 → narrowband sub-mode 5) end-to-end through the round-2
  header parser + round-3 body parser. A minimal inline Ogg
  page-walker (≈30 lines, not a general-purpose Ogg implementation)
  lifts Speex packets out of the `.spx` file to keep the test free of
  cross-crate dev-dependencies per workspace policy. The fixture
  regeneration command is recorded in `tests/fixtures/Makefile`;
  `speexenc` is invoked as an opaque binary.
- Round 3: 44 tests total (42 unit + 2 integration), up from 31 in
  round 2.

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
