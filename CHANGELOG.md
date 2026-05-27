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
- Round 4: §5.5 in-band signalling body parsers — `InbandMessage`
  (mode 14: 4-bit Table 5.1 code + `1 / 4 / 8 / 16 / 32 / 64` bits of
  payload) and `CustomInbandMessage` (mode 13: 5-bit byte-count +
  opaque payload skipped per §5.5's "decoder can skip it if it
  doesn't know how to interpret it" rule). Both consume the body
  that the round-2 dispatcher leaves immediately after the 5-bit
  frame prefix, so the cursor lands on the first bit of the next
  frame in the same packet without a manual re-sync.
- Round 4: public `INBAND_TABLE_5_1: [InbandCodeSpec; 16]` table
  staging every row of *The Speex Codec Manual* Table 5.1 verbatim
  (`PerceptualEnhancement` / `LessAggressive` / `SwitchMode` /
  `SwitchModeLowBand` / `SwitchModeHighBand` / `SwitchQualityVbr` /
  `RequestAcknowledge` / `SetRateMode` / `TransmitCharacter` /
  `IntensityStereo` / `AnnounceMaxBitrate` / `AcknowledgePacket` +
  reserved-row category). Lookup helper `inband_code_spec(code)`
  returns the row for any of the sixteen 4-bit codes.
- Round 4: public constants `CUSTOM_INBAND_SIZE_BITS` (5),
  `CUSTOM_INBAND_MAX_BYTES` (31), and `INBAND_CODE_BITS` (4) for
  downstream consumers' bit-budget math.
- Round 4: typed `Error::Signalling(SignallingError)` top-level
  variant + `From<SignallingError>` conversion matching the
  round-2 / round-3 error envelopes.
- Round 4: 22 new unit tests (66 total: 64 unit + 2 integration)
  covering Table 5.1 structural sanity (sixteen rows, codes match
  index, payload widths match the manual, reserved-row taxonomy),
  parsing every documented code category (perceptual enhancement /
  switch mode / transmit character / max bit-rate / packet ack /
  reserved 64-bit), the 32-bit + 64-bit width split paths,
  truncated-code + truncated-payload underflow diagnosis, mode-13
  size-zero / size-one / max-31-byte payload skip, and end-to-end
  round-trip parsing through `NarrowbandFrameHeader::parse` for
  mode 14 (transmit `'B'`) + mode 13 (zero-size custom message).
- Round 5: wideband high-band sub-mode table from *The Speex Codec
  Manual* §10.4 / Table 10.1 (modes 0..=4, 5 columns). Public
  `WIDEBAND_HIGH_BAND_SUBMODES: [WidebandHighBandSubmode; 5]` stages
  every column verbatim with rows `Wideband bit` / `Mode ID` (3 bits,
  not 4) / `LSP` (12-bit MSVQ for modes 1..=4, 0 for silence mode 0) /
  `Excitation gain` (sub-frame, 0/5/4/4/4 bits) / `Excitation VQ`
  (sub-frame, 0/0/20/40/80 bits) / `Total` (4/36/112/192/352 bits per
  20 ms high-band frame).
- Round 5: `WidebandHighBandFrameHeader::parse` consumes the 4-bit
  high-band prefix (1-bit wideband flag + 3-bit mode ID, distinct
  from the narrowband 5-bit prefix); `WidebandHighBandBody::parse`
  walks Table 10.1's columns in `frame-LSP || (gain || VQ) × 4
  sub-frames` order per §10.4 ("the entire narrowband frame is packed
  before the high-band is encoded. The narrowband part of the
  bit-stream is as defined in table 9.1. The high-band follows, as
  described in table 10.1.").
- Round 5: `WidebandSubmode::for_id` dispatches the 3-bit mode ID
  into `Documented(WidebandHighBandSubmode)` for modes 0..=4 and
  `ReservedHighRate(u8)` for modes 5..=7 (encodable in the 3-bit
  field but the staged Table 10.1 stops at column 4; Table 10.2 lists
  modes 5..=10 with composite bit-rates but does not detail the
  per-field budgets, so these IDs surface without a bit-budget
  contract until a follow-up docs round stages the missing columns).
- Round 5: public constants `HIGH_BAND_FRAME_PREFIX_BITS` (4) and
  `HIGH_BAND_SUBFRAMES_PER_FRAME` (4) for downstream consumers; new
  `HighBandSubFrameIndices` struct carrying per-sub-frame
  `excitation_gain_index` (u8) and `excitation_vq_index` (u128,
  sized for mode 4's 80-bit max).
- Round 5: typed `WidebandBodyError::Underflow(BitError)` +
  `ReservedHighRate(u8)` errors with `From<BitError>` conversion;
  new `Error::Wideband(WidebandBodyError)` top-level envelope
  variant + `From<WidebandBodyError>` plumbing.
- Round 5: 21 new unit tests in `src/wideband.rs` (87 unit tests
  total, up from 66) covering Table 10.1 structural sanity (five
  columns, mode IDs match index, totals match field breakdown, totals
  match the manual's verbatim `Total` row), per-column field-width
  assertions (LSP / excitation gain / excitation VQ), mode 0 silent
  high-band with empty body, mode 4 widest-fields round-trip, mode 1
  no-innovation-VQ assertion, truncated-frame underflow diagnosis,
  reserved-high-rate dispatch (modes 5..=7), out-of-3-bit-range
  rejection, and the wideband-flag-cleared surfacing.
- Round 5: 89 tests total (87 unit + 2 integration), up from 68 in
  round 4.
- Round r165: typed packet → frame iterator composing the
  round-2 / 3 / 4 / r160 primitives end-to-end per *The Speex Codec
  Manual* §5.5 — `PacketFrames::new(buf)` yields one
  `PacketFrame` per frame in the body and halts cleanly on a mode-15
  terminator or on `< NARROWBAND_FRAME_PREFIX_BITS` of remaining
  padding. New `parse_packet(buf) -> Vec<PacketFrame>` convenience.
- Round r165: `PacketFrame` enum with `Narrowband` / `Wideband` /
  `InbandSignalling` / `CustomInband` variants — each carrying the
  5-bit prefix alongside the typed body. Wideband variant captures
  both halves (narrowband body + high-band header + high-band body)
  per §10.4 *"the entire narrowband frame is packed before the
  high-band is encoded"*.
- Round r165: typed `PacketError` envelope (`Frame` /
  `NarrowbandBody` / `Wideband` / `Signalling` / `UnexpectedEnd`)
  with `From` conversions from every underlying error type; new
  top-level `Error::Packet(PacketError)` variant + `From<PacketError>`
  plumbing.
- Round r165: iterator surfaces a clean halt for the §5.5
  terminator + padding tail, and a `PacketError::Wideband(
  ReservedHighRate(id))` when the high-band 3-bit mode field lands
  in `5..=7` (the Table 10.1 docs gap).
- Round r165: 19 new unit tests in `src/packet.rs` (108 unit tests
  total, up from 89) covering empty buffer, terminator-only packet,
  single silence frame + padding, multi-frame packets, in-band
  signalling intermixed with CELP, custom in-band size-0 + silence,
  reserved-mode rejection, truncated-body underflow, error-then-None
  iterator invariant, wideband-silence round-trip, reserved-high-rate
  dispatch, and `Iterator` combinator usage (`filter` / `count`).
- Round r165: new integration test `tests/packet_iterator_fixture.rs`
  (2 tests) walks every audio packet of the existing
  `speexenc`-encoded narrowband fixture through `PacketFrames` and
  asserts every packet yields the expected mode-5 narrowband frame,
  the iterator halts cleanly, and `< 5` bits of trailing padding
  remain after the terminator.
- Round r165: 112 tests total (108 unit + 4 integration), up from
  91 in round 5.

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
