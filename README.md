# oxideav-speex

A pure-Rust Speex (CELP speech codec) NB/WB/UWB decoder + encoder for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild in progress.** The prior implementation was retired
under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md)
on **2026-05-19**: the source citations across the decoder + encoder +
table modules acknowledged that the implementation was a direct port of
an external library's codebase. The contamination was caught in the
2026-05-19 audit and master history was fully erased per the Hat-3
cold-enforcement procedure. The new master is being rebuilt from scratch
against the published Speex specifications + RFC 5574.

### What lands per round

**Round 1** wired the **Ogg/Speex stream-header packet parser** from
*The Speex Codec Manual* §7.3, Table 7.1:

- Validates the 8-byte `Speex   ` magic (with the spec-mandated three
  trailing spaces).
- Decodes all 13 little-endian `int32` fields from Table 7.1 — namely
  `speex_version_id`, `header_size`, `rate`, `mode`,
  `mode_bitstream_version`, `nb_channels`, `bitrate`, `frame_size`,
  `vbr`, `frames_per_packet`, `extra_headers`, `reserved1`, `reserved2`
  — and the `speex_version` ASCII field.
- Surfaces the parsed fields as a `SpeexHeader` struct + `mode`
  cross-check against the three modes documented in RFC 5574 §3
  (narrowband 8 kHz / wideband 16 kHz / ultra-wideband 32 kHz).
- Returns typed `HeaderError::TooShort` / `HeaderError::BadMagic` for
  bad inputs.

**Round 2** added the **per-frame leading prefix** parser
+ the typed **narrowband sub-mode table** distilled from §9.3 and
Table 9.1:

- A minimal MSB-first `BitReader` matching the Speex bit-packing
  convention (the next-bit-read is the bit immediately to the right of
  the previously-read bit within the same byte, MSB-first).
- `NarrowbandFrameHeader::parse` consumes the leading 5 bits of every
  Speex frame: the 1-bit wideband flag (§10.4) and the 4-bit mode ID
  (§9.3). Cursor is left immediately after the prefix, ready for the
  per-sub-mode body parser landing in round 3.
- `Submode` resolves the mode ID into one of:
  * a regular CELP `NarrowbandSubmode` (IDs 0..=8, Table 9.1),
  * `CustomInband` (mode 13, §5.5),
  * `InbandSignalling` (mode 14, §5.5, Table 5.1),
  * `Terminator` (mode 15, §5.5).
  IDs in the reserved range 9..=12 surface as `FrameError::ReservedMode`.
- `NARROWBAND_SUBMODES` is a `[NarrowbandSubmode; 9]` table that
  records per-column from Table 9.1 the LSP-VQ width, frame-level
  open-loop pitch / pitch-gain / excitation-gain bit counts, sub-frame
  fine-pitch / pitch-gain / innovation-gain / innovation-VQ bit counts,
  and the table's `Total` row. A self-consistency test recomputes
  `Total` from the breakdown for every column.

**Round 3** (this commit) adds the **narrowband CELP frame-body
bit-reader** — `NarrowbandFrameBody::parse` — that walks Table 9.1's
columns in the bit-stream order documented in §9.3 ("all frame-based
parameters are packed before sub-frame parameters; the parameters for
a certain sub-frame are all packed before the following sub-frame is
packed"):

- Frame-level: LSP VQ index (0 / 18 / 30 bits), open-loop pitch
  period (0 / 7 bits), open-loop pitch gain (0 / 4 bits), open-loop
  excitation gain (0 / 5 bits).
- Sub-frame ×4: fine pitch period (0 / 7 bits, de-biased per §9.2 to
  the spec's [17, 144] range via the public `PITCH_PERIOD_MIN` /
  `PITCH_PERIOD_MAX` constants), pitch-gain VQ index (0 / 5 / 7 bits),
  innovation gain (0 / 1 / 3 bits), innovation VQ raw index (up to 96
  bits per sub-frame for mode 7, surfaced as `u128`).

Eight raw-index fields per frame × four sub-frames = 12 distinct
field positions consumed per body, plus the four frame-level fields.
The struct intentionally records only the index integers; codebook
lookup (LSP VQ → 10 LSP coefficients, innovation VQ → 40-sample
sub-vector innovation signal, pitch-gain VQ → three β coefficients)
is deferred until the Speex CELP **companion-table** docs gap closes
— see "Spec gaps noted" below.

An integration test (`tests/narrowband_body_fixture.rs`) exercises the
parser against a real `speexenc`-encoded narrowband fixture: 51 audio
packets, every one parsed end-to-end through both the round-2 header
parser and the round-3 body parser, with every per-field index range
asserted against the mode-5 column of Table 9.1. The fixture
regeneration command is recorded in `tests/fixtures/Makefile`;
`speexenc` is used as an opaque binary — its source is NOT consulted.

LSP→LPC conversion, codebook lookup, and the synthesis filter still
return `Error::NotImplemented`. They land in subsequent rounds, once
the companion-table material is staged.

**Round 4** (this commit) adds the **§5.5 in-band signalling parser**
— the bodies that the round-2 dispatcher recognised but left to a
follow-up round:

- `InbandMessage::parse` walks mode 14's body: a 4-bit Table 5.1 code
  (`code`) followed by a payload of `1 / 4 / 8 / 16 / 32 / 64` bits as
  prescribed by the code's row. Surfaces the typed
  `InbandCodeSpec { code, payload_bits, kind }` + the raw `payload`
  bits zero-extended into a `u64`. The wide path (>32 bits) splits the
  payload across two `BitReader::read` calls to side-step the reader's
  32-bit width guard. Reserved rows (11 / 13 / 14 / 15) parse without
  error per §5.5's "by default ignore" rule — the cursor still
  advances past the declared payload so subsequent frames in the same
  packet stay aligned.
- `CustomInbandMessage::parse` walks mode 13's body: the 5-bit
  byte-count field (max 31 bytes per the 5-bit width) and discards
  `size_bytes * 8` opaque payload bits — exactly the behaviour
  §5.5's final paragraph specifies ("The size of the message in
  bytes is encoded with 5 bits, so that the decoder can skip it if
  it doesn't know how to interpret it.").
- The public `INBAND_TABLE_5_1: [InbandCodeSpec; 16]` array stages
  Table 5.1 verbatim: every code 0..=15 carries its `payload_bits`
  width and an [`InbandKind`] tag (`PerceptualEnhancement`,
  `LessAggressive`, `SwitchMode`, `SwitchModeLowBand`,
  `SwitchModeHighBand`, `SwitchQualityVbr`, `RequestAcknowledge`,
  `SetRateMode`, `TransmitCharacter`, `IntensityStereo`,
  `AnnounceMaxBitrate`, `AcknowledgePacket`, `Reserved`).
- New `Error::Signalling(SignallingError)` top-level error variant +
  `From<BitError>` plumbing matching the round-2/3 error envelopes.

This element is **unblocked by the round prompt** because §5.5 +
Table 5.1 are fully published in the staged Speex Codec Manual — no
CELP companion-table material (the open #969 blocker) is touched.

**Round 5** (this commit) adds the **wideband (sub-band CELP)
high-band sub-mode table** from §10.4 / Table 10.1 plus a high-band
frame-body bit-reader:

- `WIDEBAND_HIGH_BAND_SUBMODES: [WidebandHighBandSubmode; 5]` stages
  Table 10.1 verbatim for modes 0..=4 — every row captured with rows
  `Wideband bit` (1) / `Mode ID` (3 bits, *not* 4) / `LSP` (0 / 12
  bits MSVQ) / `Excitation gain` (0/5/4/4/4 per sub-frame) /
  `Excitation VQ` (0/0/20/40/80 per sub-frame) / `Total`
  (4/36/112/192/352 bits per 20 ms high-band frame).
- `WidebandHighBandFrameHeader::parse` consumes the 4-bit high-band
  prefix; `WidebandHighBandBody::parse` walks Table 10.1's columns
  in the spec-stated order (frame-LSP first, then four sub-frames of
  `excitation_gain || excitation_vq`). Per §10.4: *"the entire
  narrowband frame is packed before the high-band is encoded. The
  narrowband part of the bit-stream is as defined in table 9.1. The
  high-band follows, as described in table 10.1."*
- `WidebandSubmode::for_id` dispatches the 3-bit field into
  `Documented(...)` for modes 0..=4 and `ReservedHighRate(id)` for
  modes 5..=7 — encodable but not in the staged Table 10.1 (Table
  10.2 lists 0..=10 with composite bit-rates but does not detail
  modes 5..=10's per-field budgets; recorded as a docs gap below).
- New public constants `HIGH_BAND_FRAME_PREFIX_BITS` (4) and
  `HIGH_BAND_SUBFRAMES_PER_FRAME` (4); new
  `HighBandSubFrameIndices` struct (excitation gain + excitation VQ
  index as raw integers).
- New `Error::Wideband(WidebandBodyError)` envelope variant with
  `Underflow(BitError)` + `ReservedHighRate(u8)` + `From<BitError>`
  plumbing matching the round-2/3/4 error shape.
- 21 new unit tests in `src/wideband.rs` (89 tests total, up from
  68 in round 4) covering Table 10.1 structural sanity, per-column
  field-width assertions, mode-0 silent-body, mode-4 widest-field
  round-trip, mode-1 no-innovation-VQ, truncated-frame underflow,
  reserved-high-rate dispatch, and the cursor-after-prefix contract.

As with round 3, this round stops at the raw bit-index layer. The
high-band LSP MSVQ codebook (level-1 + level-2, both 6-bit per §10.1)
and the per-mode high-band innovation codebooks are also in the
libspeex `*_table.c` files and remain #969-blocked.

**Round r165** (this commit) adds the **typed packet → frame
iterator** that composes the round-2 / 3 / 4 / r160 primitives
end-to-end without introducing any codebook-dependent logic:

- New module `src/packet.rs` with public [`PacketFrames`] iterator,
  [`PacketFrame`] sum type, [`PacketError`] envelope, and a
  [`parse_packet`] convenience that returns `Vec<PacketFrame>`.
- Walks a Speex packet body per §5.5's *"Sometimes it is desirable
  to pack more than one frame per packet … it is possible to
  include a terminator code. That terminator consists of the code
  15 (decimal) encoded with 5 bits, as shown in Table 9.2 … calling
  speex_bits_write automatically inserts the terminator so as to
  fill the last byte."* — dispatching each successive 5-bit prefix
  into:
  * a regular narrowband CELP frame
    (`PacketFrame::Narrowband { header, body }`),
  * a wideband narrowband+high-band pair
    (`PacketFrame::Wideband { header, narrowband, high_band_header,
    high_band }`) when the narrowband prefix's wideband flag is set —
    per §10.4 *"the entire narrowband frame is packed before the
    high-band is encoded"*,
  * a §5.5 in-band signalling message
    (`PacketFrame::InbandSignalling { header, message }`),
  * a §5.5 custom in-band message
    (`PacketFrame::CustomInband { header, message }`),
  * a mode-15 terminator (yields `None` and halts iteration).
- Treats `< NARROWBAND_FRAME_PREFIX_BITS` of remaining bits as
  end-of-packet padding (clean halt, no error) — matches the §5.5
  trailing-pad convention.
- Surfaces a `PacketError::Wideband(ReservedHighRate(id))` and halts
  iteration when the high-band 3-bit mode field falls in `5..=7`
  (the Table 10.1 docs gap), letting the caller inspect what
  remains via [`PacketFrames::remaining_bits`].
- Implements [`std::iter::Iterator`] directly on `PacketFrames<'_>`,
  so combinators (`filter`, `count`, `collect`) work naturally.
- New top-level `Error::Packet(PacketError)` envelope variant +
  `From<PacketError>` plumbing.
- 19 new unit tests in `src/packet.rs` (108 unit tests total, up
  from 89) covering empty buffer, terminator-only packet, single
  silence frame + padding, two-silence + terminator multi-frame
  packet, in-band signalling intermixed with CELP, custom in-band
  size-0 + silence, reserved-mode rejection, truncated-body
  underflow, error-then-None invariant, wideband silence frame
  round-trip, reserved-high-rate dispatch, and packet structural
  combinator usage.
- New integration test `tests/packet_iterator_fixture.rs` (2 tests)
  walks every audio packet of the round-3 `speexenc`-encoded
  narrowband fixture through `PacketFrames` and asserts (a) every
  packet yields ≥ 1 narrowband frame of the expected mode 5, and
  (b) after iteration halts every packet has < 5 trailing bits
  (the §5.5 padding tail). 112 tests total (108 unit + 4
  integration), up from 91 in round 5.

This is composition only — no #969-blocked tables are touched. The
iterator's variants carry the same raw bit-index payloads that the
underlying primitives already produced; once the CELP codebooks
land, the variant payloads grow decoded values, but the dispatch
structure stays.

**Round r179** (this commit) adds the **MSB-first `BitWriter`** —
the symmetric companion to the round-2 `BitReader`:

- `BitWriter` is the bit sink an encoder needs. It is the inverse
  operation of `BitReader`: feed it the same `(value, n)` pairs a
  `BitReader` would emit from a buffer, and the bytes it produces
  round-trip back through `BitReader` to the same `(value, n)` pairs
  in the same order. The round-trip invariant is asserted by three
  tests: a curated short sequence, a per-bit walk, and a 256-step
  LCG-driven random pattern.
- API mirrors the reader's: `new()` / `with_capacity(bytes)` /
  `write_bit(b)` / `write(value, n)` / `bits_written()` /
  `bits_left_in_last_byte()` / `is_byte_aligned()` / `pad_to_byte()`
  / `as_bytes()` / `into_bytes()`. The existing `BitError::TooWide(n)`
  diagnostic is reused for `n > 32` so writer + reader share the
  same error envelope.
- The cfg-test `BitPacker` helper that `packet::tests` had been
  using to assemble synthetic Speex packets is retired in favour of
  the public `BitWriter`. The conversion is behaviour-preserving;
  every previously-passing `packet::tests` case now exercises the
  same public bit-packing routine an encoder would call. This is
  the first piece of encoder-shaped infrastructure to land in the
  rebuild.
- 15 new unit tests on `BitWriter` (123 unit tests total, up from
  108 in round r165). Test count: 127 (123 unit + 4 integration),
  up from 112.

The `BitWriter` itself depends on no companion tables and is
therefore #969-independent — it slots in below the eventual CELP
encoder.

**Round r187** (this commit) adds the **structured `write` methods
symmetric to the existing `parse` paths** for the three framing-level
types whose layout is fully published in the staged manual without
any CELP companion-table material:

- `NarrowbandFrameHeader::write` emits the 5-bit prefix (1-bit
  wideband flag + 4-bit mode ID, MSB-first per §9.3). A new
  `NarrowbandFrameHeader::new(wideband, mode_id)` constructor
  dispatches the mode ID through `Submode::for_id` and rejects the
  reserved range 9..=12 via `FrameError::ReservedMode` — the
  encoder-side counterpart of the round-2 parser's rejection on
  the same input.
- `InbandMessage::write` emits the 4-bit Table 5.1 code followed by
  the per-row payload width (1 / 4 / 8 / 16 / 32 / 64 bits). The
  wide path (>32 bits, reserved codes 14 / 15) splits the payload
  across two `BitWriter::write` calls mirroring the parser's split.
  Payload bits above the spec'd width are masked off before
  emission.
- `CustomInbandMessage::write` emits the 5-bit `size_bytes` field
  per §5.5 followed by `size_bytes` opaque payload bytes taken from
  a caller-supplied slice. `size_bytes` is masked to 5 bits so the
  field width is preserved even if the caller supplies a value
  above 31.

All three `write` methods are inverse operations of the existing
`parse` methods. The round-trip invariant — parse(write(value)) ==
value — is asserted by 17 new unit tests, including a sweep over
every Table 5.1 code (1..=64 bit payloads, including the wide-path
split) and over every documented CELP / signalling mode ID with
both wideband-flag values. An end-to-end "write header + write
inband message → parse back" test exercises the round-2
dispatcher against synthetic bytes assembled by the new writers.

The writers depend only on the round-179 `BitWriter`, the round-2
`Submode::for_id` dispatch, and the round-4 Table 5.1 staging.
**No** CELP companion-table material is touched.

**Round 191** (this commit) wires the **CELP companion tables**
into the crate as a typed pure-data surface. The clean-room CSVs
staged at `docs/audio/speex/tables/` (see the in-repo provenance
manifest `docs/audio/speex/provenance/01-speex-table-extraction.md`)
are embedded via `include_str!` and parsed on first use into
`OnceLock`-backed `&'static [Row]` slices. The public accessors
shipped this round:

- **Narrowband LSP VQ** (§9.1): `nb_lsp_stage0()` returns the
  6-bit 64 × 10 stage; `nb_lsp_low1/low2/high1/high2()` return the
  four 6-bit 64 × 5 split-band stages. `nb_lsp_scale(stage)`
  returns the documented Q-scale (`Div256` / `Div512` / `Div1024`)
  the decoder applies before adding the stage's contribution.
- **3-tap pitch-gain VQ** (§9.2): `pitch_gain_5bit()` (32 × 4) and
  `pitch_gain_7bit()` (128 × 4) — rows are
  `[g0, g1, g2, search_aid]`; consumers add the documented
  `PITCH_GAIN_BIAS` (+32) to each tap before applying.
- **Narrowband innovation codebooks** (§9.2): six accessors covering
  every shape Table 9.1 references — `innovation_5_64/_256`,
  `innovation_8_128`, `innovation_10_16/_32`, `innovation_20_32`.
- **Wideband high-band LSP MSVQ** (§10.x): `hb_lsp_stage1/stage2()`
  return the two 6-bit 64 × 8 stages; combined index space
  `64 × 64 = 2^12` matches the 12-bit `lsp_msvq_index` already
  surfaced by the round-5 high-band body bit-reader.
- **Wideband high-band innovation codebooks**: `hb_innovation_8_128`
  and `hb_innovation_10_32` for the two shapes documented in
  Table 10.1.
- **LPC analysis fixtures (Q15)**: `lpc_analysis_window_q15()` (200
  samples), `lpc_lag_window_q15()` (11 taps), `qmf_h0_q15()` (64-tap
  QMF analysis filter used by the wideband 0-4 / 4-8 kHz split per
  §10.1).

Naming follows the role (stage, dimension, regime), not the
source-side identifier; the `.meta` sidecars preserve the canonical
names. Sixteen self-checks under `codebooks::tests` verify the
embedded row counts, cross-check the codebook widths against the
existing `NarrowbandSubmode` bit budgets, and spot-check fixed
values (`nb_lsp_stage0()[0]`, `pitch_gain_5bit()[0]`,
`lpc_lag_window_q15()[0] == 32767`).

Lookup, gain scaling, LSP→LPC conversion, pitch/innovation
synthesis, and the encoder-side codebook search remain deferred to
subsequent rounds — this round ships the *table*, not the codepath
that consumes it.

### Coverage estimate

~23 % of the Speex codec surface (Ogg stream header + per-frame
leading prefix + Table 9.1 narrowband sub-mode budgets + Table 10.1
wideband high-band sub-mode budgets + narrowband frame-body
bit-reader + wideband high-band frame-body bit-reader + §5.5
in-band signalling body parser for modes 13 / 14 + typed packet →
frame iterator composing the above end-to-end + MSB-first
`BitWriter` covering the encoder's bit-sink side of the symmetry
with `BitReader` + structured `write` methods for the three
framing-level types whose layout is published without any
companion-table material + r191 CELP companion-table accessors
exposing the staged narrowband LSP VQ, 3-tap pitch-gain VQ, six
narrowband innovation codebooks, the wideband high-band LSP MSVQ
and high-band innovation codebooks, plus the Q15 LPC analysis
window / lag window / QMF analysis filter as typed
`&'static [Row]` slices; LSP→LPC + gain scaling + pitch / innovation
synthesis + ultra-wideband framing + the CELP frame-body writer +
encoder-side codebook search are the remaining pieces).

### Spec material consulted

- `docs/audio/speex/speex-manual.pdf` — *The Speex Codec Manual*
  Version 1.2 Beta 3 (Jean-Marc Valin, December 2007). §5.5
  ("Packing and in-band signalling", Table 5.1), §7.3 ("Ogg file
  format", Table 7.1), §8 (CELP overview — source/filter, LPC, pitch,
  innovation), §9.1 ("Whole-frame analysis": 160-sample frame, 4
  sub-frames of 40 samples), §9.2 ("Sub-frame analysis-by-synthesis":
  pitch period in [17, 144] encoded with 7 bits; 3-tap β
  coefficients VQ'd with 5 or 7 bits; sub-vector innovation codebook
  sizes), §9.3 (Bit allocation, Table 9.1 — bit-stream packing
  order), §10 / §10.1 / §10.4 (wideband sub-band CELP — QMF split
  into low/high 8 kHz bands; 12-bit MSVQ for high-band LSP via two
  6-bit codebooks; high-band frame layout Table 10.1; high-band
  composite bit-rates Table 10.2).
- `docs/audio/speex/rfc5574-speex.txt` — RFC 5574 *RTP Payload Format
  for the Speex Codec*, Tables 1 & 2 for the mode ↔ bit-rate mapping
  cross-reference (Table 2 confirms wideband + ultra-wideband mode
  IDs 0..=10).

No external library source (libspeex / Speex reference implementation,
FFmpeg, etc.) was consulted, paraphrased, or used as cross-check
oracle. `speexenc` was invoked as an opaque binary only for fixture
generation; its output bytes are the test input.

### Spec gaps noted

- §9.3 prose says *"only the first 7 values are used (the others are
  reserved)"* but Table 9.1 itself lists 9 columns (modes 0..=8). The
  implementation follows the table.
- The Speex Manual does **not** publish the per-mode LSP VQ codebooks,
  pitch-gain VQ codebooks, or innovation codebooks themselves — those
  live in the libspeex distribution as `*_table.c` files. Round 3
  therefore stops at the **raw bit-index** layer: every codebook
  index, pitch period offset and gain index is recovered as an
  integer, but mapping `lsp_index` → ten LSP coefficients (and
  thence into LPC) or `innovation_vq_index` → a 40-sample
  sub-vector cannot proceed until the codebook tables are staged
  under `docs/audio/speex/` for clean-room transcription. This is
  tracked as the round-prompt "#969 stage Speex CELP companion
  tables" follow-up.
- The Speex Manual's §9.4 in the staged PDF is titled "Perceptual
  enhancement" and Table 9.2 is "Quality versus bit-rate" (mode ↔
  quality ↔ mflops); neither matches the round-3 prompt's reference
  to "Manual §9.4 / Table 9.2" for the narrowband-decoder frame
  layout / LSP values. The frame-layout material the round actually
  needs lives in §9.1 + §9.2 + §9.3 + Table 9.1 of the PDF, and
  Round 3 sources its work from there.
- Table 10.1 in the staged manual only details five wideband
  high-band columns (modes 0..=4); Table 10.2 names modes 0..=10
  with composite bit-rates / quality descriptors but does not list
  the per-field bit budgets for modes 5..=10. Round 5 therefore
  surfaces modes 5..=7 (the rest of the 3-bit field's encodable
  range) as `WidebandSubmode::ReservedHighRate(id)`; modes 8..=10
  fall outside the 3-bit field entirely and are unreachable via a
  conforming bit-stream. A follow-up docs round needs to stage the
  Table 10.1 columns for modes 5..=10 from the original Speex Codec
  Manual revisions if they are documented anywhere outside the
  libspeex source tree.
- Ultra-wideband (mode 2 in the Ogg stream header) has no dedicated
  §11 chapter in the staged Speex Codec Manual; only RFC 5574 Table
  2 lists the per-mode bit-rates. The bit-stream packing layout for
  the 32 kHz high-band is therefore not in the staged spec — UWB
  framing is deferred to a follow-up round once the relevant
  material is staged (likely a triple-band QMF + per-band CELP, but
  the exact bit allocation is a docs gap).

## License

MIT — see [LICENSE](./LICENSE).
