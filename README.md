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

### Coverage estimate

~12 % of the Speex codec surface (Ogg stream header + per-frame
leading prefix + Table 9.1 sub-mode budgets + frame-body bit-reader
producing raw indices + §5.5 in-band signalling body parser for
modes 13 / 14; codebook lookup + LSP→LPC + pitch / innovation
synthesis + wideband high-band + encoder pending).

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
  order), §10.4 (wideband bit allocation).
- `docs/audio/speex/rfc5574-speex.txt` — RFC 5574 *RTP Payload Format
  for the Speex Codec*, Tables 1 & 2 for the mode ↔ bit-rate mapping
  cross-reference.

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

## License

MIT — see [LICENSE](./LICENSE).
