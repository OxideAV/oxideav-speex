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

**Round 2** (this commit) adds the **per-frame leading prefix** parser
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

No CELP frame body is decoded yet — the LSP VQ, pitch / innovation
codebooks, and synthesis filter all still return
`Error::NotImplemented`. They land in subsequent rounds.

### Coverage estimate

~5 % of the Speex codec surface (Ogg stream header + per-frame leading
prefix + Table 9.1 sub-mode budgets; CELP frame body + wideband
high-band + encoder + in-band signalling pending).

### Spec material consulted

- `docs/audio/speex/speex-manual.pdf` — *The Speex Codec Manual*
  Version 1.2 Beta 3 (Jean-Marc Valin, December 2007). §5.5
  ("Packing and in-band signalling", Table 5.1), §7.3 ("Ogg file
  format", Table 7.1), §9.1–§9.3 (narrowband mode, Table 9.1), §10.4
  (wideband bit allocation).
- `docs/audio/speex/rfc5574-speex.txt` — RFC 5574 *RTP Payload Format
  for the Speex Codec*, Tables 1 & 2 for the mode ↔ bit-rate mapping
  cross-reference.

No external library source (libspeex / Speex reference implementation,
FFmpeg, etc.) was consulted, paraphrased, or used as cross-check
oracle.

### Spec gaps noted

- §9.3 prose says *"only the first 7 values are used (the others are
  reserved)"* but Table 9.1 itself lists 9 columns (modes 0..=8). The
  implementation follows the table; round-3 should keep an eye on
  whether mode 8 turns out to be a documented-but-unused encoder
  selection or whether real encoders emit it.

## License

MIT — see [LICENSE](./LICENSE).
