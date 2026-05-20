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

### What lands in this round

Round 1 (this commit) implements the **Ogg/Speex stream-header packet
parser** from *The Speex Codec Manual* §7.3, Table 7.1. The parser:

- Validates the 8-byte `Speex   ` magic (with the spec-mandated three
  trailing spaces).
- Decodes all 13 little-endian `int32` fields from Table 7.1 — namely
  `speex_version_id`, `header_size`, `rate`, `mode`,
  `mode_bitstream_version`, `nb_channels`, `bitrate`, `frame_size`,
  `vbr`, `frames_per_packet`, `extra_headers`, `reserved1`, `reserved2`
  — and the `speex_version` ASCII field.
- Surfaces the parsed fields as a [`SpeexHeader`] struct + `mode`
  cross-check against the three modes documented in RFC 5574 §3
  (narrowband 8 kHz / wideband 16 kHz / ultra-wideband 32 kHz).
- Returns typed `HeaderError::TooShort` / `HeaderError::BadMagic` for
  bad inputs.

No frame decode is wired up yet — the CELP entropy stages, sub-mode
tables, and pitch/innovation codebooks all return
`Error::NotImplemented`. They land in subsequent rounds.

### Coverage estimate

~2 % of the Speex codec surface (stream-header parser only;
frame decode + encoder + in-band signalling pending).

### Spec material consulted

- `docs/audio/speex/speex-manual.pdf` — *The Speex Codec Manual*
  Version 1.2 Beta 3 (Jean-Marc Valin, December 2007), §7.3 + Table
  7.1, plus §2 and §5.5 for cross-reference.
- `docs/audio/speex/rfc5574-speex.txt` — RFC 5574 *RTP Payload Format
  for the Speex Codec*, Tables 1 & 2 for the mode ↔ bit-rate mapping.

No external library source (libspeex / Speex reference implementation,
FFmpeg, etc.) was consulted, paraphrased, or used as cross-check oracle.

## License

MIT — see [LICENSE](./LICENSE).
