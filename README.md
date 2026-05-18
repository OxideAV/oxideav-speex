# oxideav-speex

A pure-Rust Speex (CELP speech codec) NB/WB/UWB decoder + encoder for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Orphan-rebuild scaffold (2026-05-19).** The prior implementation was
retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md):
the source citations across the decoder + encoder + table modules
acknowledged that the implementation was a direct port of an external
library's codebase. The contamination was caught in the 2026-05-19
audit and master history was fully erased per the Hat-3 cold-enforcement
procedure.

The implementation will be re-built against the published Speex
specifications (the in-tree `docs/audio/speex/speex-manual.pdf` + RFC
5574) in a future clean-room round.

## License

MIT — see [LICENSE](./LICENSE).
