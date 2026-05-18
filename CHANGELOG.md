# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- Clean-room re-implementation against the published Speex
  specifications + RFC 5574 in a future round.
