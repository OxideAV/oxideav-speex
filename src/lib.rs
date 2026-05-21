//! # oxideav-speex
//!
//! **Status:** clean-room rebuild in progress (post 2026-05-19 audit).
//!
//! Layered scope as the rebuild advances:
//!
//! * **Round 1** — the Ogg/Speex stream-header packet parser from
//!   *The Speex Codec Manual* §7.3, Table 7.1 (validates `Speex   `
//!   magic + decodes 13 LE int32 fields). See [`SpeexHeader`].
//! * **Round 2** (this commit) — the per-frame leading prefix
//!   (1-bit wideband flag + 4-bit mode ID) from §9.3 + the typed
//!   narrowband sub-mode table from Table 9.1 (modes 0..=8, plus
//!   the three §5.5 reserved-for-signalling slots: custom in-band
//!   mode 13, in-band signalling mode 14, terminator mode 15). See
//!   [`NarrowbandFrameHeader`] and [`Submode`]. No CELP body is
//!   parsed yet.
//!
//! Frame decode, encoder, and the `Decoder` / `Encoder` trait wiring
//! against `oxideav-core` still return [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

mod bitreader;
mod frame;
mod header;
mod submode;

pub use bitreader::{BitError, BitReader};
pub use frame::{FrameError, NarrowbandFrameHeader, NARROWBAND_FRAME_PREFIX_BITS};
pub use header::{
    HeaderError, SpeexHeader, SPEEX_HEADER_LEN, SPEEX_MAGIC, SPEEX_MODE_NARROWBAND,
    SPEEX_MODE_ULTRAWIDEBAND, SPEEX_MODE_WIDEBAND, SPEEX_STRING_LEN, SPEEX_VERSION_LEN,
};
pub use submode::{LspQuant, NarrowbandSubmode, PitchGainQuant, Submode, NARROWBAND_SUBMODES};

/// Crate-local error type. Until the full clean-room rebuild lands, the
/// codec-level public API paths return [`Error::NotImplemented`]; the
/// Ogg/Speex header parser surfaces its own [`HeaderError`] and the
/// per-frame parser surfaces [`FrameError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The crate's frame decoder / encoder has not been re-implemented
    /// against the spec yet.
    NotImplemented,
    /// The Ogg/Speex stream header failed to parse — see [`HeaderError`].
    Header(HeaderError),
    /// The per-frame leading prefix failed to parse — see [`FrameError`].
    Frame(FrameError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotImplemented => write!(
                f,
                "oxideav-speex: clean-room rebuild in progress — frame decoder/encoder not yet wired up"
            ),
            Error::Header(e) => write!(f, "oxideav-speex: {}", e),
            Error::Frame(e) => write!(f, "oxideav-speex: {}", e),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::NotImplemented => None,
            Error::Header(e) => Some(e),
            Error::Frame(e) => Some(e),
        }
    }
}

impl From<HeaderError> for Error {
    fn from(e: HeaderError) -> Self {
        Error::Header(e)
    }
}

impl From<FrameError> for Error {
    fn from(e: FrameError) -> Self {
        Error::Frame(e)
    }
}

/// No-op codec registration — the rebuild has no encoder/decoder to
/// hand to the runtime context yet. Header parsing is exposed as a free
/// function via [`SpeexHeader::parse`].
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("speex", register);
