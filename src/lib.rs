//! # oxideav-speex
//!
//! **Status:** clean-room rebuild in progress (post 2026-05-19 audit).
//!
//! Round-1 scope (this commit): the Ogg/Speex stream-header packet
//! parser from *The Speex Codec Manual* §7.3, Table 7.1. The parser
//! validates the `Speex   ` magic, decodes every header field as a
//! little-endian integer per the spec, and surfaces the fields as a
//! [`SpeexHeader`] struct. No frame decode is wired up yet — that
//! comes in a future round.
//!
//! Frame decode, encoder, and the `Decoder` / `Encoder` trait wiring
//! against `oxideav-core` still return [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

mod header;

pub use header::{
    HeaderError, SpeexHeader, SPEEX_HEADER_LEN, SPEEX_MAGIC, SPEEX_MODE_NARROWBAND,
    SPEEX_MODE_ULTRAWIDEBAND, SPEEX_MODE_WIDEBAND, SPEEX_STRING_LEN, SPEEX_VERSION_LEN,
};

/// Crate-local error type. Until the full clean-room rebuild lands, the
/// codec-level public API paths return [`Error::NotImplemented`]; the
/// Ogg/Speex header parser surfaces its own [`HeaderError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The crate's frame decoder / encoder has not been re-implemented
    /// against the spec yet.
    NotImplemented,
    /// The Ogg/Speex stream header failed to parse — see [`HeaderError`].
    Header(HeaderError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotImplemented => write!(
                f,
                "oxideav-speex: clean-room rebuild in progress — frame decoder/encoder not yet wired up"
            ),
            Error::Header(e) => write!(f, "oxideav-speex: {}", e),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::NotImplemented => None,
            Error::Header(e) => Some(e),
        }
    }
}

impl From<HeaderError> for Error {
    fn from(e: HeaderError) -> Self {
        Error::Header(e)
    }
}

/// No-op codec registration — the rebuild has no encoder/decoder to
/// hand to the runtime context yet. Header parsing is exposed as a free
/// function via [`SpeexHeader::parse`].
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("speex", register);
