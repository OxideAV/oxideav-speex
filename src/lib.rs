//! # oxideav-speex
//!
//! **Status:** orphan-rebuild scaffold (post 2026-05-19 audit).
//!
//! The prior implementation was retired under the workspace clean-room
//! policy. The crate will be re-implemented from scratch against the
//! published Speex specifications + RFC 5574 in a future clean-room
//! round.
//!
//! Every public API currently returns [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

/// Crate-local error type. Until the clean-room rebuild lands every
/// public API path returns [`Error::NotImplemented`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The crate has been reset to a scaffold pending clean-room
    /// rebuild; no decoder or encoder functionality is wired up yet.
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "oxideav-speex: orphan-rebuild scaffold — no decoder/encoder wired up"
        )
    }
}

impl std::error::Error for Error {}

/// No-op codec registration — the orphan-rebuild scaffold registers
/// nothing into the runtime context.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("speex", register);
