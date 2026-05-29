//! # oxideav-speex
//!
//! **Status:** clean-room rebuild in progress (post 2026-05-19 audit).
//!
//! Layered scope as the rebuild advances:
//!
//! * **Round 1** — the Ogg/Speex stream-header packet parser from
//!   *The Speex Codec Manual* §7.3, Table 7.1 (validates `Speex   `
//!   magic + decodes 13 LE int32 fields). See [`SpeexHeader`].
//! * **Round 2** — the per-frame leading prefix (1-bit wideband flag +
//!   4-bit mode ID) from §9.3 + the typed narrowband sub-mode table
//!   from Table 9.1 (modes 0..=8, plus the three §5.5
//!   reserved-for-signalling slots: custom in-band mode 13, in-band
//!   signalling mode 14, terminator mode 15). See
//!   [`NarrowbandFrameHeader`] and [`Submode`].
//! * **Round 3** — the narrowband CELP frame-body bit-reader: walks
//!   Table 9.1's columns in the order documented in §9.3 ("All
//!   frame-based parameters are packed before sub-frame parameters.
//!   The parameters for a certain sub-frame are all packed before the
//!   following sub-frame is packed.") and surfaces every field as a
//!   raw bit-index in [`NarrowbandFrameBody`]. Codebook lookup (LSP VQ
//!   → coefficients, innovation VQ → 40-sample sub-vector) and LSP→LPC
//!   conversion stay deferred until the Speex CELP companion table
//!   docs gap closes.
//! * **Round 4** — the §5.5 in-band signalling bodies that the
//!   round-2 dispatcher leaves unparsed: [`InbandMessage`] walks
//!   mode 14's 4-bit Table 5.1 code + 1/4/8/16/32/64-bit payload;
//!   [`CustomInbandMessage`] consumes mode 13's 5-bit byte size +
//!   opaque payload. Table 5.1 itself is staged as the public
//!   [`INBAND_TABLE_5_1`] array indexed by code value. The §5.5 path
//!   needs no CELP companion tables — it is bit-stream framing only.
//! * **Round r165** (this commit) — typed packet → frame iterator
//!   composing the round-2..5 primitives end-to-end. See
//!   [`PacketFrames`], [`PacketFrame`], and the [`parse_packet`]
//!   convenience that returns a `Vec<PacketFrame>`. Walks a Speex
//!   packet body per §5.5 ("Sometimes it is desirable to pack more
//!   than one frame per packet … it is possible to include a
//!   terminator code. That terminator consists of the code 15
//!   (decimal) encoded with 5 bits"), dispatching each successive
//!   5-bit prefix into a CELP frame, a wideband narrowband+high-band
//!   pair, a §5.5 in-band signalling message, or a mode-13 custom
//!   in-band message, and terminating cleanly on mode 15 or on
//!   <5-bit padding tail.
//! * **Round 5** — the wideband high-band sub-mode
//!   table from §10.4 / Table 10.1 (modes 0..=4, 5 columns) plus a
//!   high-band frame-body bit-reader. A wideband packet is the
//!   concatenation of an embedded narrowband frame (round 3) followed
//!   by a 1-bit wideband flag, a 3-bit high-band mode ID, the 12-bit
//!   LSP MSVQ index (modes 1..=4), and four sub-frames of `gain || VQ`
//!   bits as listed in Table 10.1. See [`WidebandHighBandFrameHeader`]
//!   and [`WidebandHighBandBody`]. As with round 3 the body lands
//!   only the raw bit indices — the high-band LSP MSVQ codebook +
//!   innovation codebook are #969-blocked until staged.
//! * **Round r187** (this commit) — structured `write` methods
//!   symmetric to the existing `parse` paths for the three
//!   framing-level types whose layout is fully defined by the
//!   manual without any CELP companion-table material:
//!   [`NarrowbandFrameHeader::write`] emits the 5-bit prefix
//!   (1-bit wideband flag + 4-bit mode ID per §9.3);
//!   [`InbandMessage::write`] emits the 4-bit Table 5.1 code +
//!   1/4/8/16/32/64-bit payload (with the same >32-bit split path
//!   the parser uses for reserved codes 14 / 15);
//!   [`CustomInbandMessage::write`] emits the 5-bit `size_bytes`
//!   per §5.5 + opaque payload bytes from a caller-supplied
//!   slice. A new [`NarrowbandFrameHeader::new`] constructor is
//!   the encoder-side counterpart of the round-2 parser's
//!   reserved-mode rejection. All three writers depend only on
//!   the round-179 [`BitWriter`] + existing dispatch tables —
//!   no companion-table material is touched.
//!
//! Frame decode, encoder, and the `Decoder` / `Encoder` trait wiring
//! against `oxideav-core` still return [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

mod bitreader;
mod frame;
mod header;
mod narrowband_body;
mod packet;
mod signalling;
mod submode;
mod wideband;

pub use bitreader::{BitError, BitReader, BitWriter};
pub use frame::{FrameError, NarrowbandFrameHeader, NARROWBAND_FRAME_PREFIX_BITS};
pub use header::{
    HeaderError, SpeexHeader, SPEEX_HEADER_LEN, SPEEX_MAGIC, SPEEX_MODE_NARROWBAND,
    SPEEX_MODE_ULTRAWIDEBAND, SPEEX_MODE_WIDEBAND, SPEEX_STRING_LEN, SPEEX_VERSION_LEN,
};
pub use narrowband_body::{
    NarrowbandBodyError, NarrowbandFrameBody, NarrowbandSubFrameIndices, PITCH_PERIOD_MAX,
    PITCH_PERIOD_MIN,
};
pub use packet::{parse_packet, PacketError, PacketFrame, PacketFrames};
pub use signalling::{
    inband_code_spec, CustomInbandMessage, InbandCodeSpec, InbandKind, InbandMessage,
    SignallingError, CUSTOM_INBAND_MAX_BYTES, CUSTOM_INBAND_SIZE_BITS, INBAND_CODE_BITS,
    INBAND_TABLE_5_1,
};
pub use submode::{LspQuant, NarrowbandSubmode, PitchGainQuant, Submode, NARROWBAND_SUBMODES};
pub use wideband::{
    HighBandSubFrameIndices, WidebandBodyError, WidebandHighBandBody, WidebandHighBandFrameHeader,
    WidebandHighBandSubmode, WidebandSubmode, HIGH_BAND_FRAME_PREFIX_BITS,
    HIGH_BAND_SUBFRAMES_PER_FRAME, WIDEBAND_HIGH_BAND_SUBMODES,
};

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
    /// The narrowband CELP frame body failed to parse — see
    /// [`NarrowbandBodyError`].
    NarrowbandBody(NarrowbandBodyError),
    /// A §5.5 in-band signalling body failed to parse — see
    /// [`SignallingError`].
    Signalling(SignallingError),
    /// A wideband high-band frame body failed to parse — see
    /// [`WidebandBodyError`].
    Wideband(WidebandBodyError),
    /// A whole-packet walk failed mid-frame — see [`PacketError`].
    Packet(PacketError),
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
            Error::NarrowbandBody(e) => write!(f, "oxideav-speex: {}", e),
            Error::Signalling(e) => write!(f, "oxideav-speex: {}", e),
            Error::Wideband(e) => write!(f, "oxideav-speex: {}", e),
            Error::Packet(e) => write!(f, "oxideav-speex: {}", e),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::NotImplemented => None,
            Error::Header(e) => Some(e),
            Error::Frame(e) => Some(e),
            Error::NarrowbandBody(e) => Some(e),
            Error::Signalling(e) => Some(e),
            Error::Wideband(e) => Some(e),
            Error::Packet(e) => Some(e),
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

impl From<NarrowbandBodyError> for Error {
    fn from(e: NarrowbandBodyError) -> Self {
        Error::NarrowbandBody(e)
    }
}

impl From<SignallingError> for Error {
    fn from(e: SignallingError) -> Self {
        Error::Signalling(e)
    }
}

impl From<WidebandBodyError> for Error {
    fn from(e: WidebandBodyError) -> Self {
        Error::Wideband(e)
    }
}

impl From<PacketError> for Error {
    fn from(e: PacketError) -> Self {
        Error::Packet(e)
    }
}

/// No-op codec registration — the rebuild has no encoder/decoder to
/// hand to the runtime context yet. Header parsing is exposed as a free
/// function via [`SpeexHeader::parse`].
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("speex", register);
