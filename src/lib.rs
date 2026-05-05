//! Speex (CELP speech codec) — pure-Rust decoder + encoder with Ogg
//! integration.
//!
//! Implements:
//!   * Bit-exact 80-byte Speex header parser (Speex-in-Ogg mapping).
//!   * MSB-first bit reader / writer matching `libspeex/bits.c`.
//!   * Mode + sub-mode descriptors (NB 0..=8).
//!   * Float-mode CELP decoder for narrowband (NB) streams covering
//!     sub-modes 1..=7 (silence/vocoder, 5.95k, 8k, 11k, 15k, 18.2k,
//!     24.6k) and sub-mode 8 (3.95k vocoder + algebraic codebook).
//!   * Float-mode sub-band CELP decoder for wideband (16 kHz) streams
//!     — WB sub-modes 1..=4, QMF synthesis bank, high-band LSP & LPC
//!     synthesis, stochastic codebook + spectral-folding excitation.
//!     See [`wb_decoder`] and [`qmf`].
//!   * Ultra-wideband (32 kHz) SB-CELP decoder — stacks a second
//!     8th-order LPC + spectral-folding layer on top of the wideband
//!     decoder and runs a second QMF synthesis stage to produce 32
//!     kHz mono output. Matches `sb_uwb_mode` in `libspeex/modes.c`.
//!     Only sub-mode 1 (folding, 36 bits of UWB overhead) is defined
//!     by the reference; stochastic UWB sub-modes do not exist. See
//!     [`uwb_decoder`].
//!   * NB encoder for the **full rate ladder** — sub-modes 1..=8 (2.15
//!     kbps vocoder through 24.6 kbps high-rate; default is sub-mode 5
//!     at 15 kbps). Encode dispatch is driven by the
//!     [`submodes::NbSubmode`] descriptor so adding a decoder mode
//!     never requires a parallel encoder change.
//!   * WB encoder covering the **full WB sub-mode ladder** — sub-modes
//!     1 (folding, 336 b), 2 (LBR split-VQ, 412 b), 3 (stochastic
//!     split-VQ, 492 b — default) and 4 (double-codebook split-VQ,
//!     652 b). UWB encoder for the null layer and the folding layer
//!     (default). See [`encoder`], [`nb_encoder`], [`wb_encoder`],
//!     [`uwb_encoder`].
//!   * Intensity-stereo decode ([`stereo`]) — the 8-bit
//!     `m=14, id=9` in-band side-channel pre-pended to each frame
//!     by the reference encoder expands the mono CELP output into
//!     interleaved L/R.
//!   * Typed in-band signalling per Speex manual §5.5 / RFC 5574 —
//!     [`inband::InbandMessage`] covers all 16 `m=14` codes from
//!     Table 5.1 plus the `m=13` user payload and the `m=15` frame
//!     terminator. [`inband::pad_to_octet_boundary`] writes the
//!     RFC 5574 §3.3 padding pattern (a single `0` followed by all
//!     ones to the next octet boundary). The CELP decoders still skip
//!     unrecognised in-band requests opaquely (matching libspeex's
//!     "no callback registered" default); typed parsing is opt-in.
//!
//! Tables (LSP, gain, fixed codebooks) are transcribed from the
//! BSD-licensed Xiph reference (`libspeex/{lsp_tables_nb,gain_table,
//! exc_*_table}.c`) — values only, no derived code.
//!
//! References:
//!   * <https://www.speex.org/docs/manual/speex-manual.pdf>
//!   * RFC 5574 — RTP payload format for Speex.
//!   * <https://github.com/xiph/speex>

#![allow(
    clippy::needless_range_loop,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::manual_range_contains
)]

pub mod codec;
pub mod decoder;
pub mod encoder;
pub mod exc_tables;
pub mod gain_tables;
pub mod header;
pub mod hexc_tables;
pub mod inband;
pub mod lsp;
pub mod lsp_tables_nb;
pub mod lsp_tables_wb;
pub mod nb_decoder;
pub mod nb_encoder;
pub mod qmf;
pub mod stereo;
pub mod submodes;
pub mod uwb_decoder;
pub mod uwb_encoder;
pub mod wb_decoder;
pub mod wb_encoder;
pub mod wb_submodes;

use oxideav_core::CodecRegistry;

pub const CODEC_ID_STR: &str = "speex";

/// Register Speex with the supplied [`CodecRegistry`]. Prefer the
/// unified [`register`] entry point when you have a
/// [`oxideav_core::RuntimeContext`] in hand.
pub fn register_codecs(reg: &mut CodecRegistry) {
    codec::register_codecs(reg);
}

/// Unified registration entry point — installs Speex into the codec
/// sub-registry of the supplied [`oxideav_core::RuntimeContext`].
pub fn register(ctx: &mut oxideav_core::RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

oxideav_core::register!("speex", register);

#[cfg(test)]
mod register_tests {
    use super::*;
    use oxideav_core::CodecId;

    #[test]
    fn register_via_runtime_context_installs_codec_factory() {
        let mut ctx = oxideav_core::RuntimeContext::new();
        register(&mut ctx);
        let id = CodecId::new(CODEC_ID_STR);
        assert!(
            ctx.codecs.has_decoder(&id),
            "decoder factory not installed via RuntimeContext"
        );
        assert!(
            ctx.codecs.has_encoder(&id),
            "encoder factory not installed via RuntimeContext"
        );
    }
}
