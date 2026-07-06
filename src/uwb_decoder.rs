//! **Ultra-wideband (32 kHz) packet decoder** — walks the §2.2 embedded
//! recursion one level above wideband: each ultra-wideband frame is an
//! embedded **wideband frame** (narrowband layer + first high-band
//! layer, decoded by [`crate::WidebandDecoder`]) followed by a **second
//! high-band layer** covering 8–16 kHz, recombined to 32 kHz PCM by an
//! outer QMF synthesis bank (round r389 scope).
//!
//! ## On-wire layout (per frame)
//!
//! ```text
//!  ultra-wideband frame
//!    ├── 5-bit NB prefix (wideband flag=1 || 4-bit NB mode ID)
//!    ├── narrowband Table 9.1 body                       (0–4 kHz)
//!    ├── 4-bit HB prefix + Table 10.1 body               (4–8 kHz)
//!    └── 4-bit HB prefix + Table 10.1 body               (8–16 kHz)
//! ```
//!
//! The recursion marker and grammar are the wideband ones repeated one
//! level up ([`crate::UwbFrameLayout`]): the manual has **no** dedicated
//! ultra-wideband chapter, but the second layer's shape is pinned by
//! RFC 5574 Table 2 — the ultra-wideband bit-rate sits a constant
//! +1,800 bit/s (= 36 bits/frame at 50 frames/s) above the wideband
//! bit-rate at **every** quality, and 36 bits is exactly (and uniquely)
//! the Table 10.1 **mode-1** total (see [`crate::quality`]). A
//! conformant stream's second layer is therefore the gain-only mode-1
//! frame: 12-bit LSP MSVQ envelope + four 5-bit excitation gains, no
//! excitation VQ.
//!
//! ## Rate-class context, not in-band detection
//!
//! After the first high-band body, the bits of a second high-band
//! prefix (`1` + 3-bit mode) are indistinguishable from the next
//! frame's narrowband prefix (`1` + the top of a 4-bit mode) — the
//! §5.5 concatenation is only walkable when the packet's rate class is
//! known out-of-band, exactly what the stream header's `mode` field
//! provides (§7.3, [`crate::SpeexHeader::mode`] = 2 for
//! ultra-wideband). This decoder therefore *is* the rate-class context:
//! feed it only packets from an ultra-wideband stream.
//!
//! ## What the second layer decodes to
//!
//! The staged material pins the second layer's **framing** (prefix +
//! mode-1 field widths), its **envelope** (the same 12-bit two-stage
//! LSP MSVQ, §10.1), and its **gain** track (the 32-level 5-bit
//! folded-excitation gain quantiser, staged as `hb-fold-quant-bound`).
//! As of round r393 the mode-1 folded reconstruction **law** is pinned
//! by the wideband fixture (`crate::hb_fold`:
//! `e_hb[n] = K·g·(−1)ⁿ·e_src[n]`), and this decoder reconstructs a
//! real 8–16 kHz half-band with it. The fold **source** at the 16 kHz
//! half-band geometry is *not* covered by that (wideband-only)
//! fixture; this decoder applies the **recursion-consistent
//! generalisation** — each layer folds its embedded lower layer's
//! excitation, so the second layer's source is the QMF-synthesis
//! recombination (16 kHz) of the wideband layer's two 8 kHz excitation
//! tracks (narrowband composed excitation + first-high-band
//! excitation). That source choice is the crate's documented
//! convention, recorded as the remaining sub-gap of #170 pending an
//! ultra-wideband fixture. The excitation-VQ modes 2..=4 additionally
//! lack a pinned sub-frame geometry at the 16 kHz half-band rate
//! (Table 10.1's VQ budgets are stated for the 8 kHz half-band) and
//! surface [`UwbDecodeError::UwbLayerUndocumented`].

use crate::bitreader::BitReader;
use crate::codebooks::HB_LPC_ORDER;
use crate::decoder::ControlMessage;
use crate::frame::{NarrowbandFrameHeader, NARROWBAND_FRAME_PREFIX_BITS};
use crate::gain_reconstruction::reconstruct_hb_exc_gain;
use crate::hb_excitation_gain::HbExcitationGainIndex;
use crate::hb_fold::folded_hb_excitation_slice;
use crate::hb_lsp_interp::HbSubFrameLsp;
use crate::hb_synthesis::HbSynthesisFilter;
use crate::lsp_to_lpc::hb_subframe_lpc_set_with_base;
use crate::qmf::{QmfSynthesis, QMF_WIDEBAND_FRAME};
use crate::signalling::{CustomInbandMessage, InbandMessage};
use crate::submode::Submode;
use crate::wb_synthesis::HB_FRAME_SAMPLES;
use crate::wideband::{
    WidebandHighBandBody, WidebandHighBandFrameHeader, WidebandSubmode,
    HIGH_BAND_SUBFRAMES_PER_FRAME,
};
use crate::wideband_decoder::{WidebandDecodeError, WidebandDecoder, WidebandFrame};
use core::fmt;

/// Samples per 20 ms ultra-wideband frame at 32 kHz.
pub const UWB_FRAME_SAMPLES: usize = 2 * QMF_WIDEBAND_FRAME;

/// Samples per 16 kHz half-band frame of the outer QMF split (each half
/// of a 640-sample 32 kHz frame) — equal to one wideband frame.
pub const UWB_HALF_BAND_FRAME: usize = QMF_WIDEBAND_FRAME;

/// Sub-frames per ultra-wideband second high-band layer — the same four
/// per-frame gain slots as the wideband layer (the mode-1 36-bit total
/// pins 4 × 5 gain bits; see [`crate::quality`]).
pub const UWB_HB_SUBFRAMES: usize = HIGH_BAND_SUBFRAMES_PER_FRAME as usize;

/// One decoded ultra-wideband audio frame.
#[derive(Debug, Clone)]
pub struct UltraWidebandFrame {
    /// The embedded wideband layers (narrowband + first high band),
    /// including the recombined 16 kHz PCM that becomes the outer QMF's
    /// low band.
    pub wideband: WidebandFrame,
    /// The second high-band layer's Table 10.1 sub-mode ID (mode 1 in
    /// every conformant stream — module docs).
    pub uwb_hb_mode: u8,
    /// The second high-band layer's parsed body (12-bit LSP MSVQ index
    /// + per-sub-frame gain indices).
    pub uwb_hb_body: WidebandHighBandBody,
    /// The reconstructed per-sub-frame excitation gains of the second
    /// layer (the staged 32-level folded-gain law for mode 1; `0.0` for
    /// the silence mode) — the energy envelope the bit-stream carries
    /// for the 8–16 kHz band.
    pub uwb_gains: [f32; UWB_HB_SUBFRAMES],
    /// The reconstructed 8–16 kHz half-band signal at 16 kHz — the
    /// folded second-layer excitation through the order-8 half-band
    /// synthesis filter (module docs; fold law `crate::hb_fold`).
    pub uwb_high_band: Box<[f64; UWB_HALF_BAND_FRAME]>,
    /// The outer-QMF-recombined 640-sample 32 kHz PCM.
    pub uwb_pcm: Box<[f64; UWB_FRAME_SAMPLES]>,
}

impl UltraWidebandFrame {
    /// The 32 kHz PCM, rounded + saturated to `i16` through the
    /// crate-wide [`crate::saturate_i16`] convention.
    pub fn uwb_pcm_i16(&self) -> [i16; UWB_FRAME_SAMPLES] {
        let mut out = [0i16; UWB_FRAME_SAMPLES];
        for (o, &s) in out.iter_mut().zip(self.uwb_pcm.iter()) {
            *o = crate::saturate_i16(s);
        }
        out
    }
}

/// One decoded unit from an ultra-wideband packet: an audio frame or a
/// §5.5 control pseudo-frame (the same [`ControlMessage`] the top-level
/// [`crate::SpeexDecoder`] surfaces).
#[derive(Debug, Clone)]
pub enum UwbDecodedFrame {
    /// A decoded 640-sample 32 kHz audio frame.
    Audio(Box<UltraWidebandFrame>),
    /// A §5.5 in-band signalling / custom pseudo-frame (no audio).
    Control(ControlMessage),
}

/// Errors from [`UltraWidebandDecoder::decode_packet`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UwbDecodeError {
    /// The embedded wideband layers failed to decode.
    Wideband(WidebandDecodeError),
    /// The second high-band layer's 3-bit mode is a reserved high-rate
    /// slot (5..=7) with no documented bit budget.
    UwbLayerReserved {
        /// The reserved mode ID.
        mode_id: u8,
    },
    /// The second high-band layer uses an excitation-VQ mode (2..=4)
    /// whose sub-frame geometry at the 16 kHz half-band rate is not
    /// pinned by the staged material (module docs) — a recorded docs
    /// gap, not a malformed stream.
    UwbLayerUndocumented {
        /// The undocumented mode ID.
        mode_id: u8,
    },
    /// A low-level bit-reader / framing failure (truncated packet,
    /// malformed control body, …).
    Framing,
}

impl fmt::Display for UwbDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UwbDecodeError::Wideband(e) => write!(f, "ultra-wideband decode: {e}"),
            UwbDecodeError::UwbLayerReserved { mode_id } => write!(
                f,
                "ultra-wideband decode: second high-band mode {mode_id} is the reserved slot"
            ),
            UwbDecodeError::UwbLayerUndocumented { mode_id } => write!(
                f,
                "ultra-wideband decode: second high-band mode {mode_id} geometry is a docs gap"
            ),
            UwbDecodeError::Framing => {
                write!(f, "ultra-wideband decode: framing/bit-reader failure")
            }
        }
    }
}

impl std::error::Error for UwbDecodeError {}

impl From<WidebandDecodeError> for UwbDecodeError {
    fn from(e: WidebandDecodeError) -> Self {
        UwbDecodeError::Wideband(e)
    }
}

/// Stateful ultra-wideband (32 kHz) decoder.
///
/// Holds the embedded [`WidebandDecoder`] (narrowband + first high-band
/// state, inner QMF) plus the **outer** QMF synthesis bank recombining
/// the 16 kHz wideband PCM with the 8–16 kHz half-band into 32 kHz
/// output. Construct with [`UltraWidebandDecoder::new`], then call
/// [`UltraWidebandDecoder::decode_packet`] once per packet of an
/// ultra-wideband stream ([`crate::SpeexHeader::mode`] = 2).
#[derive(Debug, Clone)]
pub struct UltraWidebandDecoder {
    wideband: WidebandDecoder,
    outer_qmf: QmfSynthesis,
    /// QMF synthesis bank recombining the embedded wideband layer's two
    /// excitation tracks (narrowband + first high band, both 8 kHz)
    /// into the 16 kHz **fold source** for the second layer (see
    /// `decode_audio_frame`). Persistent FIR history, advanced every
    /// audio frame regardless of the second layer's sub-mode so the
    /// fold source stays continuous across mode switches.
    exc_qmf: QmfSynthesis,
    /// Second-layer order-8 synthesis filter (8–16 kHz half-band IIR
    /// memory, carried across frames).
    layer2_filter: HbSynthesisFilter,
    /// Previous frame's second-layer LSP codebook-delta vector (Q10,
    /// pre-base) for the §9.1 interpolation, mirroring the wideband
    /// high-band convention.
    prev_layer2_lsp_delta_q10: Option<[i32; HB_LPC_ORDER]>,
}

impl Default for UltraWidebandDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl UltraWidebandDecoder {
    /// A fresh decoder at stream start (zero history in every band).
    pub fn new() -> Self {
        Self {
            wideband: WidebandDecoder::new(),
            outer_qmf: QmfSynthesis::new(),
            exc_qmf: QmfSynthesis::new(),
            layer2_filter: HbSynthesisFilter::new(),
            prev_layer2_lsp_delta_q10: None,
        }
    }

    /// Decode every frame in an ultra-wideband packet.
    ///
    /// Walks the §5.5 concatenation — audio frames, in-band signalling
    /// / custom pseudo-frames, the mode-15 terminator, and the
    /// sub-prefix padding tail — decoding each audio frame's three
    /// sub-band layers and recombining to 32 kHz PCM.
    pub fn decode_packet(&mut self, packet: &[u8]) -> Result<Vec<UwbDecodedFrame>, UwbDecodeError> {
        let mut reader = BitReader::new(packet);
        let mut out = Vec::new();
        loop {
            // §5.5 padding tail: fewer bits than a prefix = end.
            if reader.remaining_bits() < NARROWBAND_FRAME_PREFIX_BITS {
                return Ok(out);
            }
            let header =
                NarrowbandFrameHeader::parse(&mut reader).map_err(|_| UwbDecodeError::Framing)?;
            match header.submode {
                Submode::Terminator => return Ok(out),
                Submode::InbandSignalling => {
                    let message =
                        InbandMessage::parse(&mut reader).map_err(|_| UwbDecodeError::Framing)?;
                    let request = message.interpret();
                    out.push(UwbDecodedFrame::Control(ControlMessage::Inband {
                        message,
                        request,
                    }));
                }
                Submode::CustomInband => {
                    let message = CustomInbandMessage::parse(&mut reader)
                        .map_err(|_| UwbDecodeError::Framing)?;
                    out.push(UwbDecodedFrame::Control(ControlMessage::Custom {
                        size_bytes: message.size_bytes,
                    }));
                }
                Submode::Celp(_) => {
                    let frame = self.decode_audio_frame(&header, &mut reader)?;
                    out.push(UwbDecodedFrame::Audio(Box::new(frame)));
                }
            }
        }
    }

    /// Decode a whole packet straight to flat 32 kHz `i16` PCM (640
    /// samples per audio frame), skipping control pseudo-frames.
    pub fn decode_packet_pcm_i16(&mut self, packet: &[u8]) -> Result<Vec<i16>, UwbDecodeError> {
        let frames = self.decode_packet(packet)?;
        let mut pcm = Vec::new();
        for f in &frames {
            if let UwbDecodedFrame::Audio(a) = f {
                pcm.extend_from_slice(&a.uwb_pcm_i16());
            }
        }
        Ok(pcm)
    }

    /// Decode one audio frame's three layers off the live cursor (the
    /// 5-bit narrowband prefix is already consumed).
    fn decode_audio_frame(
        &mut self,
        nb_header: &NarrowbandFrameHeader,
        reader: &mut BitReader<'_>,
    ) -> Result<UltraWidebandFrame, UwbDecodeError> {
        // Layers 1+2: the embedded wideband frame (narrowband + first
        // high band + inner QMF recombination).
        let wideband = self.wideband.decode_frame_after_header(nb_header, reader)?;

        // Layer 3: the second high-band frame (same Table 10.1 grammar).
        let hb_header =
            WidebandHighBandFrameHeader::parse(reader).map_err(|_| UwbDecodeError::Framing)?;
        let submode = match hb_header.submode {
            WidebandSubmode::Documented(s) => s,
            WidebandSubmode::ReservedHighRate(id) => {
                return Err(UwbDecodeError::UwbLayerReserved { mode_id: id })
            }
        };
        let body =
            WidebandHighBandBody::parse(reader, &submode).map_err(|_| UwbDecodeError::Framing)?;
        // The excitation-VQ modes' sub-frame geometry at the 16 kHz
        // half-band is not pinned (module docs) — surface the gap after
        // the parse so the cursor diagnostics stay sane.
        if submode.excitation_vq_bits > 0 {
            return Err(UwbDecodeError::UwbLayerUndocumented {
                mode_id: submode.mode_id,
            });
        }

        // Reconstruct the per-sub-frame gains the layer carries (the
        // staged folded-gain law).
        let mut uwb_gains = [0.0f32; UWB_HB_SUBFRAMES];
        if submode.excitation_gain_bits > 0 {
            for (g, sf) in uwb_gains.iter_mut().zip(body.subframes.iter()) {
                if let Some(idx) =
                    HbExcitationGainIndex::resolve(sf.excitation_gain_index, &submode)
                {
                    *g = reconstruct_hb_exc_gain(idx);
                }
            }
        }

        // --- Second-layer fold source (always advanced, every audio
        // frame, so the recombination history stays continuous across
        // sub-mode switches): the embedded wideband layer's two
        // excitation tracks recombined to 16 kHz through the QMF
        // synthesis bank. The fold *law* is the wideband-fixture-pinned
        // one (`crate::hb_fold`); this recursion-consistent *source*
        // generalisation is the crate's — the staged fixture is
        // wideband-only, so the reference's exact 16 kHz fold-source
        // geometry stays a recorded gap (module docs). ---
        let mut exc_lb64 = [0.0f64; HB_FRAME_SAMPLES];
        for (o, &e) in exc_lb64
            .iter_mut()
            .zip(self.wideband.low_band_decoder().last_frame_excitation())
        {
            *o = f64::from(e);
        }
        let mut exc_hb64 = [0.0f64; HB_FRAME_SAMPLES];
        for (o, &e) in exc_hb64.iter_mut().zip(self.wideband.last_hb_excitation()) {
            *o = f64::from(e);
        }
        let exc16 = self.exc_qmf.reconstruct_frame(&exc_lb64, &exc_hb64);

        // --- Second-layer synthesis: per-sub-frame interpolated LPC +
        // folded excitation through the order-8 half-band filter. ---
        let curr_lsp = body.reconstructed_lsp_q10(&submode);
        let lpc_sets = match curr_lsp {
            Some(curr) => {
                let sub_lsp = match self.prev_layer2_lsp_delta_q10 {
                    Some(prev) => HbSubFrameLsp::new(&prev, &curr),
                    None => HbSubFrameLsp::first_frame(&curr),
                };
                hb_subframe_lpc_set_with_base(&sub_lsp)
            }
            // Silence (mode 0, no LSP field): absent-envelope
            // convention, A(z) = 1.
            None => [[0.0; HB_LPC_ORDER]; UWB_HB_SUBFRAMES],
        };
        let subframe_len = UWB_HALF_BAND_FRAME / UWB_HB_SUBFRAMES;
        let mut uwb_high_band = Box::new([0.0f64; UWB_HALF_BAND_FRAME]);
        for sf in 0..UWB_HB_SUBFRAMES {
            let range = sf * subframe_len..(sf + 1) * subframe_len;
            let mut e = vec![0.0f64; subframe_len];
            if submode.excitation_gain_bits > 0 {
                folded_hb_excitation_slice(&exc16[range.clone()], uwb_gains[sf], &mut e);
            }
            self.layer2_filter
                .process(&lpc_sets[sf], &e, &mut uwb_high_band[range]);
        }
        if let Some(curr) = curr_lsp {
            self.prev_layer2_lsp_delta_q10 = Some(curr);
        }

        // Outer QMF: 16 kHz wideband PCM (low half) + 8–16 kHz band
        // (high half) → 640-sample 32 kHz PCM.
        let mut uwb_pcm = Box::new([0.0f64; UWB_FRAME_SAMPLES]);
        self.outer_qmf.reconstruct_slices(
            &wideband.wideband_pcm,
            uwb_high_band.as_ref(),
            uwb_pcm.as_mut(),
        );

        Ok(UltraWidebandFrame {
            wideband,
            uwb_hb_mode: submode.mode_id,
            uwb_hb_body: body,
            uwb_gains,
            uwb_high_band,
            uwb_pcm,
        })
    }

    /// Read-only view of the embedded wideband decoder (diagnostics).
    pub fn wideband_decoder(&self) -> &WidebandDecoder {
        &self.wideband
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitWriter;
    use crate::submode::NarrowbandSubmode;
    use crate::wideband::WIDEBAND_HIGH_BAND_SUBMODES;

    fn write_ones(w: &mut BitWriter, mut bits: u32) {
        while bits > 0 {
            let chunk = bits.min(16);
            w.write((1u32 << chunk) - 1, chunk).unwrap();
            bits -= chunk;
        }
    }

    /// Append one ultra-wideband frame: all-ones NB body (mode `nb`),
    /// all-ones first HB body (mode `hb1`), and a second HB layer of
    /// mode `hb2` with the given LSP index and per-sub-frame gains.
    fn push_uwb_frame(w: &mut BitWriter, nb: u8, hb1: u8, hb2: u8, lsp: u16, gains: [u8; 4]) {
        let nb_submode = NarrowbandSubmode::for_id(nb).unwrap();
        let hb1_submode = WIDEBAND_HIGH_BAND_SUBMODES[hb1 as usize];
        // NB prefix (wideband flag 1) + all-ones body.
        w.write_bit(1).unwrap();
        w.write(u32::from(nb), 4).unwrap();
        write_ones(w, u32::from(nb_submode.total_bits) - 5);
        // First HB layer, all-ones body.
        w.write_bit(1).unwrap();
        w.write(u32::from(hb1), 3).unwrap();
        write_ones(w, u32::from(hb1_submode.total_bits) - 4);
        // Second HB layer.
        w.write_bit(1).unwrap();
        w.write(u32::from(hb2), 3).unwrap();
        if hb2 == 1 {
            w.write(u32::from(lsp), 12).unwrap();
            for g in gains {
                w.write(u32::from(g), 5).unwrap();
            }
        }
    }

    fn one_frame_packet(hb2: u8) -> Vec<u8> {
        let mut w = BitWriter::new();
        push_uwb_frame(&mut w, 5, 2, hb2, 0x0ABC, [20, 25, 18, 31]);
        // Terminator + pad.
        w.write_bit(0).unwrap();
        w.write(15, 4).unwrap();
        while w.bits_written() % 8 != 0 {
            w.write_bit(0).unwrap();
        }
        w.into_bytes()
    }

    #[test]
    fn decodes_mode1_second_layer_to_32khz_frame() {
        let pkt = one_frame_packet(1);
        let mut dec = UltraWidebandDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            UwbDecodedFrame::Audio(f) => {
                assert_eq!(f.uwb_hb_mode, 1);
                assert_eq!(f.uwb_hb_body.lsp_index, 0x0ABC);
                assert_eq!(f.uwb_pcm.len(), 640);
                assert!(f.uwb_pcm.iter().all(|s| s.is_finite()));
                assert!(
                    f.uwb_pcm.iter().any(|&s| s != 0.0),
                    "wideband content reaches the 32 kHz output"
                );
                // r393: the second layer folds the embedded layers'
                // excitation — with non-zero gains and a non-silent
                // embedded frame the 8–16 kHz half-band is non-silent.
                assert!(f.uwb_high_band.iter().all(|s| s.is_finite()));
                assert!(
                    f.uwb_high_band.iter().any(|&s| s != 0.0),
                    "folded second layer should be non-silent"
                );
                // The carried gains reconstruct through the staged law.
                assert!(f.uwb_gains.iter().all(|&g| g > 0.0));
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn second_layer_silence_mode_decodes_with_zero_gains() {
        let pkt = one_frame_packet(0);
        let mut dec = UltraWidebandDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        match &frames[0] {
            UwbDecodedFrame::Audio(f) => {
                assert_eq!(f.uwb_hb_mode, 0);
                assert_eq!(f.uwb_gains, [0.0f32; 4]);
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn embedded_wideband_layers_match_standalone_wideband_decoder() {
        // Embedded scalability: a fresh WidebandDecoder fed the same
        // packet decodes the first two layers identically (it stops
        // before the second high-band layer).
        let pkt = one_frame_packet(1);

        let mut uwb = UltraWidebandDecoder::new();
        let frames = uwb.decode_packet(&pkt).unwrap();
        let uwb_frame = match &frames[0] {
            UwbDecodedFrame::Audio(f) => f,
            other => panic!("expected Audio, got {other:?}"),
        };

        let mut wb = WidebandDecoder::new();
        let wb_frame = wb.decode_packet(&pkt).unwrap();
        assert_eq!(wb_frame.low_band, uwb_frame.wideband.low_band);
        assert_eq!(wb_frame.high_band, uwb_frame.wideband.high_band);
        assert_eq!(wb_frame.wideband_pcm, uwb_frame.wideband.wideband_pcm);
    }

    #[test]
    fn multi_frame_packet_decodes_all_frames_statefully() {
        let mut w = BitWriter::new();
        push_uwb_frame(&mut w, 5, 2, 1, 0x0123, [10, 12, 14, 16]);
        push_uwb_frame(&mut w, 5, 2, 1, 0x0456, [8, 9, 10, 11]);
        w.write_bit(0).unwrap();
        w.write(15, 4).unwrap();
        while w.bits_written() % 8 != 0 {
            w.write_bit(0).unwrap();
        }
        let pkt = w.into_bytes();

        let mut dec = UltraWidebandDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        assert_eq!(frames.len(), 2);
        let (a, b) = match (&frames[0], &frames[1]) {
            (UwbDecodedFrame::Audio(a), UwbDecodedFrame::Audio(b)) => (a, b),
            other => panic!("expected two audio frames, got {other:?}"),
        };
        assert_eq!(a.uwb_hb_body.lsp_index, 0x0123);
        assert_eq!(b.uwb_hb_body.lsp_index, 0x0456);
        // State carries: the same bits decode differently on frame 2.
        assert_ne!(a.wideband.wideband_pcm, b.wideband.wideband_pcm);

        // Flat i16 output: 2 × 640 samples at 32 kHz.
        let mut dec2 = UltraWidebandDecoder::new();
        let pcm = dec2.decode_packet_pcm_i16(&pkt).unwrap();
        assert_eq!(pcm.len(), 1280);
    }

    #[test]
    fn vq_modes_surface_the_geometry_gap() {
        for hb2 in [2u8, 3] {
            let mut w = BitWriter::new();
            let nb_submode = NarrowbandSubmode::for_id(5).unwrap();
            let hb1_submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
            w.write_bit(1).unwrap();
            w.write(5, 4).unwrap();
            write_ones(&mut w, u32::from(nb_submode.total_bits) - 5);
            w.write_bit(1).unwrap();
            w.write(2, 3).unwrap();
            write_ones(&mut w, u32::from(hb1_submode.total_bits) - 4);
            // Second layer with a VQ mode + all-ones body.
            let hb2_submode = WIDEBAND_HIGH_BAND_SUBMODES[hb2 as usize];
            w.write_bit(1).unwrap();
            w.write(u32::from(hb2), 3).unwrap();
            write_ones(&mut w, u32::from(hb2_submode.total_bits) - 4);
            let pkt = w.into_bytes();

            let mut dec = UltraWidebandDecoder::new();
            let err = dec.decode_packet(&pkt).unwrap_err();
            assert_eq!(
                err,
                UwbDecodeError::UwbLayerUndocumented { mode_id: hb2 },
                "hb2 mode {hb2}"
            );
        }
    }

    #[test]
    fn reserved_second_layer_mode_is_surfaced() {
        let mut w = BitWriter::new();
        let nb_submode = NarrowbandSubmode::for_id(5).unwrap();
        let hb1_submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
        w.write_bit(1).unwrap();
        w.write(5, 4).unwrap();
        write_ones(&mut w, u32::from(nb_submode.total_bits) - 5);
        w.write_bit(1).unwrap();
        w.write(2, 3).unwrap();
        write_ones(&mut w, u32::from(hb1_submode.total_bits) - 4);
        w.write_bit(1).unwrap();
        w.write(6, 3).unwrap(); // reserved slot
        w.write(0, 7).unwrap();
        let pkt = w.into_bytes();

        let mut dec = UltraWidebandDecoder::new();
        let err = dec.decode_packet(&pkt).unwrap_err();
        assert_eq!(err, UwbDecodeError::UwbLayerReserved { mode_id: 6 });
    }

    #[test]
    fn control_frames_pass_through() {
        // In-band signalling (mode 14, code 0, payload 1) then
        // terminator.
        let mut w = BitWriter::new();
        w.write_bit(0).unwrap();
        w.write(14, 4).unwrap();
        w.write(0, 4).unwrap();
        w.write_bit(1).unwrap();
        w.write_bit(0).unwrap();
        w.write(15, 4).unwrap();
        while w.bits_written() % 8 != 0 {
            w.write_bit(0).unwrap();
        }
        let pkt = w.into_bytes();

        let mut dec = UltraWidebandDecoder::new();
        let frames = dec.decode_packet(&pkt).expect("decodes");
        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], UwbDecodedFrame::Control(_)));
        // The flat PCM path skips it.
        let mut dec2 = UltraWidebandDecoder::new();
        assert!(dec2.decode_packet_pcm_i16(&pkt).unwrap().is_empty());
    }

    #[test]
    fn empty_and_terminator_only_packets_are_clean() {
        let mut dec = UltraWidebandDecoder::new();
        assert!(dec.decode_packet(&[]).unwrap().is_empty());
        // Terminator in the top 5 bits.
        assert!(dec.decode_packet(&[0b0111_1000]).unwrap().is_empty());
    }

    #[test]
    fn decoder_is_deterministic() {
        let pkt = one_frame_packet(1);
        let mut a = UltraWidebandDecoder::new();
        let mut b = UltraWidebandDecoder::new();
        for _ in 0..3 {
            let fa = a.decode_packet(&pkt).unwrap();
            let fb = b.decode_packet(&pkt).unwrap();
            match (&fa[0], &fb[0]) {
                (UwbDecodedFrame::Audio(x), UwbDecodedFrame::Audio(y)) => {
                    assert_eq!(x.uwb_pcm, y.uwb_pcm);
                }
                other => panic!("expected audio frames, got {other:?}"),
            }
        }
    }
}
