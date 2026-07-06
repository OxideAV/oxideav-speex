//! **End-to-end wideband (sub-band CELP) decode loop** — decodes a
//! wideband packet into its two reconstructed 8 kHz half-band signals
//! (low band `x_lb[n]` + high band `x_hb[n]`).
//!
//! A wideband packet is the **embedded** concatenation of a narrowband
//! frame (the low band, 0–4 kHz) and a high-band frame (4–8 kHz folded
//! to an 8 kHz half-band), per *The Speex Codec Manual* §10:
//!
//! > "The 16 kHz signal is thus divided into two 8 kHz signals, one
//! > representing the low band (0–4 kHz), the other the high band
//! > (4–8 kHz)."
//!
//! The on-wire layout (§10.4 *"the narrowband frame is packed first,
//! then the high band"*) is:
//!
//! ```text
//!  wideband packet
//!    ├── 5-bit NB prefix (wideband flag=1 || 4-bit NB mode ID)
//!    ├── narrowband Table 9.1 body              ─► NB decode ─► x_lb[n]
//!    ├── 4-bit HB prefix (wideband flag || 3-bit HB mode ID)
//!    └── high-band Table 10.1 body              ─► HB synth ─► x_hb[n]
//! ```
//!
//! ## What this round adds
//!
//! The narrowband decode loop ([`crate::NarrowbandDecoder`]) and the
//! high-band synthesis chain ([`crate::synthesise_high_band_frame`]) both
//! already existed; this module **binds them into one wideband decoder**
//! that walks a wideband packet, decodes the embedded narrowband frame to
//! the low-band signal, then parses + synthesises the high-band frame
//! that follows, returning both half-band signals with their respective
//! persistent IIR state carried across frames.
//!
//! ## QMF synthesis recombination (round r365)
//!
//! The final recombination `(x_lb, x_hb) → 16 kHz wideband PCM` runs the
//! two half-band signals back through the QMF **synthesis** filterbank
//! ([`crate::QmfSynthesis`]). The staged material provides the 64-tap
//! prototype `h0` as pure data; the synthesis structure is the classical
//! two-band quadrature-mirror reconstruction (textbook multirate DSP,
//! see the [`crate::qmf`] module docs). This decoder now carries a
//! persistent [`QmfSynthesis`] and emits the recombined 16 kHz PCM as
//! [`WidebandFrame::wideband_pcm`] alongside the two half-band signals.
//! The bit-exact polyphase **delay** convention the reference decoder
//! uses is not pinned by the staged manual; the sample-correct
//! textbook-DSP reconstruction is what lands here.

use crate::bitreader::BitReader;
use crate::frame::NarrowbandFrameHeader;
use crate::hb_synthesis::HbSynthesisFilter;
use crate::narrowband_body::NarrowbandFrameBody;
use crate::narrowband_decoder::{
    NarrowbandDecodeError, NarrowbandDecoder, NARROWBAND_FRAME_SAMPLES,
};
use crate::qmf::{QmfSynthesis, QMF_WIDEBAND_FRAME};
use crate::submode::Submode;
use crate::wb_synthesis::{synthesise_high_band_frame_folded_exc, HB_FRAME_SAMPLES};
use crate::wideband::{
    WidebandBodyError, WidebandHighBandBody, WidebandHighBandFrameHeader, WidebandSubmode,
};
use core::fmt;

/// One decoded wideband frame: the two reconstructed 8 kHz half-band
/// signals **plus** the QMF-recombined 16 kHz wideband PCM.
///
/// `low_band` / `high_band` are 160-sample (`= NARROWBAND_FRAME_SAMPLES
/// = HB_FRAME_SAMPLES`) 8 kHz half-band signals, retained for
/// diagnostics. `wideband_pcm` is the [`crate::QmfSynthesis`]-recombined
/// 320-sample 16 kHz full-band output (the §10 final stage).
#[derive(Debug, Clone)]
pub struct WidebandFrame {
    /// Low-band (0–4 kHz) reconstructed signal `x_lb[n]` — the embedded
    /// narrowband synthesis output.
    pub low_band: [f64; NARROWBAND_FRAME_SAMPLES],
    /// High-band (4–8 kHz folded) reconstructed signal `x_hb[n]`.
    pub high_band: [f64; HB_FRAME_SAMPLES],
    /// QMF-recombined 16 kHz wideband PCM (`2 × 160 = 320` samples).
    pub wideband_pcm: [f64; QMF_WIDEBAND_FRAME],
}

impl WidebandFrame {
    /// The QMF-recombined 16 kHz wideband PCM, rounded + saturated to
    /// `i16`.
    ///
    /// Convenience over the `f64` [`Self::wideband_pcm`] field that
    /// applies the crate-wide [`crate::saturate_i16`] rounding convention
    /// (round-to-nearest, clamp to `[i16::MIN, i16::MAX]`) — the same
    /// quantiser the narrowband [`NarrowbandDecoder::decode_frame_i16`]
    /// path uses, so a low-band-only sample reduces identically whether it
    /// is read from the narrowband or the wideband output.
    pub fn wideband_pcm_i16(&self) -> [i16; QMF_WIDEBAND_FRAME] {
        let mut out = [0i16; QMF_WIDEBAND_FRAME];
        for (o, &s) in out.iter_mut().zip(self.wideband_pcm.iter()) {
            *o = crate::narrowband_decoder::saturate_i16(s);
        }
        out
    }
}

/// Errors from [`WidebandDecoder::decode_packet`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WidebandDecodeError {
    /// The embedded narrowband prefix did not parse, or its mode was not
    /// a CELP sub-mode (an in-band-signalling / terminator pseudo-frame
    /// cannot be the low band of a wideband frame).
    NotNarrowbandLowBand,
    /// The wideband flag preceding the high-band frame was `0` — there
    /// is no high-band layer, so this is a plain narrowband packet, not
    /// a wideband one.
    NoHighBandLayer,
    /// The high-band sub-mode is the reserved high-rate slot whose
    /// per-field bit budget the staged Table 10.1 does not pin.
    HighBandReserved { mode_id: u8 },
    /// The embedded narrowband frame failed to decode.
    Narrowband(NarrowbandDecodeError),
    /// The high-band frame failed to parse or synthesise.
    HighBand(WidebandBodyError),
    /// The high-band excitation codebook binding is a recorded docs gap
    /// (high-band mode 4).
    HighBandUndocumented,
    /// A low-level bit-reader / framing failure.
    Framing,
}

impl fmt::Display for WidebandDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WidebandDecodeError::NotNarrowbandLowBand => {
                write!(f, "wideband decode: embedded low band is not a narrowband CELP frame")
            }
            WidebandDecodeError::NoHighBandLayer => {
                write!(f, "wideband decode: wideband flag 0 — no high-band layer present")
            }
            WidebandDecodeError::HighBandReserved { mode_id } => write!(
                f,
                "wideband decode: high-band mode {mode_id} is the reserved high-rate slot (not in Table 10.1)"
            ),
            WidebandDecodeError::Narrowband(e) => write!(f, "wideband decode: low band: {e}"),
            WidebandDecodeError::HighBand(e) => write!(f, "wideband decode: high band: {e}"),
            WidebandDecodeError::HighBandUndocumented => {
                write!(f, "wideband decode: high-band excitation codebook binding is a docs gap")
            }
            WidebandDecodeError::Framing => write!(f, "wideband decode: framing/bit-reader failure"),
        }
    }
}

impl std::error::Error for WidebandDecodeError {}

impl From<NarrowbandDecodeError> for WidebandDecodeError {
    fn from(e: NarrowbandDecodeError) -> Self {
        WidebandDecodeError::Narrowband(e)
    }
}

impl From<WidebandBodyError> for WidebandDecodeError {
    fn from(e: WidebandBodyError) -> Self {
        WidebandDecodeError::HighBand(e)
    }
}

/// Stateful wideband (sub-band CELP) decoder.
///
/// Holds the embedded [`NarrowbandDecoder`] (low-band state) plus the
/// high-band [`HbSynthesisFilter`] IIR memory, both carried across
/// frames. Construct with [`WidebandDecoder::new`], then call
/// [`WidebandDecoder::decode_packet`] once per wideband packet.
#[derive(Debug, Clone)]
pub struct WidebandDecoder {
    low_band: NarrowbandDecoder,
    high_band_filter: HbSynthesisFilter,
    /// Previous wideband frame's reconstructed high-band LSP
    /// codebook-delta vector (Q10, pre-base), carried so the per-frame
    /// high-band LSP interpolation (§9.1 / §10.1) is continuous across
    /// frames. `None` at stream start and after a high-band silence
    /// frame that transmitted no LSP.
    prev_hb_lsp_delta_q10: Option<[i32; crate::codebooks::HB_LPC_ORDER]>,
    /// QMF synthesis filterbank state (FIR band histories) for the final
    /// half-band → 16 kHz recombination, carried across frames.
    qmf: QmfSynthesis,
    /// The high-band excitation of the most recently decoded frame (the
    /// folded excitation for the gain-only sub-mode, the gain-scaled
    /// innovation for the VQ sub-modes) — retained for the
    /// ultra-wideband second-layer fold (see [`crate::uwb_decoder`]).
    last_hb_excitation: [f32; HB_FRAME_SAMPLES],
}

impl Default for WidebandDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl WidebandDecoder {
    /// A fresh wideband decoder at stream start (zero history in both
    /// bands).
    pub fn new() -> Self {
        Self {
            low_band: NarrowbandDecoder::new(),
            high_band_filter: HbSynthesisFilter::new(),
            prev_hb_lsp_delta_q10: None,
            qmf: QmfSynthesis::new(),
            last_hb_excitation: [0.0; HB_FRAME_SAMPLES],
        }
    }

    /// The high-band excitation of the most recently decoded frame (160
    /// samples in the 8 kHz half-band geometry): the folded excitation
    /// for the gain-only sub-mode, the gain-scaled innovation for the VQ
    /// sub-modes, zero after silence / at stream start. The
    /// ultra-wideband second-layer fold reads this alongside the
    /// low band's [`NarrowbandDecoder::last_frame_excitation`].
    pub fn last_hb_excitation(&self) -> &[f32; HB_FRAME_SAMPLES] {
        &self.last_hb_excitation
    }

    /// Decode one wideband packet into its two reconstructed half-band
    /// signals.
    ///
    /// Walks the embedded narrowband frame (low band) and the high-band
    /// frame that follows (Table 10.1), advancing both bands' state.
    pub fn decode_packet(&mut self, packet: &[u8]) -> Result<WidebandFrame, WidebandDecodeError> {
        let mut reader = BitReader::new(packet);
        self.decode_frame_reader(&mut reader)
    }

    /// Decode one wideband frame **from a live bit cursor**, leaving the
    /// cursor at the first bit after the high-band body.
    ///
    /// This is the composition entry the ultra-wideband decoder walks:
    /// a UWB frame is an embedded wideband frame followed by a second
    /// high-band layer (§2.2's recursion), so the outer decoder consumes
    /// the wideband layers through this method and then claims the
    /// remaining layer itself. [`Self::decode_packet`] is the
    /// single-frame convenience over a fresh cursor.
    pub fn decode_frame_reader(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<WidebandFrame, WidebandDecodeError> {
        // --- Low band: embedded narrowband frame ---
        let nb_header =
            NarrowbandFrameHeader::parse(reader).map_err(|_| WidebandDecodeError::Framing)?;
        self.decode_frame_after_header(&nb_header, reader)
    }

    /// Decode one wideband frame whose 5-bit narrowband prefix has
    /// already been consumed (the ultra-wideband packet walk parses the
    /// prefix first to detect the §5.5 terminator / control frames).
    pub fn decode_frame_after_header(
        &mut self,
        nb_header: &NarrowbandFrameHeader,
        reader: &mut BitReader<'_>,
    ) -> Result<WidebandFrame, WidebandDecodeError> {
        let nb_submode = match nb_header.submode {
            Submode::Celp(s) => s,
            _ => return Err(WidebandDecodeError::NotNarrowbandLowBand),
        };
        let nb_body = NarrowbandFrameBody::parse(reader, &nb_submode)
            .map_err(|_| WidebandDecodeError::Framing)?;

        // --- High band: 4-bit prefix + Table 10.1 body ---
        let hb_header =
            WidebandHighBandFrameHeader::parse(reader).map_err(WidebandDecodeError::from)?;
        if !hb_header.wideband {
            return Err(WidebandDecodeError::NoHighBandLayer);
        }
        let hb_submode = match hb_header.submode {
            WidebandSubmode::Documented(s) => s,
            WidebandSubmode::ReservedHighRate(id) => {
                return Err(WidebandDecodeError::HighBandReserved { mode_id: id })
            }
        };
        let hb_body =
            WidebandHighBandBody::parse(reader, &hb_submode).map_err(WidebandDecodeError::from)?;
        self.decode_frame_bodies(&nb_body, &nb_submode, &hb_body, &hb_submode)
    }

    /// Decode one wideband frame from its **already-parsed** layer
    /// bodies (the entry the top-level [`crate::SpeexDecoder`] uses —
    /// its packet walker has parsed the layers already). Advances the
    /// full decoder state exactly as the bit-cursor entries do.
    pub fn decode_frame_bodies(
        &mut self,
        nb_body: &NarrowbandFrameBody,
        nb_submode: &crate::submode::NarrowbandSubmode,
        hb_body: &WidebandHighBandBody,
        hb_submode: &crate::wideband::WidebandHighBandSubmode,
    ) -> Result<WidebandFrame, WidebandDecodeError> {
        let low_band = self.low_band.decode_frame(nb_body, nb_submode)?;

        // The gain-only sub-mode (Table 10.1 mode 1) folds the embedded
        // narrowband frame's excitation into the high band (r393,
        // fixture-arbitrated law — see `crate::hb_fold`); the fold
        // source is the low-band decoder's just-composed excitation.
        let lb_excitation = *self.low_band.last_frame_excitation();
        let mut hb_excitation = [0.0f32; HB_FRAME_SAMPLES];
        let high_band = synthesise_high_band_frame_folded_exc(
            hb_body,
            hb_submode,
            &mut self.high_band_filter,
            &mut self.prev_hb_lsp_delta_q10,
            &lb_excitation,
            &mut hb_excitation,
        )
        .map_err(|_| WidebandDecodeError::HighBandUndocumented)?;
        self.last_hb_excitation = hb_excitation;

        // --- QMF synthesis: recombine the two half-bands → 16 kHz PCM ---
        let wideband_pcm = self.qmf.reconstruct_frame(&low_band, &high_band);

        Ok(WidebandFrame {
            low_band,
            high_band,
            wideband_pcm,
        })
    }

    /// Mutable access to the embedded narrowband decoder — the shared
    /// low-band state a mixed narrowband/wideband stream advances with
    /// its standalone narrowband frames (RFC 5574 §3.1: the wideband
    /// low band *is* an embedded narrowband frame, so one continuous
    /// narrowband state serves both). Used by [`crate::SpeexDecoder`].
    pub fn low_band_decoder_mut(&mut self) -> &mut NarrowbandDecoder {
        &mut self.low_band
    }

    /// Decode one wideband packet straight to rounded, saturated `i16`
    /// 16 kHz wideband PCM.
    ///
    /// Convenience over [`Self::decode_packet`] that returns only the
    /// QMF-recombined 320-sample 16 kHz output, quantised through the
    /// crate-wide [`crate::saturate_i16`] convention (the same one the
    /// narrowband [`NarrowbandDecoder::decode_frame_i16`] path uses). The
    /// decoder's internal half-band IIR / excitation / QMF state advances
    /// in full `f64` precision regardless, so the `i16` reduction affects
    /// only the returned PCM, not the carried-across-frame state.
    pub fn decode_packet_i16(
        &mut self,
        packet: &[u8],
    ) -> Result<[i16; QMF_WIDEBAND_FRAME], WidebandDecodeError> {
        Ok(self.decode_packet(packet)?.wideband_pcm_i16())
    }

    /// Read-only view of the embedded narrowband decoder (diagnostics).
    pub fn low_band_decoder(&self) -> &NarrowbandDecoder {
        &self.low_band
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitWriter;
    use crate::submode::NarrowbandSubmode;
    use crate::wideband::WIDEBAND_HIGH_BAND_SUBMODES;

    /// Build a wideband packet: an all-ones narrowband mode-`nb` body
    /// (wideband flag 1) followed by an all-ones high-band mode-`hb`
    /// frame (wideband flag 1 + 3-bit HB mode ID + body bits all 1).
    fn wideband_packet(nb: u8, hb: u8) -> Vec<u8> {
        let nb_submode = NarrowbandSubmode::for_id(nb).expect("valid nb mode");
        let hb_submode = WIDEBAND_HIGH_BAND_SUBMODES[hb as usize];

        let mut w = BitWriter::new();
        // NB prefix: wideband flag = 1, 4-bit NB mode ID.
        w.write_bit(1).unwrap();
        w.write(u32::from(nb), 4).unwrap();
        // NB body: every bit 1, total - 5 prefix bits.
        let nb_body_bits = u32::from(nb_submode.total_bits) - 5;
        write_ones(&mut w, nb_body_bits);

        // HB prefix: wideband flag = 1, 3-bit HB mode ID.
        w.write_bit(1).unwrap();
        w.write(u32::from(hb), 3).unwrap();
        // HB body: every bit 1, total - 4 prefix bits.
        let hb_body_bits = u32::from(hb_submode.total_bits) - 4;
        write_ones(&mut w, hb_body_bits);

        w.into_bytes()
    }

    fn write_ones(w: &mut BitWriter, mut bits: u32) {
        // Write `bits` 1-bits, chunked through write() (which takes up to
        // 32 bits). Use ≤ 16-bit chunks so the all-ones mask never needs
        // a 32-bit shift.
        while bits > 0 {
            let chunk = bits.min(16);
            let mask = (1u32 << chunk) - 1;
            w.write(mask, chunk).unwrap();
            bits -= chunk;
        }
    }

    #[test]
    fn wideband_mode5_nb_plus_mode2_hb_decodes_both_bands() {
        // NB mode 5 (full fields), HB mode 2 (4 × HbSv10_32, documented).
        let pkt = wideband_packet(5, 2);
        let mut dec = WidebandDecoder::new();
        let frame = dec.decode_packet(&pkt).expect("wideband packet decodes");
        assert_eq!(frame.low_band.len(), 160);
        assert_eq!(frame.high_band.len(), 160);
        assert_eq!(frame.wideband_pcm.len(), 320);
        for &s in &frame.low_band {
            assert!(s.is_finite(), "low-band sample not finite");
        }
        for &s in &frame.high_band {
            assert!(s.is_finite(), "high-band sample not finite");
        }
        for &s in &frame.wideband_pcm {
            assert!(s.is_finite(), "wideband PCM sample not finite");
        }
        // A non-silent wideband frame produces non-silent 16 kHz PCM.
        assert!(
            frame.wideband_pcm.iter().any(|&s| s != 0.0),
            "recombined wideband PCM should be non-silent"
        );
    }

    #[test]
    fn wideband_pcm_i16_matches_saturated_f64_path() {
        // The i16 convenience must be exactly saturate_i16() of the f64
        // wideband PCM, and the decoder's carried state must be unaffected
        // by which output form the caller reads. Decode the same packet on
        // two fresh decoders, reading f64 from one and i16 from the other.
        let pkt = wideband_packet(5, 2);

        let mut dec_f = WidebandDecoder::new();
        let frame = dec_f.decode_packet(&pkt).expect("decodes");
        let via_frame = frame.wideband_pcm_i16();

        let mut dec_i = WidebandDecoder::new();
        let direct = dec_i.decode_packet_i16(&pkt).expect("decodes");

        assert_eq!(via_frame.len(), 320);
        for (i, ((&f, &q_frame), &q_direct)) in frame
            .wideband_pcm
            .iter()
            .zip(via_frame.iter())
            .zip(direct.iter())
            .enumerate()
        {
            assert_eq!(q_frame, crate::saturate_i16(f), "frame method sample {i}");
            assert_eq!(q_direct, crate::saturate_i16(f), "direct method sample {i}");
        }
    }

    #[test]
    fn wideband_pcm_i16_saturates_out_of_range_samples() {
        // The saturation clamp pins extreme f64 values into i16 range.
        let frame = WidebandFrame {
            low_band: [0.0; NARROWBAND_FRAME_SAMPLES],
            high_band: [0.0; HB_FRAME_SAMPLES],
            wideband_pcm: {
                let mut p = [0.0; QMF_WIDEBAND_FRAME];
                p[0] = 1.0e9; // far above i16::MAX
                p[1] = -1.0e9; // far below i16::MIN
                p[2] = 0.5; // ties-away-from-zero rounds to 1
                p[3] = -0.5; // rounds to -1
                p
            },
        };
        let q = frame.wideband_pcm_i16();
        assert_eq!(q[0], i16::MAX);
        assert_eq!(q[1], i16::MIN);
        assert_eq!(q[2], 1);
        assert_eq!(q[3], -1);
        for &s in &q[4..] {
            assert_eq!(s, 0);
        }
    }

    #[test]
    fn wideband_pcm_recombination_is_continuous_across_frames() {
        // The QMF state must carry across packets: re-running frame 2 from
        // a fresh decoder differs from the continued decode (live FIR
        // history), confirming the recombination is stateful.
        let pkt = wideband_packet(5, 2);
        let mut cont = WidebandDecoder::new();
        let _ = cont.decode_packet(&pkt).unwrap();
        let f1_cont = cont.decode_packet(&pkt).unwrap();

        let mut fresh = WidebandDecoder::new();
        let f1_fresh = fresh.decode_packet(&pkt).unwrap();

        assert!(
            f1_cont
                .wideband_pcm
                .iter()
                .zip(f1_fresh.wideband_pcm.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "QMF history should influence the continued frame's PCM"
        );
    }

    #[test]
    fn wideband_high_band_silence_mode_yields_zero_high_band() {
        // NB mode 3 + HB mode 0 (silence): the high band is exactly zero
        // at stream start (no LSP, no excitation, zero filter history).
        let pkt = wideband_packet(3, 0);
        let mut dec = WidebandDecoder::new();
        let frame = dec.decode_packet(&pkt).expect("decodes");
        assert!(
            frame.high_band.iter().all(|&s| s == 0.0),
            "silence high band should be exactly zero at stream start"
        );
    }

    #[test]
    fn wideband_high_band_mode4_reports_docs_gap() {
        // HB mode 4's excitation codebook binding is a documented gap.
        let pkt = wideband_packet(5, 4);
        let mut dec = WidebandDecoder::new();
        match dec.decode_packet(&pkt) {
            Err(WidebandDecodeError::HighBandUndocumented) => {}
            other => panic!("expected HighBandUndocumented, got {other:?}"),
        }
    }

    #[test]
    fn wideband_decoder_is_deterministic() {
        let pkt = wideband_packet(5, 2);
        let mut a = WidebandDecoder::new();
        let mut b = WidebandDecoder::new();
        for _ in 0..4 {
            let fa = a.decode_packet(&pkt).unwrap();
            let fb = b.decode_packet(&pkt).unwrap();
            assert_eq!(fa.low_band, fb.low_band);
            assert_eq!(fa.high_band, fb.high_band);
        }
    }

    #[test]
    fn high_band_state_carries_across_frames() {
        // Two consecutive non-silent wideband frames: the high-band IIR
        // history advances, so the second frame's high band generally
        // differs from the first.
        let pkt = wideband_packet(5, 2);
        let mut dec = WidebandDecoder::new();
        let f0 = dec.decode_packet(&pkt).unwrap();
        let f1 = dec.decode_packet(&pkt).unwrap();
        assert!(
            f0.high_band != f1.high_band || f0.high_band.iter().all(|&v| v == 0.0),
            "high-band history should influence the second frame"
        );
    }
}
