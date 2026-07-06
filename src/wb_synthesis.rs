//! Wideband **high-band half-band synthesis** assembly + the
//! ultra-wideband (UWB) embedded-framing recursion descriptor
//! (round r340 scope).
//!
//! This module composes the per-round high-band primitives into the
//! complete high-band synthesis chain that produces the **high-band
//! 8 kHz half-band signal** — the second of the two 8 kHz signals the
//! QMF synthesis filterbank recombines into the 16 kHz wideband PCM
//! (*The Speex Codec Manual* §10: *"The 16 kHz signal is thus divided
//! into two 8 kHz signals, one representing the low band (0-4 kHz), the
//! other the high band (4-8 kHz)"*).
//!
//! ## The wideband decode path
//!
//! ```text
//!  wideband packet
//!    ├── embedded narrowband frame  ─► NB synthesis ─► low-band  x_lb[n]  (8 kHz)
//!    └── high-band frame (Table 10.1)
//!          ├── 12-bit LSP MSVQ ─► hb_lpc()           (order-8 A_hb(z))
//!          └── per sub-frame:
//!                ├── Excitation VQ ─► decode_hb_subframe()  (raw c_hb)
//!                ├── Excitation gain ─► reconstruct_hb_exc_gain()
//!                ├── e_hb[n] = g·c_hb[n]  (§10.2: no HB pitch)
//!                └── 1/A_hb(z) synthesis ─► high-band x_hb[n]  (8 kHz)
//!  x_lb[n], x_hb[n]  ─► QMF synthesis filterbank ─► 16 kHz wideband PCM
//! ```
//!
//! This module owns the **high-band branch** end-to-end (the box from
//! the high-band frame down to `x_hb[n]`). The low-band branch is the
//! existing narrowband synthesis path; the final QMF recombination is a
//! documented docs gap (see "QMF synthesis gap" below).
//!
//! ## Per-sub-frame high-band LPC (round r350)
//!
//! The high-band frame carries a **single** 12-bit LSP MSVQ index
//! ([`WidebandHighBandBody::lsp_index`]) reconstructed to one order-8
//! LSP vector. Per manual §10.1 the high-band linear prediction is
//! *"very similar to narrowband. The only difference is that we use only
//! 12 bits"* — the manual names exactly one difference, so the §9.1
//! per-sub-frame LSP interpolation applies verbatim to the high band.
//! [`synthesise_high_band_frame_interp`] therefore interpolates the
//! high-band LSP vector across the four sub-frames
//! ([`crate::hb_lsp_interp::HbSubFrameLsp`], the high-band analogue of
//! [`crate::lsp_interp::NbSubFrameLsp`]) and reconstructs a **separate
//! order-8 LPC set per sub-frame** via
//! [`crate::lsp_to_lpc::hb_subframe_lpc_set_with_base`], so the
//! high-band spectral envelope evolves smoothly within the frame exactly
//! as the narrowband path does. The previous frame's high-band LSP set
//! is carried by the caller (the wideband decoder) so the interpolation
//! is continuous across frames.
//!
//! [`synthesise_high_band_frame`] is retained as the **single-frame /
//! stateless** entry: it delegates to the interpolating path with the
//! first-frame convention (`prev = curr`), which makes every sub-frame
//! equal to the current LSP set — i.e. the previous frame-level-LPC
//! behaviour, with no spurious transient at stream start.
//!
//! ## QMF synthesis gap (documented, not fished)
//!
//! The staged material provides the 64-tap QMF prototype `h0`
//! ([`crate::qmf_h0_q15`]) as **pure data** and states structurally
//! that a QMF splits / recombines the two half-bands (§10, companion
//! §5). It does **not** specify the QMF *synthesis* filterbank
//! algorithm: the polyphase recombination structure, the `h0 → {h0,h1}`
//! analysis/synthesis pair derivation, the 2× interpolation +
//! decimation factors, and the inter-band delay alignment are not in
//! the staged manual prose. The final `(x_lb, x_hb) → 16 kHz PCM`
//! recombination is therefore a recorded docs gap; this module stops at
//! the two reconstructed half-band signals, which is the furthest the
//! staged material uniquely pins. See the README "docs gaps" tail.
//!
//! ## Ultra-wideband framing (the embedded recursion)
//!
//! The Speex bit-stream is an **embedded, scalable** structure (§2.2
//! *"Embedded wideband structure (scalable sampling rate)"*; §1.x
//! *"Integration of wideband and narrowband using an embedded
//! bit-stream"*). Each frame begins with the 1-bit **wideband flag**
//! that marks whether a higher sub-band layer follows the embedded
//! lower-rate frame:
//!
//! * **narrowband frame** — wideband flag `0`, then the 4-bit
//!   narrowband mode ID + Table 9.1 body.
//! * **wideband frame** — wideband flag `1`, then the embedded
//!   narrowband frame, then the high-band frame (1-bit flag + 3-bit
//!   mode ID + Table 10.1 body).
//! * **ultra-wideband frame** — by the same recursion, the wideband
//!   flag `1` introduces an *embedded wideband frame* followed by a
//!   second high-band layer (the 8–16 kHz band split off a 32 kHz
//!   QMF). The recursion marker and the QMF sub-band split repeat one
//!   level up.
//!
//! [`UwbFrameLayout`] captures this recursion as a typed descriptor:
//! the number of embedded sub-band layers and the per-layer sample
//! rate. As of round r389 the recursion is **walked**, not just
//! described: [`crate::UltraWidebandDecoder`] /
//! [`crate::UltraWidebandEncoder`] decode/encode the second high-band
//! layer, whose conformant shape (Table 10.1 **mode 1** at every
//! quality) is pinned by RFC 5574 Table 2's constant +36 bits/frame
//! ultra-wideband column (see [`crate::quality`]). The staged manual
//! still has no "Table 11.x": the mode-1 folded-excitation *source*
//! and the excitation-VQ modes' sub-frame geometry at the 16 kHz
//! half-band remain the recorded docs gaps (see the
//! [`crate::uwb_decoder`] module docs).

use crate::codebooks::HB_LPC_ORDER;
use crate::hb_innovation::{HbInnovationError, HB_SUBFRAME_SAMPLES};
use crate::hb_lsp_interp::HbSubFrameLsp;
use crate::hb_synthesis::HbSynthesisFilter;
use crate::lsp_to_lpc::hb_subframe_lpc_set_with_base;
use crate::wideband::{WidebandHighBandBody, WidebandHighBandSubmode};

/// Number of high-band sub-frames per 20 ms wideband frame.
pub const HB_SUBFRAMES_PER_FRAME: usize = crate::wideband::HIGH_BAND_SUBFRAMES_PER_FRAME as usize;

/// Total high-band samples per wideband frame (`4 × 40 = 160`), the
/// length of one 8 kHz half-band frame.
pub const HB_FRAME_SAMPLES: usize = HB_SUBFRAMES_PER_FRAME * HB_SUBFRAME_SAMPLES;

/// Synthesise one wideband **high-band** frame into its 160-sample
/// 8 kHz half-band signal `x_hb[n]`.
///
/// Runs the complete high-band branch of the wideband decode path for a
/// parsed [`WidebandHighBandBody`] + its resolved
/// [`WidebandHighBandSubmode`], using the supplied persistent
/// [`HbSynthesisFilter`] (whose IIR history carries across frames):
///
/// 1. reconstruct the order-8 high-band LPC set from the frame's 12-bit
///    LSP MSVQ index ([`WidebandHighBandBody::hb_lpc`]); for the silence
///    high-band sub-mode (mode 0, no LSP field) the all-zero LPC set is
///    used (`A_hb(z) = 1`, pass-through), matching the absent-envelope
///    convention;
/// 2. for each of the four sub-frames: decode + gain-scale the high-band
///    excitation `e_hb[n] = g·c_hb[n]`
///    ([`crate::gain_scaled_hb_innovation_from_body`]) — §10.2: no
///    high-band pitch term;
/// 3. filter `e_hb[n]` through `1/A_hb(z)`
///    ([`HbSynthesisFilter::process_subframe`]) and concatenate the four
///    sub-frame outputs.
///
/// Returns [`HbInnovationError::Undocumented`] for high-band mode 4
/// (whose codebook binding the staged inventory does not pin) and
/// [`HbInnovationError::IndexOutOfRange`] for a malformed excitation
/// index.
///
/// The returned `[f64; 160]` is the reconstructed high-band 8 kHz
/// half-band signal — the second input to the (gap-documented) QMF
/// synthesis filterbank. For modes 0 / 1 (silence / gain-only) the
/// high-band excitation is all-zero, so the output rings only from any
/// residual filter history (zero at stream start).
///
/// This **single-frame / stateless** entry delegates to
/// [`synthesise_high_band_frame_interp`] with the first-frame convention
/// (`prev = curr`): every sub-frame uses the current frame's LSP set, so
/// for a one-shot call the result matches the earlier frame-level-LPC
/// behaviour with no spurious transient. A continuous decoder that wants
/// per-frame interpolation across the four sub-frames threads the
/// previous frame's high-band LSP via the `_interp` entry instead.
pub fn synthesise_high_band_frame(
    body: &WidebandHighBandBody,
    submode: &WidebandHighBandSubmode,
    filter: &mut HbSynthesisFilter,
) -> Result<[f64; HB_FRAME_SAMPLES], HbInnovationError> {
    let mut prev = None;
    synthesise_high_band_frame_interp(body, submode, filter, &mut prev)
}

/// Synthesise one wideband **high-band** frame with **per-sub-frame LSP
/// interpolation** (round r350), threading the previous frame's
/// high-band LSP through `prev_hb_lsp_delta_q10` for cross-frame
/// continuity.
///
/// Identical to [`synthesise_high_band_frame`] except the order-8
/// high-band LPC is reconstructed **per sub-frame** from the §9.1/§10.1
/// four-way linear interpolation of the previous and current frame's
/// high-band LSP codebook-delta vectors
/// ([`crate::hb_lsp_interp::HbSubFrameLsp`] →
/// [`crate::lsp_to_lpc::hb_subframe_lpc_set_with_base`]), rather than a
/// single frame-level set for all four sub-frames.
///
/// `prev_hb_lsp_delta_q10` holds the previous frame's reconstructed
/// high-band LSP codebook-delta vector (Q10, pre-base). On the first
/// wideband frame (or after a silence frame that transmitted no LSP) it
/// is `None`, and the interpolation falls back to the first-frame
/// convention (`prev = curr`). After a frame that transmits an LSP field
/// this routine updates the slot to the current frame's LSP delta so the
/// next frame interpolates against it. Silence (mode 0, no LSP field)
/// leaves the slot untouched — the previous spectral envelope persists,
/// matching the narrowband silence convention.
pub fn synthesise_high_band_frame_interp(
    body: &WidebandHighBandBody,
    submode: &WidebandHighBandSubmode,
    filter: &mut HbSynthesisFilter,
    prev_hb_lsp_delta_q10: &mut Option<[i32; HB_LPC_ORDER]>,
) -> Result<[f64; HB_FRAME_SAMPLES], HbInnovationError> {
    // No fold source: the gain-only mode's excitation reconstructs to
    // zero through this legacy entry (a zero excitation folds to zero).
    synthesise_high_band_frame_folded(
        body,
        submode,
        filter,
        prev_hb_lsp_delta_q10,
        &[0.0; HB_FRAME_SAMPLES],
    )
}

/// Synthesise one wideband **high-band** frame with the **folded
/// excitation law wired** (round r393) — the full high-band branch
/// including the gain-only sub-mode 1.
///
/// Identical to [`synthesise_high_band_frame_interp`] plus the fold
/// source: `lb_excitation` is the embedded narrowband frame's composed
/// excitation ([`crate::NarrowbandDecoder::last_frame_excitation`], 160
/// samples in the same 8 kHz half-band geometry). For a sub-mode that
/// transmits an excitation gain but **no innovation vector** (Table 10.1
/// mode 1) each sub-frame's excitation is reconstructed by the
/// fixture-arbitrated folded law
/// `e_hb[n] = K·g·(−1)ⁿ·e_lb[n]` ([`crate::hb_fold`]); every other
/// documented sub-mode takes the innovation-VQ path exactly as before.
pub fn synthesise_high_band_frame_folded(
    body: &WidebandHighBandBody,
    submode: &WidebandHighBandSubmode,
    filter: &mut HbSynthesisFilter,
    prev_hb_lsp_delta_q10: &mut Option<[i32; HB_LPC_ORDER]>,
    lb_excitation: &[f32; HB_FRAME_SAMPLES],
) -> Result<[f64; HB_FRAME_SAMPLES], HbInnovationError> {
    let mut exc_out = [0.0f32; HB_FRAME_SAMPLES];
    synthesise_high_band_frame_folded_exc(
        body,
        submode,
        filter,
        prev_hb_lsp_delta_q10,
        lb_excitation,
        &mut exc_out,
    )
}

/// [`synthesise_high_band_frame_folded`] that additionally **surfaces
/// the per-sample high-band excitation** actually driving the synthesis
/// filter (`exc_out`, 160 samples: the folded excitation for the
/// gain-only sub-mode, the gain-scaled innovation for the VQ
/// sub-modes, zero for silence).
///
/// The ultra-wideband decoder consumes this track: the §2.2 recursion's
/// second layer folds the **embedded wideband layer's excitation**, so
/// the wideband decode must retain what its high band was excited with
/// (see [`crate::uwb_decoder`]).
pub fn synthesise_high_band_frame_folded_exc(
    body: &WidebandHighBandBody,
    submode: &WidebandHighBandSubmode,
    filter: &mut HbSynthesisFilter,
    prev_hb_lsp_delta_q10: &mut Option<[i32; HB_LPC_ORDER]>,
    lb_excitation: &[f32; HB_FRAME_SAMPLES],
    exc_out: &mut [f32; HB_FRAME_SAMPLES],
) -> Result<[f64; HB_FRAME_SAMPLES], HbInnovationError> {
    // Current frame's reconstructed high-band LSP codebook-delta vector
    // (Q10, pre-base). Silence mode 0 transmits no LSP field → None.
    let curr_lsp = body.reconstructed_lsp_q10(submode);

    // Build the four per-sub-frame order-8 LPC sets.
    //
    // * LSP transmitted: interpolate prev→curr across the four
    //   sub-frames (first-frame convention when there is no prev), then
    //   reconstruct one LPC set per sub-frame with the pinned base.
    // * No LSP (silence): the absent-envelope convention — A_hb(z) = 1
    //   (zero coefficients) for all four sub-frames. (The previous LSP
    //   delta is preserved for the next frame; silence does not update
    //   the spectral envelope.)
    let lpc_sets: [[f64; HB_LPC_ORDER]; HB_SUBFRAMES_PER_FRAME] = match curr_lsp {
        Some(curr) => {
            let sub_lsp = match *prev_hb_lsp_delta_q10 {
                Some(prev) => HbSubFrameLsp::new(&prev, &curr),
                None => HbSubFrameLsp::first_frame(&curr),
            };
            hb_subframe_lpc_set_with_base(&sub_lsp)
        }
        None => [[0.0; HB_LPC_ORDER]; HB_SUBFRAMES_PER_FRAME],
    };

    // The folded (gain-only) sub-mode carries a gain but no innovation
    // vector — Table 10.1 mode 1 is the only documented column with that
    // shape.
    let folded_mode = submode.excitation_vq_bits == 0 && submode.excitation_gain_bits > 0;

    let mut out = [0.0f64; HB_FRAME_SAMPLES];
    for sf in 0..HB_SUBFRAMES_PER_FRAME {
        let lpc = lpc_sets[sf];
        let e64: [f64; HB_SUBFRAME_SAMPLES] = if folded_mode {
            // Folded law: e_hb[n] = K·g·(−1)ⁿ·e_lb[n] (crate::hb_fold).
            let gain =
                crate::hb_excitation_gain::HbExcitationGainIndex::from_body(body, submode, sf)
                    .map(crate::gain_reconstruction::reconstruct_hb_exc_gain)
                    .unwrap_or(0.0);
            let mut lb = [0.0f32; HB_SUBFRAME_SAMPLES];
            lb.copy_from_slice(
                &lb_excitation[sf * HB_SUBFRAME_SAMPLES..(sf + 1) * HB_SUBFRAME_SAMPLES],
            );
            crate::hb_fold::folded_hb_excitation_subframe(&lb, gain)
        } else {
            let e_hb = crate::gain_scaled_hb_innovation::gain_scaled_hb_innovation_from_body(
                body, submode, sf,
            )?;
            // Promote the f32 excitation to f64 for the synthesis
            // recurrence.
            let mut tmp = [0.0f64; HB_SUBFRAME_SAMPLES];
            for (d, &s) in tmp.iter_mut().zip(e_hb.iter()) {
                *d = f64::from(s);
            }
            tmp
        };
        let x = filter.process_subframe(&lpc, &e64);
        out[sf * HB_SUBFRAME_SAMPLES..(sf + 1) * HB_SUBFRAME_SAMPLES].copy_from_slice(&x);
        for (slot, &e) in exc_out[sf * HB_SUBFRAME_SAMPLES..(sf + 1) * HB_SUBFRAME_SAMPLES]
            .iter_mut()
            .zip(e64.iter())
        {
            *slot = e as f32;
        }
    }

    // Update the prev-LSP state only when this frame transmitted an LSP
    // field (silence leaves it untouched).
    if let Some(curr) = curr_lsp {
        *prev_hb_lsp_delta_q10 = Some(curr);
    }

    Ok(out)
}

/// Sampling-rate class of one embedded sub-band layer in the Speex
/// scalable bit-stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubBandLayer {
    /// The base narrowband layer (0–4 kHz, 8 kHz sample rate).
    Narrowband,
    /// The wideband high-band layer (4–8 kHz, folded to an 8 kHz
    /// half-band; recombined to 16 kHz by the QMF).
    WidebandHighBand,
    /// The ultra-wideband high-band layer (8–16 kHz, folded to a 16 kHz
    /// half-band; recombined to 32 kHz by the outer QMF).
    UltraWidebandHighBand,
}

impl SubBandLayer {
    /// The reconstructed full-band sample rate after this layer is
    /// recombined (8000 / 16000 / 32000 Hz).
    pub const fn reconstructed_rate_hz(self) -> u32 {
        match self {
            SubBandLayer::Narrowband => 8_000,
            SubBandLayer::WidebandHighBand => 16_000,
            SubBandLayer::UltraWidebandHighBand => 32_000,
        }
    }
}

/// Typed descriptor of the embedded sub-band framing for a given Speex
/// rate class — the **recursion structure** of the scalable bit-stream
/// (§2.2 *"Embedded wideband structure"*).
///
/// Each higher rate class prepends a 1-bit **wideband flag** = `1`
/// followed by the lower class's whole frame, then its own high-band
/// layer; a narrowband decoder stops at the first `wideband flag == 0`
/// embedded frame. This descriptor names the layer ladder; the walk
/// itself lives in [`crate::UltraWidebandDecoder`] /
/// [`crate::UltraWidebandEncoder`] (round r389 — see module docs for
/// what remains gapped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UwbFrameLayout {
    /// The outermost (highest-rate) layer.
    pub top: SubBandLayer,
}

impl UwbFrameLayout {
    /// Narrowband: a single base layer, no embedded high band.
    pub const NARROWBAND: Self = Self {
        top: SubBandLayer::Narrowband,
    };
    /// Wideband: narrowband base + one high-band layer.
    pub const WIDEBAND: Self = Self {
        top: SubBandLayer::WidebandHighBand,
    };
    /// Ultra-wideband: wideband (narrowband + WB high band) + one more
    /// high-band layer.
    pub const ULTRA_WIDEBAND: Self = Self {
        top: SubBandLayer::UltraWidebandHighBand,
    };

    /// Map a [`crate::SpeexHeader::mode`] field value to its sub-band
    /// framing layout: `0` → [`Self::NARROWBAND`], `1` → [`Self::WIDEBAND`],
    /// `2` → [`Self::ULTRA_WIDEBAND`] (the [`crate::header::SPEEX_MODE_NARROWBAND`]
    /// / `_WIDEBAND` / `_ULTRAWIDEBAND` constants, manual §7.3). Returns
    /// `None` for an unknown mode — the same set
    /// [`crate::SpeexHeader::is_known_mode`] accepts.
    ///
    /// This links a parsed stream header to the recursion descriptor so a
    /// caller can size the embedded layer ladder (and the reconstructed
    /// rate) from the header alone, before walking any packet.
    pub fn for_header_mode(mode: u32) -> Option<Self> {
        match mode {
            crate::header::SPEEX_MODE_NARROWBAND => Some(Self::NARROWBAND),
            crate::header::SPEEX_MODE_WIDEBAND => Some(Self::WIDEBAND),
            crate::header::SPEEX_MODE_ULTRAWIDEBAND => Some(Self::ULTRA_WIDEBAND),
            _ => None,
        }
    }

    /// The embedded sub-band layers from base (narrowband) up to and
    /// including [`Self::top`], in decode order (the wideband-flag
    /// recursion is walked outermost-first on the wire, but the layers
    /// are listed base-first here to mirror the reconstruction order).
    pub fn layers(self) -> &'static [SubBandLayer] {
        match self.top {
            SubBandLayer::Narrowband => &[SubBandLayer::Narrowband],
            SubBandLayer::WidebandHighBand => {
                &[SubBandLayer::Narrowband, SubBandLayer::WidebandHighBand]
            }
            SubBandLayer::UltraWidebandHighBand => &[
                SubBandLayer::Narrowband,
                SubBandLayer::WidebandHighBand,
                SubBandLayer::UltraWidebandHighBand,
            ],
        }
    }

    /// Number of embedded high-band layers above the narrowband base
    /// (`0` narrowband, `1` wideband, `2` ultra-wideband) — the count of
    /// wideband-flag `1` recursion markers preceding the base frame.
    pub fn high_band_layer_count(self) -> usize {
        self.layers().len() - 1
    }

    /// The reconstructed full-band sample rate of the top layer.
    pub fn reconstructed_rate_hz(self) -> u32 {
        self.top.reconstructed_rate_hz()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wideband::{HighBandSubFrameIndices, WIDEBAND_HIGH_BAND_SUBMODES};

    fn mk_body(lsp_index: u16, per_subframe: [(u8, u128); 4]) -> WidebandHighBandBody {
        let mut subframes = [HighBandSubFrameIndices::default(); 4];
        for (sf, (gain, vq)) in subframes.iter_mut().zip(per_subframe) {
            sf.excitation_gain_index = gain;
            sf.excitation_vq_index = vq;
        }
        WidebandHighBandBody {
            lsp_index,
            subframes,
        }
    }

    #[test]
    fn frame_sample_count_is_160() {
        assert_eq!(HB_FRAME_SAMPLES, 160);
        assert_eq!(HB_SUBFRAMES_PER_FRAME, 4);
    }

    /// Silence high-band (mode 0) from a fresh filter is exactly zero —
    /// no LSP, no excitation, zero history → zero output.
    #[test]
    fn silence_mode_zero_from_fresh_filter() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[0];
        let body = mk_body(0, [(0, 0); 4]);
        let mut f = HbSynthesisFilter::new();
        let out = synthesise_high_band_frame(&body, &submode, &mut f).unwrap();
        assert_eq!(out.len(), HB_FRAME_SAMPLES);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Mode 1 (gain-only, no excitation VQ) — the excitation is zero, so
    /// from a fresh filter the high-band output is zero regardless of the
    /// gain index.
    #[test]
    fn mode_1_gain_only_is_zero_excitation() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[1];
        // Non-trivial LSP + gain, but no VQ field → zero innovation.
        let body = mk_body(0x0ABC, [(20, 0), (5, 0), (31, 0), (0, 0)]);
        let mut f = HbSynthesisFilter::new();
        let out = synthesise_high_band_frame(&body, &submode, &mut f).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Mode 2 (4 × HbSv10_32) with a non-zero excitation + gain produces
    /// a finite, non-silent high-band signal.
    #[test]
    fn mode_2_produces_finite_responsive_signal() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
        // Pack 4 × 5-bit indices into the 20-bit VQ field.
        let mut vq: u128 = 0;
        for &idx in &[5u128, 17, 9, 31] {
            vq = (vq << 5) | idx;
        }
        // 4-bit gain index away from the silent end of the table.
        let body = mk_body(0x0123, [(8, vq), (10, vq), (12, vq), (6, vq)]);
        let mut f = HbSynthesisFilter::new();
        let out = synthesise_high_band_frame(&body, &submode, &mut f).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "all samples finite");
        assert!(out.iter().any(|&v| v != 0.0), "signal must be non-silent");
    }

    /// Mode 4 surfaces the documented codebook-binding gap.
    #[test]
    fn mode_4_returns_undocumented() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[4];
        let body = mk_body(0, [(1, 0); 4]);
        let mut f = HbSynthesisFilter::new();
        assert_eq!(
            synthesise_high_band_frame(&body, &submode, &mut f),
            Err(HbInnovationError::Undocumented)
        );
    }

    /// The filter history carries across frames: a second mode-2 frame
    /// fed the same body but a non-fresh filter differs from the
    /// stream-start frame (the IIR memory is live).
    #[test]
    fn filter_history_carries_across_frames() {
        let submode = WIDEBAND_HIGH_BAND_SUBMODES[2];
        let mut vq: u128 = 0;
        for &idx in &[1u128, 2, 3, 4] {
            vq = (vq << 5) | idx;
        }
        let body = mk_body(0x0555, [(9, vq); 4]);

        let mut f = HbSynthesisFilter::new();
        let frame0 = synthesise_high_band_frame(&body, &submode, &mut f).unwrap();
        let frame1 = synthesise_high_band_frame(&body, &submode, &mut f).unwrap();
        // Non-fresh-filter frame should generally differ from the
        // stream-start frame because the IIR history is non-zero.
        assert!(
            frame0 != frame1 || frame0.iter().all(|&v| v == 0.0),
            "history should influence the second frame"
        );
    }

    // ---- UWB framing recursion descriptor ----

    #[test]
    fn layer_ladder_matches_rate_class() {
        assert_eq!(
            UwbFrameLayout::NARROWBAND.layers(),
            &[SubBandLayer::Narrowband]
        );
        assert_eq!(
            UwbFrameLayout::WIDEBAND.layers(),
            &[SubBandLayer::Narrowband, SubBandLayer::WidebandHighBand]
        );
        assert_eq!(
            UwbFrameLayout::ULTRA_WIDEBAND.layers(),
            &[
                SubBandLayer::Narrowband,
                SubBandLayer::WidebandHighBand,
                SubBandLayer::UltraWidebandHighBand
            ]
        );
    }

    #[test]
    fn for_header_mode_maps_mode_field_to_layout() {
        use crate::header::{SPEEX_MODE_NARROWBAND, SPEEX_MODE_ULTRAWIDEBAND, SPEEX_MODE_WIDEBAND};
        assert_eq!(
            UwbFrameLayout::for_header_mode(SPEEX_MODE_NARROWBAND),
            Some(UwbFrameLayout::NARROWBAND)
        );
        assert_eq!(
            UwbFrameLayout::for_header_mode(SPEEX_MODE_WIDEBAND),
            Some(UwbFrameLayout::WIDEBAND)
        );
        assert_eq!(
            UwbFrameLayout::for_header_mode(SPEEX_MODE_ULTRAWIDEBAND),
            Some(UwbFrameLayout::ULTRA_WIDEBAND)
        );
        // Unknown modes have no layout (matches is_known_mode).
        assert_eq!(UwbFrameLayout::for_header_mode(3), None);
        assert_eq!(UwbFrameLayout::for_header_mode(7), None);
        // The layout's reconstructed rate agrees with the header's
        // canonical mode-rate for every known mode.
        for (mode, rate) in [
            (SPEEX_MODE_NARROWBAND, 8_000),
            (SPEEX_MODE_WIDEBAND, 16_000),
            (SPEEX_MODE_ULTRAWIDEBAND, 32_000),
        ] {
            let layout = UwbFrameLayout::for_header_mode(mode).unwrap();
            assert_eq!(layout.reconstructed_rate_hz(), rate);
        }
    }

    #[test]
    fn high_band_layer_counts_are_recursion_depth() {
        assert_eq!(UwbFrameLayout::NARROWBAND.high_band_layer_count(), 0);
        assert_eq!(UwbFrameLayout::WIDEBAND.high_band_layer_count(), 1);
        assert_eq!(UwbFrameLayout::ULTRA_WIDEBAND.high_band_layer_count(), 2);
    }

    #[test]
    fn reconstructed_rates_double_per_layer() {
        assert_eq!(UwbFrameLayout::NARROWBAND.reconstructed_rate_hz(), 8_000);
        assert_eq!(UwbFrameLayout::WIDEBAND.reconstructed_rate_hz(), 16_000);
        assert_eq!(
            UwbFrameLayout::ULTRA_WIDEBAND.reconstructed_rate_hz(),
            32_000
        );
        // Each layer doubles the reconstructed rate (the QMF interpolation
        // factor of 2 at every sub-band split).
        for l in [
            SubBandLayer::Narrowband,
            SubBandLayer::WidebandHighBand,
            SubBandLayer::UltraWidebandHighBand,
        ] {
            assert_eq!(l.reconstructed_rate_hz() % 8_000, 0);
        }
    }
}
