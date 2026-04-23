//! Speex ultra-wideband encoder — float path.
//!
//! Layers a UWB extension bitstream on top of a wideband encoder. A UWB
//! frame is 640 samples at 32 kHz (40 ms). The encoder splits it with a
//! 2-band QMF analysis: the low band (320 samples at 16 kHz, 0–8 kHz)
//! feeds a [`WbEncoder`]; the high band (320 samples at 16 kHz,
//! representing 8–16 kHz) is modelled by an 8th-order LPC plus a
//! spectral-folding excitation derived from the WB high-band
//! excitation. Only the reference sub-mode (folding, 56 bits of UWB
//! extension per frame) is defined and supported here.
//!
//! Two UWB layer encodings are available:
//! * **Null layer (uwb-bit = 0)** — 1 bit of UWB overhead. The decoder
//!   zero-pads the UWB high band and reconstructs 32 kHz output from
//!   the WB decode plus silent 8–16 kHz. Matches what the reference
//!   emits in DTX / silent frames. This is what
//!   [`UwbEncoder::with_null_layer`] produces.
//! * **Folding layer (uwb-bit = 1, submode = 1)** — 1 + 3 + 12 + 4×5 =
//!   36 bits on top of the WB frame. The folding gain is inverted
//!   against the **reconstructed** WB high-band excitation
//!   (`WbEncoder::hb_innov()`, which bit-for-bit matches what
//!   `WbDecoder::hb_innov()` will recover from the emitted stream), so
//!   the 5-bit gain quantiser targets the envelope the decoder will
//!   actually fold — not an encoder-side approximation. Matches the
//!   reference's `wb_submode1.bits = 36` with UWB's 4-subframe layout.
//!   [`UwbEncoder::new`] / `with_submode(1)` selects this path.
//!
//! Sub-modes 2/3/4 are **not defined** by the Speex UWB reference —
//! UWB is folding-only.

use oxideav_core::{Error, Result};

use crate::lsp::{bw_lpc, lsp_interpolate, lsp_to_lpc};
use crate::lsp_tables_wb::{HIGH_LSP_CDBK, HIGH_LSP_CDBK2};
use crate::qmf::{H0_PROTOTYPE, QMF_ORDER};
use crate::uwb_decoder::{
    UWB_FRAME_SIZE, UWB_FULL_FRAME_SIZE, UWB_NB_SUBFRAMES, UWB_SUBFRAME_SIZE,
};
use crate::wb_decoder::{FOLDING_GAIN, LSP_MARGIN_HIGH, WB_FRAME_SIZE, WB_LPC_ORDER};
use crate::wb_encoder::{qmf_decomp, WbEncoder};
use oxideav_core::bits::BitWriter;

// The decoder's UWB folding layer reads the WB decoder's reconstructed
// 4–8 kHz excitation (`wb.hb_innov()`) as the folding source. The
// encoder reconstructs the same signal inside `WbEncoder::encode_frame`
// and exposes it via `WbEncoder::hb_innov()`, so our UWB folding gain
// can be computed against the *exact* signal the decoder will fold.

/// Default: emit the folding extension (best fidelity this encoder
/// can produce).
pub const DEFAULT_UWB_SUBMODE: u32 = 1;

/// Ultra-wideband encoder. Owns a [`WbEncoder`] for the 0–16 kHz low
/// band plus QMF analysis memory for the 16–32 kHz split.
pub struct UwbEncoder {
    /// Which UWB sub-mode to emit: 0 = null (uwb-bit=0, silent high
    /// band), 1 = folding extension.
    submode: u32,
    /// WB encoder for the 0–16 kHz band (NB + WB extension).
    wb: WbEncoder,
    qmf_mem_lo: [f32; QMF_ORDER],
    qmf_mem_hi: [f32; QMF_ORDER],
    /// Quantised high-band LSPs carried across frames for sub-frame
    /// interpolation.
    old_qlsp_high: [f32; WB_LPC_ORDER],
    first: bool,
}

impl Default for UwbEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl UwbEncoder {
    /// Default UWB encoder — folding extension on top of WB sub-mode 3.
    pub fn new() -> Self {
        Self::with_submode(DEFAULT_UWB_SUBMODE).expect("default submode supported")
    }

    /// Construct a UWB encoder that emits just the null layer — one
    /// bit of UWB overhead, silent 8–16 kHz at decode. Useful as a
    /// round-trip baseline.
    pub fn with_null_layer() -> Self {
        Self {
            submode: 0,
            wb: WbEncoder::new(),
            qmf_mem_lo: [0.0; QMF_ORDER],
            qmf_mem_hi: [0.0; QMF_ORDER],
            old_qlsp_high: initial_qlsp(),
            first: true,
        }
    }

    /// Construct a UWB encoder for a specific sub-mode.
    ///
    /// * `0` — null layer (uwb-bit = 0, 1 bit of overhead).
    /// * `1` — folding extension (56 bits of UWB overhead, default).
    ///
    /// Any other value is rejected with `Error::Unsupported` — the
    /// reference defines no other UWB sub-modes.
    pub fn with_submode(submode: u32) -> Result<Self> {
        if submode != 0 && submode != 1 {
            return Err(Error::unsupported(format!(
                "Speex UWB encoder: sub-mode {submode} is not defined by the reference \
                 (only 0 = null and 1 = folding are valid UWB layers)"
            )));
        }
        Ok(Self {
            submode,
            wb: WbEncoder::new(),
            qmf_mem_lo: [0.0; QMF_ORDER],
            qmf_mem_hi: [0.0; QMF_ORDER],
            old_qlsp_high: initial_qlsp(),
            first: true,
        })
    }

    pub fn submode(&self) -> u32 {
        self.submode
    }

    /// Total bits per encoded UWB frame (WB bits + UWB overhead).
    pub fn bits_per_frame(&self) -> u32 {
        self.wb.bits_per_frame()
            + match self.submode {
                0 => 1, // just the uwb-bit=0
                1 => 36,
                _ => 0,
            }
    }

    /// Encode one 640-sample UWB frame (f32 in int16 range).
    pub fn encode_frame(&mut self, pcm: &[f32], bw: &mut BitWriter) -> Result<()> {
        if pcm.len() != UWB_FULL_FRAME_SIZE {
            return Err(Error::invalid(format!(
                "Speex UWB encoder: expected {UWB_FULL_FRAME_SIZE}-sample frame, got {}",
                pcm.len()
            )));
        }

        // ---- 1. QMF analysis: 640 → 320 low + 320 high (16 kHz each).
        let mut low = [0.0f32; UWB_FRAME_SIZE];
        let mut high = [0.0f32; UWB_FRAME_SIZE];
        qmf_decomp(
            pcm,
            &H0_PROTOTYPE,
            &mut low,
            &mut high,
            UWB_FULL_FRAME_SIZE,
            QMF_ORDER,
            &mut self.qmf_mem_lo,
            &mut self.qmf_mem_hi,
        );

        // ---- 2. WB encode the low band -----------------------------
        self.wb.encode_frame(&low, bw)?;

        // ---- 3. UWB layer header -----------------------------------
        if self.submode == 0 {
            bw.write_bits(0, 1);
            return Ok(());
        }
        bw.write_bits(1, 1); // uwb-bit = 1
        bw.write_bits(self.submode, 3);

        // ---- 4. High-band LPC analysis on the UWB high band --------
        let windowed = hamming_window(&high);
        let mut autocorr = [0.0f32; WB_LPC_ORDER + 1];
        autocorrelate(&windowed, &mut autocorr);
        for k in 1..=WB_LPC_ORDER {
            let tau = 40.0f32;
            let w = (-0.5 * (k as f32 / tau).powi(2)).exp();
            autocorr[k] *= w;
        }
        autocorr[0] *= 1.0001;
        if autocorr[0] < 1e-6 {
            autocorr[0] = 1e-6;
        }
        let raw_lpc = levinson_durbin(&autocorr);
        let mut lpc = [0.0f32; WB_LPC_ORDER];
        bw_lpc(0.9, &raw_lpc, &mut lpc, WB_LPC_ORDER);

        // ---- 5. LPC → LSP ------------------------------------------
        let lsp = lpc_to_lsp_high(&lpc).unwrap_or_else(initial_qlsp);

        // ---- 6. Two-stage 6+6-bit LSP VQ ---------------------------
        let (qlsp, idx_a, idx_b) = quantise_lsp_high(&lsp);
        bw.write_bits(idx_a as u32, 6);
        bw.write_bits(idx_b as u32, 6);

        if self.first {
            self.old_qlsp_high = qlsp;
        }

        // ---- 7. Per-sub-frame folding gain (5 bits × 4 sub-frames) -
        //
        // The decoder reconstructs each UWB sub-frame's excitation as
        //   exc[i] = ±FG · g · wb_hb_innov[src + i]   (alternating sign)
        // where `wb_hb_innov` is the wideband decoder's reconstructed
        // 4–8 kHz excitation. We invert that directly: ask the WB
        // encoder for the `hb_innov` it produced (exposed via
        // `WbEncoder::hb_innov()`) — that's bit-for-bit what the
        // decoder will see on its own `wb.hb_innov()` after parsing
        // our bitstream. Using the exact source lets the 5-bit gain
        // quantiser nail the right envelope instead of absorbing an
        // encoder/decoder RMS mismatch.
        //
        // Filter-ratio inversion: same rl/rh formula the decoder uses,
        // with rl pulled from the WB encoder's NB sub-encoder's
        // `pi_gain` (the WB decoder does the same thing — look up
        // `wb_pi_gain = wb.nb().pi_gain()` inside
        // `UwbDecoder::decode_folding_layer`).

        let wb_hb_innov = *self.wb.hb_innov();
        let wb_pi_gain = *self.wb.nb().pi_gain();

        let mut interp_qlsp = [0.0f32; WB_LPC_ORDER];
        let mut ak = [0.0f32; WB_LPC_ORDER];
        for sub in 0..UWB_NB_SUBFRAMES {
            let off = sub * UWB_SUBFRAME_SIZE;

            lsp_interpolate(
                &self.old_qlsp_high,
                &qlsp,
                &mut interp_qlsp,
                WB_LPC_ORDER,
                sub,
                UWB_NB_SUBFRAMES,
                LSP_MARGIN_HIGH,
            );
            lsp_to_lpc(&interp_qlsp, &mut ak, WB_LPC_ORDER);

            // Filter-response ratio — mirrors the decoder's formula:
            //   rh = 1 + Σ_{even k} (ak[k+1] - ak[k])  (ω=π response)
            //   rl = wb_pi_gain[sub]                   (WB NB sub-frame)
            //   filter_ratio = (rl + .01) / (rh + .01)
            let mut rh = 1.0f32;
            let mut k = 0;
            while k + 1 < WB_LPC_ORDER {
                rh += ak[k + 1] - ak[k];
                k += 2;
            }
            let rl_idx = sub.min(wb_pi_gain.len() - 1);
            let rl = wb_pi_gain[rl_idx];
            let filter_ratio = (rl + 0.01) / (rh + 0.01);

            // Target: the LPC residual of the UWB high band on this
            // sub-frame (80 samples).
            let hb_sub = &high[off..off + UWB_SUBFRAME_SIZE];
            let mut residual = [0.0f32; UWB_SUBFRAME_SIZE];
            fir_filter_stateless(hb_sub, &ak, &mut residual, WB_LPC_ORDER);
            let hb_res_rms = rms(&residual);

            // Folding-source RMS — walk the same 80-sample window the
            // decoder walks through `wb_hb_innov`. WB hb_innov is 160
            // samples at 8 kHz decimated, so UWB's 80-sample sub-frame
            // consumes 40 source samples (2 per fold pair). Compute RMS
            // over the effective source slice the decoder will actually
            // multiply by (±FG · g).
            let src_off = (off / 2) & !1;
            let mut src_sum_sq = 0.0f64;
            let mut n_taken = 0usize;
            let mut i = 0;
            while i + 1 < UWB_SUBFRAME_SIZE {
                let src_i = (src_off + i / 2) % WB_FRAME_SIZE;
                let src_j = (src_off + i / 2 + 1) % WB_FRAME_SIZE;
                let s0 = wb_hb_innov[src_i] as f64;
                let s1 = wb_hb_innov[src_j] as f64;
                src_sum_sq += s0 * s0 + s1 * s1;
                n_taken += 2;
                i += 2;
            }
            let src_rms = if n_taken > 0 {
                (src_sum_sq / n_taken as f64).sqrt() as f32
            } else {
                0.0
            };
            let folding_rms = FOLDING_GAIN * src_rms;

            // Decoder: `g_after = exp(.125·(q-10)) / filter_ratio` and
            // `exc[i] = ±FG · g_after · src[·]`, so the post-scaling
            // RMS equals `folding_rms · g_after`. Set that equal to
            // the residual RMS and solve for q:
            //   g_after · filter_ratio = hb_res_rms / folding_rms
            //   g_raw (before filter_ratio) = hb_res_rms · filter_ratio / folding_rms
            //   q = 10 + 8 · ln(g_raw).
            let eps = 1e-6f32;
            let g_target = (hb_res_rms / folding_rms.max(eps)) * filter_ratio.max(eps);
            let q_f = 10.0 + 8.0 * g_target.max(eps).ln();
            let q = q_f.round().clamp(0.0, 31.0) as u32;
            bw.write_bits(q, 5);
        }

        self.old_qlsp_high = qlsp;
        self.first = false;
        Ok(())
    }
}

fn initial_qlsp() -> [f32; WB_LPC_ORDER] {
    let mut out = [0.0f32; WB_LPC_ORDER];
    for i in 0..WB_LPC_ORDER {
        out[i] = std::f32::consts::PI * (i as f32 + 1.0) / (WB_LPC_ORDER as f32 + 1.0);
    }
    out
}

fn hamming_window(x: &[f32]) -> [f32; UWB_FRAME_SIZE] {
    let mut out = [0.0f32; UWB_FRAME_SIZE];
    let n = UWB_FRAME_SIZE as f32 - 1.0;
    for i in 0..UWB_FRAME_SIZE {
        let w = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / n).cos();
        out[i] = x[i] * w;
    }
    out
}

fn autocorrelate(x: &[f32], r: &mut [f32]) {
    let order = r.len() - 1;
    for k in 0..=order {
        let mut s = 0.0f32;
        for i in 0..x.len() - k {
            s += x[i] * x[i + k];
        }
        r[k] = s;
    }
}

fn levinson_durbin(r: &[f32]) -> [f32; WB_LPC_ORDER] {
    let mut a = [0.0f32; WB_LPC_ORDER];
    let mut tmp = [0.0f32; WB_LPC_ORDER];
    let mut e = r[0];
    if e <= 0.0 {
        return a;
    }
    for i in 0..WB_LPC_ORDER {
        let mut k = -r[i + 1];
        for j in 0..i {
            k -= a[j] * r[i - j];
        }
        if e.abs() < 1e-12 {
            break;
        }
        k /= e;
        const K_MAX: f32 = 0.999;
        k = k.clamp(-K_MAX, K_MAX);
        tmp[i] = k;
        for j in 0..i {
            tmp[j] = a[j] + k * a[i - 1 - j];
        }
        a[..=i].copy_from_slice(&tmp[..=i]);
        e *= 1.0 - k * k;
        if e <= 0.0 {
            e = 1e-6;
        }
    }
    a
}

fn fir_filter_stateless(x: &[f32], a: &[f32], y: &mut [f32], order: usize) {
    let n = x.len();
    for i in 0..n {
        let mut acc = x[i];
        for k in 0..order {
            if i > k {
                acc += a[k] * x[i - k - 1];
            }
        }
        y[i] = acc;
    }
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    (s / x.len() as f32).sqrt()
}

fn lpc_to_lsp_high(ak: &[f32; WB_LPC_ORDER]) -> Option<[f32; WB_LPC_ORDER]> {
    let p = WB_LPC_ORDER;
    let mut a_pad = [0.0f32; WB_LPC_ORDER + 2];
    a_pad[0] = 1.0;
    a_pad[1..=p].copy_from_slice(&ak[..p]);

    let n = p + 2;
    let mut pcoef = [0.0f32; 10];
    let mut qcoef = [0.0f32; 10];
    for k in 0..n {
        let reflected = a_pad[(p + 1) - k];
        pcoef[k] = a_pad[k] + reflected;
        qcoef[k] = a_pad[k] - reflected;
    }

    let eval_p = |coeffs: &[f32; 10], omega: f32| -> f32 {
        let half = (n as f32 - 1.0) * 0.5;
        let mut s = 0.0f32;
        for k in 0..(n / 2) {
            s += 2.0 * coeffs[k] * ((half - k as f32) * omega).cos();
        }
        s
    };
    let eval_q = |coeffs: &[f32; 10], omega: f32| -> f32 {
        let half = (n as f32 - 1.0) * 0.5;
        let mut s = 0.0f32;
        for k in 0..(n / 2) {
            s += 2.0 * coeffs[k] * ((half - k as f32) * omega).sin();
        }
        s
    };

    const GRID_N: usize = 1024;
    let grid_to_omega = |i: usize| (i as f32 * std::f32::consts::PI) / GRID_N as f32;

    fn scan_roots(
        coeffs: &[f32; 10],
        grid: usize,
        to_omega: impl Fn(usize) -> f32,
        eval: impl Fn(&[f32; 10], f32) -> f32,
    ) -> Vec<f32> {
        let mut roots = Vec::with_capacity(WB_LPC_ORDER / 2);
        let eps = 1.0 / (grid as f32 * 2.0);
        let clamp_omega = |o: f32| o.clamp(eps, std::f32::consts::PI - eps);
        let mut prev_o = clamp_omega(to_omega(1));
        let mut prev_f = eval(coeffs, prev_o);
        for i in 2..grid {
            let o = clamp_omega(to_omega(i));
            let f = eval(coeffs, o);
            if prev_f * f < 0.0 {
                let mut lo = prev_o;
                let mut hi = o;
                let mut flo = prev_f;
                for _ in 0..32 {
                    let mid = 0.5 * (lo + hi);
                    let fmid = eval(coeffs, mid);
                    if fmid == 0.0 {
                        lo = mid;
                        hi = mid;
                        break;
                    }
                    if flo * fmid < 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                        flo = fmid;
                    }
                }
                roots.push(0.5 * (lo + hi));
            }
            prev_o = o;
            prev_f = f;
        }
        roots
    }

    let r_p = scan_roots(&pcoef, GRID_N, grid_to_omega, eval_p);
    let r_q = scan_roots(&qcoef, GRID_N, grid_to_omega, eval_q);
    let mut roots = Vec::with_capacity(p);
    for &r in r_p.iter().chain(r_q.iter()) {
        if r > 1e-3 && r < std::f32::consts::PI - 1e-3 {
            roots.push(r);
        }
    }
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if roots.len() < p {
        return None;
    }
    roots.truncate(p);
    let mut out = [0.0f32; WB_LPC_ORDER];
    out[..p].copy_from_slice(&roots[..p]);
    let margin = LSP_MARGIN_HIGH;
    out[0] = out[0].max(margin);
    for i in 1..WB_LPC_ORDER {
        if out[i] < out[i - 1] + margin {
            out[i] = out[i - 1] + margin;
        }
    }
    out[WB_LPC_ORDER - 1] = out[WB_LPC_ORDER - 1].min(std::f32::consts::PI - margin);
    Some(out)
}

fn quantise_lsp_high(lsp: &[f32; WB_LPC_ORDER]) -> ([f32; WB_LPC_ORDER], usize, usize) {
    let mut residual = [0.0f32; WB_LPC_ORDER];
    for i in 0..WB_LPC_ORDER {
        residual[i] = lsp[i] - (0.3125 * i as f32 + 0.75);
    }
    let idx1 = nearest_vector_scaled(&residual, 256.0, &HIGH_LSP_CDBK, 8, 64);
    for i in 0..8 {
        residual[i] -= (HIGH_LSP_CDBK[idx1 * 8 + i] as f32) / 256.0;
    }
    let idx2 = nearest_vector_scaled(&residual, 512.0, &HIGH_LSP_CDBK2, 8, 64);

    let mut qlsp = [0.0f32; WB_LPC_ORDER];
    for i in 0..WB_LPC_ORDER {
        qlsp[i] = 0.3125 * i as f32
            + 0.75
            + (HIGH_LSP_CDBK[idx1 * 8 + i] as f32) / 256.0
            + (HIGH_LSP_CDBK2[idx2 * 8 + i] as f32) / 512.0;
    }
    let margin = LSP_MARGIN_HIGH;
    qlsp[0] = qlsp[0].max(margin);
    for i in 1..WB_LPC_ORDER {
        if qlsp[i] < qlsp[i - 1] + margin {
            qlsp[i] = qlsp[i - 1] + margin;
        }
    }
    qlsp[WB_LPC_ORDER - 1] = qlsp[WB_LPC_ORDER - 1].min(std::f32::consts::PI - margin);

    (qlsp, idx1, idx2)
}

fn nearest_vector_scaled(
    target: &[f32],
    scale: f32,
    cdbk: &[i8],
    dim: usize,
    count: usize,
) -> usize {
    let inv = 1.0 / scale;
    let mut best_idx = 0usize;
    let mut best_err = f32::INFINITY;
    for idx in 0..count {
        let mut err = 0.0f32;
        let base = idx * dim;
        for i in 0..dim {
            let v = cdbk[base + i] as f32 * inv;
            let d = target[i] - v;
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best_idx = idx;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_layer_adds_one_bit() {
        let mut enc = UwbEncoder::with_null_layer();
        let mut bw = BitWriter::new();
        let mut pcm = [0.0f32; UWB_FULL_FRAME_SIZE];
        for i in 0..UWB_FULL_FRAME_SIZE {
            pcm[i] = 3000.0 * (i as f32 * 0.04).sin();
        }
        enc.encode_frame(&pcm, &mut bw).unwrap();
        // WB at submode 3 writes 492 bits; UWB null adds 1 ⇒ 493.
        assert_eq!(bw.bit_position(), 493);
    }

    #[test]
    fn folding_layer_adds_36_bits() {
        let mut enc = UwbEncoder::with_submode(1).unwrap();
        let mut bw = BitWriter::new();
        let mut pcm = [0.0f32; UWB_FULL_FRAME_SIZE];
        for i in 0..UWB_FULL_FRAME_SIZE {
            pcm[i] = 3000.0 * (i as f32 * 0.04).sin() + 1500.0 * (i as f32 * 0.12).cos();
        }
        enc.encode_frame(&pcm, &mut bw).unwrap();
        // WB at submode 3 = 492 + UWB folding = 36 ⇒ 528.
        assert_eq!(bw.bit_position(), 528);
        assert_eq!(enc.bits_per_frame(), 528);
    }

    #[test]
    fn unsupported_submode_rejected() {
        for bad in [2u32, 3, 4, 7] {
            assert!(UwbEncoder::with_submode(bad).is_err());
        }
    }
}
