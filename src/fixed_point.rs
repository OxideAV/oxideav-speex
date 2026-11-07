//! **Fixed-point decode primitives** — the integer schedules the staged
//! Tracer round documents for the reference's fixed-point build
//! (`docs/audio/speex/tracer-2026-09.md` §1.2, §3, §4), each reproduced
//! exactly against the staged fixed-point traces
//! (`tests/fixed_point_primitives.rs`):
//!
//! * the LSP cosine in Q13 and its stored 16-bit `2·cos` (Q14) form,
//!   including the wrap to −32768 at `cos = +1` (all 25 737 inputs);
//! * the narrowband / high-band LSP decode in Q13 (`π = 25736`),
//!   sub-frame interpolation, and the LSP → LPC product-of-sections
//!   schedule to Q13 LPC, checked through `A(−1)` on every traced
//!   sub-frame of both bands;
//! * the high-band gain schedule `g_c` (Q7), `R` (Q7), `S` (Q14);
//! * the integer noise draw.
//!
//! The rest of the fixed-point build (narrowband excitation, pitch and
//! gain arithmetic, the synthesis filter, QMF, output high-pass and the
//! enhancer's integer schedule) is not staged, so this crate has no
//! fixed-point *decoder*; these are its documented building blocks.

/// `π` in the Q13 LSP angle domain.
pub const LSP_PI_Q13: i32 = 25736;

#[inline]
fn r13(a: i32, b: i32) -> i32 {
    (a * b + 4096) >> 13
}

/// The fixed-point LSP cosine (tracer §3): Q13 angle → Q13 cosine,
/// an even polynomial with the breakpoint at `π/2` (12868).
pub fn lsp_cos_q13(x: i32) -> i32 {
    let bracket = |u: i32| r13(u, -4096 + r13(u, 340 + r13(-10, u)));
    if x < 12868 {
        let u = r13(x, x);
        8192 + bracket(u)
    } else {
        let y = LSP_PI_Q13 - x;
        let u = r13(y, y);
        -8192 - bracket(u)
    }
}

/// The 16-bit value the LSP → LPC recursion stores for an angle:
/// `2·cos` in Q14, i.e. `cos_q13·4` truncated to 16 bits — which wraps
/// to −32768 for `cos = +8192` (angles ≲ 0.011 rad), as the reference
/// does.
pub fn lsp_cos_stored(x: i32) -> i16 {
    (lsp_cos_q13(x) * 4) as i16
}

/// Product of a stored Q14 multiplier with a 32-bit accumulator,
/// `c·(v ≫ 14) + ((c·(v & 0x3FFF)) ≫ 14)`.
#[inline]
fn mul_q14(c: i32, v: i32) -> i32 {
    c.wrapping_mul(v >> 14)
        .wrapping_add(c.wrapping_mul(v & 0x3FFF) >> 14)
}

/// Fixed-point LSP → LPC (tracer §3): `P` over the even-indexed
/// sections, `Q` over the odd, from a `2²⁰` impulse; output
/// `a_k = (P_k + P_{k−1} + Q_k − Q_{k−1} + 128) ≫ 8`, clamped to
/// ±32767 — the Q13 coefficients of `A(z) = 1 + Σ a_k z^{−k}`.
pub fn lsp_to_lpc_q13<const P: usize>(lsp_q13: &[i32; P]) -> [i32; P] {
    const MAX: usize = 17;
    assert!(P < MAX, "LPC order");
    let mut pp = [0i32; MAX];
    let mut qq = [0i32; MAX];
    pp[0] = 1 << 20;
    qq[0] = 1 << 20;
    let (mut np, mut nq) = (1usize, 1usize);
    for (i, &w) in lsp_q13.iter().enumerate() {
        let c = i32::from(lsp_cos_stored(w));
        let (t, n) = if i % 2 == 0 {
            (&mut pp, &mut np)
        } else {
            (&mut qq, &mut nq)
        };
        // Multiply by (1 − 2cos(ω)·z⁻¹ + z⁻²), high index first.
        let len = *n + 2;
        for k in (0..len).rev() {
            let mut v = t[k];
            if k >= 1 {
                v = v.wrapping_sub(mul_q14(c, t[k - 1]));
            }
            if k >= 2 {
                v = v.wrapping_add(t[k - 2]);
            }
            t[k] = v;
        }
        *n = len;
    }
    let mut a = [0i32; P];
    for k in 1..=P {
        let v =
            i64::from(pp[k]) + i64::from(pp[k - 1]) + i64::from(qq[k]) - i64::from(qq[k - 1]) + 128;
        a[k - 1] = ((v >> 8) as i32).clamp(-32767, 32767);
    }
    a
}

/// `A(−1)` in Q13 of a Q13 LPC set: `8192 + Σ (−1)^k a_k`.
pub fn lpc_at_minus_one_q13(a: &[i32]) -> i32 {
    let mut v = 8192;
    for (k, &x) in a.iter().enumerate() {
        if k % 2 == 0 {
            v -= x;
        } else {
            v += x;
        }
    }
    v
}

/// Narrowband LSPs in Q13: `(i+1)·2048` plus the VQ stages at shifts
/// 5 / 4 / 3 (`1/256`, `1/512`, `1/1024` rad). `None` for a frame with
/// no LSP field.
pub fn nb_lsp_q13(
    body: &crate::NarrowbandFrameBody,
    submode: &crate::NarrowbandSubmode,
) -> Option<[i32; 10]> {
    use crate::codebooks::{nb_lsp_high1, nb_lsp_high2, nb_lsp_low1, nb_lsp_low2, nb_lsp_stage0};
    let st = body.lsp_stages(submode)?;
    let mut l = [0i32; 10];
    let s0 = nb_lsp_stage0().get(usize::from(st.stage0))?;
    for (i, v) in l.iter_mut().enumerate() {
        *v = ((i as i32 + 1) << 11) + (i32::from(s0[i]) << 5);
    }
    let low1 = nb_lsp_low1().get(usize::from(st.low1))?;
    let high1 = nb_lsp_high1().get(usize::from(st.high1))?;
    for i in 0..5 {
        l[i] += i32::from(low1[i]) << 4;
        l[5 + i] += i32::from(high1[i]) << 4;
    }
    if let Some(x) = st.low2 {
        let row = nb_lsp_low2().get(usize::from(x))?;
        for i in 0..5 {
            l[i] += i32::from(row[i]) << 3;
        }
    }
    if let Some(x) = st.high2 {
        let row = nb_lsp_high2().get(usize::from(x))?;
        for i in 0..5 {
            l[5 + i] += i32::from(row[i]) << 3;
        }
    }
    Some(l)
}

/// High-band LSPs in Q13: `i·2560 + 6144` plus the two MSVQ stages at
/// shifts 5 / 4. `None` for a frame with no LSP field.
pub fn hb_lsp_q13(
    body: &crate::WidebandHighBandBody,
    submode: &crate::WidebandHighBandSubmode,
) -> Option<[i32; 8]> {
    let st = body.lsp_stages(submode)?;
    let s1 = crate::codebooks::hb_lsp_stage1().get(usize::from(st.stage1))?;
    let s2 = crate::codebooks::hb_lsp_stage2().get(usize::from(st.stage2))?;
    let mut l = [0i32; 8];
    for (i, v) in l.iter_mut().enumerate() {
        *v = i as i32 * 2560 + 6144 + (i32::from(s1[i]) << 5) + (i32::from(s2[i]) << 4);
    }
    Some(l)
}

/// Sub-frame `s` (0..=3) interpolation of Q13 LSPs, weight
/// `w = (s+1)/4` on the current frame. Every decoded Q13 LSP is a
/// multiple of 8, so the quarter-step blend is exact.
pub fn interpolate_lsp_q13<const P: usize>(prev: &[i32; P], cur: &[i32; P], s: usize) -> [i32; P] {
    let k = s as i32 + 1;
    let mut out = [0i32; P];
    for i in 0..P {
        out[i] = ((4 - k) * prev[i] + k * cur[i]) >> 2;
    }
    out
}

/// High-band gain-correction boundaries in their fixed-point form
/// (staged `tables/hb-gc-quant-bound-q15.csv`; they act as Q7 here).
const HB_GC_BOUND_Q: [i64; 16] = [
    125, 164, 215, 282, 370, 484, 635, 832, 1090, 1428, 1871, 2452, 3213, 4210, 5516, 7228,
];

/// The fixed-point high-band gain schedule (tracer §1.2): returns
/// `(g_c Q7, R Q7, S Q14)` for the 4-bit correction field `q`, the
/// integer narrowband excitation RMS `e_l`, and the two bands' Q13
/// `A(−1)` values.
pub fn hb_gain_q14(q: u8, e_l: i32, a_nb_q13: i32, a_hb_q13: i32) -> (i32, i32, i32) {
    let gc = (28626 * HB_GC_BOUND_Q[usize::from(q & 15)]) >> 15;
    let b = i64::from(a_hb_q13) + 82;
    let b16 = i64::from(b as i16);
    let r = (((i64::from(a_nb_q13) + 82) * 128 + b16 / 2) / b).clamp(-32768, 32767);
    let s = ((gc * i64::from(e_l) * 8 + r / 2) / r) * 2048;
    (gc as i32, r as i32, s as i32)
}

/// The integer noise generator (tracer §4): the decoder-lifetime LCG
/// state, seeded 1000, shared with the float path's sequence.
#[derive(Debug, Clone)]
pub struct FixedNoise {
    state: u32,
}

impl Default for FixedNoise {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedNoise {
    /// Generator at its decoder-creation seed.
    pub fn new() -> Self {
        Self { state: 1000 }
    }

    /// Current 32-bit state.
    pub fn state(&self) -> u32 {
        self.state
    }

    /// One draw at standard deviation `std` (Q0):
    /// `v = σ ≫ 16` as signed 16-bit, `(v·std − ((v·std) ≫ 3) + 2¹³) ≫ 14`.
    pub fn draw(&mut self, std: i32) -> i32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        let v = i64::from((self.state >> 16) as u16 as i16);
        let p = v * i64::from(std);
        ((p - (p >> 3) + 8192) >> 14) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_endpoints_and_wrap() {
        assert_eq!(lsp_cos_q13(0), 8192);
        assert_eq!(lsp_cos_stored(0), i16::MIN, "the documented 16-bit wrap");
        assert_eq!(lsp_cos_q13(LSP_PI_Q13), -8192);
        assert!(lsp_cos_q13(12868).abs() < 8);
    }

    #[test]
    fn flat_lsps_give_a_flat_filter() {
        // LSPs at k·π/11 are the roots of 1 ± z^{-11}: A(z) ≈ 1.
        let mut l = [0i32; 10];
        for (i, v) in l.iter_mut().enumerate() {
            *v = (LSP_PI_Q13 * (i as i32 + 1)) / 11;
        }
        let a = lsp_to_lpc_q13(&l);
        assert!(a.iter().all(|&x| x.abs() < 64), "{a:?}");
    }
}
