//! Per-frame rate decisions for the stream encoder: voice-activity
//! detection, variable bit-rate (VBR) sub-mode election, average
//! bit-rate (ABR) steering and discontinuous transmission (DTX).
//!
//! ## What the staged material pins
//!
//! *The Speex Codec Manual* §2.1 defines the four behaviours and leaves
//! every decision rule to the encoder:
//!
//! * **VBR** — *"sounds like vowels and high-energy transients require a
//!   higher bit-rate to achieve good quality, while fricatives (e.g. s,f
//!   sounds) can be coded adequately with less bits"*; the VBR quality
//!   is a float 0..=10.
//! * **ABR** — *"dynamically adjusts VBR quality in order to meet a
//!   specific target bit-rate"*, open loop.
//! * **VAD** — *"detects non-speech periods and encode[s] them with just
//!   enough bits to reproduce the background noise"* (comfort noise);
//!   implicit in VBR.
//! * **DTX** — *"stop transmitting completely when the background noise
//!   is stationary"*; in a file, a 5-bit frame.
//!
//! Any stream these decisions produce is ordinary Speex: each frame
//! carries its own sub-mode IDs, so a decoder needs no notice of the
//! rate mode.
//!
//! ## The rules used here
//!
//! One [`SignalTracker`] classifies each 20 ms frame from four
//! measurements of the 0–4 kHz band: its level, a tracked noise floor
//! (falls at once, rises 2 dB/s), a tracked speech level, and the
//! frame's voicing (the best normalised autocorrelation over the 17–144
//! sample pitch range).
//!
//! * A frame is **speech** when it stands 7 dB above the floor (3 dB if
//!   strongly voiced), with a 5-frame hangover after three consecutive
//!   speech frames.
//! * VBR moves the frame's quality around the requested one: strongly
//!   voiced frames (voicing ≥ 0.8) keep it, an onset (a rise of 6 dB or
//!   more over the previous frame) adds 1.5, weaker voicing takes off
//!   up to 2 (reached at voicing 0.4), and frames 15–35 dB under the
//!   running speech level lose up to 2 more. Non-speech frames take
//!   the lowest ladder step (the vocoder mode — comfort noise).
//! * ABR runs VBR at a quality steered by the accumulated bit surplus:
//!   one second's worth of surplus lowers the quality by six steps.
//! * DTX sends null frames once two consecutive comfort-noise frames
//!   agree in level within 2 dB, until the level leaves a 3 dB corridor
//!   around the last transmitted comfort-noise frame.

use crate::narrowband_body::{PITCH_PERIOD_MAX, PITCH_PERIOD_MIN};
use crate::quality::MAX_QUALITY;

/// Samples of the 8 kHz analysis signal per frame.
const ANALYSIS_FRAME: usize = 160;
/// Level a speech frame must stand above the noise floor, dB.
const SPEECH_MARGIN_DB: f64 = 7.0;
/// The same margin for a strongly voiced frame.
const VOICED_MARGIN_DB: f64 = 3.0;
/// Voicing above which a frame counts as strongly voiced.
const STRONG_VOICING: f64 = 0.7;
/// Frames kept as speech after speech ends.
const HANGOVER_FRAMES: u32 = 5;
/// Noise-floor rise per frame, dB (2 dB/s).
const FLOOR_RISE_DB: f64 = 0.04;

/// What one frame is, for the rate decisions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FrameClass {
    /// Frame level, dB over one LSB² of mean power.
    pub level_db: f64,
    /// Speech (true) or background (false), hangover included.
    pub speech: bool,
    /// Best normalised autocorrelation over the pitch range, 0..=1.
    pub voicing: f64,
    /// Level rise over the previous frame, dB (negative for a fall).
    pub rise_db: f64,
    /// Level relative to the running speech level, dB (≤ 0 mostly).
    pub below_speech_db: f64,
}

/// Running signal statistics behind [`FrameClass`].
#[derive(Debug, Clone)]
pub(crate) struct SignalTracker {
    /// The previous frames' 8 kHz analysis samples (pitch look-back).
    hist: Vec<f64>,
    floor_db: Option<f64>,
    speech_db: f64,
    prev_db: f64,
    run: u32,
    hangover: u32,
}

impl SignalTracker {
    pub(crate) fn new() -> Self {
        Self {
            hist: vec![0.0; usize::from(PITCH_PERIOD_MAX)],
            floor_db: None,
            speech_db: 0.0,
            prev_db: 0.0,
            run: 0,
            hangover: 0,
        }
    }

    /// Classify one frame (`pcm` at 8, 16 or 32 kHz, mono).
    pub(crate) fn classify(&mut self, pcm: &[i16]) -> FrameClass {
        // The 0–4 kHz band at 8 kHz: average each run of 1 / 2 / 4
        // input samples.
        let step = (pcm.len() / ANALYSIS_FRAME).max(1);
        let frame: Vec<f64> = pcm
            .chunks(step)
            .map(|c| c.iter().map(|&s| f64::from(s)).sum::<f64>() / c.len() as f64)
            .collect();
        let power = frame.iter().map(|v| v * v).sum::<f64>() / frame.len().max(1) as f64;
        let level_db = 10.0 * (power + 1.0).log10();

        // Voicing: the best normalised correlation of the frame with
        // itself one pitch period earlier.
        let hlen = self.hist.len();
        let at = |i: isize| -> f64 {
            if i >= 0 {
                frame[i as usize]
            } else {
                self.hist[(hlen as isize + i) as usize]
            }
        };
        let mut voicing = 0.0f64;
        if power > 1.0 {
            for lag in PITCH_PERIOD_MIN..=PITCH_PERIOD_MAX {
                let (mut xy, mut yy) = (0.0f64, 0.0f64);
                for (n, &x) in frame.iter().enumerate() {
                    let y = at(n as isize - isize::from(lag as i16));
                    xy += x * y;
                    yy += y * y;
                }
                if xy > 0.0 {
                    voicing = voicing.max(xy / (yy * power * frame.len() as f64).sqrt().max(1e-9));
                }
            }
        }
        let voicing = voicing.min(1.0);
        let keep = hlen.saturating_sub(frame.len());
        self.hist.drain(..hlen - keep);
        let tail = frame.len().saturating_sub(hlen);
        self.hist.extend_from_slice(&frame[tail..]);

        // Noise floor: follows a fall at once, rises slowly.
        let floor = match self.floor_db {
            None => level_db.min(30.0),
            Some(f) if level_db < f => level_db,
            Some(f) => f + FLOOR_RISE_DB,
        };
        self.floor_db = Some(floor);
        let over = level_db - floor;
        let raw = over > SPEECH_MARGIN_DB || (voicing > STRONG_VOICING && over > VOICED_MARGIN_DB);
        let speech = if raw {
            self.run += 1;
            if self.run >= 3 {
                self.hangover = HANGOVER_FRAMES;
            }
            true
        } else {
            self.run = 0;
            if self.hangover > 0 {
                self.hangover -= 1;
                true
            } else {
                false
            }
        };
        // Speech level: rises quickly to loud speech, decays slowly.
        if raw {
            self.speech_db = if level_db > self.speech_db {
                0.6 * self.speech_db + 0.4 * level_db
            } else {
                self.speech_db - 0.02
            };
        }
        let rise_db = level_db - self.prev_db;
        self.prev_db = level_db;
        FrameClass {
            level_db,
            speech,
            voicing,
            rise_db,
            below_speech_db: level_db - self.speech_db,
        }
    }
}

/// What the encoder should spend on one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameSpend {
    /// A coded frame at this ladder quality.
    Quality(u8),
    /// Comfort noise: the lowest ladder step.
    ComfortNoise,
    /// A null (no transmission) frame.
    Null,
}

/// The rate mode and its running state.
#[derive(Debug, Clone)]
pub(crate) struct RateController {
    tracker: SignalTracker,
    /// VBR quality, when VBR is on.
    pub vbr_quality: Option<f32>,
    /// ABR target in bit/s, when ABR is on.
    pub abr_bps: Option<u32>,
    /// Ceiling for any single frame's rate in VBR / ABR, bit/s.
    pub max_bps: Option<u32>,
    /// Comfort noise for non-speech frames in constant-quality mode.
    pub vad: bool,
    /// Null frames while the background is stationary.
    pub dtx: bool,
    /// Bits spent beyond the ABR target so far.
    surplus_bits: f64,
    /// ABR's starting quality (set with the target).
    abr_base: f32,
    /// Consecutive comfort-noise frames and the level they agree on.
    noise_run: u32,
    noise_db: f64,
}

impl RateController {
    pub(crate) fn new() -> Self {
        Self {
            tracker: SignalTracker::new(),
            vbr_quality: None,
            abr_bps: None,
            max_bps: None,
            vad: false,
            dtx: false,
            surplus_bits: 0.0,
            abr_base: 8.0,
            noise_run: 0,
            noise_db: 0.0,
        }
    }

    /// Whether any per-frame decision is on (otherwise every frame is
    /// coded at the configured quality and the signal is not analysed).
    pub(crate) fn is_adaptive(&self) -> bool {
        self.vbr_quality.is_some() || self.abr_bps.is_some() || self.vad || self.dtx
    }

    /// Start ABR at `bps` from the quality `base` (the highest ladder
    /// step not above the target).
    pub(crate) fn set_abr(&mut self, bps: Option<u32>, base: u8) {
        self.abr_bps = bps;
        self.abr_base = f32::from(base);
        self.surplus_bits = 0.0;
    }

    /// The VBR quality in force for the next frame (ABR steers it).
    pub(crate) fn steering_quality(&self) -> Option<f32> {
        match self.abr_bps {
            Some(target) => {
                let q =
                    f64::from(self.abr_base) - 6.0 * self.surplus_bits / f64::from(target.max(1));
                Some(q.clamp(0.0, f64::from(MAX_QUALITY)) as f32)
            }
            None => self.vbr_quality,
        }
    }

    /// Decide one frame. `cbr_quality` is the configured constant
    /// quality; `rate_of(q)` the ladder's bit-rate at quality `q`.
    pub(crate) fn decide(
        &mut self,
        pcm: &[i16],
        cbr_quality: u8,
        rate_of: impl Fn(u8) -> u32,
    ) -> FrameSpend {
        if !self.is_adaptive() {
            return FrameSpend::Quality(cbr_quality);
        }
        let class = self.tracker.classify(pcm);
        let variable = self.steering_quality();
        if !class.speech && (self.vad || self.dtx || variable.is_some()) {
            // Background: comfort noise, or nothing while it holds still.
            let steady = (class.level_db - self.noise_db).abs();
            if self.dtx && self.noise_run >= 2 && steady < 3.0 {
                self.noise_run += 1;
                return FrameSpend::Null;
            }
            if self.noise_run > 0 && steady < 2.0 {
                self.noise_run += 1;
            } else {
                self.noise_run = 1;
            }
            self.noise_db = class.level_db;
            return FrameSpend::ComfortNoise;
        }
        self.noise_run = 0;
        let Some(q) = variable else {
            return FrameSpend::Quality(cbr_quality);
        };
        let mut q = f64::from(q);
        // Vowels hold the requested quality, onsets go above it,
        // unvoiced and quiet frames below.
        if class.rise_db >= 6.0 {
            q += 1.5;
        }
        q -= 2.0 * ((0.8 - class.voicing) / 0.4).clamp(0.0, 1.0);
        q -= 2.0 * ((-class.below_speech_db - 15.0) / 20.0).clamp(0.0, 1.0);
        let mut q = q.round().clamp(1.0, f64::from(MAX_QUALITY)) as u8;
        if let Some(max) = self.max_bps {
            while q > 0 && rate_of(q) > max {
                q -= 1;
            }
        }
        FrameSpend::Quality(q)
    }

    /// Account the bits one frame took (ABR bookkeeping).
    pub(crate) fn account(&mut self, bits: usize) {
        if let Some(target) = self.abr_bps {
            self.surplus_bits +=
                bits as f64 - f64::from(target) / f64::from(crate::quality::FRAMES_PER_SECOND);
            // Bound the integrator (two seconds of target bits) so it
            // recovers quickly after long pauses.
            let lim = f64::from(target) * 2.0;
            self.surplus_bits = self.surplus_bits.clamp(-lim, lim);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(amp: f64, n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| (amp * (2.0 * core::f64::consts::PI * 200.0 * i as f64 / 8000.0).sin()) as i16)
            .collect()
    }

    fn hiss(amp: f64, n: usize, seed: &mut u32) -> Vec<i16> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (amp * (f64::from(*seed >> 8) / f64::from(1u32 << 23) - 1.0)) as i16
            })
            .collect()
    }

    #[test]
    fn a_periodic_frame_is_voiced_and_noise_is_not() {
        let mut t = SignalTracker::new();
        t.classify(&tone(8000.0, 160));
        let c = t.classify(&tone(8000.0, 160));
        assert!(c.voicing > 0.9, "tone voicing {}", c.voicing);
        let mut seed = 7;
        let mut t = SignalTracker::new();
        t.classify(&hiss(8000.0, 160, &mut seed));
        let c = t.classify(&hiss(8000.0, 160, &mut seed));
        assert!(c.voicing < 0.4, "noise voicing {}", c.voicing);
    }

    #[test]
    fn speech_stands_out_of_the_floor_with_a_hangover() {
        let mut t = SignalTracker::new();
        let mut seed = 3;
        for _ in 0..20 {
            assert!(!t.classify(&hiss(30.0, 160, &mut seed)).speech);
        }
        for _ in 0..10 {
            assert!(t.classify(&tone(6000.0, 160)).speech);
        }
        let after: Vec<bool> = (0..12)
            .map(|_| t.classify(&hiss(30.0, 160, &mut seed)).speech)
            .collect();
        assert!(after[..HANGOVER_FRAMES as usize].iter().all(|&s| s));
        assert!(!after[11]);
    }

    #[test]
    fn constant_quality_passes_through_untouched() {
        let mut rc = RateController::new();
        assert!(!rc.is_adaptive());
        assert_eq!(
            rc.decide(&[0; 160], 6, |_| 0),
            FrameSpend::Quality(6),
            "no analysis without a rate mode"
        );
    }

    #[test]
    fn dtx_goes_quiet_on_a_steady_background_and_wakes_for_speech() {
        let mut rc = RateController::new();
        rc.vad = true;
        rc.dtx = true;
        let mut seed = 11;
        let spends: Vec<FrameSpend> = (0..10)
            .map(|_| rc.decide(&hiss(40.0, 160, &mut seed), 8, |_| 0))
            .collect();
        assert_eq!(spends[0], FrameSpend::ComfortNoise);
        assert!(spends[4..].iter().all(|&s| s == FrameSpend::Null));
        let mut woke = false;
        for _ in 0..3 {
            woke |= rc.decide(&tone(7000.0, 160), 8, |_| 0) == FrameSpend::Quality(8);
        }
        assert!(woke);
    }

    #[test]
    fn abr_steers_quality_against_the_surplus() {
        let mut rc = RateController::new();
        rc.set_abr(Some(10_000), 4);
        assert_eq!(rc.steering_quality(), Some(4.0));
        for _ in 0..50 {
            rc.account(300); // 15 kbit/s against a 10 kbit/s target
        }
        assert!(rc.steering_quality().unwrap() < 1.5);
        for _ in 0..200 {
            rc.account(40);
        }
        assert!(rc.steering_quality().unwrap() > 6.0);
    }

    #[test]
    fn vbr_spends_more_on_vowels_than_on_hiss() {
        let mut rc = RateController::new();
        rc.vbr_quality = Some(6.0);
        let mut seed = 5;
        // Establish a floor, then compare a voiced and an unvoiced
        // frame at the same level.
        for _ in 0..10 {
            rc.decide(&hiss(20.0, 160, &mut seed), 8, |_| 0);
        }
        let mut voiced = FrameSpend::Null;
        for _ in 0..6 {
            voiced = rc.decide(&tone(6000.0, 160), 8, |_| 0);
        }
        let mut unvoiced = FrameSpend::Null;
        for _ in 0..6 {
            unvoiced = rc.decide(&hiss(8000.0, 160, &mut seed), 8, |_| 0);
        }
        let (FrameSpend::Quality(v), FrameSpend::Quality(u)) = (voiced, unvoiced) else {
            panic!("expected coded frames, got {voiced:?} / {unvoiced:?}");
        };
        assert!(v > u, "voiced q{v} vs unvoiced q{u}");
    }

    #[test]
    fn the_frame_ceiling_caps_vbr() {
        let mut rc = RateController::new();
        rc.vbr_quality = Some(9.0);
        rc.max_bps = Some(8_000);
        let rate = |q: u8| 2_000 * u32::from(q);
        let mut top = 0;
        for _ in 0..8 {
            if let FrameSpend::Quality(q) = rc.decide(&tone(9000.0, 160), 8, rate) {
                top = top.max(q);
            }
        }
        assert_eq!(top, 4);
    }
}
