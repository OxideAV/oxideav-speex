# High-band mode-4 gain-base probe fixtures (round r446)

Black-box encode/decode reference I/O for the mode-4 gain-base
discrimination gate (`tests/hb_mode4_gain_probe_fixture.rs`) — the
"fixture pair separating the drivers" that
`docs/audio/speex/provenance/08-qmf-recovered-hb-excitation.md` names
as what would close #329 residual 1, generated locally per the r410
conformance-fixture precedent. Per the workspace clean-room policy,
`speexenc` / `speexdec` are invoked as **opaque binaries**; their
source is NOT consulted.

- Toolchain: `speexenc` / `speexdec` **1.2.1** (Homebrew build, macOS
  arm64). Reference decodes use `--no-enh`.
- Both streams: 16 kHz wideband, `--quality 10 --comp 10` → every
  frame NB submode 7 + HB submode 4 (81 audio frames; the reference
  PCM is source-length-trimmed to 25 600 samples).
- The two sources share one deterministic construction (script below,
  seed `0x5EEC`) and differ **only** in which band's amplitude steps
  ×{0.05, 0.12, 0.3, 0.75, 1.9} across five 16-frame segments:
  - `lbvar` — high-band content **fixed**, low-band amplitude steps
    (38× ≈ 31.6 dB span);
  - `hbvar` — low-band amplitude **fixed**, high-band amplitude steps.

## What the pair discriminates (r446 findings, gate-pinned)

Recovering the 4–8 kHz excitation from each reference decode (staged
QMF prototype, global order-8 LPC — legitimate: the high-band spectral
*shape* is constant per stream — projection onto the innovation
rebuilt from the transmitted indices, the provenance/08 route):

1. **The reference gain base is NOT the same frame's low-band level.**
   On `lbvar` the recovered per-sub-frame gain moves by only ≈ 2 dB
   (per-segment medians 28.5 → 36.2) across the 31.6 dB low-band
   sweep, and the transmitted 4-bit correction drifts only 3–4 grid
   steps. A causal `lb²` base would have required a ≈ 63 dB swing.
   The provenance/08 low-band correlation (R² 0.70 on natural speech)
   is **co-variation**, not causation — natural speech moves both
   bands together.
2. **The base is backward-adaptive (decoder-state gain memory).** On
   `hbvar` the recovered gain rises ≈ 18 dB across the high-band
   sweep while the transmitted correction stays parked at the **grid
   bottom** (per-segment median gc index 0–2): the adaptation is done
   entirely by a base the decoder derives from its own recent
   high-band excitation history (visible at segment transitions as a
   ≈ 1–2-frame settling), not by the transmitted field and not by the
   low band.
3. A third stream (envelope sweep at fixed levels, `envvar` — not
   committed, same generator with per-segment resonator centres
   4.6/5.3/6.0/6.7/7.4 kHz) showed only a weak, non-monotonic
   envelope effect (±0.5 log₁₀): the transmitted LSP envelope is not
   the base's driver either.

The reference's exact predictor **update rule** (time constant,
log-domain vs energy-domain averaging, cold-start value, and how the
correction feeds back into the memory) is not recoverable from
steady-state black-box probing — a backward-adaptive loop with wrong
constants accumulates multiplicative drift — so the crate's decode
law is unchanged and the update rule is the recorded docs ask (crate
README).

The crate's r440 fixture-fitted `(gc·lb_rms)²` law consequently
diverges on these off-manifold streams (4–8 kHz per-segment mean band
error 6–27 dB, pinned by the gate as a **known divergence**) while
remaining the best available fit on natural speech, where the
low band tracks the true state closely (wb/uwb q10 gates: ≈ 5.5–6 dB).

## Source generator (deterministic, seed 0x5EEC)

```sh
python3 -c "
import numpy as np, wave
sr=16000; frames=80; N=frames*320
rng=np.random.RandomState(0x5EEC)
def resonator(x,f0,bw):
    r=np.exp(-np.pi*bw/sr); th=2*np.pi*f0/sr
    a1,a2=2*r*np.cos(th),-r*r
    y=np.zeros_like(x)
    for n in range(len(x)):
        y[n]=x[n]+(a1*y[n-1] if n>=1 else 0)+(a2*y[n-2] if n>=2 else 0)
    return y
def bpnoise(lo,hi,n):
    z=rng.randn(n); X=np.fft.rfft(z); f=np.fft.rfftfreq(n,1/sr)
    X[(f<lo)|(f>hi)]=0
    return np.fft.irfft(X,n)
def lb_buzz(n):
    p=np.zeros(n); p[::114]=1.0
    y=resonator(p,500,120)+0.6*resonator(p,1400,180)
    y+=0.05*bpnoise(100,3500,n)
    return y/np.sqrt(np.mean(y**2))
def hb_fix(n,f0=6300):
    y=bpnoise(5200,7200,n)+0.8*resonator(0.3*rng.randn(n),f0,250)
    return y/np.sqrt(np.mean(y**2))
seg=np.repeat([0.05,0.12,0.3,0.75,1.9],N//5)
lb=lb_buzz(N); hb=hb_fix(N)
A_LB=0.22*32768; A_HB=0.055*32768
for name,sig in (('lbvar',seg*A_LB*lb+A_HB*hb),('hbvar',A_LB*lb+seg*A_HB*hb)):
    s=np.clip(sig,-30000,30000).astype('<i2')
    w=wave.open(name+'.wav','wb')
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(s.tobytes()); w.close()
"
```

NOTE: `hb_fix` must be evaluated after `lb_buzz` (both draw from the
one seeded RNG stream, in that order).

```sh
speexenc -w --quality 10 --comp 10 lbvar.wav lbvar.spx
speexdec --no-enh lbvar.spx lbvar.noenh.pcm      # likewise hbvar
```

## SHA-256

| file | SHA-256 |
| --- | --- |
| `lbvar.wav` (not committed) | `5db516b14a867d9f13ead08ec70bdffd8fcbcf3bde090258d099667c76b22dd4` |
| `lbvar.spx` | `8ca3b76af6d846bd9cf696dc9481b1b8260882c47f64a51b1e7ec87607a6737c` |
| `lbvar.noenh.pcm` | `0923090176e1b89569c06212c67e86867924b43b4000764a5f0bc6a7ca6fd31c` |
| `hbvar.wav` (not committed) | `d22f0ac21e9555519375411580cd98c4b2c775f2bacea23c774db96eebc36b01` |
| `hbvar.spx` | `1067e7f30c416ac4144003a7feb3ebaa1bb25201ff5f419c2441f859777cd4a7` |
| `hbvar.noenh.pcm` | `a7f9a3bf1b3789ad48b0a5bebe7c38d23fee9a5c135fbf83d291a771bf530f65` |
| `envvar.wav` (not committed) | `67c55b4b07d948ca8a0d21732801e1ec9baf0c557b483a15785c659b67bb2c84` |
| `envvar.spx` (not committed) | `f2fa7276a4a5b8fbbccf77dfed54fed8294b8fa6026caafdad173815f9a86e9a` |
| `envvar.noenh.pcm` (not committed) | `b13dd732fd01566eca7e973dcc722d68db57d67348cefd2ae798d11892634bc3` |

The reference decoder is deterministic per build but not bit-exact
across compilers/libm; `*.noenh.pcm` is byte-exact for the recorded
build host and should be compared to ±2 LSB cross-platform (same
posture as the staged fixtures). The analysis scripts are
session-scratch tooling (not tracked) per the fixture-generation
convention; this note plus the gate are the audit trail.
