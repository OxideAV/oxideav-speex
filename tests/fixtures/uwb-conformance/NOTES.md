# Ultra-wideband speech-material fixture (round r410)

Black-box encode/decode reference I/O for the ultra-wideband tracking
gate (`tests/uwb_conformance_fixture.rs`). Per the workspace clean-room
policy, `speexenc` / `speexdec` (1.2.1, Homebrew build, macOS arm64)
are invoked as **opaque binaries**; their source is NOT consulted.

## Source (`speechy32k.raw`, 2.0 s, 64 000 samples)

Pitch-gliding harmonic stack (f0 40–180 Hz, 59 harmonics, formant-like
weighting up to ~14 kHz) with syllabic AM:

```sh
python3 -c "
import math, struct
sr=32000; n=sr*2
s=[]
phase=0.0
for i in range(n):
    t=i/sr
    f0=110+70*math.sin(2*math.pi*0.9*t)
    phase+=2*math.pi*f0/sr
    v=0.0
    for h in range(1,60):
        w=math.exp(-((h*f0-700)/500)**2)+0.6*math.exp(-((h*f0-1800)/600)**2)+0.35*math.exp(-((h*f0-5200)/1500)**2)+0.18*math.exp(-((h*f0-11000)/3000)**2)+0.08
        v+=w*math.sin(h*phase)
    env=0.55+0.45*math.sin(2*math.pi*2.3*t+1.0)
    v*=env*0.08
    s.append(int(max(-1,min(1,v))*22000))
open('speechy32k.raw','wb').write(struct.pack('<%dh'%n,*s))
"
```

Encode / reference decode:

```sh
speexenc --ultra-wideband --rate 32000 --le --16bit --quality 4 --comp 10 speechy32k.raw uwb_q4.spx
speexdec --no-enh uwb_q4.spx uwb_q4.noenh.pcm
```

## SHA-256

| file | SHA-256 |
| --- | --- |
| `speechy32k.raw` | `16b18872bebe75654020f25758a1e0ca1d4ad410cdbf79c00e3cda77a9cb852f` |
| `uwb_q4.spx` | `780fe310cc37cc5a99f0ec16c65a421c0a76a2b9dda3ee55f212891e0e9b90e5` |
| `uwb_q4.noenh.pcm` | `128a15e26f3768a7b9204101a76f927cb098665aad48867878f872fb896f5454` |

## Known divergence (r410)

Measured r410: **2.0 dB / corr 0.78 / energy 1.66** full-signal at the
160-sample look-ahead alignment (best two-sided lag) — far below the
19.1 dB tone fixture. Band split: the embedded 0–8 kHz half scores
2.3 dB at 1.6× energy (vs 15.6 dB for the same code on the standalone
wideband speech fixture); the 8–16 kHz folded layer overshoots ≈10×.
Applying the inner crossover-shaped fold law to the **outer** fold
(slope scaled by analogy, ceiling 1/16) barely moves this fixture and
regresses the tone fixture — the outer-layer speech behaviour is a
distinct, unpinned mechanism. Recorded follow-up; the gate holds
tracking floors so a regression or a fix both surface.
