# Narrowband decode-conformance fixtures (round r410)

Black-box encode/decode reference I/O for the narrowband
decode-conformance matrix (`tests/nb_conformance_fixture.rs`). Per the
workspace clean-room policy, `speexenc` / `speexdec` are invoked as
**opaque binaries** (decode oracle); their source is NOT consulted.

- Toolchain: `speexenc` / `speexdec` **1.2.1** (Homebrew build,
  `speexenc --version` → "speexenc (Speex encoder) version 1.2.1"),
  macOS arm64. Reference decodes use `--no-enh` (perceptual enhancer
  off — the core synthesis output; the codec's default output
  high-pass remains active in the reference decoder).
- Sources are deterministic synthetic PCM (generation scripts below),
  headerless s16le mono 8 kHz.
- Each `*.noenh.pcm` is the byte-exact `speexdec --no-enh` decode of
  its `*.spx`. The decode is 40 samples longer than the source (the
  reference codec's look-ahead padding); the reference **leads** the
  crate's decode by those 40 samples.

## Tone-mix source (`src8k.raw`, 1.0 s, 8 000 samples)

```sh
python3 -c "
import math, struct
sr=8000; n=sr
s=[]
for i in range(n):
    t=i/sr
    v=(0.45*math.sin(2*math.pi*440*t)
       +0.25*math.sin(2*math.pi*1200*t)*(0.6+0.4*math.sin(2*math.pi*3*t))
       +0.15*math.sin(2*math.pi*2500*t))
    s.append(int(max(-1,min(1,v))*20000))
open('src8k.raw','wb').write(struct.pack('<%dh'%n,*s))
"
```

Encodes (`nb_qQ.spx`, Q ∈ {1, 2, 3, 5, 7, 9} — Table 9.2 sub-modes
8, 2, 3, 4, 5, 6):

```sh
speexenc --narrowband --rate 8000 --le --16bit --quality Q --comp 10 src8k.raw nb_qQ.spx
speexdec --no-enh nb_qQ.spx nb_qQ.noenh.pcm
```

## Speech-like source (`speechy8k.raw`, 2.0 s, 16 000 samples)

Pitch-gliding harmonic stack (f0 40–180 Hz) with formant-ish harmonic
weighting and syllabic AM — exercises time-varying pitch periods both
above and below the 40-sample sub-frame length:

```sh
python3 -c "
import math, struct
sr=8000; n=sr*2
s=[]
phase=0.0
for i in range(n):
    t=i/sr
    f0=110+70*math.sin(2*math.pi*0.9*t)
    phase+=2*math.pi*f0/sr
    v=0.0
    for h in range(1,9):
        w=math.exp(-((h*f0-700)/500)**2)+0.6*math.exp(-((h*f0-1800)/600)**2)+0.25
        v+=w*math.sin(h*phase)
    env=0.55+0.45*math.sin(2*math.pi*2.3*t+1.0)
    v*=env*0.16
    s.append(int(max(-1,min(1,v))*22000))
open('speechy8k.raw','wb').write(struct.pack('<%dh'%n,*s))
"
```

Encodes (`sp_qQ.spx`, Q ∈ {1, 2, 3, 7} — sub-modes 8, 2, 3, 5), same
commands as above with `speechy8k.raw`.

## SHA-256

| file | SHA-256 |
| --- | --- |
| `src8k.raw` | `95e7be32a2da8a101c9b50bd7e7f5b718185ecd6627b3f6775f52e90b30e4ae8` |
| `nb_q1.spx` | `8c3861285c409fa5e8d86f264dc544d92b1eb15ff43e40297929442096a13e60` |
| `nb_q1.noenh.pcm` | `250d6c05c222a723f23bbf61b34a17697c459b0607ae5893b466089e14cc6fc3` |
| `nb_q2.spx` | `ad70ae1ae999fb8057a6aa89021317189b5777a8a1787276ccacb5c39d00ac16` |
| `nb_q2.noenh.pcm` | `e674ceb9319ce26a916cf0ea3cb2630a7d6cc4edbd685c91a6f95f59c370f953` |
| `nb_q3.spx` | `b23da0144674dec9f6c8c60b372612938ceb542c32cc75c92b0f35110208f0e4` |
| `nb_q3.noenh.pcm` | `50f293c9bc33f5275e6f106297caa6a70f56b3e3a60e30e9fb088e2576b3ea12` |
| `nb_q5.spx` | `6d0725f768e578a3510b3352a48456b8cfef29faed503b42559c10592578e0e3` |
| `nb_q5.noenh.pcm` | `8d669eb6b17984b6d69c1459437fb426c7f4b92dabca298db57f20f026cc627d` |
| `nb_q7.spx` | `900c3dd2473492a0b16fc6b52043bd68d5567fd69603589bfaceaa91536b0ef4` |
| `nb_q7.noenh.pcm` | `b9abfa57c823553cacfa0f507f07cf051c06e7102ee98becbf6bb7e0b31d84a8` |
| `nb_q9.spx` | `7cc48f21686454364acdb5b7b65722008ab852f77ffa2f8d549db1015e1d4528` |
| `nb_q9.noenh.pcm` | `1e03d65ee051bc61ff6601ddc4f2dcd0d6288df273448a0825789e1a8372d765` |

| `speechy8k.raw` | `f557da3cfb08b1978c8cfb70016f412dc7294f26516924f47e4173158f4450d0` |
| `sp_q1.spx` | `84f6fbaf1561d699da38710bdff3897a2978102f1226c74d3307fb27171aff1d` |
| `sp_q1.noenh.pcm` | `4744588e91382ae2c325eb87158480c84ce2b25ed4ee213286de30171288929f` |
| `sp_q2.spx` | `d242ad48e1c4200b08acc4ff216662572e820446f4b4a2db9371d4f7530d49c2` |
| `sp_q2.noenh.pcm` | `365fc005e04c16ac726dadf93a8a6d7229a89b1287a28d25e2696ed5f033f0e9` |
| `sp_q3.spx` | `ed1f52601052fe03227cbeafba4e8a778b1f0ea891a17252e8d50771e2b8bfae` |
| `sp_q3.noenh.pcm` | `659634f0f07ed5d3515f36ade4075b336b015ff9bb0faec8c3910c7e72576eda` |
| `sp_q7.spx` | `b7c0d31fb54a2a651d5563e7d8aaf48f6903a3ea069d8ba7d69684079e538467` |
| `sp_q7.noenh.pcm` | `082e3f8295ceab5ae79986c59b07d8e040ea6ee87b7598b6bbcdc19e9f151e26` |
