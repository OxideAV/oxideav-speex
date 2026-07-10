# Wideband decode-conformance fixtures (round r410)

Black-box encode/decode reference I/O for the wideband
decode-conformance matrix (`tests/wb_conformance_fixture.rs`). Per the
workspace clean-room policy, `speexenc` / `speexdec` are invoked as
**opaque binaries** (decode oracle); their source is NOT consulted.

- Toolchain: `speexenc` / `speexdec` **1.2.1** (Homebrew build, macOS
  arm64). Reference decodes use `--no-enh`.
- Source: deterministic synthetic 16 kHz PCM (script below),
  headerless s16le mono.
- Alignment (measured, pinned by the gate's alignment test): the q4
  reference decode leads ours by **80** samples (the QMF/look-ahead
  padding, same as the staged `wb-mode1-folded` fixture); the q6/q8
  reference decodes are 32 000 samples (vs q4's 32 080) and **trail**
  our decode by 143 samples.

## Source (`speechy16k.raw`, 2.0 s, 32 000 samples)

Pitch-gliding harmonic stack (f0 40–180 Hz, 25 harmonics) with
formant-like weighting up to ~6 kHz and syllabic AM:

```sh
python3 -c "
import math, struct
sr=16000; n=sr*2
s=[]
phase=0.0
for i in range(n):
    t=i/sr
    f0=110+70*math.sin(2*math.pi*0.9*t)
    phase+=2*math.pi*f0/sr
    v=0.0
    for h in range(1,26):
        w=math.exp(-((h*f0-700)/500)**2)+0.6*math.exp(-((h*f0-1800)/600)**2)+0.35*math.exp(-((h*f0-5200)/1500)**2)+0.12
        v+=w*math.sin(h*phase)
    env=0.55+0.45*math.sin(2*math.pi*2.3*t+1.0)
    v*=env*0.11
    s.append(int(max(-1,min(1,v))*22000))
open('speechy16k.raw','wb').write(struct.pack('<%dh'%n,*s))
"
```

Encodes (`wb_qQ.spx`, Q ∈ {4, 6, 8} — Table 10.2 per-layer ladders
NB 4 + HB 1, NB 5 + HB 2, NB 6 + HB 3):

```sh
speexenc --wideband --rate 16000 --le --16bit --quality Q --comp 10 speechy16k.raw wb_qQ.spx
speexdec --no-enh wb_qQ.spx wb_qQ.noenh.pcm
```

## SHA-256

| file | SHA-256 |
| --- | --- |
| `speechy16k.raw` | `10ac4a38e14f7661cb90f83ab8e6007c10855252c150e54fed26f00a870a83e1` |
| `wb_q4.spx` | `6e2f6bee0e3bab5b8653678beb85f7aee54cc7d51a9c1744e26fbfead6aac7b6` |
| `wb_q4.noenh.pcm` | `38b9d9ec464d6d8361464e458cb80a447db5a0026dd36b189fb485f179eb2ea6` |
| `wb_q6.spx` | `a3a68ad4b7d2b092162f0dc7ecbe9629b3a59e8e91801381a3343c1a04801e67` |
| `wb_q6.noenh.pcm` | `17f8f6471c578b800264f811db3333a9a0fe937af50be7e258487b56e48d9a17` |
| `wb_q8.spx` | `dc00f090ca75b8eab96032319ab09847b5ab5cc9a99434e63b0563cbb5f2ccfe` |
| `wb_q8.noenh.pcm` | `d3902e3cadb8781ffb5483116ae71d1a1477c9c0f098a1c1c2860bb353d97604` |

## Fold-law oracle probing (round r410, method record)

The `wb_q4` fixture exposed a catastrophic per-frame divergence
(up to 130× energy overshoot in envelope troughs) in the r393 flat
folded high-band law. The responsible normalisation was then pinned by
**synthetic single-fold oracle streams**: hand-assembled wideband
mode-1 packets (this crate's Table 9.1 / 10.1 wire writers) with

- a deterministic, pitch-free narrowband layer (mode 8, forced pitch
  gain index 0, fixed innovation index) whose excitation is exactly
  computable,
- swept fields: the 5-bit high-band fold-gain index (0..=31), the NB
  OL excitation-gain index, the 12-bit high-band LSP index (16
  envelope points), and the NB LSP index (8 envelope points),

wrapped in Ogg (stdlib CRC page writer, header/comment packets cloned
from `wb_q4.spx`) and decoded by `speexdec --no-enh` as a black-box
oracle. The reference high-band excitation was recovered from each
decode by QMF-splitting the PCM and inverse-filtering with the
transmitted (known) high-band envelope. Findings (recorded in
`src/hb_fold.rs`):

- reference high-band excitation is **linear in the fold gain level
  and in the low-band excitation** (constant ratio across both
  sweeps once measurement floors are avoided);
- across 16 high-band envelopes with `|A_hb(π)| ≤ 1.4` the effective
  fold scale tracks `C·|A_hb(π)|` (the envelope's magnitude response
  at the 4 kHz QMF crossover), measured slope `C ≈ 0.171…0.189`;
  it tracks *no* power of the filter's gross (impulse-energy) gain;
- both real-stream anchors (`wb-mode1-folded`, `uwb-fold-geometry`)
  sit at `|A_hb(π)| ≥ 2.4` and match the **flat** `0.35355` law
  sample-accurately, pinning a saturation ceiling;
- the NB-envelope sweep showed **no** clean dependence on the
  low-band envelope's crossover response or prediction gain
  (candidate laws dividing by `|A_lb(π)|` break the tone anchor).

The verifier/generator scripts are session-scratch tooling (not
tracked), per the fixture-generation convention; this note plus the
sweep results embedded in `src/hb_fold.rs` docs are the audit trail.
