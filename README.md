# oxideav-speex

Pure-Rust Speex (CELP speech codec) — header parser + bit reader + float-mode
decoder and encoder for narrowband / wideband streams.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a
100% pure Rust media transcoding and streaming stack. No C libraries, no FFI
wrappers, no `*-sys` crates.

## Features

### Decoder

- **Narrowband** (8 kHz) — sub-modes 0..=8 covering the full Speex NB
  rate ladder from 2.15 kbps vocoder up to 24.6 kbps high-quality CELP.
- **Wideband** (16 kHz) — sub-band CELP layer (QMF + 8th-order high-band
  LPC + stochastic / spectral-folding excitation) for WB sub-modes 1..=4.
- Ultra-wideband (32 kHz) not yet implemented.

### Encoder

- **Narrowband** (8 kHz) — two NB CELP sub-modes are supported:

  | NB sub-mode | Bits/frame | Nominal rate | LSP VQ | Pitch | Gain | Innovation CB |
  |-------------|-----------:|-------------:|--------|-------|------|---------------|
  | **3** | 160 | 8 kbps | LBR 3-stage, 18 bits | 7-bit lag + 5-bit LBR gain | 1-bit sub-frame scalar | 4×10 × 5-bit (`EXC_10_32_TABLE`) |
  | **5** *(default)* | 300 | 15 kbps | NB 5-stage, 30 bits | 7-bit lag + 7-bit NB gain | 3-bit sub-frame scalar | 8×5 × 6-bit (`EXC_5_64_TABLE`) |

  Select the NB sub-mode via `CodecParameters::bit_rate`:
  - `bit_rate ≤ 12_000` → sub-mode 3 (8 kbps, 160 bits/frame).
  - `bit_rate > 12_000` or `None` → sub-mode 5 (15 kbps, default).

  NB sub-modes 1/2/4/6/7/8 are *not* implemented by the encoder and
  return `Error::Unsupported` if explicitly constructed via
  `NbEncoder::with_submode`.

- **Wideband** (16 kHz) — NB mode 5 layered with one of two WB extensions:
  | WB sub-mode | Bits/frame | Total rate | Technique |
  |-------------|-----------:|-----------:|-----------|
  | **1** | 336 | ~16.8 kbps | Spectral folding + 5-bit folding gain |
  | **3** *(default)* | 492 | ~24.6 kbps | Stochastic split-VQ (5×8 × 7-bit shape + sign) + 4-bit gain |

  WB sub-modes 2 and 4 are *not* implemented by the encoder and return
  `Error::Unsupported` if requested. The WB encoder always uses NB mode
  5 as its low-band — there is no direct way to combine it with NB
  mode 3 yet.
- Select the WB sub-mode via `CodecParameters::bit_rate`:
  - `bit_rate ≤ 20_000` → sub-mode 1 (folding).
  - `bit_rate > 20_000` or `None` → sub-mode 3 (stochastic, default).

## Usage

```toml
[dependencies]
oxideav-speex = "0.0"
```

## License

MIT — see [LICENSE](LICENSE).
