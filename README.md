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

- **Narrowband** (8 kHz) — **sub-mode 5** (15 kbps NB CELP). Other NB
  sub-modes return `Error::Unsupported` from the factory.
- **Wideband** (16 kHz) — NB mode 5 layered with one of two WB extensions:
  | WB sub-mode | Bits/frame | Total rate | Technique |
  |-------------|-----------:|-----------:|-----------|
  | **1** | 336 | ~16.8 kbps | Spectral folding + 5-bit folding gain |
  | **3** *(default)* | 492 | ~24.6 kbps | Stochastic split-VQ (5×8 × 7-bit shape + sign) + 4-bit gain |

  WB sub-modes 2 and 4 are *not* implemented by the encoder and return
  `Error::Unsupported` if requested.
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
