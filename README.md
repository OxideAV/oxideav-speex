# oxideav-speex

Pure-Rust **Speex** (CELP speech codec) — float-mode decoder and encoder
covering the full narrowband / wideband / ultra-wideband rate ladder.
Parses the 80-byte Speex-in-Ogg header, streams Ogg packets through the
stacked NB + SB-CELP pipeline, and produces S16 PCM output. Zero C
dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.0"
oxideav-codec = "0.0"
oxideav-speex = "0.0"

# Optional — for demuxing Speex from Ogg files
oxideav-ogg = "0.0"
oxideav-container = "0.0"
```

## Decoder coverage

| Speex mode | Sample rate | Sub-modes supported | Notes |
|------------|------------:|---------------------|-------|
| Narrowband | 8 kHz | **0..=8** (full NB rate ladder) | 2.15 kbps vocoder (sm0/sm1) through 24.6 kbps (sm7); sm8 = 3.95 kbps algebraic. |
| Wideband | 16 kHz | **1..=4** | QMF + 8th-order high-band LPC + spectral-folding (sm1) / stochastic split-VQ (sm2/3/4) excitation. |
| Ultra-wideband | 32 kHz | **1** (folding — the only UWB sub-mode the reference defines) | Stacks a second 8th-order LPC + folding layer on top of WB, second QMF synthesis for 16→32 kHz. |

Output is mono S16 at each mode's native sample rate. Stereo Speex
(intensity side-channel) is not implemented — `nb_channels > 1`
returns `Error::Unsupported`.

## Encoder coverage

### Narrowband (8 kHz)

| NB sub-mode | Bits/frame | Nominal rate | Selection |
|-------------|-----------:|-------------:|-----------|
| **3** | 160 | 8 kbps | `bit_rate ≤ 12_000` |
| **5** *(default)* | 300 | 15 kbps | `bit_rate > 12_000` or `None` |

NB sub-modes 1/2/4/6/7/8 are not implemented by the encoder and
return `Error::Unsupported` if explicitly constructed via
`NbEncoder::with_submode`.

### Wideband (16 kHz)

Always stacks NB sub-mode 5 as the low band; the WB extension picks
one of:

| WB sub-mode | Extra bits | Total bits/frame | Total rate | Selection |
|-------------|-----------:|-----------------:|-----------:|-----------|
| **1** | 36 | 336 | ~16.8 kbps | `bit_rate ≤ 20_000` |
| **3** *(default)* | 192 | 492 | ~24.6 kbps | `bit_rate > 20_000` or `None` |

Sub-modes 2 and 4 are recognised by the decoder tables but not
emitted by the encoder (return `Error::Unsupported`).

### Ultra-wideband (32 kHz)

Always stacks WB mode-3 underneath; the UWB extension picks one of:

| UWB layer | Extra bits | Total bits/frame | Total rate | Selection |
|-----------|-----------:|-----------------:|-----------:|-----------|
| **Null** (uwb-bit = 0) | 1 | 493 | ~24.65 kbps | `bit_rate ≤ 25_000` |
| **Folding** *(default)* | 36 | 528 | ~26.4 kbps | `bit_rate > 25_000` or `None` |

The folding layer follows the reference's `sb_uwb_mode`: 4 sub-frames of
80 samples each at the 16 kHz decimated rate, 12-bit two-stage LSP VQ,
4 × 5-bit folding gain. Sub-modes other than 1 are **not defined** by
the Speex UWB reference.

Encoder input must be mono S16 at the band's native rate (8000 /
16000 / 32000 Hz). A downstream muxer sees the chosen mode reflected
in the 80-byte Speex header written to `CodecParameters::extradata`.

## Quick use

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{Frame, MediaType};

let mut codecs = CodecRegistry::new();
oxideav_speex::register(&mut codecs);

// Open an Ogg-wrapped .spx file (NB/WB/UWB all handled identically).
let input: Box<dyn oxideav_container::ReadSeek> = Box::new(
    std::io::Cursor::new(std::fs::read("clip.spx")?),
);
let mut dmx = oxideav_ogg::demux::open(input)?;
let stream = &dmx.streams()[0];
let mut dec = codecs.make_decoder(&stream.params)?;

loop {
    match dmx.next_packet() {
        Ok(pkt) => {
            dec.send_packet(&pkt)?;
            while let Ok(Frame::Audio(af)) = dec.receive_frame() {
                // af.format == SampleFormat::S16
                // af.sample_rate == 8000 / 16000 / 32000
                // af.data[0] = S16 LE interleaved mono
            }
        }
        Err(oxideav_core::Error::Eof) => break,
        Err(e) => return Err(e.into()),
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Encoder (UWB example)

```rust
use oxideav_core::{CodecId, CodecParameters, SampleFormat};

let mut params = CodecParameters::audio(CodecId::new("speex"));
params.sample_rate = Some(32_000);
params.channels = Some(1);
params.sample_format = Some(SampleFormat::S16);
// params.bit_rate = None → folding UWB layer, ~26.4 kbps.

let mut enc = codecs.make_encoder(&params)?;
enc.send_frame(&Frame::Audio(audio_frame_s16_32k))?;
enc.flush()?;
while let Ok(pkt) = enc.receive_packet() {
    // pkt.data = encoded Speex frame (NB + WB + UWB layers).
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Codec ID

- Codec: `"speex"`; accepted sample format `S16`; channels 1 (mono).
- Header is carried in `CodecParameters::extradata` as an 80-byte
  Speex-in-Ogg struct (signature `"Speex   "`).

## Tables + license

LSP, gain, and fixed codebook tables are transcribed values-only from
the BSD-licensed Xiph reference (`libspeex/lsp_tables_nb.c`,
`libspeex/gain_table.c`, `libspeex/exc_*_table.c`, `libspeex/hexc_*.c`,
`libspeex/lsp_tables_wb.c`). No reference code is wrapped.

## References

- [Speex manual](https://www.speex.org/docs/manual/speex-manual.pdf)
- [RFC 5574 — RTP payload format for Speex](https://www.rfc-editor.org/rfc/rfc5574)
- [xiph/speex reference](https://github.com/xiph/speex)

## License

MIT — see [LICENSE](LICENSE).
