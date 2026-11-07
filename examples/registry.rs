//! Use the Speex codec through the `oxideav-core` framework traits.
//!
//! ```text
//! cargo run --example registry
//! ```
//!
//! `register` installs the factories under the codec id `"speex"`;
//! `make_encoder` / `make_decoder` are the same factories without a
//! registry. The encoder takes interleaved S16 frames of any length and
//! emits one packet per 20 ms; its `output_params().extradata` is the
//! 80-byte stream header the decoder accepts as `extradata`.

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, RuntimeContext, SampleFormat,
};

fn main() -> Result<(), Error> {
    let mut ctx = RuntimeContext::new();
    oxideav_speex::register(&mut ctx);

    let rate = 8_000;
    let mut params = CodecParameters::audio(CodecId::new("speex"));
    params.sample_rate = Some(rate);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    params.options.insert("quality", "8");
    let mut enc = ctx.codecs.first_encoder(&params)?;

    // The decoder is configured from the encoder's output parameters
    // (extradata = stream header).
    let mut dec = ctx.codecs.first_decoder(enc.output_params())?;

    // One second of a 440 Hz tone, sent in 1000-sample chunks.
    let pcm: Vec<i16> = (0..rate as usize)
        .map(|i| {
            (8000.0 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / f64::from(rate)).sin())
                as i16
        })
        .collect();
    for chunk in pcm.chunks(1000) {
        enc.send_frame(&Frame::Audio(AudioFrame {
            samples: chunk.len() as u32,
            pts: None,
            data: vec![chunk.iter().flat_map(|s| s.to_le_bytes()).collect()],
        }))?;
    }
    enc.flush()?;

    let (mut packets, mut decoded) = (0, 0u64);
    loop {
        match enc.receive_packet() {
            Ok(packet) => {
                packets += 1;
                dec.send_packet(&packet)?;
                while let Ok(frame) = dec.receive_frame() {
                    if let Frame::Audio(a) = frame {
                        decoded += u64::from(a.samples);
                    }
                }
            }
            Err(Error::NeedMore) => break,
            Err(e) => return Err(e),
        }
    }
    println!("{packets} packets -> {decoded} decoded samples at {rate} Hz");
    Ok(())
}
