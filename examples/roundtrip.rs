//! Encode a synthetic signal to Speex packets and decode it back with the
//! direct (registry-free) API.
//!
//! ```text
//! cargo run --example roundtrip -- [RATE] [QUALITY] [CHANNELS]
//! cargo run --example roundtrip -- 32000 10 1
//! ```
//!
//! `RATE` is 8000 / 16000 / 32000 (default 16000), `QUALITY` 0..=10
//! (default 8), `CHANNELS` 1 or 2 (default 1; 2 = intensity stereo).

use oxideav_speex::{SpeexHeader, SpeexStreamDecoder, SpeexStreamEncoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let rate: u32 = args.next().map_or(Ok(16_000), |s| s.parse())?;
    let quality: u8 = args.next().map_or(Ok(8), |s| s.parse())?;
    let channels: u16 = args.next().map_or(Ok(1), |s| s.parse())?;

    // 1. Encoder: one per stream. 20 ms frames in, one packet per frame out.
    let mut enc = match channels {
        1 => SpeexStreamEncoder::new(rate, quality)?,
        2 => SpeexStreamEncoder::new_stereo(rate, quality)?,
        n => return Err(format!("CHANNELS must be 1 or 2, got {n}").into()),
    };
    let frame = enc.frame_samples(); // per channel: 160 / 320 / 640

    // 2. The 80-byte stream header: Ogg packet 0, or codec extradata.
    let header_bytes = enc.header().write_bytes();

    // 3. Decoder: built from the header (or SpeexStreamDecoder::for_sample_rate
    //    when the transport fixes the rate, e.g. RTP).
    let header = SpeexHeader::parse(&header_bytes)?;
    let mut dec = SpeexStreamDecoder::for_header(&header)?;

    // Two seconds of a 300 Hz + 1 kHz mix (right channel at half level).
    let n_frames = 100;
    let input: Vec<i16> = (0..frame * n_frames)
        .flat_map(|i| {
            let t = i as f64 / f64::from(rate);
            let v = 6000.0 * (2.0 * std::f64::consts::PI * 300.0 * t).sin()
                + 3000.0 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin();
            let s = v as i16;
            if channels == 2 {
                vec![s, s / 2]
            } else {
                vec![s]
            }
        })
        .collect();

    let mut bytes = 0usize;
    let mut output: Vec<i16> = Vec::with_capacity(input.len());
    for block in input.chunks_exact(frame * usize::from(channels)) {
        let packet = enc.encode_frame(block)?;
        bytes += packet.len();
        let pcm = if channels == 2 {
            dec.decode_packet_stereo_i16(&packet)?
        } else {
            dec.decode_packet_pcm_i16(&packet)?
        };
        output.extend_from_slice(&pcm);
    }

    let secs = (frame * n_frames) as f64 / f64::from(rate);
    println!(
        "{rate} Hz, quality {quality}, {channels} ch: {n_frames} packets, {bytes} bytes \
         ({:.1} kbit/s; nominal {} bit/s), decoded {} samples",
        bytes as f64 * 8.0 / secs / 1000.0,
        enc.nominal_bitrate_bps().unwrap_or(0),
        output.len()
    );
    Ok(())
}
