//! Speex in-band signalling — RFC 5574 §3 + Speex manual §5.5.
//!
//! In-band messages are 4-bit "mode" tags `m=14` that precede a sub-mode
//! selector and carry a small control payload. They share the same
//! bitstream slot as a CELP frame, so a decoder must consume them
//! before reading the next 4-bit `m` selector. The full code table is
//! transcribed from the Speex manual Table 5.1 and the
//! `speex_callbacks.h` constants:
//!
//! ```text
//!   code | size | meaning
//!   -----+------+--------------------------------------------------
//!     0  |  1   | perceptual enhancement off (0) / on (1)
//!     1  |  1   | encoder be less aggressive on packet loss (0/1)
//!     2  |  4   | switch to mode N (0..=15)
//!     3  |  4   | switch to low-band mode N
//!     4  |  4   | switch to high-band mode N
//!     5  |  4   | switch to quality N for VBR
//!     6  |  4   | request acknowledge: 0=no, 1=all, 2=in-band only
//!     7  |  4   | rate-control mode: 0=CBR, 1=VAD, 3=DTX, 5=VBR, 7=VBR+DTX
//!     8  |  8   | transmit one 8-bit character
//!     9  |  8   | intensity-stereo side channel (1 sign + 5 dexp + 2 ratio)
//!    10  | 16   | maximum acceptable bit-rate, in bytes/second
//!    11  | 16   | reserved
//!    12  | 32   | acknowledge receipt of packet N
//!    13  | 32   | reserved
//!    14  | 64   | reserved
//!    15  | 64   | reserved
//! ```
//!
//! On top of those there are two "pseudo-modes" the manual describes
//! alongside Table 5.1:
//!
//! * `m=13` — application-defined data: 5-bit length in **bytes**
//!   followed by that many octets. Decoders that don't recognise the
//!   payload still know how many bits to skip from the length prefix.
//! * `m=15` — frame terminator. Only emitted when the encoder wants to
//!   pad a packet to an octet boundary; the decoder treats it as
//!   "no more frames" and stops reading. RFC 5574 §3.3 specifies the
//!   padding pattern as a single `0` followed by all-ones to the next
//!   octet boundary, LSB-aligned.
//!
//! This module provides:
//! * [`InbandMessage`] — typed enum covering every code in Table 5.1,
//!   plus the mode-13 user payload and the mode-15 terminator.
//! * [`encode_inband`] / [`InbandMessage::encode`] — append a typed
//!   message to a [`BitWriter`], including the leading `m=14` (or
//!   `m=13` / `m=15`) marker. Useful when an authored stream needs to
//!   emit a control message in front of a CELP frame.
//! * [`decode_inband`] — read a single in-band message starting from a
//!   `m=14`/`m=13`/`m=15` marker the caller has already peeked at.
//! * [`pad_to_octet_boundary`] — write the RFC 5574 §3.3 padding
//!   terminator. Only required for the last frame of a packet.
//!
//! ### Why both encode + decode here
//!
//! The intensity-stereo side-channel ([`crate::stereo`]) is the only
//! in-band code we previously handled with semantic intent. Every other
//! code was opaquely skipped via [`crate::stereo::inband_skip_bits`].
//! That keeps the bit reader synchronised but loses the actual control
//! information, which a downstream player or RTP receiver may want to
//! act on (e.g. CBR↔VBR switch, mode bumps from a remote endpoint).
//! Exposing typed encode/decode here lets callers process those
//! messages without re-deriving the bit layout, and gives the encoder
//! a way to emit them — which the reference's
//! `speex_bits_pack`/`speex_inband_handler` pair handles for libspeex
//! callers but had no equivalent here.
//!
//! The CELP frame readers in [`crate::nb_decoder`] and
//! [`crate::wb_decoder`] still skip unrecognised in-band requests
//! silently (matching libspeex's "no callback registered" default);
//! [`decode_inband`] is the typed path callers can hook in when they
//! want to *observe* those messages.

use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_core::{Error, Result};

/// 4-bit mode tag for an in-band signalling request — Speex manual §5.5
/// pseudo-mode 14.
pub const M_INBAND_REQUEST: u32 = 14;

/// 4-bit mode tag for an application-defined in-band payload — Speex
/// manual §5.5 pseudo-mode 13.
pub const M_USER_INBAND: u32 = 13;

/// 4-bit mode tag for the frame terminator — Speex manual §5.5
/// pseudo-mode 15. Decoders treat this as "no more frames".
pub const M_TERMINATOR: u32 = 15;

/// Speex manual Table 5.1 request codes. `id` is the 4-bit value
/// transmitted right after the `m=14` marker; `payload_bits` is the
/// width of the message body that follows that 4-bit id.
pub mod codes {
    /// Decoder perceptual enhancement on/off (1-bit payload).
    pub const PERCEPTUAL_ENHANCE: u32 = 0;
    /// Encoder packet-loss aggressiveness (1-bit payload).
    pub const PACKET_LOSS: u32 = 1;
    /// Encoder mode switch (4-bit payload, mode id 0..=15).
    pub const MODE_SWITCH: u32 = 2;
    /// Encoder low-band mode switch (4-bit payload).
    pub const LOW_BAND_MODE: u32 = 3;
    /// Encoder high-band mode switch (4-bit payload).
    pub const HIGH_BAND_MODE: u32 = 4;
    /// VBR quality switch (4-bit payload).
    pub const VBR_QUALITY: u32 = 5;
    /// Request acknowledgement policy (4-bit payload):
    /// 0 = none, 1 = all packets, 2 = only for in-band data.
    pub const REQUEST_ACK: u32 = 6;
    /// Rate-control mode (4-bit payload): 0=CBR, 1=VAD, 3=DTX, 5=VBR,
    /// 7=VBR+DTX.
    pub const RATE_CONTROL: u32 = 7;
    /// Transmit one 8-bit character to the peer.
    pub const CHAR_TRANSMIT: u32 = 8;
    /// Intensity-stereo side-channel — see [`crate::stereo`] for the
    /// per-bit decode.
    pub const STEREO: u32 = 9;
    /// Announce maximum acceptable bit-rate (16-bit payload, bytes/s).
    pub const MAX_BITRATE: u32 = 10;
    /// Reserved (16-bit payload).
    pub const RESERVED_11: u32 = 11;
    /// Acknowledge receipt of packet N (32-bit payload).
    pub const ACK_PACKET: u32 = 12;
    /// Reserved (32-bit payload).
    pub const RESERVED_13: u32 = 13;
    /// Reserved (64-bit payload).
    pub const RESERVED_14: u32 = 14;
    /// Reserved (64-bit payload).
    pub const RESERVED_15: u32 = 15;
}

/// Width in bits of the payload that follows the 4-bit id of an `m=14`
/// in-band request, indexed by id (0..=15). Mirrors the ladder in
/// `libspeex/speex_callbacks.c` and matches Table 5.1 of the manual.
///
/// Kept as a thin alternative to [`crate::stereo::inband_skip_bits`]
/// for callers that want to look the width up directly without
/// branching.
pub const INBAND_REQUEST_PAYLOAD_BITS: [u32; 16] =
    [1, 1, 4, 4, 4, 4, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64];

/// Acknowledgement policy enum for [`InbandMessage::RequestAck`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AckPolicy {
    /// `0` — do not acknowledge any packet.
    None,
    /// `1` — acknowledge every packet.
    All,
    /// `2` — only acknowledge packets that carry in-band data.
    InBandOnly,
    /// Any other 4-bit value — kept as a raw integer so a decoder can
    /// faithfully report what was on the wire. Reference implementation
    /// would clamp to one of the three known policies.
    Other(u8),
}

impl AckPolicy {
    /// Wire encoding of an [`AckPolicy`] — exactly the 4-bit value the
    /// reference reads from the bitstream.
    pub fn as_u4(self) -> u32 {
        match self {
            AckPolicy::None => 0,
            AckPolicy::All => 1,
            AckPolicy::InBandOnly => 2,
            AckPolicy::Other(v) => (v as u32) & 0xF,
        }
    }

    fn from_u4(v: u32) -> Self {
        match v & 0xF {
            0 => AckPolicy::None,
            1 => AckPolicy::All,
            2 => AckPolicy::InBandOnly,
            other => AckPolicy::Other(other as u8),
        }
    }
}

/// Rate-control mode enum for [`InbandMessage::RateControl`]. The wire
/// values come from Table 5.1 row 7.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateControl {
    /// `0` — constant bit-rate.
    Cbr,
    /// `1` — voice-activity detection (CBR + silence frames).
    Vad,
    /// `3` — discontinuous transmission.
    Dtx,
    /// `5` — variable bit-rate.
    Vbr,
    /// `7` — VBR + DTX.
    VbrDtx,
    /// Any other 4-bit value — kept as raw so decoders can faithfully
    /// report what was on the wire (Table 5.1 only defines five values).
    Other(u8),
}

impl RateControl {
    /// Wire encoding (Table 5.1 row 7).
    pub fn as_u4(self) -> u32 {
        match self {
            RateControl::Cbr => 0,
            RateControl::Vad => 1,
            RateControl::Dtx => 3,
            RateControl::Vbr => 5,
            RateControl::VbrDtx => 7,
            RateControl::Other(v) => (v as u32) & 0xF,
        }
    }

    fn from_u4(v: u32) -> Self {
        match v & 0xF {
            0 => RateControl::Cbr,
            1 => RateControl::Vad,
            3 => RateControl::Dtx,
            5 => RateControl::Vbr,
            7 => RateControl::VbrDtx,
            other => RateControl::Other(other as u8),
        }
    }
}

/// Typed Speex in-band message. Each variant maps 1:1 to a row in
/// Table 5.1 (`m=14` requests), the application-defined `m=13` payload,
/// or the `m=15` frame terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InbandMessage {
    /// Code 0 — toggle perceptual enhancement at the decoder.
    PerceptualEnhance(bool),
    /// Code 1 — request that the encoder be less aggressive due to
    /// high packet loss.
    PacketLossLessAggressive(bool),
    /// Code 2 — switch encoder mode N (0..=15).
    ModeSwitch(u8),
    /// Code 3 — switch low-band mode N.
    LowBandMode(u8),
    /// Code 4 — switch high-band mode N.
    HighBandMode(u8),
    /// Code 5 — switch VBR quality N (0..=10 in the reference).
    VbrQuality(u8),
    /// Code 6 — request acknowledgement policy.
    RequestAck(AckPolicy),
    /// Code 7 — rate-control mode.
    RateControl(RateControl),
    /// Code 8 — transmit one 8-bit character.
    Character(u8),
    /// Code 9 — intensity-stereo side channel. The 8 bits are
    /// `1-bit sign | 5-bit dexp | 2-bit e_ratio_idx`. We surface the
    /// raw byte; callers wanting the unpacked exponents should drive
    /// [`crate::stereo::StereoState::read_side_channel`] directly.
    StereoSideChannel(u8),
    /// Code 10 — announce maximum acceptable bit-rate, in bytes/second
    /// (16-bit payload).
    MaxBitrate(u16),
    /// Code 11 — reserved (16-bit payload). Body kept verbatim so
    /// decoders that observe a future use can report it without losing
    /// data.
    Reserved11(u16),
    /// Code 12 — acknowledge receipt of packet N (32-bit payload).
    AckPacket(u32),
    /// Code 13 — reserved (32-bit payload).
    Reserved13(u32),
    /// Code 14 — reserved (64-bit payload).
    Reserved14(u64),
    /// Code 15 — reserved (64-bit payload).
    Reserved15(u64),
    /// `m=13` pseudo-mode — application-defined opaque data (length
    /// prefix is 5-bit `len_bytes`, payload follows octet-aligned).
    /// Wire byte budget is `1 + len_bytes` (the leading `m=13` marker
    /// is owned by [`InbandMessage::encode`] / [`decode_inband`]).
    UserPayload(Vec<u8>),
    /// `m=15` pseudo-mode — frame terminator. Decoders stop reading at
    /// this marker. Encoders insert it before [`pad_to_octet_boundary`]
    /// when finishing a packet.
    Terminator,
}

impl InbandMessage {
    /// Total bit cost of the wire form, **including** the 4-bit
    /// `m=14`/`m=13`/`m=15` marker.
    pub fn wire_bits(&self) -> u32 {
        match self {
            // 4-bit m + 4-bit id + payload.
            InbandMessage::PerceptualEnhance(_) | InbandMessage::PacketLossLessAggressive(_) => {
                4 + 4 + 1
            }
            InbandMessage::ModeSwitch(_)
            | InbandMessage::LowBandMode(_)
            | InbandMessage::HighBandMode(_)
            | InbandMessage::VbrQuality(_)
            | InbandMessage::RequestAck(_)
            | InbandMessage::RateControl(_) => 4 + 4 + 4,
            InbandMessage::Character(_) | InbandMessage::StereoSideChannel(_) => 4 + 4 + 8,
            InbandMessage::MaxBitrate(_) | InbandMessage::Reserved11(_) => 4 + 4 + 16,
            InbandMessage::AckPacket(_) | InbandMessage::Reserved13(_) => 4 + 4 + 32,
            InbandMessage::Reserved14(_) | InbandMessage::Reserved15(_) => 4 + 4 + 64,
            // m=13: 4-bit marker + 5-bit length + 8·len bits payload.
            InbandMessage::UserPayload(p) => 4 + 5 + (p.len() as u32) * 8,
            // m=15 marker only.
            InbandMessage::Terminator => 4,
        }
    }

    /// Append the message (with its leading `m=14`/`m=13`/`m=15`
    /// marker) to `bw`. Mirrors `speex_inband_handler`'s emission path.
    ///
    /// Errors only when the encoded form would require more than 31
    /// bytes of `UserPayload` (Speex codes the length in 5 bits).
    pub fn encode(&self, bw: &mut BitWriter) -> Result<()> {
        match self {
            InbandMessage::PerceptualEnhance(b) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::PERCEPTUAL_ENHANCE, 4);
                bw.write_bits(if *b { 1 } else { 0 }, 1);
            }
            InbandMessage::PacketLossLessAggressive(b) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::PACKET_LOSS, 4);
                bw.write_bits(if *b { 1 } else { 0 }, 1);
            }
            InbandMessage::ModeSwitch(n) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::MODE_SWITCH, 4);
                bw.write_bits((*n as u32) & 0xF, 4);
            }
            InbandMessage::LowBandMode(n) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::LOW_BAND_MODE, 4);
                bw.write_bits((*n as u32) & 0xF, 4);
            }
            InbandMessage::HighBandMode(n) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::HIGH_BAND_MODE, 4);
                bw.write_bits((*n as u32) & 0xF, 4);
            }
            InbandMessage::VbrQuality(n) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::VBR_QUALITY, 4);
                bw.write_bits((*n as u32) & 0xF, 4);
            }
            InbandMessage::RequestAck(p) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::REQUEST_ACK, 4);
                bw.write_bits(p.as_u4(), 4);
            }
            InbandMessage::RateControl(r) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::RATE_CONTROL, 4);
                bw.write_bits(r.as_u4(), 4);
            }
            InbandMessage::Character(c) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::CHAR_TRANSMIT, 4);
                bw.write_bits(*c as u32, 8);
            }
            InbandMessage::StereoSideChannel(b) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::STEREO, 4);
                bw.write_bits(*b as u32, 8);
            }
            InbandMessage::MaxBitrate(rate) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::MAX_BITRATE, 4);
                bw.write_bits(*rate as u32, 16);
            }
            InbandMessage::Reserved11(v) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::RESERVED_11, 4);
                bw.write_bits(*v as u32, 16);
            }
            InbandMessage::AckPacket(n) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::ACK_PACKET, 4);
                bw.write_u64(*n as u64, 32);
            }
            InbandMessage::Reserved13(v) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::RESERVED_13, 4);
                bw.write_u64(*v as u64, 32);
            }
            InbandMessage::Reserved14(v) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::RESERVED_14, 4);
                bw.write_u64(*v, 64);
            }
            InbandMessage::Reserved15(v) => {
                bw.write_bits(M_INBAND_REQUEST, 4);
                bw.write_bits(codes::RESERVED_15, 4);
                bw.write_u64(*v, 64);
            }
            InbandMessage::UserPayload(bytes) => {
                if bytes.len() > 31 {
                    return Err(Error::invalid(format!(
                        "Speex inband: user payload length {} exceeds 5-bit cap of 31 bytes",
                        bytes.len()
                    )));
                }
                bw.write_bits(M_USER_INBAND, 4);
                bw.write_bits(bytes.len() as u32, 5);
                for b in bytes {
                    bw.write_bits(*b as u32, 8);
                }
            }
            InbandMessage::Terminator => {
                bw.write_bits(M_TERMINATOR, 4);
            }
        }
        Ok(())
    }
}

/// Convenience: build + emit an [`InbandMessage`] in one call.
pub fn encode_inband(msg: &InbandMessage, bw: &mut BitWriter) -> Result<()> {
    msg.encode(bw)
}

/// Decode the next in-band message from `br`. The caller is responsible
/// for having confirmed (e.g. via [`BitReader::peek_u32`]) that the
/// next 4 bits are one of `m=13`/`m=14`/`m=15` — this function will
/// read the marker as part of its work.
///
/// Returns:
/// * `Ok(Some(msg))` — a decoded message; bit cursor has advanced past
///   the entire wire form.
/// * `Ok(None)` — the next 4 bits were not an in-band marker (i.e. the
///   bitstream points at an actual CELP frame); the cursor is left
///   unchanged.
/// * `Err(_)` — the bitstream ran out mid-message.
pub fn decode_inband(br: &mut BitReader<'_>) -> Result<Option<InbandMessage>> {
    if br.bits_remaining() < 4 {
        return Ok(None);
    }
    let marker = br.peek_u32(4)?;
    match marker {
        M_INBAND_REQUEST => {
            br.read_u32(4)?; // consume marker
            decode_request(br).map(Some)
        }
        M_USER_INBAND => {
            br.read_u32(4)?;
            decode_user_payload(br).map(Some)
        }
        M_TERMINATOR => {
            br.read_u32(4)?;
            Ok(Some(InbandMessage::Terminator))
        }
        _ => Ok(None),
    }
}

fn decode_request(br: &mut BitReader<'_>) -> Result<InbandMessage> {
    if br.bits_remaining() < 4 {
        return Err(Error::invalid(
            "Speex inband: truncated `m=14` request (no id)",
        ));
    }
    let id = br.read_u32(4)?;
    let payload_bits = INBAND_REQUEST_PAYLOAD_BITS[id as usize] as u64;
    if br.bits_remaining() < payload_bits {
        return Err(Error::invalid(format!(
            "Speex inband: truncated payload for id {id} (need {payload_bits} bits)"
        )));
    }
    Ok(match id {
        codes::PERCEPTUAL_ENHANCE => InbandMessage::PerceptualEnhance(br.read_u32(1)? != 0),
        codes::PACKET_LOSS => InbandMessage::PacketLossLessAggressive(br.read_u32(1)? != 0),
        codes::MODE_SWITCH => InbandMessage::ModeSwitch(br.read_u32(4)? as u8),
        codes::LOW_BAND_MODE => InbandMessage::LowBandMode(br.read_u32(4)? as u8),
        codes::HIGH_BAND_MODE => InbandMessage::HighBandMode(br.read_u32(4)? as u8),
        codes::VBR_QUALITY => InbandMessage::VbrQuality(br.read_u32(4)? as u8),
        codes::REQUEST_ACK => InbandMessage::RequestAck(AckPolicy::from_u4(br.read_u32(4)?)),
        codes::RATE_CONTROL => InbandMessage::RateControl(RateControl::from_u4(br.read_u32(4)?)),
        codes::CHAR_TRANSMIT => InbandMessage::Character(br.read_u32(8)? as u8),
        codes::STEREO => InbandMessage::StereoSideChannel(br.read_u32(8)? as u8),
        codes::MAX_BITRATE => InbandMessage::MaxBitrate(br.read_u32(16)? as u16),
        codes::RESERVED_11 => InbandMessage::Reserved11(br.read_u32(16)? as u16),
        codes::ACK_PACKET => InbandMessage::AckPacket(br.read_u32(32)?),
        codes::RESERVED_13 => InbandMessage::Reserved13(br.read_u32(32)?),
        codes::RESERVED_14 => InbandMessage::Reserved14(br.read_u64(64)?),
        codes::RESERVED_15 => InbandMessage::Reserved15(br.read_u64(64)?),
        _ => unreachable!("id is 4-bit (0..=15) and table covers all entries"),
    })
}

fn decode_user_payload(br: &mut BitReader<'_>) -> Result<InbandMessage> {
    if br.bits_remaining() < 5 {
        return Err(Error::invalid(
            "Speex inband: truncated `m=13` user payload (length prefix)",
        ));
    }
    let len_bytes = br.read_u32(5)? as usize;
    let needed = (len_bytes as u64) * 8;
    if br.bits_remaining() < needed {
        return Err(Error::invalid(format!(
            "Speex inband: truncated `m=13` user payload (need {len_bytes} bytes)"
        )));
    }
    let mut out = Vec::with_capacity(len_bytes);
    for _ in 0..len_bytes {
        out.push(br.read_u32(8)? as u8);
    }
    Ok(InbandMessage::UserPayload(out))
}

/// Pad the bit cursor to the next octet boundary using the RFC 5574
/// §3.3 pattern: a single `0` followed by all-ones until the byte
/// boundary. If the cursor is already byte-aligned, no padding is
/// emitted (the RFC says padding "is only required for the last frame
/// in the packet, and only to ensure the packet contents end on an
/// octet boundary").
///
/// Returns the number of padding bits written (0..=7).
pub fn pad_to_octet_boundary(bw: &mut BitWriter) -> u32 {
    if bw.is_byte_aligned() {
        return 0;
    }
    let bits_in_byte = (bw.bit_position() % 8) as u32;
    let pad_bits = 8 - bits_in_byte;
    // First a `0`, then `pad_bits - 1` ones.
    bw.write_bits(0, 1);
    for _ in 1..pad_bits {
        bw.write_bits(1, 1);
    }
    pad_bits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(msg: InbandMessage) {
        let mut bw = BitWriter::new();
        msg.encode(&mut bw).expect("encode");
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let got = decode_inband(&mut br).expect("decode").expect("some");
        assert_eq!(got, msg);
    }

    #[test]
    fn roundtrip_perceptual_enhance() {
        roundtrip(InbandMessage::PerceptualEnhance(true));
        roundtrip(InbandMessage::PerceptualEnhance(false));
    }

    #[test]
    fn roundtrip_mode_switch() {
        for n in 0u8..16 {
            roundtrip(InbandMessage::ModeSwitch(n));
        }
    }

    #[test]
    fn roundtrip_low_high_band_mode() {
        for n in [0u8, 3, 5, 15] {
            roundtrip(InbandMessage::LowBandMode(n));
            roundtrip(InbandMessage::HighBandMode(n));
        }
    }

    #[test]
    fn roundtrip_request_ack_all_policies() {
        for p in [
            AckPolicy::None,
            AckPolicy::All,
            AckPolicy::InBandOnly,
            AckPolicy::Other(7),
        ] {
            roundtrip(InbandMessage::RequestAck(p));
        }
    }

    #[test]
    fn roundtrip_rate_control_all_modes() {
        for r in [
            RateControl::Cbr,
            RateControl::Vad,
            RateControl::Dtx,
            RateControl::Vbr,
            RateControl::VbrDtx,
            RateControl::Other(2),
        ] {
            roundtrip(InbandMessage::RateControl(r));
        }
    }

    #[test]
    fn roundtrip_char_transmit() {
        for c in [0u8, 0x41, 0xFF] {
            roundtrip(InbandMessage::Character(c));
        }
    }

    #[test]
    fn roundtrip_stereo_side_channel_byte() {
        // Same byte the existing stereo round-trip tests use.
        roundtrip(InbandMessage::StereoSideChannel(0x12));
    }

    #[test]
    fn roundtrip_max_bitrate_16bit() {
        roundtrip(InbandMessage::MaxBitrate(2_000));
        roundtrip(InbandMessage::MaxBitrate(65_535));
    }

    #[test]
    fn roundtrip_ack_packet_32bit() {
        roundtrip(InbandMessage::AckPacket(0xDEAD_BEEF));
    }

    #[test]
    fn roundtrip_reserved_64bit() {
        roundtrip(InbandMessage::Reserved14(0x0123_4567_89AB_CDEF));
        roundtrip(InbandMessage::Reserved15(0xFFFF_FFFF_FFFF_FFFF));
    }

    #[test]
    fn roundtrip_user_payload_lengths() {
        roundtrip(InbandMessage::UserPayload(Vec::new()));
        roundtrip(InbandMessage::UserPayload(b"hello".to_vec()));
        let max: Vec<u8> = (0..31u8).collect();
        roundtrip(InbandMessage::UserPayload(max));
    }

    #[test]
    fn user_payload_over_31_bytes_fails() {
        let too_big = vec![0u8; 32];
        let mut bw = BitWriter::new();
        let err = InbandMessage::UserPayload(too_big).encode(&mut bw);
        assert!(err.is_err(), "32-byte user payload must fail");
    }

    #[test]
    fn terminator_is_four_bits() {
        let msg = InbandMessage::Terminator;
        assert_eq!(msg.wire_bits(), 4);
        let mut bw = BitWriter::new();
        msg.encode(&mut bw).unwrap();
        assert_eq!(bw.bit_position(), 4);
    }

    #[test]
    fn decode_returns_none_for_non_inband_marker() {
        // m=5 is a regular CELP sub-mode selector — not an in-band code.
        let mut bw = BitWriter::new();
        bw.write_bits(5, 4);
        bw.write_bits(0, 4); // pad so the buffer has data
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let got = decode_inband(&mut br).unwrap();
        assert!(got.is_none(), "non-inband marker must yield Ok(None)");
        // Cursor unchanged.
        assert_eq!(br.bit_position(), 0);
    }

    #[test]
    fn decode_terminator_consumes_four_bits() {
        let mut bw = BitWriter::new();
        InbandMessage::Terminator.encode(&mut bw).unwrap();
        // Pad with junk so the reader has data to refuse to consume.
        bw.write_bits(0xFF, 8);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let got = decode_inband(&mut br).unwrap().unwrap();
        assert_eq!(got, InbandMessage::Terminator);
        assert_eq!(br.bit_position(), 4);
    }

    #[test]
    fn payload_bit_table_matches_skip_ladder() {
        // Sanity: the typed table here must agree bit-for-bit with the
        // existing `inband_skip_bits` ladder (used by NbDecoder to keep
        // the bit reader aligned when no semantic handler is registered).
        for id in 0u32..16 {
            assert_eq!(
                INBAND_REQUEST_PAYLOAD_BITS[id as usize],
                crate::stereo::inband_skip_bits(id),
                "id {id} disagrees between inband + stereo tables",
            );
        }
    }

    #[test]
    fn wire_bits_matches_actual_emission() {
        let cases: [InbandMessage; 9] = [
            InbandMessage::PerceptualEnhance(true),
            InbandMessage::ModeSwitch(7),
            InbandMessage::RequestAck(AckPolicy::All),
            InbandMessage::Character(b'X'),
            InbandMessage::StereoSideChannel(0x42),
            InbandMessage::MaxBitrate(12_345),
            InbandMessage::AckPacket(99),
            InbandMessage::Reserved14(123),
            InbandMessage::Terminator,
        ];
        for msg in cases.iter() {
            let mut bw = BitWriter::new();
            msg.encode(&mut bw).unwrap();
            assert_eq!(bw.bit_position() as u32, msg.wire_bits(), "{msg:?}");
        }
    }

    #[test]
    fn pad_byte_aligned_writes_nothing() {
        let mut bw = BitWriter::new();
        // Write exactly 8 bits.
        bw.write_bits(0xAB, 8);
        let pad = pad_to_octet_boundary(&mut bw);
        assert_eq!(pad, 0);
        let bytes = bw.finish();
        assert_eq!(bytes, vec![0xAB]);
    }

    #[test]
    fn pad_three_bits_writes_five_bit_marker() {
        // RFC 5574 §3.3: 0 followed by all ones to next octet boundary.
        // After 3 written bits, we need 5 padding bits = 0b01111.
        let mut bw = BitWriter::new();
        bw.write_bits(0b101, 3);
        let pad = pad_to_octet_boundary(&mut bw);
        assert_eq!(pad, 5);
        let bytes = bw.finish();
        // MSB-first packing: 3-bit 0b101 then 5-bit 0b01111 →
        // 1010_1111 = 0xAF. (The leading "0" of the padding aligns with
        // bit position 3; the four ones fill bits 4..=7.)
        assert_eq!(bytes, vec![0xAF]);
    }

    #[test]
    fn pad_seven_bits_writes_one_bit_marker() {
        // After 7 bits, need 1 padding bit = "0".
        let mut bw = BitWriter::new();
        bw.write_bits(0b101_0101, 7);
        let pad = pad_to_octet_boundary(&mut bw);
        assert_eq!(pad, 1);
        let bytes = bw.finish();
        // 7-bit 0b101_0101 then 1-bit 0 → 1010_1010 = 0xAA.
        assert_eq!(bytes, vec![0xAA]);
    }

    #[test]
    fn padding_then_terminator_is_recoverable() {
        // Encoder side: emit a terminator, pad to byte. Decoder side:
        // reads terminator and stops. Padding bits must NOT be parsed
        // as another in-band marker (because their leading "0" decodes
        // as `m=0`, not 13/14/15 — `decode_inband` returns None there).
        let mut bw = BitWriter::new();
        InbandMessage::Terminator.encode(&mut bw).unwrap();
        pad_to_octet_boundary(&mut bw);
        let bytes = bw.finish();
        // Should be exactly one byte: 4-bit 0b1111 then 4 padding bits
        // 0b0111 = 1111_0111 = 0xF7.
        assert_eq!(bytes, vec![0xF7]);
        let mut br = BitReader::new(&bytes);
        let term = decode_inband(&mut br).unwrap().unwrap();
        assert_eq!(term, InbandMessage::Terminator);
        // The next decode returns None (the leading "0" is `m=0`).
        let after = decode_inband(&mut br).unwrap();
        assert!(after.is_none(), "padding must not be parsed as inband");
    }

    #[test]
    fn decode_truncated_request_errors_cleanly() {
        // Just the 4-bit m=14 marker, no id.
        let mut bw = BitWriter::new();
        bw.write_bits(M_INBAND_REQUEST, 4);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let err = decode_inband(&mut br);
        assert!(err.is_err(), "no id after m=14 must error");
    }

    #[test]
    fn decode_truncated_payload_errors_cleanly() {
        // m=14, id=10 (MaxBitrate) needs 16 payload bits — supply none.
        let mut bw = BitWriter::new();
        bw.write_bits(M_INBAND_REQUEST, 4);
        bw.write_bits(codes::MAX_BITRATE, 4);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let err = decode_inband(&mut br);
        assert!(err.is_err(), "no payload after id must error");
    }
}
