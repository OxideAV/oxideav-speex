//! # oxideav-speex
//!
//! Pure-Rust Speex (CELP speech codec): decoder and encoder for all three
//! rate classes — narrowband 8 kHz, wideband 16 kHz, ultra-wideband
//! 32 kHz — at every quality 0..=10, mono or intensity stereo, with the
//! perceptual enhancer. Implemented from *The Speex Codec Manual* and
//! RFC 5574. The float decode matches reference decodes to ±1 LSB; the
//! encoder produces conformant streams (it is not a bit-stream twin of
//! the reference encoder).
//!
//! ## Which API to use
//!
//! | you have | use |
//! | --- | --- |
//! | raw 20 ms PCM frames, want packets (any rate, mono / stereo) | [`SpeexStreamEncoder`] |
//! | Speex packets + an Ogg/Speex header (or a known rate) | [`SpeexStreamDecoder`] |
//! | the `oxideav-core` [`Decoder`](oxideav_core::Decoder) / [`Encoder`](oxideav_core::Encoder) traits | [`register`], or [`make_decoder`] / [`make_encoder`] without a registry |
//!
//! Everything else exported here (per-band encoders/decoders, bit-stream
//! parsers, codebooks, gain laws) is the lower layer those types are
//! built from; integrations do not need it.
//!
//! ## Direct API
//!
//! One packet per 20 ms frame (160 / 320 / 640 samples per channel at
//! 8 / 16 / 32 kHz). The header is the Ogg/Speex packet 0 (Table 7.1),
//! 80 bytes via [`SpeexHeader::write_bytes`].
//!
//! ```
//! use oxideav_speex::{SpeexHeader, SpeexStreamDecoder, SpeexStreamEncoder};
//!
//! let mut enc = SpeexStreamEncoder::new(32_000, 10).unwrap(); // UWB, ~44 kbit/s
//! let header: [u8; 80] = enc.header().write_bytes();
//!
//! let mut dec = SpeexStreamDecoder::for_header(&SpeexHeader::parse(&header).unwrap()).unwrap();
//! // No header on the wire (e.g. RTP)? `SpeexStreamDecoder::for_sample_rate(32_000)`.
//!
//! let frame = vec![0i16; enc.frame_samples()]; // 640 samples = 20 ms
//! let packet: Vec<u8> = enc.encode_frame(&frame).unwrap();
//! let pcm: Vec<i16> = dec.decode_packet_pcm_i16(&packet).unwrap();
//! assert_eq!(pcm.len(), 640);
//! ```
//!
//! Stereo: [`SpeexStreamEncoder::new_stereo`] takes interleaved L/R
//! frames (`2 × frame_samples()` samples) and
//! [`SpeexStreamDecoder::decode_packet_stereo_i16`] returns interleaved
//! L/R — the Speex in-band intensity-stereo extension.
//!
//! ## Framework API
//!
//! ```
//! use oxideav_core::{CodecId, CodecParameters, RuntimeContext, SampleFormat};
//!
//! let mut ctx = RuntimeContext::new();
//! oxideav_speex::register(&mut ctx);
//!
//! let mut params = CodecParameters::audio(CodecId::new("speex"));
//! params.sample_rate = Some(16_000);
//! params.channels = Some(1);
//! params.sample_format = Some(SampleFormat::S16);
//! params.options.insert("quality", "8");
//! let enc = ctx.codecs.first_encoder(&params).unwrap();
//! // extradata = the 80-byte stream header; the decoder accepts it, a
//! // bare `sample_rate`, or the header in-band as the first packet.
//! let dec = ctx.codecs.first_decoder(enc.output_params()).unwrap();
//! # let _ = dec;
//! ```
//!
//! The encoder takes interleaved S16 frames of any length and re-blocks
//! them into 20 ms packets; the decoder emits interleaved S16, one audio
//! frame per packet. Options: encoder `quality` (0..=10, default 8),
//! decoder `enhancer` (default on). Runnable versions of both paths are
//! in `examples/roundtrip.rs` and `examples/registry.rs`.
//!
//! ## Limits
//!
//! * The fixed-point decode build is not reproduced (the float decode is
//!   within a few LSB of it).
//! * The encoder's searches are its own analysis-by-synthesis without a
//!   perceptual weighting filter: conformant, input-tracking streams,
//!   not the reference encoder's exact bits.
//! * Ogg encapsulation lives in the container layer (`oxideav-ogg`);
//!   this crate handles the header and packets.

#![warn(missing_debug_implementations)]

mod abs_search;
mod adaptive_codebook;
mod adaptive_contribution;
mod bitreader;
mod codebooks;
mod decoder;
mod encoder_nb;
mod encoder_uwb;
mod encoder_wb;
mod enhancer;
mod excitation;
mod fixed_codebook_gain;
mod fixed_point;
mod forced_pitch_gain;
mod frame;
mod framework;
mod gain_reconstruction;
mod gain_scaled_excitation;
mod gain_scaled_hb_innovation;
mod gain_scaled_innovation;
mod gain_scaled_pitch;
mod hb_encode;
mod hb_excitation_gain;
mod hb_fold;
mod hb_innovation;
mod hb_innovation_search;
mod hb_lsp;
mod hb_lsp_interp;
mod hb_synthesis;
mod header;
mod innovation;
mod innovation_search;
mod lpc_analysis;
mod lpc_to_lsp;
mod lsp;
mod lsp_base;
mod lsp_interp;
mod lsp_pi_domain;
mod lsp_quant;
mod lsp_to_lpc;
mod narrowband_body;
mod narrowband_decoder;
mod nb_encode;
mod ol_pitch;
mod output_highpass;
mod packet;
mod pitch_gain;
mod qmf;
mod quality;
mod signalling;
mod stereo;
mod stream_decoder;
mod stream_encoder;
mod submode;
mod synthesis;
mod uwb_decoder;
mod vad;
mod wb_synthesis;
mod weighting;
mod wideband;
mod wideband_decoder;

// internal: closed-loop pitch-search plumbing, exposed for tests only
#[doc(hidden)]
pub use abs_search::{
    ClosedLoopPitch, WeightFactors, closed_loop_pitch_search, weighted_synthesis_zero_state,
};
// internal: long-term-predictor index/buffer plumbing, exposed for tests only
#[doc(hidden)]
pub use adaptive_codebook::{
    ADAPTIVE_CODEBOOK_TAPS, EXCITATION_HISTORY_LEN, ExcitationBuffer, ExcitationError,
    TAP_PITCH_OFFSETS, resolve_lookback, sample_lookback_indices, subframe_lookback_indices,
};
// internal: adaptive-codebook contribution plumbing, exposed for tests only
#[doc(hidden)]
pub use adaptive_contribution::{
    AdaptiveContributionError, adaptive_contribution_sample, adaptive_contribution_subframe,
};
pub use bitreader::{BitError, BitReader, BitWriter};
// internal: raw codebook-table accessors, exposed for tests only
#[doc(hidden)]
pub use codebooks::{
    HB_LSP_STAGE_ENTRIES, InnovationShape, LPC_ANALYSIS_WINDOW_LEN, LPC_LAG_WINDOW_LEN,
    NB_LSP_SPLIT_HALF, NB_LSP_STAGE_ENTRIES, NbLspScale, PITCH_GAIN_5BIT_ENTRIES,
    PITCH_GAIN_7BIT_ENTRIES, PITCH_GAIN_BIAS, PITCH_GAIN_COLS, QMF_FILTER_LEN, hb_innovation_8_128,
    hb_innovation_10_32, hb_lsp_scale, hb_lsp_stage1, hb_lsp_stage2, innovation_5_64,
    innovation_5_256, innovation_8_128, innovation_10_16, innovation_10_32, innovation_20_32,
    lpc_analysis_window_float, lpc_analysis_window_q15, lpc_lag_window_float, lpc_lag_window_q15,
    nb_lsp_high1, nb_lsp_high2, nb_lsp_low1, nb_lsp_low2, nb_lsp_scale, nb_lsp_stage0,
    pitch_gain_5bit, pitch_gain_7bit, qmf_h0_float, qmf_h0_q15,
};
// stable: these two order constants appear in visible signatures
// ([`NarrowbandFrameBody::reconstructed_lsp_q10`] / [`WidebandHighBandBody::hb_lpc`])
pub use codebooks::{HB_LPC_ORDER, NB_LSP_ORDER};
pub use decoder::{ControlMessage, DecodeError, DecodedFrame, SpeexDecoder};
pub use encoder_nb::{EncodeError, NB_FRAME_SAMPLES, NarrowbandEncoder};
pub use encoder_uwb::{
    UWB_HB_SUBFRAME_SAMPLES, UltraWidebandEncoder, UwbEncodeError, UwbFrameBodies,
};
pub use encoder_wb::{WbEncodeError, WidebandEncoder, WidebandFrameBodies, hb_encoder_gain_index};
// internal: raw excitation composition plumbing, exposed for tests only
#[doc(hidden)]
pub use excitation::{RAW_EXCITATION_SAMPLES, raw_excitation_sample, raw_excitation_subframe};
// stable: typed gain-index pair returned by
// [`NarrowbandFrameBody::fixed_codebook_gain_indices`] (plus its two field types)
pub use fixed_codebook_gain::{
    FixedCodebookGainIndices, FrameInnovationGainIndex, SubFrameInnovationGainCorrection,
};
// internal: gain-index batch helper + bit-budget constants, exposed for tests only
#[doc(hidden)]
pub use fixed_codebook_gain::{
    FRAME_OL_EXC_GAIN_BITS, FRAME_OL_EXC_GAIN_ENTRIES, fixed_codebook_gain_indices,
};
// internal: mode-7 forced pitch-gain grid plumbing, exposed for tests only
#[doc(hidden)]
pub use forced_pitch_gain::{
    FORCED_PITCH_GAIN_BITS, FORCED_PITCH_GAIN_LEVELS, FORCED_PITCH_GAIN_STEP, forced_pitch_coef,
    forced_pitch_gain_taps,
};
pub use frame::{FrameError, NARROWBAND_FRAME_PREFIX_BITS, NarrowbandFrameHeader};
pub use framework::{
    SpeexDecoderOptions, SpeexEncoderOptions, SpeexFrameworkDecoder, SpeexFrameworkEncoder,
    make_decoder, make_encoder, register,
};
// internal: scalar gain quantiser tables/helpers, exposed for tests only
#[doc(hidden)]
pub use gain_reconstruction::{
    HB_FOLDED_GAIN_LEVELS, HB_GAIN_CORRECTION_LEVELS, NB_OL_EXC_GAIN_LEVELS,
    NB_SUBFRAME_GAIN_LEVELS_1BIT, NB_SUBFRAME_GAIN_LEVELS_3BIT, hb_folded_gain_levels,
    hb_gain_correction_levels, hb_gc_quant_bound, nb_ol_exc_gain_levels,
    nb_subframe_gain_bound_1bit, nb_subframe_gain_bounds_3bit, nb_subframe_gain_levels_1bit,
    nb_subframe_gain_levels_3bit, quantise_frame_ol_exc_gain, quantise_frame_ol_exc_gain_exact,
    quantise_hb_exc_gain, quantise_subframe_gain_correction, reconstruct_fixed_codebook_gain,
    reconstruct_frame_ol_exc_gain, reconstruct_hb_exc_gain, reconstruct_subframe_gain_correction,
};
// internal: gain-scaled excitation plumbing, exposed for tests only
#[doc(hidden)]
pub use gain_scaled_excitation::{
    GAIN_SCALED_EXCITATION_SAMPLES, gain_scaled_excitation_sample, gain_scaled_excitation_subframe,
};
// internal: gain-scaled high-band innovation plumbing, exposed for tests only
#[doc(hidden)]
pub use gain_scaled_hb_innovation::{
    GAIN_SCALED_HB_INNOVATION_SAMPLES, HB_INNOVATION_POLARITY, gain_scaled_hb_innovation_from_body,
    gain_scaled_hb_innovation_from_body_leveled, gain_scaled_hb_innovation_sample,
    gain_scaled_hb_innovation_subframe, hb_gc_crossover_gain,
};
pub use gain_scaled_hb_innovation::{
    HB_CROSSOVER_REGULARISER, HB_RMS_FLOOR, floored_rms, hb_gain_lb_base,
};
// internal: gain-scaled innovation plumbing, exposed for tests only
#[doc(hidden)]
pub use gain_scaled_innovation::{
    GAIN_SCALED_INNOVATION_SAMPLES, INNOVATION_CODEBOOK_SCALE, gain_scaled_innovation_from_indices,
    gain_scaled_innovation_from_shape_f32, gain_scaled_innovation_sample,
    gain_scaled_innovation_subframe,
};
// internal: gain-scaled pitch contribution plumbing, exposed for tests only
#[doc(hidden)]
pub use gain_scaled_pitch::{
    GAIN_SCALED_PITCH_SAMPLES, PITCH_GAIN_SCALING, gain_scaled_pitch_sample,
    gain_scaled_pitch_subframe, gain_scaled_pitch_subframe_forced,
    gain_scaled_pitch_subframe_repeat,
};
// internal: high-band frame-writer plumbing behind [`WidebandEncoder`], exposed for tests only
#[doc(hidden)]
pub use hb_encode::{encode_wideband_frame, write_high_band_body, write_high_band_frame};
// stable: typed gain index returned by [`WidebandHighBandBody::hb_excitation_gain_indices`]
pub use hb_excitation_gain::HbExcitationGainIndex;
// internal: gain-index batch helper + bit-width constants, exposed for tests only
#[doc(hidden)]
pub use hb_excitation_gain::{
    HB_EXC_GAIN_BITS_MODE_1, HB_EXC_GAIN_BITS_MODES_2_TO_4, hb_excitation_gain_indices,
};
// internal: folded high-band excitation law plumbing, exposed for tests only
#[doc(hidden)]
pub use hb_fold::{
    HB_FOLD_CROSSOVER_SCALE, HB_FOLD_ENVELOPE_COMPRESSION_GAMMA, HB_FOLD_RECONSTRUCTION_MULT,
    UWB_FOLD_RECONSTRUCTION_MULT, folded_hb_excitation_slice, folded_hb_excitation_subframe,
    folded_hb_excitation_subframe_crossover, folded_uwb_excitation_slice, hb_crossover_response,
    hb_crossover_response_bwexp, hb_crossover_response_from_lsp, upsample_hb_excitation_linear,
};
#[doc(hidden)]
pub use hb_fold::{
    UWB_FOLD_CROSSOVER_SCALE, fold_gain_from_index, fold_index_for_gain,
    folded_uwb_excitation_subframe_crossover, hb_fold_ratio_inv,
};
// stable: error + sub-frame length used in the visible
// [`WidebandHighBandBody::hb_innovation_sub_vector`] signature
pub use hb_innovation::{HB_SUBFRAME_SAMPLES, HbInnovationError};
// internal: high-band innovation codebook plumbing, exposed for tests only
#[doc(hidden)]
pub use hb_innovation::{
    HB_MODE4_STAGE2_WEIGHT, HbInnovationCodebook, HbInnovationMapping, decode_hb_subframe,
    decode_hb_subframe_mode4_f32, hb_innovation_sub_vector,
};
// internal: high-band innovation search plumbing, exposed for tests only
#[doc(hidden)]
pub use hb_innovation_search::{HbInnovationChoice, search_hb_innovation};
// stable: typed MSVQ stage pair returned by [`WidebandHighBandBody::lsp_stages`]
pub use hb_lsp::HbLspStages;
// internal: high-band LSP MSVQ plumbing, exposed for tests only
#[doc(hidden)]
pub use hb_lsp::{
    HB_LSP_INDEX_MASK, HB_LSP_OUTPUT_Q, HB_LSP_PACKED_BITS, HB_LSP_STAGE_BITS, pack_hb_lsp_index,
    quantise_q10 as quantise_hb_lsp_q10, reconstruct_q10 as reconstruct_hb_lsp_q10,
};
// internal: high-band LSP interpolation plumbing, exposed for tests only
#[doc(hidden)]
pub use hb_lsp_interp::{HB_LSP_INTERP_OUTPUT_Q, HB_LSP_SUBFRAMES_PER_FRAME, HbSubFrameLsp};
// internal: high-band synthesis filter plumbing, exposed for tests only
#[doc(hidden)]
pub use hb_synthesis::HbSynthesisFilter;
pub use header::{
    HeaderError, SPEEX_HEADER_LEN, SPEEX_MAGIC, SPEEX_MODE_NARROWBAND, SPEEX_MODE_ULTRAWIDEBAND,
    SPEEX_MODE_WIDEBAND, SPEEX_STRING_LEN, SPEEX_VERSION_LEN, SpeexHeader,
};
// stable: error + sub-frame length used in the visible
// [`NarrowbandFrameBody::innovation_sub_vector`] signature
pub use innovation::{InnovationError, SUBFRAME_SAMPLES};
// internal: narrowband innovation codebook plumbing, exposed for tests only
#[doc(hidden)]
pub use innovation::{
    InnovationCodebook, InnovationMapping, NB_MODE7_STAGE2_WEIGHT,
    decode_subframe as decode_innovation_subframe, decode_subframe_f32,
    sub_vector as innovation_sub_vector,
};
// internal: narrowband innovation search plumbing, exposed for tests only
#[doc(hidden)]
pub use innovation_search::{InnovationChoice, search_innovation};
// internal: encoder LPC analysis plumbing, exposed for tests only
#[doc(hidden)]
pub use lpc_analysis::{
    AUTOCORR_LAGS, HB_AUTOCORR_LAGS, HbLpcCoefficients, LPC_NOISE_FLOOR, LpcAnalysisError,
    LpcCoefficients, analyse as lpc_analyse, analyse_hb as hb_lpc_analyse, apply_analysis_window,
    autocorrelate, levinson_durbin, stabilise_autocorrelation,
};
// internal: encoder LPC->LSP root-find plumbing, exposed for tests only
#[doc(hidden)]
pub use lpc_to_lsp::{LSP_BISECT_ITERS, LSP_SCAN_STEPS, LpcToLspError, hb_lpc_to_lsp, lpc_to_lsp};
// stable: typed VQ stage split returned by [`NarrowbandFrameBody::lsp_stages`]
pub use lsp::NbLspStages;
// internal: narrowband LSP VQ plumbing, exposed for tests only
#[doc(hidden)]
pub use lsp::{
    NB_LSP_INDEX_MASK, NB_LSP_OUTPUT_Q, NB_LSP_STAGE_BITS, NB_LSP_STAGES_18BIT,
    NB_LSP_STAGES_30BIT, reconstruct_q10 as reconstruct_nb_lsp_q10,
};
// internal: LSP base-vector/margin plumbing, exposed for tests only
#[doc(hidden)]
pub use lsp_base::{
    HB_LSP_BASE_INTERCEPT_Q10, HB_LSP_BASE_SLOPE_Q10, HB_LSP_MARGIN_Q10, HB_LSP_MARGIN_RADIANS,
    NB_LSP_BASE_INTERCEPT_Q10, NB_LSP_BASE_SLOPE_Q10, NB_LSP_MARGIN_Q10, NB_LSP_MARGIN_RADIANS,
    add_hb_lsp_base, add_nb_lsp_base, enforce_lsp_margin_radians, hb_lsp_base_q10,
    hb_lsp_base_radians, hb_lsp_margin_radians, nb_lsp_base_q10, nb_lsp_base_radians,
    nb_lsp_margin_radians,
};
// stable: per-sub-frame LSP matrix returned by
// [`NarrowbandFrameBody::interpolated_lsp_q12`] (+ its public array-length constant)
pub use lsp_interp::{NB_LSP_SUBFRAMES_PER_FRAME, NbSubFrameLsp};
// internal: interpolation Q-format constant, exposed for tests only
#[doc(hidden)]
pub use lsp_interp::NB_LSP_INTERP_OUTPUT_Q;
// internal: LSP_PI storage-domain plumbing, exposed for tests only
#[doc(hidden)]
pub use lsp_pi_domain::{
    LSP_PI, hb_lsp_linear_storage, lsp_storage_to_q10, lsp_storage_to_radians,
    nb_lsp_linear_storage, radians_to_lsp_storage,
};
// internal: encoder LSP-VQ quantiser plumbing, exposed for tests only
#[doc(hidden)]
pub use lsp_quant::{pack_lsp_index, quantise_lsp_q10};
// stable: LPC order constant used in the visible
// [`NarrowbandDecoder::synthesis_history`] signature
pub use lsp_to_lpc::LPC_ORDER;
// internal: LSP->LPC conversion plumbing, exposed for tests only
// internal: the staged fixed-point primitives (no fixed-point decoder), exposed for tests only
pub use enhancer::{
    ENHANCER_BLOCK_SAMPLES, EnhancerBlock, enhance_block, enhancer_comb_gain, enhancer_period,
};
#[doc(hidden)]
pub use fixed_point::{
    FixedNoise, LSP_PI_Q13, hb_gain_q14, hb_lsp_q13, interpolate_lsp_q13, lpc_at_minus_one_q13,
    lsp_cos_q13, lsp_cos_stored, lsp_to_lpc_q13, nb_lsp_q13,
};
#[doc(hidden)]
pub use lsp_to_lpc::{
    HB_LPC_ORDER_OUT, hb_lsp_q10_to_radians, hb_lsp_to_lpc, hb_lsp_with_base_q10,
    hb_subframe_lpc_set_with_base, lpc_from_hb_lsp_delta_q10, lpc_from_hb_lsp_q10,
    lpc_from_hb_subframe_lsp_q12, lpc_from_lsp_delta_q10, lpc_from_lsp_q10,
    lpc_from_subframe_lsp_q12, lsp_q10_to_radians, lsp_qn_to_radians, lsp_to_lpc,
    lsp_vector_q10_to_radians, lsp_vector_radians_to_q10, nb_lsp_with_base_q10, radians_to_lsp_q10,
    subframe_lpc_set, subframe_lpc_set_with_base,
};
pub use narrowband_body::{
    NarrowbandBodyError, NarrowbandFrameBody, NarrowbandSubFrameIndices, PITCH_PERIOD_MAX,
    PITCH_PERIOD_MIN,
};
pub use narrowband_decoder::{
    NARROWBAND_FRAME_SAMPLES, NB_MODE1_NOISE_SCALE, NB_OUTPUT_HIGHPASS, NarrowbandDecodeError,
    NarrowbandDecoder, WB_LOW_BAND_OUTPUT_HIGHPASS, saturate_i16,
};
// internal: float LSP-path primitives shared with the high band, exposed for tests only
#[doc(hidden)]
pub use narrowband_decoder::{enforce_interpolated_margin, lsp_cos_poly, lsp_to_lpc_poly};
// internal: narrowband frame-writer plumbing behind [`NarrowbandEncoder`], exposed for tests only
#[doc(hidden)]
pub use nb_encode::{encode_narrowband_frame, write_narrowband_body, write_packet_terminator};
// internal: open-loop pitch estimation plumbing, exposed for tests only
#[doc(hidden)]
pub use ol_pitch::{OpenLoopPitch, estimate_open_loop_pitch, estimate_open_loop_pitch_range};
pub use output_highpass::{OUTPUT_HIGHPASS_CUTOFF_HZ, OutputHighpass};
pub use packet::{FrameKind, PacketError, PacketFrame, PacketFrames, PacketSummary, parse_packet};
// stable: 3-tap gain triple returned by [`NarrowbandSubFrameIndices::pitch_gain_taps`]
// (+ its public array-length constant)
pub use pitch_gain::{PITCH_GAIN_TAPS, PitchGainTaps};
// internal: pitch-gain VQ reconstruction helper, exposed for tests only
#[doc(hidden)]
pub use pitch_gain::reconstruct as reconstruct_pitch_gain;
// stable: wideband frame length used in visible PCM signatures
// ([`WidebandFrame::wideband_pcm`] / [`DecodedFrame::Wideband`])
pub use qmf::QMF_WIDEBAND_FRAME;
// internal: QMF filterbank plumbing, exposed for tests only
#[doc(hidden)]
pub use qmf::{QMF_HALF_BAND_FRAME, QmfAnalysis, QmfSynthesis};
pub use quality::{
    FRAMES_PER_SECOND, MAX_QUALITY, UWB_HIGH_BAND_MODE, UwbQualityModes, WbQualityModes,
    nb_mode_for_quality, uwb_bitrate_bps, uwb_modes_for_quality, wb_bitrate_bps,
    wb_modes_for_quality,
};
pub use signalling::{
    AcknowledgePolicy, CUSTOM_INBAND_MAX_BYTES, CUSTOM_INBAND_SIZE_BITS, CustomInbandMessage,
    INBAND_CODE_BITS, INBAND_TABLE_5_1, InbandCodeSpec, InbandKind, InbandMessage, InbandRequest,
    RateModeConfig, SignallingError, inband_code_spec,
};
pub use stereo::{
    STEREO_E_RATIO, STEREO_INTERP_A, StereoDecoder, StereoGains, downmix_mean,
    encode_stereo_payload, pack_stereo_payload, stereo_gains,
};
pub use stream_decoder::{SpeexStreamDecoder, StereoFrame, StreamDecodeError};
pub use stream_encoder::{SpeexStreamEncoder, StreamEncodeError};
pub use submode::{LspQuant, NARROWBAND_SUBMODES, NarrowbandSubmode, PitchGainQuant, Submode};
// internal: narrowband LPC synthesis filter plumbing, exposed for tests only
#[doc(hidden)]
pub use synthesis::SynthesisFilter;
pub use uwb_decoder::{
    UWB_FRAME_SAMPLES, UWB_HALF_BAND_FRAME, UWB_HB_SUBFRAMES, UltraWidebandDecoder,
    UltraWidebandFrame, UwbDecodeError, UwbDecodedFrame,
};
pub use vad::{DTX_MODE, EnergyVad};
// stable: high-band frame length used in visible signatures
// ([`WidebandFrame::high_band`] / [`WidebandDecoder::last_hb_excitation`])
pub use wb_synthesis::HB_FRAME_SAMPLES;
// internal: high-band synthesis assembly plumbing, exposed for tests only
#[doc(hidden)]
pub use wb_synthesis::{
    HB_SUBFRAMES_PER_FRAME, SubBandLayer, UwbFrameLayout, synthesise_high_band_frame,
    synthesise_high_band_frame_folded, synthesise_high_band_frame_folded_exc,
    synthesise_high_band_frame_interp, synthesise_high_band_frame_leveled,
};
// internal: perceptual-weighting filter plumbing, exposed for tests only
#[doc(hidden)]
pub use weighting::{PerceptualWeighting, WEIGHT_GAMMA1, WEIGHT_GAMMA2, weighted_coeffs};
pub use wideband::{
    HIGH_BAND_FRAME_PREFIX_BITS, HIGH_BAND_SUBFRAMES_PER_FRAME, HighBandSubFrameIndices,
    WIDEBAND_HIGH_BAND_SUBMODES, WidebandBodyError, WidebandHighBandBody,
    WidebandHighBandFrameHeader, WidebandHighBandSubmode, WidebandSubmode,
};
pub use wideband_decoder::{WidebandDecodeError, WidebandDecoder, WidebandFrame};

/// Crate-local error type aggregating the parse-layer errors. The
/// decode/encode paths surface their own typed errors
/// ([`DecodeError`], [`EncodeError`], …); the framework
/// [`make_decoder`] / [`make_encoder`] surface maps everything into
/// `oxideav_core::Error`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The Ogg/Speex stream header failed to parse — see [`HeaderError`].
    Header(HeaderError),
    /// The per-frame leading prefix failed to parse — see [`FrameError`].
    Frame(FrameError),
    /// The narrowband CELP frame body failed to parse — see
    /// [`NarrowbandBodyError`].
    NarrowbandBody(NarrowbandBodyError),
    /// A §5.5 in-band signalling body failed to parse — see
    /// [`SignallingError`].
    Signalling(SignallingError),
    /// A wideband high-band frame body failed to parse — see
    /// [`WidebandBodyError`].
    Wideband(WidebandBodyError),
    /// A whole-packet walk failed mid-frame — see [`PacketError`].
    Packet(PacketError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Header(e) => write!(f, "oxideav-speex: {}", e),
            Error::Frame(e) => write!(f, "oxideav-speex: {}", e),
            Error::NarrowbandBody(e) => write!(f, "oxideav-speex: {}", e),
            Error::Signalling(e) => write!(f, "oxideav-speex: {}", e),
            Error::Wideband(e) => write!(f, "oxideav-speex: {}", e),
            Error::Packet(e) => write!(f, "oxideav-speex: {}", e),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Header(e) => Some(e),
            Error::Frame(e) => Some(e),
            Error::NarrowbandBody(e) => Some(e),
            Error::Signalling(e) => Some(e),
            Error::Wideband(e) => Some(e),
            Error::Packet(e) => Some(e),
        }
    }
}

impl From<HeaderError> for Error {
    fn from(e: HeaderError) -> Self {
        Error::Header(e)
    }
}

impl From<FrameError> for Error {
    fn from(e: FrameError) -> Self {
        Error::Frame(e)
    }
}

impl From<NarrowbandBodyError> for Error {
    fn from(e: NarrowbandBodyError) -> Self {
        Error::NarrowbandBody(e)
    }
}

impl From<SignallingError> for Error {
    fn from(e: SignallingError) -> Self {
        Error::Signalling(e)
    }
}

impl From<WidebandBodyError> for Error {
    fn from(e: WidebandBodyError) -> Self {
        Error::Wideband(e)
    }
}

impl From<PacketError> for Error {
    fn from(e: PacketError) -> Self {
        Error::Packet(e)
    }
}

oxideav_core::register!("speex", register);
