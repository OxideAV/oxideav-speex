//! # oxideav-speex
//!
//! **Status:** clean-room rebuild in progress (post 2026-05-19 audit).
//!
//! Layered scope as the rebuild advances:
//!
//! * **Round 1** — the Ogg/Speex stream-header packet parser from
//!   *The Speex Codec Manual* §7.3, Table 7.1 (validates `Speex   `
//!   magic + decodes 13 LE int32 fields). See [`SpeexHeader`].
//! * **Round 2** — the per-frame leading prefix (1-bit wideband flag +
//!   4-bit mode ID) from §9.3 + the typed narrowband sub-mode table
//!   from Table 9.1 (modes 0..=8, plus the three §5.5
//!   reserved-for-signalling slots: custom in-band mode 13, in-band
//!   signalling mode 14, terminator mode 15). See
//!   [`NarrowbandFrameHeader`] and [`Submode`].
//! * **Round 3** — the narrowband CELP frame-body bit-reader: walks
//!   Table 9.1's columns in the order documented in §9.3 ("All
//!   frame-based parameters are packed before sub-frame parameters.
//!   The parameters for a certain sub-frame are all packed before the
//!   following sub-frame is packed.") and surfaces every field as a
//!   raw bit-index in [`NarrowbandFrameBody`]. Codebook lookup (LSP VQ
//!   → coefficients, innovation VQ → 40-sample sub-vector) and LSP→LPC
//!   conversion stay deferred until the Speex CELP companion table
//!   docs gap closes.
//! * **Round 4** — the §5.5 in-band signalling bodies that the
//!   round-2 dispatcher leaves unparsed: [`InbandMessage`] walks
//!   mode 14's 4-bit Table 5.1 code + 1/4/8/16/32/64-bit payload;
//!   [`CustomInbandMessage`] consumes mode 13's 5-bit byte size +
//!   opaque payload. Table 5.1 itself is staged as the public
//!   [`INBAND_TABLE_5_1`] array indexed by code value. The §5.5 path
//!   needs no CELP companion tables — it is bit-stream framing only.
//! * **Round r165** (this commit) — typed packet → frame iterator
//!   composing the round-2..5 primitives end-to-end. See
//!   [`PacketFrames`], [`PacketFrame`], and the [`parse_packet`]
//!   convenience that returns a `Vec<PacketFrame>`. Walks a Speex
//!   packet body per §5.5 ("Sometimes it is desirable to pack more
//!   than one frame per packet … it is possible to include a
//!   terminator code. That terminator consists of the code 15
//!   (decimal) encoded with 5 bits"), dispatching each successive
//!   5-bit prefix into a CELP frame, a wideband narrowband+high-band
//!   pair, a §5.5 in-band signalling message, or a mode-13 custom
//!   in-band message, and terminating cleanly on mode 15 or on
//!   <5-bit padding tail.
//! * **Round 5** — the wideband high-band sub-mode
//!   table from §10.4 / Table 10.1 (modes 0..=4, 5 columns) plus a
//!   high-band frame-body bit-reader. A wideband packet is the
//!   concatenation of an embedded narrowband frame (round 3) followed
//!   by a 1-bit wideband flag, a 3-bit high-band mode ID, the 12-bit
//!   LSP MSVQ index (modes 1..=4), and four sub-frames of `gain || VQ`
//!   bits as listed in Table 10.1. See [`WidebandHighBandFrameHeader`]
//!   and [`WidebandHighBandBody`]. As with round 3 the body lands
//!   only the raw bit indices — the high-band LSP MSVQ codebook +
//!   innovation codebook are #969-blocked until staged.
//! * **Round r191** (this commit) — CELP companion-table accessors.
//!   The clean-room codebook arrays staged under
//!   `docs/audio/speex/tables/` (LSP VQ stages, 3-tap pitch-gain VQ,
//!   six narrowband innovation codebooks, the two wideband
//!   high-band LSP MSVQ stages + two high-band innovation
//!   codebooks, plus the Q15 LPC analysis window, autocorrelation
//!   lag window, and QMF analysis filter) are now embedded into
//!   the crate via [`include_str!`] and parsed on first use into
//!   typed `&'static [Row]` slices. Dimension constants are
//!   cross-checked against the existing
//!   [`NarrowbandSubmode`] bit-budget table. Lookup, scaling, and
//!   the downstream LSP→LPC / synthesis-filter path stay deferred.
//!   See [`codebooks`] (re-exported at the crate root).
//! * **Round r187** — structured `write` methods
//!   symmetric to the existing `parse` paths for the three
//!   framing-level types whose layout is fully defined by the
//!   manual without any CELP companion-table material:
//!   [`NarrowbandFrameHeader::write`] emits the 5-bit prefix
//!   (1-bit wideband flag + 4-bit mode ID per §9.3);
//!   [`InbandMessage::write`] emits the 4-bit Table 5.1 code +
//!   1/4/8/16/32/64-bit payload (with the same >32-bit split path
//!   the parser uses for reserved codes 14 / 15);
//!   [`CustomInbandMessage::write`] emits the 5-bit `size_bytes`
//!   per §5.5 + opaque payload bytes from a caller-supplied
//!   slice. A new [`NarrowbandFrameHeader::new`] constructor is
//!   the encoder-side counterpart of the round-2 parser's
//!   reserved-mode rejection. All three writers depend only on
//!   the round-179 [`BitWriter`] + existing dispatch tables —
//!   no companion-table material was touched at that point.
//!
//! * **Round r194** — first companion-table → decoder
//!   pipeline wiring. The 5-stage narrowband LSP-VQ codebooks staged
//!   in [`codebooks`] now drive a reconstructed
//!   ten-coefficient LSP frequency vector in a common Q10
//!   fixed-point format. Given a parsed [`NarrowbandFrameBody`],
//!   [`NarrowbandFrameBody::reconstructed_lsp_q10`] resolves the
//!   raw `lsp_index` field into per-stage 6-bit indices and sums
//!   the contributions per the `.meta`-documented per-stage scaling
//!   factors (1/256, 1/512, 1/1024). See the new [`lsp`] module for
//!   the documented bit-stream-stage-ordering convention used to
//!   split the packed 18-bit / 30-bit LSP field into per-stage
//!   indices. LSP→LPC conversion, sub-frame interpolation, and
//!   downstream synthesis filtering stay deferred.
//!
//! * **Round r208** (this commit) — narrowband **3-tap pitch-gain VQ
//!   reconstruction** per Speex manual Eq. 9.1 / companion §2.2. The
//!   r191 5-bit (32 × 4) and 7-bit (128 × 4) pitch-gain VQ codebooks
//!   are now resolved through a typed accessor: given a per-sub-frame
//!   raw VQ index parsed by the round-3 frame-body bit-reader and the
//!   active sub-mode's [`PitchGainQuant`] regime, the new
//!   [`pitch_gain`] module returns [`PitchGainTaps`] — the three β tap
//!   coefficients `(g0, g1, g2)` of the long-term predictor
//!   convolution `ea[n] = g0·e[n-T-1] + g1·e[n-T] + g2·e[n-T+1]`,
//!   with the documented `+32` codebook bias already applied. Mode 0
//!   silence (`PitchGainQuant::None`) returns the all-zero
//!   [`PitchGainTaps::SILENCE`] without consulting any codebook. A
//!   [`NarrowbandSubFrameIndices::pitch_gain_taps`] convenience
//!   method wires the lookup off the existing per-sub-frame indices.
//!   The long-term predictor convolution itself stays deferred — it
//!   needs the per-sub-frame pitch period (already surfaced by
//!   [`NarrowbandSubFrameIndices::pitch_period`]) plus the historical
//!   excitation buffer, which lands once the innovation-codebook
//!   path is also wired.
//!
//! * **Round r200** — narrowband **sub-frame LSP
//!   interpolation** per the Speex manual §9.1 ("The LSP's are
//!   considered to be associated to the 4th sub-frames and the LSP's
//!   associated to the first 3 sub-frames are linearly interpolated
//!   using the current and previous LSP coefficients"). The new
//!   [`lsp_interp`] module takes the previous + current frame's
//!   reconstructed Q10 LSPs (from r194) and produces a
//!   `[[i32; 10]; 4]` matrix of per-sub-frame LSP vectors in Q12
//!   fixed-point — the unique linear-interpolation weight set
//!   `(3·prev + 1·curr) / 4`, `(2·prev + 2·curr) / 4`,
//!   `(1·prev + 3·curr) / 4`, `(0·prev + 4·curr) / 4 = curr`. The
//!   output is emitted in Q12 (not Q10-after-division) so the
//!   interpolation is exact integer arithmetic with no rounding
//!   direction choice for the spec to be silent about; the
//!   downstream LSP→LPC conversion can rescale with a single
//!   arithmetic shift. A new [`NarrowbandFrameBody::interpolated_lsp_q12`]
//!   convenience method composes the r194 reconstruction with this
//!   sub-frame interpolation. A `first_frame` constructor handles
//!   the stream-start case (prev = curr → no spurious transient).
//!   LSP→LPC conversion stays deferred — the in-repo Speex manual
//!   §9.1 + the staged CELP companion are both silent on the
//!   specific algorithm (the manual only states "converted back to
//!   the LPC filter Â(z)" without giving the polynomial root-find
//!   procedure). Reported as a docs gap.
//!
//! * **Round r220** (this commit) — narrowband **innovation sub-vector
//!   lookup primitive + per-mode dispatcher** for the two modes whose
//!   codebook binding is grounded by the staged material (Speex Codec
//!   Manual §9.2 + CELP companion §2.3 worked examples). The new
//!   [`innovation`] module exposes [`InnovationCodebook`] selecting one
//!   of the six documented sub-vector codebook shapes (5 / 8 / 10 / 20
//!   sample sub-vectors at 4 / 5 / 6 / 7 / 8 bits per index),
//!   [`innovation_sub_vector`] returning the `&'static [i16]` slice for
//!   one row, [`InnovationMapping::for_mode`] dispatching mode 0 to
//!   `Silence`, modes 6 and 8 to `Documented`, and the rest to
//!   `Undocumented`, plus [`decode_innovation_subframe`] decoding the
//!   40-sample fixed-codebook `c[n]` sub-vector for one CELP sub-frame
//!   by concatenating `count` MSB-first sub-vector lookups off the
//!   `innovation_vq_index` field. Wired off
//!   [`NarrowbandSubFrameIndices::innovation_sub_vector`]. Per-mode
//!   binding for modes 1 / 2 / 3 / 4 / 5 / 7 stayed deferred behind a
//!   docs gap until r277 narrowed it to modes 1 / 7 (see below).
//!
//! * **Round r230** (this commit) — wideband **high-band innovation
//!   sub-vector lookup primitive + per-mode dispatcher** mirroring
//!   r220's narrowband [`innovation`] path for the sub-band-CELP
//!   high band. Spec basis: Speex Codec Manual §10.3 *"the encoding
//!   of the high-band excitation is done in a way similar to that
//!   of the narrowband innovation"*, Table 10.1's per-sub-frame
//!   `excitation_vq_bits` widths (0 / 0 / 20 / 40 / 80 for modes
//!   0..=4), and the two staged high-band codebook shapes in
//!   `docs/audio/speex/tables/README.md`. The new [`hb_innovation`]
//!   module exposes [`HbInnovationCodebook`] selecting between
//!   `HbSv8_128` (8-sample × 7-bit index + 1-bit sign, 128 entries)
//!   and `HbSv10_32` (10-sample × 5-bit, 32 entries),
//!   [`hb_innovation_sub_vector`] returning the `&'static [i16]`
//!   slice for one row, [`HbInnovationMapping::for_mode`]
//!   dispatching modes 0 / 1 to `Silence`, mode 2 to `Documented`
//!   (4 × `HbSv10_32`), mode 3 to `Documented` (5 × `HbSv8_128`
//!   with per-sub-vector sign), mode 4 to `Undocumented` (no
//!   unique decomposition over the staged inventory), plus
//!   [`decode_hb_subframe`] decoding the 40-sample fixed-codebook
//!   excitation sub-vector for one high-band CELP sub-frame. Wired
//!   off [`WidebandHighBandBody::hb_innovation_sub_vector`].
//!   High-band fixed-codebook gain scaling + excitation buffer
//!   state + mode-4 codebook binding stay deferred.
//!
//! * **Round r214** — wideband **high-band LSP MSVQ
//!   reconstruction** per Speex manual §10.1 / companion §9. The
//!   r191 two-stage 6-bit MSVQ codebooks (`hb_lsp_stage1`,
//!   `hb_lsp_stage2`; 64 × 8 each) now resolve through a typed
//!   accessor: given the 12-bit packed `lsp_index` field already
//!   surfaced by [`WidebandHighBandBody::lsp_index`] (round-r160)
//!   plus the resolved sub-mode, the new [`hb_lsp`] module returns
//!   an eight-coefficient Q10 fixed-point LSP frequency vector —
//!   matching the r194 narrowband Q-format so both bands speak the
//!   same downstream Q-format. [`HbLspStages::from_packed`] splits
//!   the packed field into per-stage 6-bit indices (top 6 bits →
//!   level-1 codebook, bottom 6 bits → level-2 residual codebook);
//!   [`hb_lsp::reconstruct_q10`] sums the two staged contributions
//!   with the `.meta`-documented per-stage scaling (`1/256` → ×4
//!   and `1/512` → ×2). New
//!   [`WidebandHighBandBody::lsp_stages`] and
//!   [`WidebandHighBandBody::reconstructed_lsp_q10`] convenience
//!   methods wire the new module off the existing parsed body,
//!   mirroring r194's narrowband path. Silence mode 0 propagates
//!   `None`. High-band LSP→LPC conversion + the equivalent
//!   r200-style sub-frame interpolation rule (if any) stay
//!   deferred — the in-repo manual §10 is silent on whether the
//!   high band uses the same interpolation scheme as the
//!   narrowband or none at all.
//!
//! * **Round r241** (this commit) — narrowband **adaptive-codebook
//!   contribution sum** composing r208 + r234 into the per-sub-frame
//!   `[i32; 40]` evaluation of Speex Manual §9.2 Eq. 9.1
//!   `ea[n] = g0·e[n−T−1] + g1·e[n−T] + g2·e[n−T+1]`. The new
//!   [`adaptive_contribution`] module exposes
//!   [`adaptive_contribution_subframe`] (whole sub-frame batch) and
//!   [`adaptive_contribution_sample`] (per-sample helper) returning
//!   the raw integer dot product of the post-bias gain triple with
//!   the three substituted historical samples — no Q-format scaling,
//!   no rounding, no widening shift. Output is in Q-format-agnostic
//!   raw integer units in `i32` (documented headroom: ≤ 3 × 159 ×
//!   `i16::MAX` ≈ 1.6e7, well below `i32::MAX`). Stream-start case is
//!   handled implicitly by the all-zero default [`ExcitationBuffer`]
//!   from r234 — every historical lookup returns `0`, yielding an
//!   all-zero contribution and the documented "no spurious transient"
//!   envelope at stream start. The downstream `e[n] = p[n] + c[n]`
//!   composition + the Q-format choice for `p[n] = ea[n]` stay
//!   deferred behind the documented pitch-gain Q-format gap (see the
//!   [`pitch_gain`] module docs).
//!
//! * **Round r234** — narrowband **adaptive-codebook
//!   (long-term predictor) index resolution + excitation history
//!   buffer**, Q-format-agnostic by design. Spec basis: Speex Codec Manual §9.2 Eq. 9.1
//!   `ea[n] = g0·e[n−T−1] + g1·e[n−T] + g2·e[n−T+1]` plus the
//!   explicit excitation-repeat rule for short pitches:
//!   *"when the pitch is smaller than the sub-frame size, we repeat
//!   the excitation at a period T. For example, when `n − T + 1 ≥ 0`,
//!   we use `n − 2T + 1` instead."* The new [`adaptive_codebook`]
//!   module exposes [`resolve_lookback`] (the typed repeat-rule
//!   helper), [`sample_lookback_indices`] (per-sample three-tap
//!   offsets), [`subframe_lookback_indices`] (a
//!   `[[i32; 3]; 40]` matrix per pitch period), plus the typed
//!   [`ExcitationBuffer`] holding the
//!   [`EXCITATION_HISTORY_LEN`] = 145 past samples the deepest
//!   conformant tap (`e[n − T − 1]` at `n = 0`, `T = 144`) reads.
//!   The module is **Q-format-agnostic** by design — the gain
//!   multiplication `g·e[k]` is deferred until the documented β
//!   Q-format pin (see [`crate::pitch_gain`] module docs for the
//!   recorded gap), so the index resolution + buffer state machine
//!   land independently and the eventual long-term-predictor sum
//!   can pin its scaling in a single follow-up round.
//!
//! * **Round r244** — narrowband **raw excitation composition primitive**
//!   composing r241 + r220 into the per-sub-frame `[i32; 40]`
//!   raw-integer evaluation of Speex Codec Manual §8.4 / CELP companion
//!   §2.3 `e[n] = p[n] + c[n]` where `p[n] = ea[n]` is the r241
//!   adaptive-codebook contribution and `c[n]` is the r220 innovation
//!   sub-vector. The new [`excitation`] module exposes
//!   [`raw_excitation_subframe`] (whole sub-frame batch) and
//!   [`raw_excitation_sample`] (per-sample helper) returning the raw
//!   widening sum `ea[n] + i32::from(c[n])` — Q-format-agnostic by
//!   design, matching the r234 / r241 convention. The fixed-codebook
//!   gain composition + the saturating `i32 → i16` buffer-push step +
//!   the final Q-format pin stay deferred behind the documented
//!   pitch-gain Q-format gap (see the [`adaptive_contribution`] module
//!   docs) and the CELP companion §9 fixed-codebook gain scalar
//!   quantiser gap.
//!
//! * **Round r261** (this commit) — narrowband **fixed-codebook gain
//!   index composition primitive** surfacing the Speex Codec Manual
//!   §9.2 / CELP companion §2.3 product structure
//!   `fixed-codebook gain = g_frame × g_subf` at the typed-index layer.
//!   The new [`fixed_codebook_gain`] module exposes
//!   [`FrameInnovationGainIndex`] (typed wrapper over the 5-bit
//!   frame-level OL excitation-gain field with a [`FrameInnovationGainIndex::Silence`]
//!   variant for mode 0),
//!   [`SubFrameInnovationGainCorrection`] (typed wrapper over the
//!   0 / 1 / 3-bit per-sub-frame correction field with an
//!   [`SubFrameInnovationGainCorrection::Absent`] variant for the
//!   0-bit-budget modes 0, 2, 8), and the composed
//!   [`FixedCodebookGainIndices`] pair. A [`fixed_codebook_gain_indices`]
//!   helper produces `[FixedCodebookGainIndices; 4]` per frame; a
//!   [`NarrowbandFrameBody::fixed_codebook_gain_indices`] convenience
//!   method wires it off the existing parsed body. The numeric
//!   `g_frame × g_subf` reconstruction is gap-blocked behind the CELP
//!   companion §9 "computed, not a lookup array" open-loop scalar
//!   quantiser note; this primitive surfaces the algebra of the
//!   composition at the index layer, leaving the Q-format / quantiser
//!   pin for a future docs round.
//!
//! * **Round r269** (this commit) — wideband **high-band
//!   fixed-codebook gain index primitive**, the high-band counterpart
//!   of r261's narrowband composition. Per Speex Codec Manual §10.4 /
//!   Table 10.1 the high band carries exactly one gain field per
//!   sub-frame (the `Excitation gain` row: 0 / 5 / 4 / 4 / 4 bits for
//!   modes 0..=4) and **no frame-level factor**, so the §9.2
//!   `g_frame × g_subf` product structure reduces to a single typed
//!   index. The new [`hb_excitation_gain`] module exposes
//!   [`HbExcitationGainIndex`] (`Absent` for mode 0 / `FiveBit` for
//!   mode 1 / `FourBit` for modes 2..=4), the
//!   [`hb_excitation_gain_indices`] batch helper, and the
//!   [`WidebandHighBandBody::hb_excitation_gain_indices`] convenience
//!   method. The numeric gain magnitude stays gap-blocked behind the
//!   documented "computed, not a lookup array" open-loop scalar
//!   quantiser note (`tables/README.md` "Not extracted").
//!
//! * **Round r277** (this commit) — narrowband **per-mode innovation
//!   codebook binding for modes 2 / 3 / 4 / 5**. The staged
//!   per-codebook innovation bit-rate annotations
//!   (`docs/audio/speex/tables/innovation-cdbk-*.meta` `role:`
//!   fields plus the staged `tables/README.md` inventory) combined
//!   with the Table 9.1 "Innovation VQ" row (`bits/sub-frame × 200
//!   = innovation bps`) uniquely pin mode 2 → 4 × `Sv10_16`, mode
//!   3 → 4 × `Sv10_32`, mode 4 → 5 × `Sv8_128`, mode 5 → 8 ×
//!   `Sv5_64`.
//!   [`InnovationMapping::for_mode`] now surfaces `Documented` for
//!   modes 2 / 3 / 4 / 5 / 6 / 8; `Undocumented` narrows to mode 1
//!   (0-bit VQ field, vocoder excitation generation unstaged) and
//!   mode 7 (96 bits → 19 200 bps, no annotated codebook). The real
//!   mode-5 fixture now decodes every sub-frame's 40-sample `c[n]`
//!   innovation vector end-to-end.
//!
//! * **Round r286** (this commit) — narrowband **LSP→LPC conversion +
//!   the LPC synthesis filter**, closing the decoder README's
//!   "lacks LSP→LPC + synthesis filter" tail. The new [`lsp_to_lpc`]
//!   module turns a set of LSP angular frequencies into the ten
//!   linear-prediction coefficients of the analysis filter
//!   `A(z) = 1 − Σ a[i]·z⁻¹⁻ⁱ` via the standard auxiliary-polynomial
//!   identity `A(z) = (P(z) + Q(z)) / 2` (P/Q built by convolving the
//!   per-LSP second-order sections `[1, −2cos(ωₖ), 1]` and applying
//!   the `(1 ± z⁻¹)` boundary factors). Spec basis: *The Speex Codec
//!   Manual* §9.1 ("converted back to the LPC filter Â(z)") + §9.4
//!   ("the synthesis filter S(z) = 1/A(z)"). [`lsp_to_lpc`] is the
//!   general float transform; [`lsp_q10_to_radians`] /
//!   [`lsp_vector_q10_to_radians`] / [`lpc_from_lsp_q10`] bridge the
//!   r194/r200 Q10 reconstruction under a documented angular-unit
//!   assumption (the LSP Q-format pin for *bit-exactness* against
//!   reference output is a recorded docs gap). The new [`synthesis`]
//!   module's [`SynthesisFilter`] runs the decoder recurrence
//!   `x[n] = e[n] + Σ a[i]·x[n−1−i]` (manual §8.2 prediction-error
//!   inverse) with persistent IIR history across sub-frames and
//!   frames, emitting `f64` or rounded/saturated `i16` PCM. A new
//!   integration test (`tests/synthesis_pcm_fixture.rs`) runs the
//!   real mode-5 fixture end-to-end (LSP → interp → LSP→LPC →
//!   innovation `c[n]` → synthesis) to produce finite, responsive,
//!   non-silent PCM. The adaptive-codebook (pitch) contribution is
//!   not yet folded into the excitation feeding the filter — its gain
//!   Q-format stays gap-blocked — so the PCM is correct-by-construction
//!   but not yet bit-exact.
//!
//! * **Round r296** (this commit) — narrowband **per-sub-frame LSP→LPC
//!   conversion for a full frame**, bridging the r200 sub-frame LSP
//!   interpolation (Q12 [`crate::lsp_interp::NbSubFrameLsp`]) and the
//!   r286 LSP→LPC core. The r286 `lpc_from_lsp_q10` path only consumed
//!   the per-frame Q10 vector; the interpolated per-sub-frame vectors
//!   carry two extra sub-binary-point bits (Q12), so a Q-shift-aware
//!   conversion was missing. This round generalises
//!   [`lsp_q10_to_radians`] into the Q-shift-parameterised
//!   [`lsp_qn_to_radians`] (the Q10 helper now delegates to it), adds
//!   [`lpc_from_subframe_lsp_q12`] for one Q12 sub-frame vector, and
//!   [`subframe_lpc_set`] returning the four per-sub-frame LPC sets the
//!   [`synthesis`] filter consumes (manual §9.1: "each sub-frame is
//!   filtered with its own interpolated LPC set"; the 4th sub-frame
//!   carries the current LSPs unchanged). The angular-unit assumption
//!   stays pinned in the single shared [`lsp_qn_to_radians`] helper, so
//!   the recorded LSP Q-format docs gap still has one fix-site.
//!
//! * **Round r321** (this commit) — **exact scalar gain reconstruction
//!   tables**, wiring the previously index-only excitation-gain fields
//!   (r261 narrowband frame-level OL excitation
//!   gain + r269 high-band per-sub-frame excitation gain) into
//!   reconstructed scalar magnitudes. Spec basis: the **exact**
//!   reconstruction tables staged under `docs/audio/speex/tables/`
//!   (`provenance/02-speex-gain-quant.md`) — `ol_gain_table` (32-level
//!   5-bit NB OL excitation gain, float law `exp(qe/3.5)`),
//!   `exc_gain_quant_scal3` / `_scal1` (8-/2-level NB per-sub-frame
//!   innovation-gain correction `g_subf`), `gc_quant_bound` (16-level
//!   4-bit HB gain-correction, reconstruction `0.87360·bound`), and
//!   `fold_quant_bound` (32-level 5-bit HB folded gain). The
//!   [`gain_reconstruction`] module embeds these vendored tables and
//!   exposes the raw level accessors plus the dispatch helpers
//!   [`reconstruct_frame_ol_exc_gain`] (silence → `0.0`),
//!   [`reconstruct_subframe_gain_correction`] (absent → identity
//!   `1.0`), [`reconstruct_fixed_codebook_gain`] (the composed
//!   `g_frame · g_subf` fixed-codebook gain), and
//!   [`reconstruct_hb_exc_gain`] (absent → `0.0`). This **replaces** the
//!   earlier parametric `10^((index − offset)/slope)` placeholder grid:
//!   the gain magnitudes are now exact reconstruction-level lookups, so
//!   the codec author's quantiser points are pinned (no remaining
//!   behavioural-trace gap for these tables). Excitation scaling
//!   (multiplying the §8.4 innovation by the gain) remains a downstream
//!   synthesis layer.
//!
//! Frame decode, encoder, and the `Decoder` / `Encoder` trait wiring
//! against `oxideav-core` still return [`Error::NotImplemented`]; the
//! r191 codebook accessors + r194 narrowband LSP reconstruction +
//! r200 sub-frame LSP interpolation + r208 pitch-gain VQ
//! reconstruction + r214 high-band LSP MSVQ reconstruction +
//! r220 narrowband innovation sub-vector dispatch + r230 high-band
//! innovation sub-vector dispatch + r234 adaptive-codebook index
//! resolution + excitation history buffer + r241 adaptive-codebook
//! contribution sum + r244 raw excitation composition primitive +
//! r261 fixed-codebook gain index composition primitive + r269
//! high-band excitation-gain index primitive + r277 per-mode
//! innovation binding for modes 2 / 3 / 4 / 5 + r286 LSP→LPC
//! conversion + LPC synthesis filter now compose into a real PCM
//! path for one narrowband fixture, though the public
//! `Decoder`/`Encoder` trait surface is not yet wired.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

mod abs_search;
mod adaptive_codebook;
mod adaptive_contribution;
mod bitreader;
mod codebooks;
mod decoder;
mod encoder_nb;
mod encoder_wb;
mod excitation;
mod fixed_codebook_gain;
mod forced_pitch_gain;
mod frame;
mod gain_reconstruction;
mod gain_scaled_excitation;
mod gain_scaled_hb_innovation;
mod gain_scaled_innovation;
mod gain_scaled_pitch;
mod hb_encode;
mod hb_excitation_gain;
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
mod packet;
mod pitch_gain;
mod qmf;
mod quality;
mod signalling;
mod submode;
mod synthesis;
mod wb_synthesis;
mod weighting;
mod wideband;
mod wideband_decoder;

pub use abs_search::{
    closed_loop_pitch_search, weighted_synthesis_zero_state, ClosedLoopPitch, WeightFactors,
};
pub use adaptive_codebook::{
    resolve_lookback, sample_lookback_indices, subframe_lookback_indices, ExcitationBuffer,
    ExcitationError, ADAPTIVE_CODEBOOK_TAPS, EXCITATION_HISTORY_LEN, TAP_PITCH_OFFSETS,
};
pub use adaptive_contribution::{
    adaptive_contribution_sample, adaptive_contribution_subframe, AdaptiveContributionError,
};
pub use bitreader::{BitError, BitReader, BitWriter};
pub use codebooks::{
    hb_innovation_10_32, hb_innovation_8_128, hb_lsp_scale, hb_lsp_stage1, hb_lsp_stage2,
    innovation_10_16, innovation_10_32, innovation_20_32, innovation_5_256, innovation_5_64,
    innovation_8_128, lpc_analysis_window_float, lpc_analysis_window_q15, lpc_lag_window_float,
    lpc_lag_window_q15, nb_lsp_high1, nb_lsp_high2, nb_lsp_low1, nb_lsp_low2, nb_lsp_scale,
    nb_lsp_stage0, pitch_gain_5bit, pitch_gain_7bit, qmf_h0_float, qmf_h0_q15, InnovationShape,
    NbLspScale, HB_LPC_ORDER, HB_LSP_STAGE_ENTRIES, LPC_ANALYSIS_WINDOW_LEN, LPC_LAG_WINDOW_LEN,
    NB_LSP_ORDER, NB_LSP_SPLIT_HALF, NB_LSP_STAGE_ENTRIES, PITCH_GAIN_5BIT_ENTRIES,
    PITCH_GAIN_7BIT_ENTRIES, PITCH_GAIN_BIAS, PITCH_GAIN_COLS, QMF_FILTER_LEN,
};
pub use decoder::{ControlMessage, DecodeError, DecodedFrame, SpeexDecoder};
pub use encoder_nb::{EncodeError, NarrowbandEncoder, NB_FRAME_SAMPLES};
pub use encoder_wb::{WbEncodeError, WidebandEncoder, WidebandFrameBodies};
pub use excitation::{raw_excitation_sample, raw_excitation_subframe, RAW_EXCITATION_SAMPLES};
pub use fixed_codebook_gain::{
    fixed_codebook_gain_indices, FixedCodebookGainIndices, FrameInnovationGainIndex,
    SubFrameInnovationGainCorrection, FRAME_OL_EXC_GAIN_BITS, FRAME_OL_EXC_GAIN_ENTRIES,
};
pub use forced_pitch_gain::{
    forced_pitch_coef, forced_pitch_gain_taps, FORCED_PITCH_GAIN_BITS, FORCED_PITCH_GAIN_LEVELS,
    FORCED_PITCH_GAIN_STEP,
};
pub use frame::{FrameError, NarrowbandFrameHeader, NARROWBAND_FRAME_PREFIX_BITS};
pub use gain_reconstruction::{
    hb_folded_gain_levels, hb_gain_correction_levels, hb_gc_quant_bound, nb_ol_exc_gain_levels,
    nb_subframe_gain_bound_1bit, nb_subframe_gain_bounds_3bit, nb_subframe_gain_levels_1bit,
    nb_subframe_gain_levels_3bit, quantise_frame_ol_exc_gain, quantise_hb_exc_gain,
    quantise_subframe_gain_correction, reconstruct_fixed_codebook_gain,
    reconstruct_frame_ol_exc_gain, reconstruct_hb_exc_gain, reconstruct_subframe_gain_correction,
    HB_FOLDED_GAIN_LEVELS, HB_GAIN_CORRECTION_LEVELS, NB_OL_EXC_GAIN_LEVELS,
    NB_SUBFRAME_GAIN_LEVELS_1BIT, NB_SUBFRAME_GAIN_LEVELS_3BIT,
};
pub use gain_scaled_excitation::{
    gain_scaled_excitation_sample, gain_scaled_excitation_subframe, GAIN_SCALED_EXCITATION_SAMPLES,
};
pub use gain_scaled_hb_innovation::{
    gain_scaled_hb_innovation_from_body, gain_scaled_hb_innovation_sample,
    gain_scaled_hb_innovation_subframe, GAIN_SCALED_HB_INNOVATION_SAMPLES,
};
pub use gain_scaled_innovation::{
    gain_scaled_innovation_from_indices, gain_scaled_innovation_sample,
    gain_scaled_innovation_subframe, GAIN_SCALED_INNOVATION_SAMPLES,
};
pub use gain_scaled_pitch::{
    gain_scaled_pitch_sample, gain_scaled_pitch_subframe, GAIN_SCALED_PITCH_SAMPLES,
    PITCH_GAIN_SCALING,
};
pub use hb_encode::{encode_wideband_frame, write_high_band_body, write_high_band_frame};
pub use hb_excitation_gain::{
    hb_excitation_gain_indices, HbExcitationGainIndex, HB_EXC_GAIN_BITS_MODES_2_TO_4,
    HB_EXC_GAIN_BITS_MODE_1,
};
pub use hb_innovation::{
    decode_hb_subframe, hb_innovation_sub_vector, HbInnovationCodebook, HbInnovationError,
    HbInnovationMapping, HB_SUBFRAME_SAMPLES,
};
pub use hb_innovation_search::{search_hb_innovation, HbInnovationChoice};
pub use hb_lsp::{
    pack_hb_lsp_index, quantise_q10 as quantise_hb_lsp_q10,
    reconstruct_q10 as reconstruct_hb_lsp_q10, HbLspStages, HB_LSP_INDEX_MASK, HB_LSP_OUTPUT_Q,
    HB_LSP_PACKED_BITS, HB_LSP_STAGE_BITS,
};
pub use hb_lsp_interp::{HbSubFrameLsp, HB_LSP_INTERP_OUTPUT_Q, HB_LSP_SUBFRAMES_PER_FRAME};
pub use hb_synthesis::HbSynthesisFilter;
pub use header::{
    HeaderError, SpeexHeader, SPEEX_HEADER_LEN, SPEEX_MAGIC, SPEEX_MODE_NARROWBAND,
    SPEEX_MODE_ULTRAWIDEBAND, SPEEX_MODE_WIDEBAND, SPEEX_STRING_LEN, SPEEX_VERSION_LEN,
};
pub use innovation::{
    decode_subframe as decode_innovation_subframe, sub_vector as innovation_sub_vector,
    InnovationCodebook, InnovationError, InnovationMapping, SUBFRAME_SAMPLES,
};
pub use innovation_search::{search_innovation, InnovationChoice};
pub use lpc_analysis::{
    analyse as lpc_analyse, analyse_hb as hb_lpc_analyse, apply_analysis_window, autocorrelate,
    levinson_durbin, stabilise_autocorrelation, HbLpcCoefficients, LpcAnalysisError,
    LpcCoefficients, AUTOCORR_LAGS, HB_AUTOCORR_LAGS, LPC_NOISE_FLOOR,
};
pub use lpc_to_lsp::{hb_lpc_to_lsp, lpc_to_lsp, LpcToLspError, LSP_BISECT_ITERS, LSP_SCAN_STEPS};
pub use lsp::{
    reconstruct_q10 as reconstruct_nb_lsp_q10, NbLspStages, NB_LSP_INDEX_MASK, NB_LSP_OUTPUT_Q,
    NB_LSP_STAGES_18BIT, NB_LSP_STAGES_30BIT, NB_LSP_STAGE_BITS,
};
pub use lsp_base::{
    add_hb_lsp_base, add_nb_lsp_base, enforce_lsp_margin_radians, hb_lsp_base_q10,
    hb_lsp_base_radians, hb_lsp_margin_radians, nb_lsp_base_q10, nb_lsp_base_radians,
    nb_lsp_margin_radians, HB_LSP_BASE_INTERCEPT_Q10, HB_LSP_BASE_SLOPE_Q10, HB_LSP_MARGIN_Q10,
    HB_LSP_MARGIN_RADIANS, NB_LSP_BASE_INTERCEPT_Q10, NB_LSP_BASE_SLOPE_Q10, NB_LSP_MARGIN_Q10,
    NB_LSP_MARGIN_RADIANS,
};
pub use lsp_interp::{NbSubFrameLsp, NB_LSP_INTERP_OUTPUT_Q, NB_LSP_SUBFRAMES_PER_FRAME};
pub use lsp_pi_domain::{
    hb_lsp_linear_storage, lsp_storage_to_q10, lsp_storage_to_radians, nb_lsp_linear_storage,
    radians_to_lsp_storage, LSP_PI,
};
pub use lsp_quant::{pack_lsp_index, quantise_lsp_q10};
pub use lsp_to_lpc::{
    hb_lsp_q10_to_radians, hb_lsp_to_lpc, hb_lsp_with_base_q10, hb_subframe_lpc_set_with_base,
    lpc_from_hb_lsp_delta_q10, lpc_from_hb_lsp_q10, lpc_from_hb_subframe_lsp_q12,
    lpc_from_lsp_delta_q10, lpc_from_lsp_q10, lpc_from_subframe_lsp_q12, lsp_q10_to_radians,
    lsp_qn_to_radians, lsp_to_lpc, lsp_vector_q10_to_radians, lsp_vector_radians_to_q10,
    nb_lsp_with_base_q10, radians_to_lsp_q10, subframe_lpc_set, subframe_lpc_set_with_base,
    HB_LPC_ORDER_OUT, LPC_ORDER,
};
pub use narrowband_body::{
    NarrowbandBodyError, NarrowbandFrameBody, NarrowbandSubFrameIndices, PITCH_PERIOD_MAX,
    PITCH_PERIOD_MIN,
};
pub use narrowband_decoder::{
    saturate_i16, NarrowbandDecodeError, NarrowbandDecoder, NARROWBAND_FRAME_SAMPLES,
};
pub use nb_encode::{encode_narrowband_frame, write_narrowband_body, write_packet_terminator};
pub use ol_pitch::{estimate_open_loop_pitch, estimate_open_loop_pitch_range, OpenLoopPitch};
pub use packet::{parse_packet, FrameKind, PacketError, PacketFrame, PacketFrames, PacketSummary};
pub use pitch_gain::{reconstruct as reconstruct_pitch_gain, PitchGainTaps, PITCH_GAIN_TAPS};
pub use qmf::{QmfAnalysis, QmfSynthesis, QMF_HALF_BAND_FRAME, QMF_WIDEBAND_FRAME};
pub use quality::{
    nb_mode_for_quality, uwb_bitrate_bps, uwb_modes_for_quality, wb_bitrate_bps,
    wb_modes_for_quality, UwbQualityModes, WbQualityModes, FRAMES_PER_SECOND, MAX_QUALITY,
    UWB_HIGH_BAND_MODE,
};
pub use signalling::{
    inband_code_spec, AcknowledgePolicy, CustomInbandMessage, InbandCodeSpec, InbandKind,
    InbandMessage, InbandRequest, RateModeConfig, SignallingError, CUSTOM_INBAND_MAX_BYTES,
    CUSTOM_INBAND_SIZE_BITS, INBAND_CODE_BITS, INBAND_TABLE_5_1,
};
pub use submode::{LspQuant, NarrowbandSubmode, PitchGainQuant, Submode, NARROWBAND_SUBMODES};
pub use synthesis::SynthesisFilter;
pub use wb_synthesis::{
    synthesise_high_band_frame, synthesise_high_band_frame_interp, SubBandLayer, UwbFrameLayout,
    HB_FRAME_SAMPLES, HB_SUBFRAMES_PER_FRAME,
};
pub use weighting::{weighted_coeffs, PerceptualWeighting, WEIGHT_GAMMA1, WEIGHT_GAMMA2};
pub use wideband::{
    HighBandSubFrameIndices, WidebandBodyError, WidebandHighBandBody, WidebandHighBandFrameHeader,
    WidebandHighBandSubmode, WidebandSubmode, HIGH_BAND_FRAME_PREFIX_BITS,
    HIGH_BAND_SUBFRAMES_PER_FRAME, WIDEBAND_HIGH_BAND_SUBMODES,
};
pub use wideband_decoder::{WidebandDecodeError, WidebandDecoder, WidebandFrame};

/// Crate-local error type. Until the full clean-room rebuild lands, the
/// codec-level public API paths return [`Error::NotImplemented`]; the
/// Ogg/Speex header parser surfaces its own [`HeaderError`] and the
/// per-frame parser surfaces [`FrameError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The crate's frame decoder / encoder has not been re-implemented
    /// against the spec yet.
    NotImplemented,
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
            Error::NotImplemented => write!(
                f,
                "oxideav-speex: clean-room rebuild in progress — frame decoder/encoder not yet wired up"
            ),
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
            Error::NotImplemented => None,
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

/// No-op codec registration — the rebuild has no encoder/decoder to
/// hand to the runtime context yet. Header parsing is exposed as a free
/// function via [`SpeexHeader::parse`].
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("speex", register);
