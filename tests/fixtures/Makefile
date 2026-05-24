# Black-box fixture-regeneration commands.
#
# The .spx fixtures committed here are checked in directly so the
# crate's CI does not need speexenc on the runner; this Makefile is the
# audit trail for how to reproduce them. Per the workspace clean-room
# policy, speexenc is invoked as an opaque binary — its source is NOT
# consulted.
#
# Run:
#   make -C crates/oxideav-speex/tests/fixtures regen
#
# Requires:
#   - speexenc (Speex reference command-line encoder, version >= 1.2)
#   - python3 (for the synthetic PCM generator)

PYTHON   ?= python3
SPEEXENC ?= speexenc

.PHONY: regen clean

regen: nb_440hz_q8.spx

# Synthesise 1 s of 8 kHz mono 440 Hz tone as 16-bit signed LE PCM,
# then encode as narrowband Speex at the default quality (8). Quality 8
# resolves to narrowband sub-mode 5 (15 kbps / 300 bits per 20 ms
# frame, per *Speex Codec Manual* Tables 9.1 + 9.2).
nb_440hz_q8.tmp.raw:
	$(PYTHON) -c "import math, struct; \
	  sr=8000; n=sr; samples=[int(8000*math.sin(2*math.pi*440*i/sr)) for i in range(n)]; \
	  open('$@','wb').write(struct.pack('<%dh'%n, *samples))"

nb_440hz_q8.spx: nb_440hz_q8.tmp.raw
	$(SPEEXENC) --narrowband --rate 8000 --le --16bit --quality 8 $< $@
	rm -f $<

clean:
	rm -f nb_440hz_q8.spx nb_440hz_q8.tmp.raw
