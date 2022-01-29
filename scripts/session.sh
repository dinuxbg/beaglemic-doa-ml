#!/bin/bash

# Play a given raw PCM audio.
# Simultaneously record from BeagleMic.

#PLAY="ffplay  -f s16le -ar 16k -ac 1 -i -"

# OUT_DEV="-D hw:CARD=UAC2Gadget"
PLAY="aplay --quiet ${OUT_DEV} -t raw -c1 -f S16_LE -r16000"

set -e

die()
{
  echo "ERROR: $@"
  exit 1
}

[ $# = 2 ] || die "Usage: $0 <input-toplay.raw> <output-recorded.raw>"

F_PLAY="${1}"
shift
F_REC="${1}"
shift

# Start recording
arecord --quiet -D hw:CARD=BeagleMic -c8 -t raw -f S32_LE -r24000 "${F_REC}" &
REC_PID=$!

# Play the prepared data.
cat "${F_PLAY}" | ${PLAY}

# Stop recording
kill -s SIGINT ${REC_PID}
