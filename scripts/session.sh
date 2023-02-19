#!/bin/bash

# Play a given raw PCM audio. Simultaneously record from BeagleMic.
# This is used to record data for training the DOA NN model.

# Output to my cheap USB Audio dongle. Comment out to use the default for your host.
OUT_DEV="-D hw:CARD=Device"

# Define how to play the recorded audio. See below how these two variables are used.
PLAY_CONVERT="sox --no-show-progress  -b 16 -c 1 -e signed-integer -r 16000 -t raw - -b 16 -c 2 -e signed-integer -r 44100 -t raw  -"
PLAY="aplay --quiet ${OUT_DEV} -t raw -c2 -f S16_LE -r44100"

# Alternative, if the above does not work.
## PLAY_CONVERT=cat
## NCHANNELS=1
## PLAY="aplay --quiet ${OUT_DEV} -t raw -c${NCHANNELS} -f S16_LE -r16000"

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
cat "${F_PLAY}" | ${PLAY_CONVERT} | ${PLAY}

# Stop recording
kill -s SIGINT ${REC_PID}
