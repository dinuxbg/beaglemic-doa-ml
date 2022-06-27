#!/bin/bash

# Generate raw PCM for playback. Start with short 1khz
# pulse to help with playback-recording delay estimation later.

NCHANNELS=1

# Output format.
FF_FMT="-f s16le -acodec pcm_s16le -ac ${NCHANNELS} -ar 16000"

set -e

die()
{
  echo "ERROR: $@"
  exit 1
}

[ $# = 0 ] || die "Usage: find  -name '*.flac' | sort | $0 > data.raw"

# Start 2s silence
dd if=/dev/zero bs=1 count=$((16000 * 2 * 2 * NCHANNELS))

# Play a 1s marker. Later it will help to calculate the delay
# between playback and recorded audio.
ffmpeg -loglevel error -f lavfi -i "sine=frequency=1000:duration=1" ${FF_FMT} - </dev/null

# 1s silence
dd if=/dev/zero bs=1 count=$((16000 * 2 * 1 * NCHANNELS))

# Now play the given sampls.
while read INFILE
do
  # Don't let ffmpeg corrupt stdin stream, since that's where we keep the list
  # of input files.
  ffmpeg -loglevel error -i "${INFILE}" ${FF_FMT}  - </dev/null
done
