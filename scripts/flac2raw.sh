#!/bin/bash

# Convert all FLAC files to 16000Hz 16bps raw audio files.
# Perhaps in the future the main stand application should
# do the conversion itself.

set -e

die()
{
  echo "ERROR: $@"
  exit 1
}

find . -name '*.flac' | while read FLACF
do
  O=`basename "${FLACF}" .flac`.raw
  [ -f "${O}" ] && die "duplicate filename detected"
  # Don't let ffmpeg corrupt stdin stream, since that's where we keep the list
  # of input files.
  ffmpeg -loglevel error -i "${FLACF}" -f s16le -acodec pcm_s16le -ac 1 -ar 16000  "${O}" </dev/null
done

