#!/bin/bash

# Debug tool to convert raw microphone recording files to
# WAV files.

set -e

[ $# = 2 ] || { echo "Usage: $0 <in.raw> <out.wav>"; exit 1; }

sox --no-show-progress  -b 32 -c 8 -e signed-integer -r 24000 -t raw "${1}" -b 32 -c 8 -e signed-integer -r 24000 -t wav  "${2}"


