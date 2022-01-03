#!/bin/bash

# Alternative:
#  cat $1.raw | sox --no-show-progress  -b 32 -c 8 -e signed-integer -r 24000 -t raw - -b 32 -c 8 -e signed-integer -r 24000 -t wav -  | aplay -V mono -

ffplay  -f s24le -ar 24k -ac 8 ${1}


