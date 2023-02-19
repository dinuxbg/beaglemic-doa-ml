#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2022-2023 Dimitar Dimitrov <dimitar@dinux.eu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Plot the DOA estimation model.
#
# Example invocations:
#    $ ./plot-model.py -m model.hdf5 -o model.png

import argparse
import sys

import tensorflow as tf
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser(description='Plot the DOA estimation model.')
    parser.add_argument('-m', '--model', required=True,
        help = 'NN model file to use')
    parser.add_argument('-o', '--output', required=True,
        help = 'Output PNG file')
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    keras.utils.plot_model(model, to_file=args.output, show_shapes=True)
    sys.exit(0)

main()
