#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2022-2023 Dimitar Dimitrov <dimitar@dinux.eu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Test the DOA estimation model by running it against file datasets.
#
# Example invocations:
#    $ ./test-model.py -i data.hdf5 -m model.hdf5

import numpy as np
import argparse
import random
import glob
import json
import sys
import os

import tensorflow as tf
from tensorflow import keras

# The datasets audio parameters.
NCHANNELS = 8;
BITS_PER_SAMPLE = 32;
SAMPLES_PER_SECOND = 24000;

# Neural Network's input parameters.
DATASET_NSAMPLES = 512;

BATCH_SIZE = 32;

def run_sample(model, a, idstr, labels):
    ds = tf.data.Dataset.from_tensor_slices(([a], [idstr])).batch(BATCH_SIZE)

    predict = model.predict(ds, verbose=0)
    angle_id_predict = np.argmax(predict)
    print('Expected: ' + idstr + ', got: ' + labels[angle_id_predict])

def path_to_audio(path):
    audio = np.fromfile(path, dtype=np.dtype('<i4'))
    return audio

# Load the mapping of NN output class IDs to their human-readable strings.
def load_class_names(input_filename):
    with open(input_filename, 'r') as f:
        json_str = f.read()
    return json.loads(json_str)['class_names']

def main():
    parser = argparse.ArgumentParser(description='Test the DOA estimation model.')
    parser.add_argument('-i', '--input', required=True,
        help = 'Directory with test vectors of audio chunks')
    parser.add_argument('-m', '--model', required=True,
        help = 'NN model file to use')
    parser.add_argument('-n', '--niterations', required=False,
        default=10, type=int,
        help = 'How much test iterations to do')
    args = parser.parse_args()

    dir_class_names = os.listdir(args.input)
    dataset_paths = []
    dataset_classes = []

    # Enumerate the available datasets.
    for name in dir_class_names:
        print("Processing dataset {}".format(name,))
        dirpath = os.path.join(args.input, name)
        fpaths = glob.glob(dirpath + '/**/*raw_*', recursive=True)
        dataset_paths += fpaths
        dataset_classes += [name] * len(fpaths)
    print("Found {} files.".format(len(dataset_paths), ))

    model = keras.models.load_model(args.model)
    class_names = load_class_names(os.path.splitext(args.model)[0] + '.json')

    for testi in range(0, args.niterations):
        rnd_i = random.randint(0, len(dataset_paths)-1)
        path = dataset_paths[rnd_i]
        a = path_to_audio(path)

        run_sample(model, a, dataset_classes[rnd_i], class_names)

    sys.exit(0)

main()
