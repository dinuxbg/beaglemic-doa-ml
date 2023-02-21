#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2022-2023 Dimitar Dimitrov <dimitar@dinux.eu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Test the DOA estimation model by running it against file datasets.
#
# Example invocations:
#    $ ./test-model.py -i dataset -m model.h5

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

# Threshold for considering a resulting angle
# as a "loose" (i.e. not exact) match. In degrees.
LOOSE_MATCH_DEGS = 15;

def run_sample(model, a, idstr, labels):
    ds = tf.data.Dataset.from_tensor_slices(([a], [idstr])).batch(1)

    predict = model.predict(ds, verbose=0)
    angle_id_predict = np.argmax(predict)
    print('Expected: ' + idstr + ', got: ' + labels[angle_id_predict])

    s_expected = idstr
    s_got = labels[angle_id_predict]
    if s_expected == 'silence' and s_got != 'silence':
        return False, False
    if s_expected != 'silence' and s_got == 'silence':
        return False, False
    if s_expected == 'silence' and s_got == 'silence':
        return True, True
    f_expected = float(s_expected)
    f_got = float(s_got)
    return idstr == labels[angle_id_predict], abs(f_got - f_expected) < LOOSE_MATCH_DEGS 

def path_to_audio(path):
    audio = np.fromfile(path, dtype=np.dtype('<i4'))
    audio = np.divide(audio, 2**31)
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

    n_total = 0
    n_exact = 0
    n_loose = 0
    for testi in range(0, args.niterations):
        rnd_i = random.randint(0, len(dataset_paths)-1)
        path = dataset_paths[rnd_i]
        a = path_to_audio(path)

        (exact, loose) = run_sample(model, a, dataset_classes[rnd_i], class_names)
        if exact:
            n_exact += 1
        if loose:
            n_loose += 1
        n_total += 1

    print("Total samples: {}, exact match {}%, loose match {}%".format(n_total, (n_exact * 100) // n_total, (n_loose * 100) // n_total))

    sys.exit(0)

main()
