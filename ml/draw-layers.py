#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2022-2023 Dimitar Dimitrov <dimitar@dinux.eu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Draw the kernel weights as grayscale bitmap file.
#
# Example invocation:
#    $ ./test-model.py -o output-dir -m model.h5

import numpy as np
import argparse
import random
import glob
import json
import sys
import os

import tensorflow as tf
from tensorflow import keras

NCHANNELS = 8;

def normalize_to_16bit(tensor):
    a = np.asarray(tensor)
    return ((a - a.min()) / (a.max() - a.min())) * 65535 

def normalize_relu_to_16bit(tensor):
    a = np.asarray(tensor)
    return (np.maximum(0, a) / a.max()) * 65535

def save_pgm(name, a):
    xres,yres = a.shape
    with open(name + '.pgm', "wb") as f:
        f.write(bytes('P5\n' + str(xres) + ' ' + str(yres) + '\n65535\n', 'ascii'))
        for y in range(0, yres):
            for x in range(0, xres):
                val = int(a[x, y])
                f.write(val.to_bytes(2, 'big'))

def dump_output(name, kernel):
    kernel = normalize_to_16bit(kernel)
    kernel = np.reshape(kernel, (len(kernel)//64, 64))
    save_pgm(name, kernel)

def dump_dense(name, layer):
    print('Dumping ' + layer.name)
    kernel = normalize_relu_to_16bit(layer.kernel)

    save_pgm(name + '-linear', kernel)

    DATASET_NSAMPLES = kernel.shape[0] // NCHANNELS

    # [64x64 for sample 0, ch0] [64x64 for sample 1, ch0] ... [64x64 for sample 511, ch0]
    # [64x64 for sample 0, ch1] [64x64 for sample 1, ch1] ... [64x64 for sample 511, ch1]
    a = np.empty((NCHANNELS * 64, DATASET_NSAMPLES * kernel.shape[0] // 64))
    for ch in range(0, NCHANNELS):
        for sample in range(0, DATASET_NSAMPLES):
            for x in range(0, 64):
                for y in range(0, kernel.shape[0] // 64):
                    ax = 64 * sample + x
                    ay = ch * kernel.shape[0] // 64 + y
                    a[ay][ax] = kernel[ch * DATASET_NSAMPLES + sample][x + y * kernel.shape[0] // 64]
    save_pgm(name + '-sample-outputs', kernel)

    # ---------------------
    a = np.empty((NCHANNELS * 64, DATASET_NSAMPLES * kernel.shape[0] // 64))
    for ch in range(0, NCHANNELS):
        for sample in range(0, DATASET_NSAMPLES):
            for x in range(0, 64):
                for y in range(0, kernel.shape[0] // 64):
                    ax = 64 * sample + x
                    ay = ch * kernel.shape[0] // 64 + y
                    a[ay][ax] = kernel[ch * DATASET_NSAMPLES + sample][x * 64 + y]
    save_pgm(name + '-sample-outputs2', kernel)

    # ---------------------
    a = np.empty((NCHANNELS * 64, DATASET_NSAMPLES * kernel.shape[0] // 64))
    for ch in range(0, NCHANNELS):
        for sample in range(0, DATASET_NSAMPLES):
            for x in range(0, 64):
                for y in range(0, kernel.shape[0] // 64):
                    ax = 64 * sample + x
                    ay = ch * kernel.shape[0] // 64 + y
                    a[ay][ax] = kernel[ch + sample * NCHANNELS][x * 64 + y]
    save_pgm(name + '-sample-outputs3', kernel)

    a = np.empty((1024,1024))
    for x in range(0,1024):
        for y in range(0,1024):
            a[y][x] = ((x+y) * 65535) // 2 // 1024
    save_pgm(name + '-pgm-test', a)


# Load the mapping of NN output class IDs to their human-readable strings.
def load_class_names(input_filename):
    with open(input_filename, 'r') as f:
        json_str = f.read()
    return json.loads(json_str)['class_names']

def main():
    parser = argparse.ArgumentParser(description='Draw the Dense layers in a model.')
    parser.add_argument('-o', '--output', required=True,
        help = 'Directory to save the drawings to.')
    parser.add_argument('-m', '--model', required=True,
        help = 'NN model file to use')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    model = keras.models.load_model(args.model)
    class_names = load_class_names(os.path.splitext(args.model)[0] + '.json')

    for layer in model.layers:
        # For clarity, layer.bias is not considered.
        if layer.name == 'output':
            for i in range(0, len(class_names)):
                dump_output(args.output + '/' + class_names[i], layer.kernel[:, i])
        elif layer.name == 'dense':
            dump_dense(args.output + '/' + 'dense', layer)

    sys.exit(0)

main()
