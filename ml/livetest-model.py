#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2023 Dimitar Dimitrov <dimitar@dinux.eu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Test the DOA estimation model by running it against live recording stream.
#
# Example invocations:
#    $ alias micrec='arecord --quiet -D hw:CARD=BeagleMic -c8 -t raw -f S32_LE -r24000'
#    $ micreg | ./livetest-model.py -m model.h5

import numpy as np
import argparse
import json
import socket
import time
import sys
import os

import tensorflow as tf
from tensorflow import keras

# The audio parameters.
NCHANNELS = 8;
# Neural Network's input parameters.
DATASET_NSAMPLES = 512;

class angle_printer:
    def __init__(self, _maxn):
        self.maxn = _maxn
        self.marker_space = self.maxn // 8

    def print_header(self):
        fmts = '%-' + str(self.marker_space) + '.3f'
        a_step = 360.0 / (self.maxn // self.marker_space)
        for i in range(0, self.maxn // self.marker_space):
            print(fmts % (i * a_step), end='')
        print('')
        s = ''
        for i in range(0, self.maxn // self.marker_space):
            s += '|' + ' ' * (self.marker_space -1)
        print(s + '|')

    def update_angle(self, A):
        s = ''
        x = int((A * self.maxn) / 360.0)
        if x > self.maxn:
            raise Exception('Invalid angle ' + str(A))
        for i in range(0,self.maxn):
            if i == x:
                s += '#'
            else:
                s += ' '
        print('\r' + s, end='', flush=True)

    def update_silence(self):
        print('\r' + '.' * self.maxn, end='', flush=True)

    def demo(self):
        self.print_header()
        i = 0.0
        while True:
            time.sleep(0.021)
            self.update_angle(i)
            i += (360.0 / 64)
            if i >= 360.0:
                i = 0
                time.sleep(0.021)
                self.update_silence()

class data_reader:
    def __init__(self, _tcp_port):
        self.tcp_port = _tcp_port
        if (self.tcp_port > 0):
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.bind(('0.0.0.0', self.tcp_port))
            self.s.listen()
            self.conn, addr = self.s.accept()
        else:
            self.s = None

    def read(self):
        nbytes = NCHANNELS * DATASET_NSAMPLES * 4
        data = b''
        while nbytes > 0:
            if (self.tcp_port > 0):
                buf = self.conn.recv(nbytes)
            else:
                buf = sys.stdin.buffer.read(nbytes)
            nbytes -= len(buf)
            data += buf
        audio = np.frombuffer(data, dtype=np.dtype('<i4'))
        audio = np.divide(audio, 2**31)
        return audio

def run_sample(model, a, labels):
    ds = tf.data.Dataset.from_tensor_slices(([a], [''])).batch(1)

    predict = model.predict(ds, verbose=0)
    angle_id_predict = np.argmax(predict)
    a = labels[angle_id_predict]

    if a == 'silence':
        return -1
    else:
        return float(a)

# Load the mapping of NN output class IDs to their human-readable strings.
def load_class_names(input_filename):
    with open(input_filename, 'r') as f:
        json_str = f.read()
    return json.loads(json_str)['class_names']

def main():
    parser = argparse.ArgumentParser(description='Test the DOA estimation model.')
    parser.add_argument('-m', '--model', required=True,
        help = 'NN model file to use')
    parser.add_argument('-n', '--niterations', required=False,
        default=100, type=int,
        help = 'How much test iterations to do')
    parser.add_argument('-p', '--port', required=False,
        default=-1, type=int,
        help = 'Use the given TCP port for incoming data instead of stdin.')
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    class_names = load_class_names(os.path.splitext(args.model)[0] + '.json')

    data = data_reader(args.port)

    aprinter = angle_printer(len(class_names) - 1)
    aprinter.print_header()

    for testi in range(0, args.niterations):
        achunk = data.read()

        angle = run_sample(model, achunk, class_names)
        if angle < 0.0:
            aprinter.update_silence()
        else:
            aprinter.update_angle(angle)

    sys.exit(0)

main()
