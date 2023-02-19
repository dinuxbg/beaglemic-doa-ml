#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2022-2023 Dimitar Dimitrov <dimitar@dinux.eu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Train DOA estimation model.
#
# Example invocation:
#    $ ./train.py -i dataset-directory -o model.h5

import numpy as np
import argparse
import random
import glob
import json
import sys
import os

import tensorflow as tf
from tensorflow import keras

# Percentage of samples to use for validation
VALID_SPLIT = 0.1
BATCH_SIZE = 32
EPOCHS = 100

SHUFFLE_SEED = 42

# The datasets audio parameters.
NCHANNELS = 8;
SAMPLES_PER_SECOND = 24000;

# Neural Network's input parameters.
DATASET_NSAMPLES = 512;

class train_state:
    def __init__(self):
        self.class_names = None
        self.dataset_paths = []
        self.labels = []
        # TF datasets with audio samples and integer labels.
        self.train_ds = None
        self.validation_ds = None
        self.model_filename = None

def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    # TODO - revise!
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(4096, activation="relu")(x)
    x = keras.layers.Dense(4096, activation="relu")(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(256, activation="relu")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)

def do_training(trst):
    model = build_model((NCHANNELS * DATASET_NSAMPLES, 1), len(trst.class_names))
    model.summary()

    # Compile the model using Adam's default learning rate
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Add callbacks:
    # 'EarlyStopping' to stop training when the model is not enhancing anymore
    # 'ModelCheckpoint' to always keep the model that has the best val_accuracy
    if not trst.model_filename.endswith('.h5'):
        print('ERROR: Output filename must with with .h9')
        sys.exit(1)

    earlystopping_cb = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
        trst.model_filename, monitor="val_accuracy", save_best_only=True
    )

    history = model.fit(
        trst.train_ds,
        epochs=EPOCHS,
        validation_data=trst.validation_ds,
        callbacks=[earlystopping_cb, mdlcheckpoint_cb],
    )
    print(model.evaluate(trst.validation_ds))
    model.save(trst.model_filename)

def path_to_audio(path):
    """Reads a raw audio file."""
    audio = tf.io.read_file(path)
    audio = tf.io.decode_raw(audio, tf.int32)
    audio = tf.cast(audio, tf.float32) / 2**31

    return audio

def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(
        lambda x: path_to_audio(x), num_parallel_calls=tf.data.AUTOTUNE
    )
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))

def prepare_datasets(trst, input_dirname):
    # We'll classify per angle of arrival and silence.
    # For now the elevation and distance will not be taken into account.
    trst.class_names = os.listdir(input_dirname)

    # Enumerate the available datasets.
    for label, name in enumerate(trst.class_names):
        print("Processing dataset {}".format(name,))
        dirpath = os.path.join(input_dirname, name)
        fpaths = glob.glob(dirpath + '/**/*raw_*', recursive=True)
        trst.dataset_paths += fpaths
        trst.labels += [label] * len(fpaths)
    print("Found {} files belonging to {} classes.".format(len(trst.dataset_paths), len(trst.class_names)))

    # Shuffle
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(trst.dataset_paths)
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(trst.labels)

    # Split into training and validation
    num_val_samples = int(VALID_SPLIT * len(trst.dataset_paths))
    print("Using {} files for training.".format(len(trst.dataset_paths) - num_val_samples))
    train_ds_paths = trst.dataset_paths[:-num_val_samples]
    train_labels = trst.labels[:-num_val_samples]

    print("Using {} files for validation.".format(num_val_samples))
    valid_ds_paths = trst.dataset_paths[-num_val_samples:]
    valid_labels = trst.labels[-num_val_samples:]

    # Create 2 datasets, one for training and the other for validation
    trst.train_ds = paths_and_labels_to_dataset(train_ds_paths, train_labels)
    trst.train_ds = trst.train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)

    trst.validation_ds = paths_and_labels_to_dataset(valid_ds_paths, valid_labels)
    trst.validation_ds = trst.validation_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
    
    trst.train_ds = trst.train_ds.prefetch(tf.data.AUTOTUNE)
    trst.validation_ds = trst.validation_ds.prefetch(tf.data.AUTOTUNE)

def save_class_names(trst, output_filename):
    root = {"class_names" : trst.class_names}
    root_str = json.dumps(root)
    with open(output_filename, 'w') as o:
        o.write(root_str)

def main():
    tf.config.experimental.set_memory_growth = True
    parser = argparse.ArgumentParser(description='Train a DOA estimation model.')
    parser.add_argument('-i', '--input', required=True,
        help = 'Directory with audio datasets.')
    parser.add_argument('-o', '--output', required=True,
        help = 'File to write the final model.')
    parser.add_argument('-d', '--debug', required=False,
        help = 'Directory to write debug TF logs to.')
    args = parser.parse_args()

    if args.debug is not None:
        tf.debugging.experimental.enable_dump_debug_info(
            args.debug,
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1)

    trst = train_state()
    trst.model_filename = args.output

    prepare_datasets(trst, args.input)

    # I'm not sure why there is no standard method to save
    # the output label strings in the model itself.
    # So save in a separate json file.
    save_class_names(trst, os.path.splitext(args.output)[0] + '.json')

    do_training(trst)

    sys.exit(0)

main()
