#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2023 Dimitar Dimitrov <dimitar@dinux.eu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Check if GPU is detected and can be used by TensorFlow.

import tensorflow as tf
print("GPU available: {}".format(tf.test.is_gpu_available(),))
