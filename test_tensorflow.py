#!/usr/bin/env python3

import tensorflow as tf


print(tf.__version__)
print()
print()

if tf.test.is_gpu_available():
    print("GPU is available")

else:
    print("no gpu found :(")




