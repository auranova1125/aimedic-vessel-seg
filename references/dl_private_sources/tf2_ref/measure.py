"""Measure the DMPHN_124 model on a fixed 3400 by 3400 single-channel input."""

import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from nets import DMPHN


INPUT_HEIGHT = 3400
INPUT_WIDTH = 3400
WARMUP_RUNS = 10
MEASUREMENT_RUNS = 10


def print_flops(model):
    """Build the model in a TF1 graph and print the profiled floating-point ops."""
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        model(np.zeros((1, INPUT_HEIGHT, INPUT_WIDTH, 1)))
        print(model.summary())

        run_metadata = tf.compat.v1.RunMetadata()
        profile_options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=graph,
            run_meta=run_metadata,
            cmd="op",
            options=profile_options,
        )
        print("FLOPS: {}".format(flops.total_float_ops))


def print_inference_speed(model):
    """Print the mean inference time and frames per second over ten runs."""
    image = np.zeros((1, INPUT_HEIGHT, INPUT_WIDTH, 1), dtype=np.float32)

    for _ in range(WARMUP_RUNS):
        model(image)

    start_time = time.time()
    for _ in range(MEASUREMENT_RUNS):
        model(image)
    elapsed_time = time.time() - start_time
    print(
        "Time: {:.6f}, FPS: {:.6f}".format(
            elapsed_time / MEASUREMENT_RUNS,
            MEASUREMENT_RUNS / elapsed_time,
        )
    )


if __name__ == "__main__":
    print_flops(DMPHN.DMPHN_124())
