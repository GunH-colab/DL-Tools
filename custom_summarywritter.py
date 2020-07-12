import numpy as np
import os
import time
import tensorflow as tf

from directory_for_every_run import create_dir_cur

test_log_dir = create_dir_cur()

writer = tf.summary.create_file_writer(test_log_dir)

with writer.as_default():
    for step in range(1, 1000 + 1):
        tf.summary.scalar("keras_model2_scalar", np.sin(step/10), step=step)
        data = (np.random.rand(100) + 2) * (step/100)
        tf.summary.histogram("keras_model2_hist", data, buckets=50, step=step)
        images = np.random.rand(2, 32, 32, 3)
        tf.summary.image("keras_model2_image", images * step/1000, step=step)
        texts = ["The step is " + str(step), "Its square is " + str(step**2)]
        tf.summary.text("keras_model2_text", texts, step=step)
        sine_w = tf.math.sin(tf.range(12000)/48000*2*np.pi*step)
        audio = tf.reshape(tf.cast(sine_w, tf.float32), [1, -1, 1])
        tf.summary.audio("keras_model2_audio", audio, sample_rate=48000, step=step)