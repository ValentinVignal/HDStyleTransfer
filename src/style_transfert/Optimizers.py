import numpy as np
import tensorflow as tf

from . import variables as var


class Optimizers:
    def __init__(self, shape=(1,), lr=var.param.lr.value):
        self.optimizers = None
        self.lr = lr
        self.init_optimizers(shape=shape)

    @property
    def shape(self):
        return self.optimizers.shape

    def init_optimizers(self, shape=(1,)):
        optimizers = np.ndarray(
            shape=shape,
            dtype=tf.optimizers.Optimizer
        ).flatten()
        for i in range(len(optimizers)):
            optimizers[i] = tf.optimizers.Adam(
                learning_rate=self.lr,
                beta_1=0.99,
                epsilon=1e-1
            )
        self.optimizers = np.reshape(optimizers, shape)
