import numpy as np
import tensorflow as tf

from . import variables as var


class Optimizers:
    def __init__(self):
        self.optimizers = None
        self.init_optimizers()

    def init_optimizers(self):
        print('nb offsets', var.gv.nb_offsets)
        self.optimizers = np.ndarray(
            shape=(var.gv.ratio_size, var.gv.nb_offsets, var.gv.nb_offsets),
            dtype=tf.optimizers.Optimizer
        )
        for r in range(var.gv.ratio_size):
            for o_i in range(var.gv.nb_offsets):
                for o_j in range(var.gv.nb_offsets):
                    self.optimizers[r, o_i, o_j] = tf.optimizers.Adam(learning_rate=var.p.lr, beta_1=0.99, epsilon=1e-1)
