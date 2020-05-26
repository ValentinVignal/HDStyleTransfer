import numpy as np
import tensorflow as tf

from . import global_variables as gv
from . import parameters as p


class Optimizers:
    def __init__(self):
        self.optimizers = None
        self.init_optimizers()

    def init_optimizers(self):
        self.optimizers = np.ndarray(
            shape=(gv.ratio_size, gv.nb_offsets, gv.nb_offsets),
            dtype=tf.optimizers.Optimizer
        )
        for r in range(gv.ratio_size):
            for o_i in range(gv.nb_offsets):
                for o_j in range(gv.nb_offsets):
                    self.optimizers[r, o_i, o_j] = tf.optimizers.Adam(learning_rate=p.lr, beta_1=0.99, epsilon=1e-1)
