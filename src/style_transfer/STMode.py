from enum import Enum


class STMode(Enum):
    Direct = 'direct'   # Directly put the image in the NN
    Noise = 'noise'     # Put reduced noisy image in the NN
    Hub = 'hub'     # Tensorflow hub
