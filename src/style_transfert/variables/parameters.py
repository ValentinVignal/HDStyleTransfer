import sys
from epicpath import EPath
import json
import re

from . import utils
from .Parameter import Parameter
from .ParametersManager import ParametersManager


# Parameters

p = ParametersManager()
p.img_size = 1024       # Size of one dimension of an image
p.img_size_nn = 512       # Size of the image given to the nn for the noise mode
p.dim_size = 'min'  # To choose whether the image size if the biggest or smallest axis
p.style_weight = 1e-2  # Importance of style
p.content_weight = 1e4  # Importance of content
p.content_gram_weight = 1e-4
p.content_weight_multiplicator = 10
p.total_variation_weight = 30  # How much to reduce high frequencies

p.epochs = 10
p.steps_per_epoch = 20
p.lr = 2e-2

p.content_layers = [
    'block5_conv2'
]

p.content_gram_layers = [
    'block5_conv2',
    'block5_conv4'
]

p.style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]


p.loss = 'l1'

colab = 'google.colab' in sys.modules

if not colab:
    # on my pc
    p.img_size = 128
    p.img_size_nn = 64
    p.epochs = 4
    p.steps_per_epoch = 5

style_transfert_parameters_path = EPath('style_transfert_parameters.json')

if style_transfert_parameters_path.exists():
    # Then update the default parameters
    with open(style_transfert_parameters_path, 'r') as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            if value is not None:
                p.update(key, value)

style_transfert_parameters_grid_path = EPath('style_transfert_parameters_grid.json')

if style_transfert_parameters_grid_path.exists():
    # Create the grid
    with open(style_transfert_parameters_grid_path, 'r') as json_file:
        data = json.load(json_file)
        for key, values in data.items():
            if value is not None:
                p.set_grid_values(key, values)




#
# num_content_layers = len(content_layers)
# num_content_gram_layers = len(content_gram_layers)
# num_style_layers = len(style_layers)
