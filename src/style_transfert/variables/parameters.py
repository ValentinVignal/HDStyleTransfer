import sys
from epicpath import EPath
import json

from . import utils
from ..STMode import STMode

# Parameters

st_mode = STMode.Noise.value     # Mode of the style transfert

img_size = 1024  # Size of one dimension of an image
img_size_nn = 512       # Size of the image given to the nn for the noise mode
dim_size = 'min'  # To choose whether the image size if the biggest or smallest axis
style_weight = 1e-2  # Importance of style
content_weight = 1e4  # Importance of content
content_gram_weight = 1e-4
content_weight_multiplicator = 10
total_variation_weight = 30  # How much to reduce high frequencies

epochs = 10
steps_per_epoch = 20
lr = 2e-2

content_layers = [
    'block5_conv3'
]

content_gram_layers = [
    'block5_conv2',
    'block5_conv4'
]

style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

image_start = ['content', 'style']  # all_content, style, all_style, all

loss = 'l1'

colab = 'google.colab' in sys.modules

if not colab:
    # on my pc
    img_size = 128
    img_size_nn = 64
    epochs = 4
    steps_per_epoch = 5

json_path = EPath('style_transfert_parameters.json')

if json_path.exists():
    # If a json file exists, then take the variables
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        st_mode = utils.get_key(data, 'st_mode', st_mode)
        img_size = utils.get_key(data, 'img_size', img_size)
        img_size_nn = utils.get_key(data, 'img_size_nn', img_size_nn)
        dim_size = utils.get_key(data, 'dim_size', dim_size)
        style_weight = utils.get_key(data, 'style_weight', style_weight)
        content_weight = utils.get_key(data, 'content_weight', content_weight)
        content_gram_weight = utils.get_key(data, 'content_gram_weight', content_gram_weight)
        content_weight_multiplicator = utils.get_key(data, 'content_weight_multiplicator',
                                                     content_weight_multiplicator)
        total_variation_weight = utils.get_key(data, 'total_variation_weight', total_variation_weight)
        epochs = utils.get_key(data, 'epochs', epochs)
        steps_per_epoch = utils.get_key(data, 'steps_per_epoch', steps_per_epoch)
        lr = utils.get_key(data, 'lr', lr)
        content_layers = utils.get_key(data, 'content_layers', content_layers)
        content_gram_layers = utils.get_key(data, 'content_gram_layers', content_gram_layers)
        style_layers = utils.get_key(data, 'style_layers', style_layers)
        image_start = utils.get_key(data, 'image_start', image_start)
        loss = utils.get_key(data, 'loss', loss)

num_content_layers = len(content_layers)
num_content_gram_layers = len(content_gram_layers)
num_style_layers = len(style_layers)
