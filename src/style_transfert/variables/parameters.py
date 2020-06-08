import sys
from epicpath import EPath
import json

from . import utils

# Parameters
img_size = 1024     # Size of one dimension of an image
dim_size = 'min'    # To choose whether the image size if the biggest or smallest axis
style_weight = 1e-2     # Importance of style
content_weight = 1e4        # Importance of content
content_weight_other_image = 1e5        # Importance of the content weight when the start image is not the content one
total_variation_weight = 30        # How much to reduce high frequencies

epochs = 10
steps_per_epoch = 20
lr = 2e-2

content_layers = [
    'block5_conv2'
]
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

image_start = ['content', 'style']       # all_content, style, all_style, all

colab = 'google.colab' in sys.modules

if not colab:
    # on my pc
    img_size = 128
    epochs = 4
    steps_per_epoch = 5

json_path = EPath('style_transfert_parameters.json')

if json_path.exists():
    # If a json file exists, then take the variables
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        img_size = utils.get_key(data, 'img_size', img_size)
        dim_size = utils.get_key(data, 'dim_size', dim_size)
        style_weight = utils.get_key(data, 'style_weight', style_weight)
        content_weight = utils.get_key(data, 'content_weight', content_weight)
        content_weight_other_image = utils.get_key(data, 'content_weight_other_image', content_weight_other_image)
        total_variation_weight = utils.get_key(data, 'total_variation_weight', total_variation_weight)
        epochs = utils.get_key(data, 'epochs', epochs)
        steps_per_epoch = utils.get_key(data, 'steps_per_epoch', steps_per_epoch)
        lr = utils.get_key(data, 'lr', lr)
        content_layers = utils.get_key(data, 'content_layers', content_layers)
        style_layers = utils.get_key(data, 'style_layers', style_layers)
        image_start = utils.get_key(data, 'image_start', image_start)

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
