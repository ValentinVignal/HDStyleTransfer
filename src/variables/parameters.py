import platform
from epicpath import EPath
import json

from . import utils

# Parameters
style_weight = 1e-2     # Importance of style
content_weight = 1e5        # Importance of content
total_variation_weight = 1e3        # How much to reduce high freauencies

ratio_weight = 1e2      # Reduction of smaller sub-images


epochs = 10
steps_per_epoch = 5
lr = 1e-2

style_division = False      # Should we use sub-images of the style image


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

if platform.system() == 'Windows':
    # on my pc
    epochs = 2
    steps_per_epoch = 2

json_path = EPath('parameters.json')

if json_path.exists():
    # If a json file exists, then take the variables
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        style_weight = utils.get_key(data, 'style_weight', style_weight)
        content_weight = utils.get_key(data, 'content_weight', content_weight)
        total_variation_weight = utils.get_key(data, 'total_variation_weight', total_variation_weight)
        ratio_weight = utils.get_key(data, 'ratio_weight', ratio_weight)
        epochs = utils.get_key(data, 'epochs', epochs)
        steps_per_epoch = utils.get_key(data, 'steps_per_epoch', steps_per_epoch)
        lr = utils.get_key(data, 'lr', lr)
        style_division = utils.get_key(data, 'style_division', style_division)
        content_layers = utils.get_key(data, 'content_layers', content_layers)
        style_layers = utils.get_key(data, 'style_layers', style_layers)
        lr = utils.get_key(data, 'lr', lr)
        lr = utils.get_key(data, 'lr', lr)
        lr = utils.get_key(data, 'lr', lr)
        lr = utils.get_key(data, 'lr', lr)

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
