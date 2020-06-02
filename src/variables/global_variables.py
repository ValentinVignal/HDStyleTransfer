import sys
from epicpath import EPath
import json

from . import utils

img_size_nn = 512  # Max size of the input of the nn
img_size_hd = 1024  # TODO: change it to 4096
nb_offsets = 4  # number of offsets to do overlapping

colab = 'google.colab' in sys.modules

if not colab:
    # on my pc
    img_size_nn = 64
    img_size_hd = 128
    nb_offsets = 2

json_path = EPath('global_variables.json')

if json_path.exists():
    # If a json file exists, then take the variables
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        img_size_nn = utils.get_key(data, 'img_size_nn', img_size_nn)
        img_size_hd = utils.get_key(data, 'img_size_hd', img_size_hd)
        nb_offsets = utils.get_key(data, 'nb_offsets', nb_offsets)

ratio_size = img_size_hd // img_size_nn
