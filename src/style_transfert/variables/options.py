import sys
from epicpath import EPath
import json

from . import utils
from ..STMode import STMode


st_mode = STMode.Noise.value     # Mode of the style transfert

image_start = ['content']#, 'style']  # all_content, style, all_style, all

colab = 'google.colab' in sys.modules
if not colab:
    pass

options_json_path = EPath('style_transfert_options.json')

if options_json_path.exists():
    with open(options_json_path, 'r') as json_file:
        data = json.load(json_file)

        st_mode = utils.get_key(data, 'st_mode', st_mode)
        image_start = utils.get_key(data, 'image_start', image_start)
