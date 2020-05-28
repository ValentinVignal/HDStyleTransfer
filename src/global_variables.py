import sys

img_size_nn = 512       # Max size of the input of the nn
img_size_hd = 1024      # TODO: change it to 4096
nb_offsets = 4           # number of offsets to do overlapping

colab = 'google.colab' in sys.modules

if not colab:
    # on my pc
    img_size_nn = 64
    img_size_hd = 128
    nb_offsets = 2


ratio_size = img_size_hd // img_size_nn



