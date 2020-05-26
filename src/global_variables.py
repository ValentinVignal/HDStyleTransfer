import platform

img_size_nn = 512       # Max size of the input of the nn
img_size_hd = 2048      # TODO: change it to 4096
nb_offsets = 4           # number of offsets to do overlapping

if platform.system() == 'Windows':
    # on my pc
    img_size_nn = 128
    img_size_hd = 512
    nb_offsets = 2


ratio_size = img_size_hd // img_size_nn



