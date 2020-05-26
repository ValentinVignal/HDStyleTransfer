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



# Parameters
style_weight = 1e-2     # Importance of style
content_weight = 1e5        # Importance of content
total_variation_weight = 1e3        # How much to reduce high freauencies

ration_weight = 1e2      # Reduction of smaller sub-images

epochs = 1e2
steps_per_epoch = 5
lr = 1e-2

style_division = False      # Should we use sub-images of the style image


