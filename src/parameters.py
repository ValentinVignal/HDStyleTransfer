# Parameters
style_weight = 1e-2     # Importance of style
content_weight = 1e5        # Importance of content
total_variation_weight = 1e3        # How much to reduce high freauencies

ratio_weight = 1e2      # Reduction of smaller sub-images

epochs = 1e2
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
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
