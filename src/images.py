import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt

from . import global_variables as gv
from .ImageCouple import ImageCouple


def tensor_to_image(tensor):
    """
    take a tensor with 0 <= values <= 1 and return the corresponding image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img, max_dim=gv.img_size_nn):
    """
    function to load an image and limit its maximum dimension to 512 pixels.
    arg: path of the image
    return: a tensor of shape (1, h, l, 3)
    """
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)  # (h, l, 3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # (h, l 3)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)  # (512, 512, 3)
    img = img[tf.newaxis, :]  # (1, 512, 512, 3)
    return img


def imshow(image, title=None):
    """
    Show a tensor as an image
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)  # squeeze = remove shape of 1

    plt.imshow(image)
    if title is not None:
        plt.title(title)


def load_content_style_img(content_path, style_path, plot_it=False):
    content_image = load_img(content_path, max_dim=gv.img_size_hd)
    # gv.real_shape_hd_content = content_image.shape[1:3]
    # gv.real_shape_nn_content = (
    #     int(content_image.shape[1] / gv.ratio_size), int(content_image.shape[2] / gv.ratio_size))
    style_image = load_img(style_path, max_dim=gv.img_size_hd)
    # gv.real_shape_hd_style = style_image.shape[1:3]
    # gv.real_shape_nn_style = (content_image.shape[1] // gv.ratio_size, content_image.shape[2] // gv.ratio_size)

    content_style_images = ImageCouple(
        content_image=content_image,
        style_image=style_image
    )

    if plot_it:
        plt.close('all')
        plt.ion()
        plt.show()
        plt.subplot(1, 2, 1)
        imshow(content_image, 'Content Image')

        plt.subplot(1, 2, 2)
        imshow(style_image, 'Style Image')
        plt.draw()
        plt.pause(0.001)
    # return content_image, style_image
    return content_style_images
