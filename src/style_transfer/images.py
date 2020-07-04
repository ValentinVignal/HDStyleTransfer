import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt

from . import variables as var
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


def reshape_keep_dim(image, size):
    """

    :param image: [batch, x, y, 3]
    :param reshape_value:
    :return:
    """
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    dim = max(shape) if var.param.dim_size.value == 'max' else min(shape)
    scale = size / dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)  # (512, 512, 3)
    return image


def reshape_to_exact_dim(image, exact_size):
    """

    :param image:
    :param exact_size:
    :return:
    """
    if len(tf.shape(image)) == 3:
        # (x, y, 3)
        image = image[tf.newaxis, :]  # (1, x, y, 3)
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    # First resize so axis >= exact size
    scale_x = exact_size[0] / shape[0]
    scale_y = exact_size[1] / shape[1]
    scale = tf.math.maximum(scale_x, scale_y)

    new_shape = tf.cast(shape * scale, tf.int32)

    # image = tf.image.resize(image, new_shape)  # (x, y, 3) here image is > exact size with one axis ==
    # Actually no need to reshape

    diff_x = (new_shape[1] - exact_size[1]) / new_shape[1]
    diff_y = (new_shape[0] - exact_size[0]) / new_shape[0]
    y1 = diff_y / 2
    x1 = diff_x / 2
    y2 = 1 - diff_y / 2
    x2 = 1 - diff_x / 2
    boxes = tf.stack([y1, x1, y2, x2], axis=0),

    image = tf.image.crop_and_resize(
        image=image,
        boxes=boxes,
        box_indices=[0],
        crop_size=exact_size
    )
    return image


def load_img(path_to_img, size=var.param.img_size.value, exact_size=None):
    """
    function to load an image and limit its maximum dimension to 512 pixels.
    arg: path of the image
    return: a tensor of shape (1, h, l, 3)
    """
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)  # (h, l, 3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # (h, l 3)

    if exact_size is None:
        # dim = max(shape) if dim_size == 'max' else min(shape)
        # scale = size / dim
        #
        # new_shape = tf.cast(shape * scale, tf.int32)
        #
        # img = tf.image.resize(img, new_shape)  # (512, 512, 3)
        # img = img[tf.newaxis, :]  # (1, 512, 512, 3)
        img = reshape_keep_dim(img[tf.newaxis, :], size=size)
    else:
        img = reshape_to_exact_dim(img, exact_size)
        # if len(tf.shape(img)) == 3:
        #     # (x, y, 3)
        #     img = img[tf.newaxis, :]  # (1, x, y, 3)
        # shape = tf.cast(tf.shape(img)[1:-1], tf.float32)
        # # First resize so axis >= exact size
        # scale_x = exact_size[0] / img.shape[0]
        # scale_y = exact_size[1] / img.shape[1]
        # scale = tf.math.maximum(scale_x, scale_y)
        #
        # new_shape = tf.cast(shape * scale, tf.int32)
        #
        # # img = tf.image.resize(img, new_shape)  # (x, y, 3) here img is > exact size with one axis ==
        # # Actually no need to reshape
        #
        # diff_x = (new_shape[1] - exact_size[1]) / new_shape[1]
        # diff_y = (new_shape[0] - exact_size[0]) / new_shape[0]
        # y1 = diff_y / 2
        # x1 = diff_x / 2
        # y2 = 1 - diff_y / 2
        # x2 = 1 - diff_x / 2
        # boxes = tf.concat([y1, x1, y2, x2], axis=0),
        #
        # img = tf.image.crop_and_resize(
        #     image=img,
        #     boxes=boxes,
        #     box_indices=[0],
        #     crop_size=exact_size
        # )
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


def load_content_style_img(content_path, style_path, plot_it=False, start_path=None):
    content_image = load_img(content_path, size=var.param.img_size.value)
    # gv.real_shape_hd_content = content_image.shape[1:3]
    # gv.real_shape_nn_content = (
    #     int(content_image.shape[1] / gv.ratio_size), int(content_image.shape[2] / gv.ratio_size))
    style_image = load_img(style_path, size=var.param.img_size.value)
    # gv.real_shape_hd_style = style_image.shape[1:3]
    # gv.real_shape_nn_style = (content_image.shape[1] // gv.ratio_size, content_image.shape[2] // gv.ratio_size)
    if start_path is None:
        start_image = content_image
    else:
        start_image = load_img(start_path,
                               exact_size=content_image.shape[1:3])  # Will resize if to the content image size

    content_style_images = ImageCouple(
        content_image=content_image,
        style_image=style_image,
        start_image=start_image
    )

    if plot_it:
        plt.close('all')
        plt.ion()
        plt.show()

        plt.subplot(1, 3, 1)
        imshow(content_image, 'Content Image')

        plt.subplot(1, 3, 2)
        imshow(style_image, 'Style Image')

        plt.subplot(1, 3, 3)
        imshow(start_image, 'Start Image')

        plt.draw()
        plt.pause(0.001)
    # return content_image, style_image
    return content_style_images
