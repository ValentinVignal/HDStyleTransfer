import tensorflow as tf

from . import variables as var


class ImageCouple:
    def __init__(self, content_image, style_image):
        """

        :param content_image: tensor
        :param style_image: tensor
        """
        self.content_image = content_image
        self.style_image = style_image

    @property
    def content_hd_shape(self):
        return self.content_image.shape[1: 3]

    @property
    def content_nn_shape(self):
        return tuple(s // var.gv.ratio_size for s in self.content_image.shape[1: 3])

    @property
    def style_hd_shape(self):
        return self.style_image.shape[1: 3]

    @property
    def style_nn_shape(self):
        return self.content_nn_shape

    def get_start_image(self, image_start=None):
        """

        :param image_start:
        :return:
        """
        if image_start is None or image_start == 'content':
            return tf.Variable(self.content_image)
        elif image_start == 'style':
            return tf.Variable(tf.image.resize(self.style_image, self.content_hd_shape))
        elif image_start == 'grey':
            return tf.Variable(0.5 * tf.ones(shape=(1, *self.content_hd_shape, 3)))
