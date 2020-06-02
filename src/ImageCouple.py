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
