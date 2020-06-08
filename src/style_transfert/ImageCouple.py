class ImageCouple:
    def __init__(self, content_image, style_image, start_image=None):
        """

        :param content_image: tensor
        :param style_image: tensor
        """
        self.content_image = content_image
        self.style_image = style_image
        self.start_image = content_image if start_image is None else start_image

    @property
    def content_shape(self):
        return self.content_image.shape[1: 3]

    @property
    def style_shape(self):
        return self.style_image.shape[1: 3]

    @property
    def start_shape(self):
        return self.start_image.shape[1: 3]
