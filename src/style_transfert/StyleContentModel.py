import tensorflow as tf
from . import model
from . import variables as var


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers=var.style_layers, content_layers=var.content_layers,
                 content_gram_layers=var.content_gram_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = model.vgg_layers(style_layers + content_layers + content_gram_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.content_gram_layers = content_gram_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.num_content_gram_layers = len(content_gram_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in [0,1]"""
        inputs = inputs * 255.0
        # preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(inputs)
        style_outputs, content_outputs, content_gram_outputs = (outputs[:self.num_style_layers],
                                                                outputs[self.num_style_layers:],
                                                                outputs[:self.num_content_gram_layers])

        style_outputs = [model.gram_matrix(style_output)
                         for style_output in style_outputs]

        content_gram_outputs = [model.gram_matrix(content_gram_output)
                                for content_gram_output in content_gram_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        content_gram_dict = {
            content_gram_name: value for content_gram_name, value in zip(self.content_gram_layers, content_gram_outputs)
        }

        return {'content': content_dict, 'style': style_dict, 'content_gram': content_gram_dict}
