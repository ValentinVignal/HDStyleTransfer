import tensorflow as tf
import tensorflow_hub as hub
import gc
import loadbar
import random
import numpy as np

from . import variables as var
from . import images
from . import plot
from .STMode import STMode


def clip_0_1(image):
    """

    :param image:
    :return: image with 0 <= values <= 1
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def loss_function(tensor1, tensor2, loss_type=var.param.loss.value):
    """

    :param tensor1:
    :param tensor2:
    :param loss_type:
    :return:
    """
    if loss_type == 'l2':
        return tf.reduce_mean((tensor1 - tensor2) ** 2)
    elif loss_type == 'l1':
        return tf.reduce_mean(tf.abs(tensor1 - tensor2))


def style_content_loss(outputs, content_targets, style_targets, content_gram_targets=None, is_start_content=True):
    """

    :param outputs:
    :param content_targets:
    :param style_targets:
    :return: the loss for the style transfert
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([loss_function(style_outputs[name], style_targets[name])
                           for name in style_outputs.keys()]) / var.param.style_layers.num
    style_loss *= var.param.style_weight.value

    # content_loss = var.param.content_weight * tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
    #                                               for name in content_outputs.keys()]) / var.param.num_content_layers
    content_loss = var.param.content_weight.value * tf.add_n([
        loss_function(content_outputs[name], content_targets[name])
         for name in content_outputs.keys()
    ]) / var.param.content_layers.num
    if content_gram_targets is not None and var.param.content_gram_weight.value != 0:
        content_gram_outputs = outputs['content_gram']
        content_loss += var.param.content_gram_weight.value * tf.add_n([
            loss_function(content_gram_outputs[name], content_gram_targets[name])
            for name in content_gram_outputs.keys()
        ]) / var.param.content_gram_layers.num

    if not is_start_content:
        content_loss *= var.param.content_weight_multiplicator.value
    loss = style_loss + content_loss
    return loss


def deform_image(image):
    """

    :param image:
    :return:
    """

    # Deform the image and resize it to a smaller image
    # Make sure the all the image is given to the nn

    final_shape = tuple([int((var.param.img_size_nn.value / var.param.img_size.value) * s) for s in image.shape[1:3]])
    batch = image.shape[0]

    # Padding
    padding_size = 32
    padding = tf.pad(
        tensor=tf.random.uniform(
            shape=(2, 2),
            minval=padding_size,
            maxval=2 * padding_size + 1,
            dtype=tf.int32
        ),
        paddings=[[1, 1], [0, 0]]
    )

    padded_image = tf.pad(
        tensor=image,
        paddings=padding,
        mode='CONSTANT',
        constant_values=0
    )

    # Crop and resize

    boxe_sizes = tf.random.uniform(
        shape=(batch, 2, 2),
        minval=0,
        maxval=padding_size / var.param.img_size.value,
        dtype=tf.float32
    )
    boxes = tf.concat(
        [
            boxe_sizes[:, :, 0],
            1 - boxe_sizes[:, :, 1]
        ],
        axis=1,
    )  # (batch, 4)

    reshaped_image = tf.image.crop_and_resize(
        image=padded_image,
        boxes=boxes,
        box_indices=tf.range(batch),
        crop_size=final_shape
    )

    # reshaped_image = tf.image.resize(
    #     padded_image,
    #     final_shape
    # )
    return reshaped_image


def create_train_step(extractor, optimizers, content_image, style_image, content_gram_targets,
                      is_start_content=True, st_mode=var.options.st_mode):
    """
    Creates and returns the train step function ton do style transfert iteration
    :param extractor: model used to extract the features layers
    :param optimizers: optimizers
    :param image_couple: images of content and style
    :return: train step
    """

    @tf.function
    def train_step(image):
        """
        update the image

        :param image: Image to modify   [-1, 1]
        :param content_image:
        :param style_image:
        :return: None
        """
        with tf.GradientTape() as tape:
            if st_mode == STMode.Noise.value:
                # Deform the image and resize it to a smaller image
                # Make sure the all the image is given to the nn
                style_image_reshaped = images.reshape_to_exact_dim(style_image, image.shape[1:-1])
                deformed_images = deform_image(tf.concat([image, content_image, style_image_reshaped], axis=0))

                image_nn, content_nn, style_nn = deformed_images[0:1], deformed_images[1:2], deformed_images[2:3]
            else:
                image_nn = image
                content_nn = content_image
                style_nn = style_image
            outputs = extractor(image_nn)
            outputs_content = extractor(content_nn)
            outputs_style = extractor(style_nn)
            loss = style_content_loss(
                outputs=outputs,
                content_targets=outputs_content['content'],
                style_targets=outputs_style['style'],
                content_gram_targets=content_gram_targets,
                is_start_content=is_start_content
            )
            loss += var.param.total_variation_weight.value * tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        optimizers.optimizers[0].apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    return train_step


def style_transfert(file_combination, extractor, optimizers, epochs=var.param.epochs.value,
                    steps_per_epoch=var.param.steps_per_epoch.value, st_mode=var.options.st_mode):
    """

    :param content_path:
    :param style_path:
    :param extractor:
    :param optimizers:
    :param image_start:
    :param epochs:
    :param steps_per_epoch:
    :return:
    """
    image_couple = images.load_content_style_img(
        content_path=file_combination.content_path.str,
        style_path=file_combination.style_path.str,
        start_path=file_combination.start_path.str,
        plot_it=True
    )
    file_combination.results_folder.mkdir(exist_ok=True, parents=True)
    if st_mode == STMode.Hub.value:
        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
        stylized_image = hub_module(tf.constant(image_couple.content_image), tf.constant(image_couple.style_image))[0]
        # stylized_image = image_couple.content_image
        plot.display(stylized_image)
        file_path = file_combination.results_folder / f'{file_combination.result_stem}.png'
        images.tensor_to_image(stylized_image).save(file_path.str)
    else:
        content_gram_targets = extractor(image_couple.content_image)['content_gram']

        train_step = create_train_step(
            extractor=extractor,
            optimizers=optimizers,
            content_image=image_couple.content_image,
            style_image=image_couple.style_image,
            content_gram_targets=content_gram_targets,
            is_start_content=file_combination.is_start_content,
            st_mode=st_mode
        )
        image = tf.Variable(image_couple.start_image)
        bar_epoch = loadbar.ColorBar(color=loadbar.Colors.cyan, max=epochs, title='Epoch', show_eta=False,
                                     show_time=True)
        bar_epoch.start()
        nb_steps = 0
        for n in range(epochs):
            # pb = ProgressBar(max_iteration=(n + 1) * varsteps_per_epoch, title=f'Epoch {n + 1}/{varepochs}')
            bar_epoch.update(step=n, end='\n')

            bar_step = loadbar.LoadBar(max=(n + 1) * steps_per_epoch, title='Step')
            bar_step.start()
            for m in range((n + 1) * steps_per_epoch):
                train_step(image=image)
                nb_steps += 1
                bar_step.update()
            bar_step.end()
            plot.display(image)
            file_name = file_combination.results_folder / f'{file_combination.result_stem}_step_{nb_steps}.png'
            images.tensor_to_image(image).save(file_name.str)
        bar_epoch.end()
        del image_couple, image, train_step
        gc.collect()
