import tensorflow as tf
import gc
import loadbar

from . import variables as var
from . import images
from . import plot


def clip_0_1(image):
    """

    :param image:
    :return: image with 0 <= values <= 1
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(outputs, content_targets, style_targets, is_start_content):
    """

    :param outputs:
    :param content_targets:
    :param style_targets:
    :return: the loss for the style transfert
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= var.style_weight / var.num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    if is_start_content:
        content_loss *= var.content_weight / var.num_content_layers
    else:
        content_loss *= var.content_weight_other_image / var.num_content_layers
    loss = style_loss + content_loss
    return loss


def create_train_step(extractor, optimizers, content_targets, style_targets, is_start_content=True):
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

        :param image: Image to modify
        :param content_image:
        :param style_image:
        :return: None
        """
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(
                outputs=outputs,
                content_targets=content_targets,
                style_targets=style_targets,
                is_start_content=is_start_content
            )
            loss += var.total_variation_weight * tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        optimizers.optimizers[0].apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    return train_step


def style_transfert(file_combination, extractor, optimizers, epochs=var.epochs,
                    steps_per_epoch=var.steps_per_epoch):
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
    content_targets = extractor(image_couple.content_image)['content']
    style_targets = extractor(image_couple.style_image)['style']

    file_combination.results_folder.mkdir(exist_ok=True, parents=True)
    train_step = create_train_step(
        extractor=extractor,
        optimizers=optimizers,
        content_targets=content_targets,
        style_targets=style_targets,
        is_start_content=file_combination.is_start_content
    )
    image = tf.Variable(image_couple.start_image)
    bar_epoch = loadbar.ColorBar(color=loadbar.Colors.cyan, max=epochs, title='Epoch', show_eta=False, show_time=True)
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
