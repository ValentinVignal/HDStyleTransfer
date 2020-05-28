import numpy as np
import random
import tensorflow as tf
from epicpath import EPath
import gc
import loadbar

from . import global_variables as gv
from . import images
from . import parameters as p
from . import plot


def get_ij(i, j, o_i=0, o_j=0, ratio_size=gv.ratio_size, img_type='content', image_couple=None):
    """
    return i_start, i_stop, j_start, j_stop
    """
    if img_type == 'content':
        # shape_hd = gv.real_shape_hd_content
        # shape_nn = gv.real_shape_nn_content
        shape_hd = image_couple.content_hd_shape
        shape_nn = image_couple.content_nn_shape
    elif img_type == 'style':
        # shape_hd = gv.real_shape_hd_style
        # shape_nn = gv.real_shape_nn_style
        shape_hd = image_couple.style_hd_shape
        shape_nn = image_couple.style_nn_shape
    i_offset = (o_i * shape_hd[0]) // (gv.nb_offsets * ratio_size)
    j_offset = (o_j * shape_hd[1]) // (gv.nb_offsets * ratio_size)

    i_start = i * shape_hd[0] // ratio_size + i_offset
    i_stop = i_start + shape_hd[0] // ratio_size
    j_start = j * shape_hd[1] // ratio_size + j_offset
    j_stop = j_start + shape_hd[1] // ratio_size
    return i_start, i_stop, j_start, j_stop


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def get_optimizers():
    optimizers = np.ndarray(
        shape=(gv.ratio_size, gv.nb_offsets, gv.nb_offsets),
        dtype=tf.optimizers.Optimizer
    )

    for r in range(gv.ratio_size):
        for o_i in range(gv.nb_offsets):
            for o_j in range(gv.nb_offsets):
                optimizers[r, o_i, o_j] = tf.optimizers.Adam(learning_rate=gv.lr, beta_1=0.99, epsilon=1e-1)

    return optimizers


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var


# TODO Make it work is combinaison of images
def style_content_loss(outputs, content_targets, style_targets):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= p.style_weight / p.num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= p.content_weight / p.num_content_layers
    loss = style_loss + content_loss
    return loss


def total_variation_loss(image, content_targets):
    x_deltas, y_deltas = high_pass_x_y(image, content_targets)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


# TODO: maybe create a function which creates this function
def create_train_step(extractor, optimizers, image_couple):
    @tf.function
    def train_step(image, content_image, style_image):
        for r in range(1, gv.ratio_size + 1):
            # ! For all size of sub-images
            offsets_i = list(range(gv.nb_offsets))
            random.shuffle(offsets_i)
            for o_i in offsets_i:
                # for all the offsets on first axis
                offsets_j = list(range(gv.nb_offsets))
                random.shuffle(offsets_j)
                for o_j in offsets_j:
                    # for all offsets on second axis
                    loss = tf.zeros(shape=(1,))
                    has_loss = False
                    with tf.GradientTape() as tape:
                        for i in range(r):
                            # For all the sub-images on first axis
                            for j in range(r):
                                # For all the sub-images on the second axis
                                if (o_i == 0 or i < r - 1) and (o_j == 0 or j < r - 1):
                                    has_loss = True
                                    i_start, i_stop, j_start, j_stop = get_ij(
                                        i, j, o_i, o_j,
                                        ratio_size=r,
                                        img_type='content',
                                        image_couple=image_couple
                                    )
                                    img = image[:, i_start:i_stop, j_start:j_stop]
                                    # img = tf.image.resize(img, gv.real_shape_nn_content)
                                    img = tf.image.resize(img, image_couple.content_nn_shape)
                                    cont = content_image[:, i_start:i_stop, j_start:j_stop]
                                    # cont = tf.image.resize(cont, gv.real_shape_nn_content)
                                    cont = tf.image.resize(cont, image_couple.style_nn_shape)

                                    outputs = extractor(img)
                                    content_targets = extractor(cont)['content']

                                    if p.style_division:
                                        i_start_style, i_stop_style, j_start_style, j_stop_style = get_ij(
                                            i, j, o_i, o_j,
                                            ratio_size=r,
                                            img_type='style',
                                            image_couple=image_couple
                                        )
                                        style = style_image[:, i_start_style:i_stop_style, j_start_style:j_stop_style]
                                    else:
                                        style = style_image
                                    # style = tf.image.resize(style, gv.real_shape_nn_style)
                                    style = tf.image.resize(style, image_couple.style_nn_shape)
                                    style_targets = extractor(style)['style']

                                    loss += style_content_loss(outputs, content_targets, style_targets)
                                    loss += p.total_variation_weight * tf.image.total_variation(img)
                        loss *= p.ratio_weight ** (1 - r)

                    if has_loss:
                        grad = tape.gradient(
                            loss,
                            image
                        )
                        optimizers.optimizers[r - 1, o_i, o_j].apply_gradients([(grad, image)])
                        image.assign(clip_0_1(image))

    return train_step


def style_transfert(content_path, style_path, extractor, optimizers):
    # content_image, style_image = images.load_content_style_img(content_path.as_posix(), style_path.as_posix(), plot_it=True)
    image_couple = images.load_content_style_img(content_path.as_posix(), style_path.as_posix(), plot_it=True)
    image = tf.Variable(image_couple.content_image)

    results_folder = EPath('results') / content_path.stem / style_path.stem
    results_folder.mkdir(exist_ok=True, parents=True)
    train_step = create_train_step(
        extractor=extractor,
        optimizers=optimizers,
        image_couple=image_couple
    )
    bar_epoch = loadbar.ColorBar(color=loadbar.Colors.cyan, max=p.epochs)
    bar_epoch.start()
    for n in range(p.epochs):
        # pb = ProgressBar(max_iteration=(n + 1) * p.steps_per_epoch, title=f'Epoch {n + 1}/{p.epochs}')
        print(f'Epoch {n + 1}/{p.epochs}')
        bar_epoch.update(step=n, end='\n')

        bar_step = loadbar.LoadBar(max=(n + 1) * p.steps_per_epoch)
        bar_step.start()
        for m in range((n + 1) * p.steps_per_epoch):
            train_step(
                image=image,
                content_image=image_couple.content_image,
                style_image=image_couple.style_image
            )
            bar_step.update()
        bar_step.end()
        # plot.clear_output(wait=True)
        plot.display(images.tensor_to_image(image).resize(tuple(s // 2 for s in image_couple.content_nn_shape)))
        file_name = results_folder / f'step_{(n + 1) * (n + 2) * p.steps_per_epoch // 2}.png'
        images.tensor_to_image(image).save(file_name.str)
    bar_epoch.end()
    del image_couple, image, train_step
    gc.collect()
