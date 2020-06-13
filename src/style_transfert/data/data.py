from zipfile import ZipFile
from epicpath import EPath
import tensorflow as tf
import shutil
import functools

from .. import variables as var
from .FileCombination import FileCombination
from ..STMode import STMode


def extract_data():
    """
    Construct the data files
    :return: 2 List<EP>: files of content and style
    """

    cp = EPath('content')
    sp = EPath('style')
    if cp.exists() and sp.exists():
        # Check if I have to reconstruct it or not
        cp_list = cp.listdir(concat=True)
        sp_list = sp.listdir(concat=True)
        if len(cp_list) > 0 and len(sp_list) > 0:
            # Data is already there
            return cp_list, sp_list
        else:
            # Remove everything to do it again
            cp.rmdir()
            sp.rmdir()

    EPath('content').mkdir()
    EPath('style').mkdir()
    possible_parents = ['.', '..']  # The working folder (my computer) or the parent folder (colab)
    data_found = False
    for possible_parent in possible_parents:
        cp_zip = EPath(possible_parent, 'content.zip')
        sp_zip = EPath(possible_parent, 'style.zip')
        if cp_zip.exists() and sp_zip.exists() and not data_found:
            data_found = True
            # the zip files are provided in the project folder
            with ZipFile(cp_zip, 'r') as zip_ref:
                zip_ref.extractall('./')
            with ZipFile(sp_zip, 'r') as zip_ref:
                zip_ref.extractall('./')
    if not data_found:
        # no image provided
        cp = tf.keras.utils.get_file(
            'YellowLabradorLooking_new.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
        )
        sp = tf.keras.utils.get_file(
            'kandinsky5.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
        )
        shutil.move(cp, 'content/YellowLabradorLooking_new.jpg')
        shutil.move(sp, 'style/kandinsky5.jpg')
    content_path_list = EPath('content').listdir(concat=True)
    style_path_list = EPath('style').listdir(concat=True)
    return content_path_list, style_path_list


def get_data():
    """
    Construct the data files
    :return: 2 List<EP>: files of content and style
    """

    cp = EPath('content')
    sp = EPath('style')
    if cp.exists() and sp.exists():
        cp_list = cp.listdir(concat=True)
        sp_list = sp.listdir(concat=True)
        if len(cp_list) > 0 and len(sp_list) > 0:
            # Data is already there
            return cp_list, sp_list
    return extract_data()


def get_nb_combinations():
    content_list, style_list = get_data()
    nb_combinations = len(content_list) * len(style_list)
    if var.st_mode == STMode.Hub:
        return nb_combinations
    num_image_start = get_num_image_start(
        num_content=len(content_list),
        num_style=len(style_list),
        image_start_list_option=var.image_start
    )
    return nb_combinations * num_image_start


def get_num_image_start(num_content, num_style, image_start_list_option=var.image_start):
    if var.st_mode == STMode.Hub:
        return 1
    if 'all' in image_start_list_option:
        return num_content * num_style
    else:
        num_image_start = 0
        if 'all_content' in image_start_list_option:
            num_image_start += num_content
        elif 'content' in image_start_list_option:
            num_image_start += 1
        if 'all_style' in image_start_list_option:
            num_image_start += num_style
        elif 'style' in image_start_list_option:
            num_image_start += 1
        return num_image_start


def get_start_path_list(content_path, style_path, image_start=var.image_start, data_path=None):
    """

    :param content_path:
    :param style_path:
    :param image_start:
    :return: The list of image to start style transfert from
    """
    if var.st_mode == STMode.Hub:
        return [content_path]
    all_content = 'all' in image_start or 'all_content' in image_start
    all_style = 'all' in image_start or 'all_style' in image_start
    if (all_content or all_style) and data_path is None:
        data_path = get_data()  # content_path_list, style_path_list
    start_path_list = []
    # Content
    if all_content:
        start_path_list.extend(data_path[0])
    elif 'content' in image_start:
        # try only the first image
        start_path_list.append(content_path)
    if all_style:
        start_path_list.extend(data_path[1])
    elif 'style' in image_start:
        start_path_list.append(style_path)
    return start_path_list


def get_next_files(content_path_list, style_path_list, image_start=var.image_start, data_path=None):
    """

    :param content_path_list: List of the content_path
    :param style_path_list: List of the style_path
    :return:
    """

    for content_path in content_path_list:
        for style_path in style_path_list:
            result_path = EPath('results', content_path.stem, style_path.stem)
            start_path_list = get_start_path_list(
                content_path=content_path,
                style_path=style_path,
                image_start=image_start,
                data_path=data_path
            )
            if not result_path.exists():
                return FileCombination(
                    content_path=content_path,
                    style_path=style_path,
                    start_path=start_path_list[0]
                )
            else:
                files = result_path.listdir(t='str')  # existing files in the result folder
                for start_path in start_path_list:
                    file_combination = FileCombination(
                        content_path=content_path,
                        style_path=style_path,
                        start_path=start_path
                    )
                    if not functools.reduce(lambda x, y: x or y.startswith(file_combination.result_stem), files, False):
                        return file_combination
    return None
