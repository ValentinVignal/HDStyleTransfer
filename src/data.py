from zipfile import ZipFile
from epicpath import EPath
import tensorflow as tf
import platform
import os


def get_data():
    """
    Construct the data files
    :return: 2 List<EP>: files of content and style
    """
    EPath('content').mkdir()
    EPath('style').mkdir()
    if EPath('content.zip').exists() and EPath('style.zip').exists():
        # the zip files are provided
        with ZipFile('content.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
        with ZipFile('style.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
        content_path_list = EPath('content').listdir(concat=True)
        style_path_list = EPath('style').listdir(concat=True)
    else:
        # no image provided
        content_path_list = tf.keras.utils.get_file(
            'YellowLabradorLooking_new.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
        )
        style_path_list = tf.keras.utils.get_file(
            'kandinsky5.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
        )
        if platform.system() == 'Windows':
            # My pc
            # TODO: find where is goes
            os.rename('/root/.keras/datasets/YellowLabradorLooking_new.jpg', 'content/YellowLabradorLooking_new.jpg')
            os.rename('/root/.keras/datasets/kandinsky5.jpg', 'style/kandinsky5.jpg')
        else:
            # On colab
            os.rename('/root/.keras/datasets/YellowLabradorLooking_new.jpg', 'content/YellowLabradorLooking_new.jpg')
            os.rename('/root/.keras/datasets/kandinsky5.jpg', 'style/kandinsky5.jpg')
    return content_path_list, style_path_list


def get_next_files(content_path_list, style_path_list):
    """

    :param content_path_list: List of the content_path
    :param style_path_list: List of the style_path
    :return:
    """
    for content_path in content_path_list:
        for style_path in style_path_list:
            result_path = EPath('results', content_path.stem, style_path.stem)
            if not result_path.exists():
                return content_path, style_path, result_path
    return None, None, None




