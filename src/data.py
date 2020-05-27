from zipfile import ZipFile
from epicpath import EPath
import tensorflow as tf
import shutil


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
        else:
            cp.rmdir()
            sp.rmdir()
    EPath('content').mkdir()
    EPath('style').mkdir()
    if EPath('content.zip').exists() and EPath('style.zip').exists():
        # the zip files are provided
        with ZipFile('content.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
        with ZipFile('style.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
    else:
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




