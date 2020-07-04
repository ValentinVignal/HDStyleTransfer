def is_defined(dictionary, key):
    """

    :param dictionary:
    :return: boolean true if key is in dict and not None
    """
    return key in dictionary and dictionary[key] is not None


def get_key(dictionary, key, default=None):
    """

    :param default:
    :param dictionary:
    :param key:
    :return:
    """
    if is_defined(dictionary, key):
        return dictionary[key]
    else:
        return default


