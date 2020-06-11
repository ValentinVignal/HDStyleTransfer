from epicpath import EPath
from .. import variables as var


class FileCombination:
    def __init__(self, content_path, style_path, start_path=None):
        self.content_path = EPath(content_path)
        self.style_path = EPath(style_path)
        self.start_path = self.content_path if start_path is None else EPath(start_path)

    @property
    def results_folder(self):
        return EPath('results') / self.content_path.rstem / self.style_path.rstem

    @staticmethod
    def is_content(path):
        return EPath(path).parent.stem == 'content'

    @staticmethod
    def is_style(path):
        return EPath(path).parent.stem == 'style'

    @property
    def is_start_content(self):
        return self.start_path == self.content_path

    @property
    def result_stem(self):
        if var.use_tf_hub:
            return 'hub'
        prefix = ''
        if self.is_content(self.start_path):
            prefix = '(content)_'
        elif self.is_style(self.start_path):
            prefix = '(style)_'
        return prefix + self.start_path.rstem




