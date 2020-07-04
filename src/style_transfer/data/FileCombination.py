from epicpath import EPath
from ..variables import options
from ..variables import param
from ..STMode import STMode


class FileCombination:
    def __init__(self, content_path, style_path, start_path=None, n=0):
        self.content_path = EPath(content_path)
        self.style_path = EPath(style_path)
        self.start_path = self.content_path if start_path is None else EPath(start_path)
        self.n = n

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
        if options.st_mode == STMode.Hub.value:
            return 'hub'
        prefix = ''
        if len(param) > 1:
            prefix += f'p{self.n}_'
        if self.is_content(self.start_path):
            prefix += '(content)_'
        elif self.is_style(self.start_path):
            prefix += '(style)_'
        return prefix + self.start_path.rstem




