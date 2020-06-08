from epicpath import EPath


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
    def result_stem(self):
        prefix = ''
        if self.is_content(self.start_path):
            prefix = '(content)_'
        elif self.is_style(self.start_path):
            prefix = '(style)_'
        return prefix + self.start_path.rstem




