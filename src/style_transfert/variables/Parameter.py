class Parameter:
    def __init__(self, name, default_value):
        self.name = name
        self.values = [default_value]
        self._grid_p = 0

    @property
    def grid_p(self):
        return self._grid_p

    @grid_p.setter
    def grid_p(self, grid_p):
        assert(grid_p < self.length)
        self._grid_p = grid_p

    @property
    def length(self):
        return len(self.values)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.values[item]

    @property
    def value(self):
        return self.values[self._grid_p]

    def update(self, updated_value):
        self.values = [updated_value]

    def set_grid_values(self, grid_values):
        """

        :param grid_values: Already a list of values
        :return:
        """
        self.values = grid_values

    @property
    def num(self):
        return len(self.value)

