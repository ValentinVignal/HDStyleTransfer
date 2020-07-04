class Parameter:
    def __init__(self, name, default_value):
        self.default_value = default_value
        self.name = name
        self.values = []
        self._n = 0

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        assert(n < self.length)
        self._n = n

    @property
    def length(self):
        return int(max(len(self.values), 1))

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if len(self.values) == 0:
            return self.default_value
        return self.values[item]

    @property
    def value(self):
        if len(self.values) == 0:
            return self.default_value
        return self.values[self._n]

    def update(self, updated_value):
        self.default_value = updated_value
        self.values = []

    def set_grid_values(self, grid_values):
        """

        :param grid_values: Already a list of values
        :return:
        """
        self.values = grid_values

    def append_list_value(self, value):
        if value is None:
            self.values.append(self.default_value)
        else:
            self.values.append(value)

    @property
    def num(self):
        return len(self.value)

    def is_constant(self):
        if self.length == 1:
            return True
        first_value = self.values[0]
        for value in self.values[1:]:
            if value != first_value:
                return False
        return True




