from .Parameter import Parameter


class ParametersManager:
    def __init__(self):
        self._parameters = dict()
        self._grid_p = 0

    @property
    def grid_p(self):
        return self._grid_p

    @grid_p.setter
    def grid_p(self, grid_p):
        assert(grid_p < self.length)
        self._grid_p = grid_p
        # Set to 0
        for i in range(grid_p):
            # Increment 1 grip_p times
            for key, value in self._parameters.items():
                if not value.length == 1:
                    # This variable can change
                    if value.grid_p + 1 == value.length:
                        value.grid_p = 0
                    else:
                        # I only have to increment this value
                        value.grid_p = value.grid_p + 1
                        break

    def __setattr__(self, key, value):
        if key not in ['_parameters', '_grid_p', 'grid_p']:
            self.add(key, value)
        else:
            object.__setattr__(self, key, value)

    def __getitem__(self, item):
        return self._parameters[item]

    def __getattr__(self, item):
        parameter_objet = object.__getattribute__(self, '_parameters')
        if item in parameter_objet:
            return parameter_objet[item]
        else:
            return object.__getattribute__(self, item)

    def add(self, name, default_value):
        self._parameters[name] = Parameter(name, default_value)

    @property
    def length(self):
        l = 1
        for key, value in self._parameters.items():
            l *= len(value)
        return l

    def __len__(self):
        return self.length

    def update(self, key, value):
        self._parameters[key].update(value)

    def set_grid_values(self, key, values):
        self._parameters[key].set_grid_values(values)

    def num(self, key):
        return self._parameters[key].num()

    def save_all_txt(self, path):
        s = '\t\tParameters\n\n'
        s += 'Constant parameters:\n'
        for key, value in self._parameters.items():
            if value.length == 1:
                s += f'\t{key}: {value.value}\n'
        s += 'Moving Parameters:\n'
        for key, value in self._parameters.items():
            if value.length > 1:
                s += f'\t{key}: {value.values}\n'
        with open(path, 'w') as file:
            file.write(s)

    def save_current_txt(self, path):
        s = f'\t\tParameters {self.grid_p}\n\n'
        for key, value in self._parameters.items():
            s += f'\t{key}: {value.value}\n'
        with open(path, 'w') as file:
            file.write(s)


