from enum import Enum

from .Parameter import Parameter


class ParametersManager:
    def __init__(self):
        self._n = 0
        self.parameters_manager_mode = ParametersManagerModes.List
        self._parameters = dict()

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        assert(n < self.length)
        if self.parameters_manager_mode == ParametersManagerModes.List:
            self._n = n
            for key, value in self._parameters.items():
                value.n = n
        elif self.parameters_manager_mode == ParametersManagerModes.Grid:
            # Set to 0
            self._n = n
            for i in range(n):
                # Increment 1 grip_p times
                for key, value in self._parameters.items():
                    if not value.length == 1:
                        # This variable can change
                        if value.n + 1 == value.length:
                            value.n = 0
                        else:
                            # I only have to increment this value
                            value.n = value.n + 1
                            break

    def __setattr__(self, key, value):
        if key not in ['_parameters', 'n', '_n', 'parameters_manager_mode']:
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
        if self.parameters_manager_mode is ParametersManagerModes.List:
            for key, value in self._parameters.items():
                return len(value)
        elif self.parameters_manager_mode is ParametersManagerModes.Grid:
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

    def append_list_value(self, obj):
        for key, value in self._parameters.items():
            if key in obj:
                value.append_list_value(obj[key])
            else:
                value.append_list_value(value[-1])

    def num(self, key):
        return self._parameters[key].num()

    def save_all_txt(self, path):
        s = '\t\tParameters\n\n'
        s += 'Constant parameters:\n'
        for key, value in self._parameters.items():
            if value.is_constant():
                s += f'\t{key}: {value.value}\n'
        s += 'Moving Parameters:\n'
        for key, value in self._parameters.items():
            if not value.is_constant():
                s += f'\t{key}: {value.values}\n'
        with open(path, 'w') as file:
            file.write(s)

    def save_current_txt(self, path):
        s = f'\t\tParameters {self.n}\n\n'
        for key, value in self._parameters.items():
            s += f'\t{key}: {value.value}\n'
        with open(path, 'w') as file:
            file.write(s)

    def list_mode(self):
        self.parameters_manager_mode = ParametersManagerModes.List

    def grid_mode(self):
        self.parameters_manager_mode = ParametersManagerModes.Grid


class ParametersManagerModes(Enum):

    List = 0
    Grid = 1
