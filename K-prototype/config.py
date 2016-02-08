class config:
    attributes = ['float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                  'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']


def get_arrays():
        numbers = []
        strings = []
        for i in range(len(config.attributes)):
            if config.attributes[i] == 'float':
                numbers.append(i)
            elif config.attributes[i] == 'string':
                strings.append(i)

        return [numbers, strings]

