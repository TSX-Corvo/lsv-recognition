import numpy as np

actions = np.array([
    'aceleracion',
    'calculo',
    'computadora',
    'constante',
    'derivada',
    'ecuacion',
    'fisica',
    'formula',
    'inversa',
    'software',
])

# Thirty videos worth of data
n_sequences = 90

# Videos are going to be 30 frames in length
sequence_length = 29

label_map = {label:num for num, label in enumerate(actions)}