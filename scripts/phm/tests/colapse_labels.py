import numpy as np
import torch
def collapse_to_interval(value, lower_bound, upper_bound):
    return torch.round(torch.clamp(value, lower_bound, upper_bound))



# Tu vector original
original_vector = [-1/2, 1.2, 3.2, 4.1, 5.4, 6.5000000001, 12]

# Definir el intervalo [0, 10]
lower_bound = 0
upper_bound = 10

# Aplicar la función de mapeo a cada elemento del vector
collapsed_vector = collapse_to_interval(torch.tensor(original_vector), lower_bound, upper_bound)

# Pasar a entero al entero más cercano
print(collapsed_vector)


print(np.round(-0.499) == 0)