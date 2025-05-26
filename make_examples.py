import numpy as np
from module import get_initial_conditions, greedy_algorithm

initial_conditions = get_initial_conditions()
examples = greedy_algorithm(initial_conditions, 100000)
np.save("texture_field/examples.npy", examples)