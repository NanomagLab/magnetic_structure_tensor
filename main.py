import numpy as np
import matplotlib.pyplot as plt
import module as m
import tensorflow as tf
from module import *

print(tf.__version__)

examples = np.load("texture_field_examples.npy")
antiskyrmion_components = spin_to_antiskyrmion_components(examples)
for i in range(examples.shape[0]):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(spin_to_rgb(examples[i]))
    axes[1].imshow(xy_to_rgb(antiskyrmion_components[i]))
    plt.show()
