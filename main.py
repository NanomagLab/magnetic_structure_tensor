import numpy as np
import matplotlib.pyplot as plt
import module as m
import tensorflow as tf
import os
from module import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

examples = make_examples()
np.save("texture_field_examples.npy", np.array(examples))
examples = np.load("texture_field_examples.npy")

antiskyrmion_components = spin_to_antiskyrmion_components(examples)
defects = antiskyrmion_components_to_defects(antiskyrmion_components)
for i in range(examples.shape[0]):
    fig, axes = plt.subplots(nrows=1, ncols=3)
    axes[0].imshow(spin_to_rgb(examples[i]))
    axes[1].imshow(xy_to_rgb(antiskyrmion_components[i]))
    plt.show()
