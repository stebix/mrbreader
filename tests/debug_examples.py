import sys
import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tqdm

sys.path.append('C:/Users/Jannik/Desktop/mrbreader')

from reader.utils import (is_binary, is_onehot,
                          convert_to_intlabel,
                          convert_to_onehot)


n_c = (3,)
shape = (10, 10)
array = np.random.default_rng().normal(size=n_c + shape)
argmax_channel_idcs = np.argmax(array, axis=0)
onehot_array = np.zeros(array.shape, dtype=np.int)
for idx in range(array.shape[0]):
    onehot_array[idx, argmax_channel_idcs == idx] = 1
# some sanity checks

assert np.allclose(onehot_array.sum(axis=0), 1)

ilbl = convert_to_intlabel(onehot_array)

fig, axes = plt.subplots(ncols=3)
axes = axes.flat

for idx, ax in enumerate(axes):
    ax.imshow(onehot_array[idx, ...])

fig, ax = plt.subplots()
ax.imshow(ilbl)

plt.show()