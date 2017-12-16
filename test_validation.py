import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('X_valid_Normalized.pickle', 'rb') as handle:
    X_valid = pickle.load(handle)
with open('y_valid_Normalized.pickle', 'rb') as handle:
    y_valid = pickle.load(handle)

for i in range(220,260):

    im = np.uint8((X_valid[i, :, :, :] * 128) + 128)
    plt.imshow(im)
    plt.show()