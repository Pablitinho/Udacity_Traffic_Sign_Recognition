import numpy as np
import pickle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------
i=0
with open('Sub_Augmentation/X_train_Augmented_sub' + str(i) + '.pickle', 'rb') as handle:
    X_train_Augmentation = pickle.load(handle)
with open('Sub_Augmentation/Y_train_Augmented_sub' + str(i) + '.pickle', 'rb') as handle:
    Y_train_Augmentation = pickle.load(handle)

for i in range(1,43):
    print("Subset: ",i)
    with open('Sub_Augmentation/X_train_Augmented_sub' + str(i) + '.pickle', 'rb') as handle:
        X_train = pickle.load(handle)
    with open('Sub_Augmentation/Y_train_Augmented_sub' + str(i) + '.pickle', 'rb') as handle:
        y_train = pickle.load(handle)

    X_train_Augmentation = np.concatenate([X_train_Augmentation, X_train])
    Y_train_Augmentation = np.concatenate([Y_train_Augmentation, y_train])


with open('X_train_Augmented_Final.pickle', 'wb') as handle:
     pickle.dump(X_train_Augmentation, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Y_train_Augmented_Final.pickle', 'wb') as handle:
    pickle.dump(Y_train_Augmentation, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Shuffle the data
X_train_Augmentation, Y_train_Augmentation = shuffle(X_train_Augmentation, Y_train_Augmentation)

with open('X_train_Augmented_Final_Shuffle.pickle', 'wb') as handle:
    pickle.dump(X_train_Augmentation, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Y_train_Augmented_Final_Shuffle.pickle', 'wb') as handle:
    pickle.dump(Y_train_Augmentation, handle, protocol=pickle.HIGHEST_PROTOCOL)