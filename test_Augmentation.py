import numpy as np
import pickle
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------
#-------------------------------------------------------------------------------------
def plt_histogram(y_train):

    classes = [np.where(r == 1)[0][0] for r in y_train]
    # plot the number of examples of training
    unique, counts = np.unique(classes, return_counts=True)
    plt.bar(unique, counts, 1 / 1.5, color="green")
    plt.title("Number of samples per class")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.show()
    return counts
#-------------------------------------------------------------------------------------
with open('X_train_Augmented_Final.pickle', 'rb') as handle:
    X_train = pickle.load(handle)
with open('Y_train_Augmented_Final.pickle', 'rb') as handle:
    y_train = pickle.load(handle)

plt_histogram(y_train)

classes = [np.where(r == 1)[0][0] for r in y_train]
# plot the number of examples of training
unique, counts = np.unique(classes, return_counts=True)

classes_array = np.array(classes)
class_id=10
class_indexs = np.argwhere(classes_array == class_id)
for i in class_indexs:

    im = np.uint8((X_train[i, :, :, :] * 128) + 128)
    im = np.squeeze(im, axis=0)
    plt.imshow(im)
    plt.show()