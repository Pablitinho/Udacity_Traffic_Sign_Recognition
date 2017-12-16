import Augmentation as augmen
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
#-------------------------------------------------------------------------------------
# Plot Histogram
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
#-------------------------------------------------------------------------------------
with open('X_train_Normalized.pickle', 'rb') as handle:
    X_train = pickle.load(handle)
with open('y_train_Normalized.pickle', 'rb') as handle:
    y_train = pickle.load(handle)

plt_histogram(y_train)
# convert hot ones to classes

X_train_Augmented = X_train
y_train_Augmented = y_train

with open('Y_train_Augmented_Final2.pickle', 'rb') as handle:
    y_train_A = pickle.load(handle)

plt_histogram(y_train_A)
for iter in range(150):
    # X_train_Augmented = [X_train[0,:,:,:]] #np.concatenate([[], [X_train[0,:,:,:]]])
    # y_train_Augmented = [y_train[0]] # np.concatenate([[], [y_train[0]]])
    # # #
    idx = np.random.permutation(X_train.shape[0])[:100]
    cnt=0
    for i in idx:
        class_= np.argwhere(y_train[i] == 1)
        if counts[class_]<1900:
            print("idx %", i)
            print("cnt %",cnt)
            cnt+=1

            im = np.uint8((X_train[i,:,:,:]*128)+128)

            img_aug=augmen.rotate_image(im, 20)
            X_train_Augmented = np.concatenate([X_train_Augmented, [(img_aug-128)/128]])#np.append(X_train, np.atleast_3d((img_aug-128)/128), axis=0)
            y_train_Augmented = np.concatenate([y_train_Augmented, [y_train[i]]])

            img_aug = augmen.shear_image(img_aug, 0.1)
            X_train_Augmented = np.concatenate([X_train_Augmented, [(img_aug-128)/128]])#np.append(X_train, np.atleast_3d((img_aug-128)/128), axis=0)
            y_train_Augmented = np.concatenate([y_train_Augmented, [y_train[i]]])

            img_aug = augmen.shift_image(img_aug, 0.05,0.05)
            X_train_Augmented = np.concatenate([X_train_Augmented, [(img_aug-128)/128]])#np.append(X_train, np.atleast_3d((img_aug-128)/128), axis=0)
            y_train_Augmented = np.concatenate([y_train_Augmented, [y_train[i]]])
        # # #
        #     img_aug = augmen.zoom_image(img_aug, [0.9, 0.9])
        #     X_train_Augmented = np.concatenate([X_train_Augmented, [(img_aug-128)/128]])#np.append(X_train, np.atleast_3d((img_aug-128)/128), axis=0)
        #     y_train_Augmented = np.concatenate([y_train_Augmented, [y_train[i]]])
            counts[classes[i]]+=3
            # plt.imshow(img_aug)
            # plt.show()

    #-----------------------------------------------------------------------------------------
    #classes = [np.where(r == 1)[0][0] for r in y_train_Augmented]
    #unique, counts = np.unique(classes, return_counts=True)

    my_file = Path("X_train_Augmented_Final2.pickle")
    if my_file.is_file():
        with open('X_train_Augmented_Final2.pickle', 'rb') as handle:
            X_train_Final = pickle.load(handle)
        with open('Y_train_Augmented_Final2.pickle', 'rb') as handle:
            y_train_Final = pickle.load(handle)

        X_train_Augmented_Final = np.concatenate([X_train_Augmented, X_train_Final])
        y_train_Augmented_Final = np.concatenate([y_train_Augmented, y_train_Final])

        with open('X_train_Augmented_Final2.pickle', 'wb') as handle:
            pickle.dump(X_train_Augmented_Final, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Y_train_Augmented_Final2.pickle', 'wb') as handle:
            pickle.dump(y_train_Augmented_Final, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('X_train_Augmented_Final2.pickle', 'wb') as handle:
            pickle.dump(X_train_Augmented, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Y_train_Augmented_Final2.pickle', 'wb') as handle:
            pickle.dump(y_train_Augmented, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #-----------------------------------------------------------------------------------------
    # Temporal
    with open('X_train_Augmented_tmp.pickle', 'wb') as handle:
         pickle.dump(X_train_Augmented, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Y_train_Augmented_tmp.pickle', 'wb') as handle:
         pickle.dump(y_train_Augmented, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #-----------------------------------------------------------------------------------------
