import Augmentation as augmen
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import time
import cv2
import matplotlib.cm as cm
def augment_brightness_camera_images(image):

    image1 = np.array(image, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:] = image1[:,:]*random_bright
    image1[:,:][image1[:,:]>255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    return image1


def transform_image(image, ang_range, shear_range, trans_range):
    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = image.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, Rot_M, (cols, rows))
    image = cv2.warpAffine(image, Trans_M, (cols, rows))
    image = cv2.warpAffine(image, shear_M, (cols, rows))

    # Brightness augmentation
    image = augment_brightness_camera_images(image)

    return image
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
    return counts
#-------------------------------------------------------------------------------------
with open('X_train_Normalized.pickle', 'rb') as handle:
    X_train = pickle.load(handle)
with open('y_train_Normalized.pickle', 'rb') as handle:
    y_train = pickle.load(handle)

im_idx=1589
im = np.uint8((X_train[im_idx, :, :, :]+0.5) * 255)

# plt.imsave('./images/writeup/before_augmentation.png', np.uint8(np.squeeze(im,axis=2)), cmap=cm.gray)
# img_aug = transform_image(im, 30, 5, 5)
# plt.imsave('./images/writeup/after_augmentation.png', img_aug, cmap=cm.gray)

#plt_histogram(y_train)
# convert hot ones to classes
classes = [np.where(r == 1)[0][0] for r in y_train]
# plot the number of examples of training
unique, counts = np.unique(classes, return_counts=True)

classes_array = np.array(classes)
# with open('X_train_Augmented_Final2.pickle', 'wb') as handle:
#     pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('Y_train_Augmented_Final2.pickle', 'wb') as handle:
#     pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('Y_train_Augmented_Final2.pickle', 'rb') as handle:
#     y_train_A = pickle.load(handle)
#
# plt_histogram(y_train_A)

#counts[0:43] = 0
# # #
idx = np.random.permutation(X_train.shape[0])[:100]
cnt=0

X_train_Augmented = np.array([X_train[0, :, :, :]])  # np.concatenate([[], [X_train[0,:,:,:]]])
y_train_Augmented = np.array([y_train[0]])  # np.concatenate([[], [y_train[0]]])

First_time = True
for class_id in range(43):
    class_indexs = np.argwhere(classes_array == class_id)
    X_train_Augmented = (X_train[class_indexs, :, :, :])  # np.concatenate([[], [X_train[0,:,:,:]]])
    y_train_Augmented = (y_train[class_indexs])  # np.concatenate([[], [y_train[0]]])

    X_train_Augmented = np.squeeze(X_train_Augmented, axis=1)
    y_train_Augmented = np.squeeze(y_train_Augmented, axis=1)
    not_filled = True

    while not_filled is True:
        for id_image in class_indexs:

            if counts[class_id] < 4200:
               First_time = False

               im = np.uint8((X_train[id_image,:,:,:]+0.5)*255)
               im = np.squeeze(im, axis=0)

               if class_id == 8:
                   pp = 0

               img_aug = transform_image(im, 30, 5, 5)
               # img_aug = np.squeeze(img_aug, axis=2)
               img_aug = np.expand_dims(img_aug, axis=2)
               # plt.imshow(img_aug[:,:,0])
               # plt.show()

               X_train_Augmented = np.concatenate([X_train_Augmented, [(img_aug / 255.0)-0.5]])
               y_train_Augmented = np.concatenate([y_train_Augmented, y_train[id_image]])

               counts[class_id] += 1

            else:
               if First_time is True:

                  X = X_train[class_indexs]
                  X = np.squeeze(X, axis=1)
                  X_train_Augmented = X #np.concatenate([X_train_Augmented, [X]])

                  Y = y_train[class_indexs]
                  Y = np.squeeze(Y, axis=1)
                  y_train_Augmented = Y #np.concatenate([y_train_Augmented, [Y]])

               not_filled = False
               print("Filled Class: ", class_id)
               print("Number of elements: ", counts[class_id])

               time.sleep(2)
               First_time = True

               with open('./Sub_Augmentation/X_train_Augmented_sub'+str(class_id)+'.pickle', 'wb') as handle:
                   pickle.dump(X_train_Augmented, handle, protocol=pickle.HIGHEST_PROTOCOL)
               with open('./Sub_Augmentation/Y_train_Augmented_sub'+str(class_id)+'.pickle', 'wb') as handle:
                   pickle.dump(y_train_Augmented, handle, protocol=pickle.HIGHEST_PROTOCOL)

               X_train_Augmented = np.array([X_train[id_image, :, :, :]])
               y_train_Augmented = np.array([y_train[id_image]])

               X_train_Augmented = np.squeeze(X_train_Augmented, axis=1)
               y_train_Augmented = np.squeeze(y_train_Augmented, axis=1)
               break
    #-----------------------------------------------------------------------------------------
    #classes = [np.where(r == 1)[0][0] for r in y_train_Augmented]
    #unique, counts = np.unique(classes, return_counts=True)

# X_train_Augmented_Final = np.concatenate([X_train_Augmented, X_train])
# y_train_Augmented_Final = np.concatenate([y_train_Augmented, y_train])
#
# with open('X_train_Augmented_Final3.pickle', 'wb') as handle:
#     pickle.dump(X_train_Augmented_Final, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('Y_train_Augmented_Final3.pickle', 'wb') as handle:
#     pickle.dump(y_train_Augmented_Final, handle, protocol=pickle.HIGHEST_PROTOCOL)

