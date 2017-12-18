from __future__ import print_function
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Conv_NN as cnn
from pathlib import Path
import logging
import sys
import cv2
import Augmentation as augmen
# ------------------------------------------------------------------------------------
def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

# ------------------------------------------------------------------------------------
def normalize_data(dataset):
  dataset = dataset.astype(np.float32)

  #dataset = np.uint8(np.sum(dataset/3, axis=3, keepdims=True))
  # Equalization
  # for i in range(dataset.shape[0]):
  #     for channel in range(3):
  #         c=dataset[i, :, :, channel]
  #         c=np.uint8(np.reshape(c,(dataset.shape[1],dataset.shape[2],1)))
  #         im =cv2.equalizeHist(c[:,:,0])
  #         dataset[i, :, :,channel] = im
  #grayscale
  dataset = np.uint8(np.sum(dataset / 3, axis=3, keepdims=True))
  for i in range(dataset.shape[0]):
      dataset[i, :, :,0] =  cv2.equalizeHist(dataset[i, :, :,0])

  #normalization
  dataset = (dataset/255)-0.5
  return dataset
# ------------------------------------------------------------------------------------
def create_hot_ones(labels, num_labels):
  labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
  return labels
# ------------------------------------------------------------------------------------
# Estimate the accuracy
# ------------------------------------------------------------------------------------
def get_accuracy(X, Y, session, train, accuracy,labels_pred_softmax,batch_size):

    num_batches = X.shape[0]/batch_size

    final_accuracy = 0
    i_start = 0
    i_end = 0
    accur = 0
    for i in range(np.int(np.ceil(num_batches))):

        if i*batch_size+batch_size < X.shape[0]:
            i_start = i*batch_size
            i_end = i*batch_size+batch_size
        else:
            i_start = i*batch_size
            i_end = X.shape[0]
        x_test= X[i_start:i_end]
        if i == 75:
            pp=0

        x_= X[i_start:i_end]
        y_ = Y[i_start:i_end]
        feed_dict = {ph_train: X[i_start:i_end], ph_train_labels: Y[i_start:i_end], keep_prob: 1.0}
        accur, labels_ = session.run([accuracy, labels_pred_softmax], feed_dict=feed_dict)
        final_accuracy += accur

    return (100.0 * final_accuracy) / np.int(np.ceil(num_batches))
# ------------------------------------------------------------------------------------
def get_accuracy2(X, Y, session, train, accuracy):

    final_accuracy = 0
    for i in range(X.shape[0]):
        x_ = np.reshape(X[i],(1,32,32,3))
        y_ = np.reshape(Y[i], (1, 43))
        feed_dict = {ph_train: x_, ph_train_labels: y_, keep_prob: 1.0}
        _, accur = session.run([train, accuracy], feed_dict=feed_dict)
        final_accuracy += accur

    return 100.0 * (final_accuracy / X.shape[0])
# ----------------------------------------------------------------------------
# Parameters used in this project
# ----------------------------------------------------------------------------
print("Start")
logging.getLogger('tensorflow').disabled = True
# ----------------------------------------------------------------------------
# File names
# ----------------------------------------------------------------------------
training_file = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"
# ----------------------------------------------------------------------------
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# TODO: Number of training examples
n_train = train['features'].shape[0]

# TODO: Number of validation examples
n_validation = valid['features'].shape[0]

# TODO: Number of testing examples.
n_test = test['features'].shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = [train['features'].shape[1], train['features'].shape[2], train['features'].shape[3]]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.max(train['labels'])+1

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# --------------------------------------------------
# plot the number of examples of training
# unique, counts = np.unique(train['labels'], return_counts=True)
# plt.bar(unique, counts, 1/1.5, color="green")
# plt.title("Number of samples per class")
# plt.xlabel("Class")
# plt.ylabel("Number of samples")
# plt.show()
# # --------------------------------------------------
# # plot the number of examples of training
# unique, counts = np.unique(valid['labels'], return_counts=True)
# plt.bar(unique, counts, 1/1.5, color="red")
# plt.title("Number of samples per class")
# plt.xlabel("Class")
# plt.ylabel("Number of samples")
# plt.show()
# # --------------------------------------------------
# # plot the number of examples of training
# unique, counts = np.unique(test['labels'], return_counts=True)
# plt.bar(unique, counts, 1/1.5, color="blue")
# plt.title("Number of samples per class")
# plt.xlabel("Class")
# plt.show()
# plt.ylabel("Number of samples")
#--------------------------------------------------
#plt.axis([0, n_classes-1, 0, np.max(hist[0])])

# Plot Randon data
# idx = np.random.permutation(train['features'].shape[0])[:5]
# for i in idx:
#     im = np.uint8((train['features'][i,:,:,:]))
#     plt.title(train['labels'][i])
#     plt.imshow(im)
#     plt.show()
print("Normalize")
image_shape[2] = 1
# Normalize the images and generate the hot-ones
# X_train, Y_train = normalize_data(train['features']), create_hot_ones(train['labels'], n_classes)
# X_valid, Y_valid = normalize_data(valid['features']), create_hot_ones(valid['labels'], n_classes)
# X_test, Y_test = normalize_data(test['features']), create_hot_ones(test['labels'], n_classes)
# # print("Save")
# with open('X_train_Normalized.pickle', 'wb') as handle:
#      pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('Y_train_Normalized.pickle', 'wb') as handle:
#      pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # #  #
# with open('X_valid_Normalized.pickle', 'wb') as handle:
#      pickle.dump(X_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('Y_valid_Normalized.pickle', 'wb') as handle:
#      pickle.dump(Y_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # # # #
# with open('X_test_Normalized.pickle', 'wb') as handle:
#      pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('Y_test_Normalized.pickle', 'wb') as handle:
#      pickle.dump(Y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the normalized images
with open('X_train_Normalized.pickle', 'rb') as handle:
     X_train = pickle.load(handle)
with open('Y_train_Normalized.pickle', 'rb') as handle:
     Y_train = pickle.load(handle)
#
with open('X_valid_Normalized.pickle', 'rb') as handle:
     X_valid = pickle.load(handle)
with open('Y_valid_Normalized.pickle', 'rb') as handle:
     Y_valid = pickle.load(handle)

with open('X_test_Normalized.pickle', 'rb') as handle:
     X_test = pickle.load(handle)
with open('Y_test_Normalized.pickle', 'rb') as handle:
     Y_test = pickle.load(handle)

print("Done")
# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
# idx = np.random.permutation(X_valid.shape[0])[:115]
# for i in idx:
#     im = np.uint8((X_valid[i,:,:,:]*128)+128)
#     im = np.squeeze(im, axis=2)
#     plt.imshow(im)
#     plt.show()
# ----------------------------------------------------------------------------
# Augmentation example
# ----------------------------------------------------------------------------
# idx = np.random.permutation(X_train.shape[0])[:5]
# for i in idx:
#     im = np.uint8((X_train[i,:,:,:]*128)+128)
#     img_aug=augmen.rotate_image(im,10);
#     img_aug = augmen.shift_image(img_aug, 0.1,0.1)
#     img_aug = augmen.shear_image(img_aug, 0.1)
#     img_aug = augmen.zoom_image(img_aug, [0.9, 0.9])
#     im = np.squeeze(im, axis=2)
#     img_aug = np.squeeze(img_aug, axis=2)
#     f, axarr = plt.subplots(2, 2)
#     axarr[0,0].imshow(im)
#     axarr[0,1].imshow(img_aug)
#     plt.show()


# print('Loading Augmented data ...')
with open('X_train_Augmented_Final_Shuffle.pickle', 'rb') as handle:
     X_train = pickle.load(handle)
# # #
with open('Y_train_Augmented_Final_Shuffle.pickle', 'rb') as handle:
     Y_train = pickle.load(handle)


# for i in range(18):
#     im = np.uint8((X_train_Augmented_1[i,:,:,:]*128)+128)
#     im = np.squeeze(im, axis=2)
#     plt.imshow(im)
#     plt.show()

print('Augmented data loaded')
# ----------------------------------------------------------------------------
# Parameters used in this project
# ----------------------------------------------------------------------------
num_epochs = 14 # num of iterations
num_iters = 1000
num_steps = 100 # each x number of epochs, It will be displayed the loss
batch_size = 128 # Size of the batch
dropout_value = 0.5 # Percentage  to apply in the dropout layer
# ----------------------------------------------------------------------------
# Create the graph
# ----------------------------------------------------------------------------
ph_train, ph_train_labels, output_nn, graph_1, loss, train,keep_prob,accuracy_op,labels_pred_softmax,layer_3 = cnn.generate_graph_cnn(image_shape, n_classes)

Debug = 0
tf.logging.set_verbosity(tf.logging.INFO)
cnt = 0
# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
with tf.Session(graph=graph_1) as session:
  tf.global_variables_initializer().run()
  merged_summary = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter('./Summary', graph_1)
  print('Initialized')
  my_file = Path("./Models/Project2.ckpt.index")
  #if my_file.is_file():
  #    tf.train.Saver().restore(session, "./Models/Project2.ckpt")

  shape_X = X_train.shape
  if Debug == 0:
      for epoch in range(num_epochs):
          # for iter in range(num_iters):
          for offset in range(0, X_train.shape[0], batch_size):
              end = offset + batch_size
              X_Batch, Y_Batch = X_train[offset:end], Y_train[offset:end]
              # idx = np.random.permutation(X_train.shape[0])[:batch_size]
              feed_dict = {ph_train: X_Batch, ph_train_labels: Y_Batch, keep_prob: dropout_value}
              # feed_dict = {ph_train: X_train[idx], ph_train_labels: Y_train[idx], keep_prob: dropout_value}
              _, l, predictions = session.run([train, loss, output_nn], feed_dict=feed_dict)

              if offset % (X_train.shape[0] / 2) == 0:
                  feed_dict = {ph_train: X_Batch, ph_train_labels: Y_Batch, keep_prob: 1}
                  # feed_dict = {ph_train: X_train[idx], ph_train_labels: Y_train[idx], keep_prob: 1}
                  summary, _, l, predictions, accur = session.run(
                      [merged_summary, train, loss, output_nn, accuracy_op], feed_dict=feed_dict)
                  summary_writer.add_summary(summary, epoch)

          print("EPOCH: ", epoch)
          print("Train accuracy: %.1f%%" % get_accuracy(X_train, Y_train, session, train, accuracy_op,
                                                        labels_pred_softmax, batch_size))
          print("Valid accuracy: %.1f%%" % get_accuracy(X_valid, Y_valid, session, train, accuracy_op,
                                                        labels_pred_softmax, batch_size))
          print("Test accuracy: %.1f%%" % get_accuracy(X_test, Y_test, session, train, accuracy_op,
                                                        labels_pred_softmax, batch_size))

          tf.train.Saver().save(session, "./Models/Project2.ckpt")
      summary_writer.close()

  print("Train accuracy: %.1f%%" % get_accuracy(X_train, Y_train, session, train, accuracy_op, labels_pred_softmax,
                                                batch_size))
  print("Valid accuracy: %.1f%%" % get_accuracy(X_valid, Y_valid, session, train, accuracy_op, labels_pred_softmax,
                                                batch_size))
  print("Test accuracy: %.1f%%" % get_accuracy(X_test, Y_test, session, train, accuracy_op, labels_pred_softmax,
                                                batch_size))
# feed_dict = {ph_train: X_valid, ph_train_labels: Y_valid, keep_prob: 1}
# accur,labels_pred_softmax = session.run([accuracy_op,labels_pred_softmax], feed_dict = feed_dict)
# accur*=100
# print("Valid accuracy: %.1f%%" % accur)

classes_gt = [np.where(r == 1)[0][0] for r in Y_valid]
print("Classes GT", classes_gt)

classes_validation = [np.where(np.max(r))[0][0] for r in labels_pred_softmax]
print("Classes Validation", classes_validation)

tf.contrib.metrics.confusion_matrix(classes_gt,classes_validation)
