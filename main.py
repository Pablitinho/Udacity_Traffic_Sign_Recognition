from __future__ import print_function
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Conv_NN as cnn
from pathlib import Path
import Augmentation as augmen
# ------------------------------------------------------------------------------------
def normalize_data(dataset):
  dataset = dataset.astype(np.float32)
  dataset = np.uint8(np.sum(dataset/3, axis=3, keepdims=True))

  dataset = (dataset-128)/128
  return dataset
# ------------------------------------------------------------------------------------
def create_hot_ones(labels, num_labels):
  labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
  return labels
# ------------------------------------------------------------------------------------
# Estimate the accuracy
# ------------------------------------------------------------------------------------
def accuracy(predictions, labels):
  sum_ = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
  return (100.0 * sum_) / predictions.shape[0]
# ------------------------------------------------------------------------------------
# Estimate the accuracy of test
# ------------------------------------------------------------------------------------
def accuracy_test(predictions, labels,size):
  test_sum = np.argmax(predictions, 1)
  test_label = np.argmax(labels, 1)
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / size)
# ----------------------------------------------------------------------------
# Parameters used in this project
# ----------------------------------------------------------------------------
num_epochs = 5000 # num of iterations
num_steps = 20 # each x number of epochs, It will be displayed the loss
batch_size = 140 # Size of the batch
dropout_value = 0.5 # Percentage to apply in the dropout layer
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
unique, counts = np.unique(train['labels'], return_counts=True)
plt.bar(unique, counts, 1/1.5, color="green")
plt.title("Number of samples per class")
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.show()
# --------------------------------------------------
# plot the number of examples of training
unique, counts = np.unique(valid['labels'], return_counts=True)
plt.bar(unique, counts, 1/1.5, color="red")
plt.title("Number of samples per class")
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.show()
# --------------------------------------------------
# plot the number of examples of training
unique, counts = np.unique(test['labels'], return_counts=True)
plt.bar(unique, counts, 1/1.5, color="blue")
plt.title("Number of samples per class")
plt.xlabel("Class")
plt.show()
plt.ylabel("Number of samples")
#--------------------------------------------------
#plt.axis([0, n_classes-1, 0, np.max(hist[0])])

# Plot Randon data
idx = np.random.permutation(train['features'].shape[0])[:5]
for i in idx:
    im = np.uint8((train['features'][i,:,:,:]))
    plt.title(train['labels'][i])
    plt.imshow(im)
    plt.show()

image_shape[2] = 1
# Normalize the images and generate the hot-ones
X_train, y_train = normalize_data(train['features']), create_hot_ones(train['labels'], n_classes)
#X_valid, y_valid = normalize_data(valid['features']), create_hot_ones(valid['labels'], n_classes)
#X_test, y_test = normalize_data(test['features']), create_hot_ones(test['labels'], n_classes)
# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
idx = np.random.permutation(X_train.shape[0])[:5]
for i in idx:
    im = np.uint8((X_train[i,:,:,:]*128)+128)

    img_aug=augmen.rotate_image(im,10);

    im = np.squeeze(im, axis=2)
    img_aug = np.squeeze(img_aug, axis=2)
    f, axarr = plt.subplots(2, 2)
    axarr[0,0].imshow(im)
    axarr[0,1].imshow(img_aug)
    plt.show()
# ----------------------------------------------------------------------------
# Create the graph
# ----------------------------------------------------------------------------
ph_train, ph_train_labels, output_nn, graph_1, loss, train,keep_prob = cnn.generate_graph_cnn(image_shape, n_classes, dropout_value)

Debug=0
# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
with tf.Session(graph=graph_1) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  my_file = Path("./Models/Project2.ckpt.index")
  if my_file.is_file():
     tf.train.Saver().restore(session, "./Models/Project2.ckpt")

  if Debug == 0:
      for epoch in range(num_epochs):
          idx = np.random.permutation(X_train.shape[0])[:batch_size]
          feed_dict = {ph_train: X_train[idx], ph_train_labels: y_train[idx], keep_prob: dropout_value}
          _, l, predictions = session.run([train, loss, output_nn], feed_dict=feed_dict)
          if (epoch % num_steps == 0):
            #idx_validation = np.random.permutation(valid_dataset.shape[0])[:batch_size]
            #_, l, predictions_validation = session.run([optimizer, loss, train_prediction],{tf_train_dataset: valid_dataset[idx_validation], tf_train_labels: valid_labels[idx_validation], keep_prob: 1.0})
            feed_dict = {ph_train: X_train[idx], ph_train_labels: y_train[idx], keep_prob: 1}
            _, l, predictions = session.run([train, loss, output_nn], feed_dict=feed_dict)
            print("Minibatch loss at epoch %d: %f" % (epoch, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, y_train[idx]))
            #print("Validation accuracy: %.1f%%" % accuracy(predictions_validation, valid_labels[idx_validation]))

      tf.train.Saver().save(session, "./Models/Project2.ckpt")

  feed_dict = {ph_train: X_valid, ph_train_labels: y_valid,keep_prob: 1.0}
  _, l, predictions = session.run([train, loss, output_nn], feed_dict=feed_dict)
  print("Minibatch accuracy: %.1f%%" % accuracy(predictions, y_valid))
