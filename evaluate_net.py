import Conv_NN as cnn
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv
import matplotlib.gridspec as gridspec
# ----------------------------------------------------------------------------
def normalize_data(dataset):
  dataset = dataset.astype(np.float32)

  #grayscale
  dataset = np.uint8(np.sum(dataset / 3, axis=3, keepdims=True))
  for i in range(dataset.shape[0]):
      dataset[i, :, :,0] =  cv2.equalizeHist(dataset[i, :, :,0])

  #normalization
  dataset = (dataset/255)-0.5
  return dataset
# ----------------------------------------------------------------------------
def print_prob_im(im_color,top_5):
    plt.figure(figsize=(5, 1.5))
    gridsp = gridspec.GridSpec(1, 2, width_ratios=[2, 3])
    plt.subplot(gridsp[0])
    plt.imshow(im_color)
    plt.axis('off')
    plt.subplot(gridsp[1])
    list_prob=np.array(top_5[0][0][:])
    print("Prob: ", list_prob[:])
    plt.barh(6 - np.arange(5), list_prob[:], align='center')
    for i_label in range(5):
        plt.text(top_5[0][0][i_label] + .02, 6 - i_label - .25, sign_names[top_5[1][0][i_label]+1,1])
    plt.axis('off');
    plt.show();
# ----------------------------------------------------------------------------

with open('./signnames.csv') as f:
    reader = csv.reader(f)
    sign_names = list(reader)
sign_names = np.array(sign_names)
max_score=14

image_shape=[32,32,1]
n_classes=43

image_list =['./images/stop_signal_small.bmp','./images/120_small.bmp','./images/general_caution_small.bmp','./images/Turn_right_ahead_small.bmp','./images/priority_road_small.bmp','./images/no_entry_small.bmp','./images/no_passing_small.bmp','./images/keep_right_small.bmp']

ph_train, ph_train_labels, output_nn, graph_1, loss, train,keep_prob,accuracy_op,labels_pred_softmax = cnn.generate_graph_cnn(image_shape, n_classes)

with tf.Session(graph=graph_1) as session:
    tf.global_variables_initializer().run()
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./Summary', graph_1)
    print('Initialized')
    my_file = Path("./Models/Project2.ckpt.index")
    if my_file.is_file():
        tf.train.Saver().restore(session, "./Models/Project2.ckpt")
    # -------------------------------------------------------
    for im_name in image_list:
        im_color = plt.imread(im_name)

        im = np.expand_dims(im_color, axis=0)
        im = normalize_data(im)
        # -------------------------------------------------------
        feed_dict = {ph_train: im, keep_prob: 1.0}
        predictions = session.run(labels_pred_softmax, feed_dict=feed_dict)
        top_5 = session.run(tf.nn.top_k(tf.constant(predictions), k=5))

        print_prob_im(im_color, top_5)
    # plt.figure(figsize=(5, 1.5))
    # gridsp = gridspec.GridSpec(1, 2, width_ratios=[2, 3])
    # plt.subplot(gridsp[0])
    # plt.imshow(im_color)
    # plt.axis('off')
    # plt.subplot(gridsp[1])
    # list_prob=np.array(top_5[0][0][:])
    # print("Prob: ", list_prob[:])
    # plt.barh(6 - np.arange(5), list_prob[:], align='center')
    # for i_label in range(5):
    #     plt.text(top_5[0][0][i_label] + .02, 6 - i_label - .25, sign_names[top_5[1][0][i_label]+1,1])
    # plt.axis('off');
    # plt.show();
