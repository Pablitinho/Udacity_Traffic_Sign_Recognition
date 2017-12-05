from PIL import Image
import math
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
import keras as ke

def rotate_image(image, angle):
    return tf.contrib.keras.preprocessing.image.random_rotation(image, angle, row_axis=0, col_axis=1, channel_axis=2)