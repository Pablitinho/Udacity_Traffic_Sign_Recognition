from PIL import Image
import math
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
# ----------------------------------------------------------------------------
def rotate_image(image, angle):
    return tf.contrib.keras.preprocessing.image.random_rotation(image, angle, row_axis=0, col_axis=1, channel_axis=2)
# ----------------------------------------------------------------------------
def shift_image(image, shift_x,shift_y):
    return tf.contrib.keras.preprocessing.image.random_shift(image, shift_y, shift_x, row_axis=0, col_axis=1, channel_axis=2)
# ----------------------------------------------------------------------------
def zoom_image(image,zoom_factor):
    return tf.contrib.keras.preprocessing.image.random_zoom(image, zoom_factor, row_axis=0, col_axis=1, channel_axis=2)
# ----------------------------------------------------------------------------
def shear_image(image,factor):
    return tf.contrib.keras.preprocessing.image.random_shear(image, factor, row_axis=0, col_axis=1, channel_axis=2)
# ----------------------------------------------------------------------------
def shift_image(image,x_shift,y_shift):
    return tf.contrib.keras.preprocessing.image.random_shift(image, x_shift,y_shift, row_axis=0, col_axis=1, channel_axis=2)
