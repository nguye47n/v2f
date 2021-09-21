from skimage import io, img_as_float
import numpy as np
import os
from natsort import natsorted
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from defines import *
from tensorflow.keras.applications.vgg16 import preprocess_input


# load images with skimage and rescale them using img_as_float
# append them to np array and split array according to train/dev/test split ratio

def load_dataset(img_path, lbl_path):
    '''
    Load images in .png format and labels from a .npy file.
    '''
    # load images and labels from path
    images = []
    image_files = natsorted(os.listdir(img_path))
    for file in image_files:
        path = os.path.join(img_path, file)
        image = io.imread(path)
        #image = img_as_float(image)
        images.append(image)

    labels = np.load(lbl_path, allow_pickle=True)

    # use vgg16.preprocess_input() on images per keras' recommendation
    images = np.array(images)
    preprocess_input(images)

    # split data into train/dev/test (0.64/0.16/0.2)
    num_images = len(images)
    train_images = np.array(images[ : math.floor(0.64 * num_images)])
    val_images = np.array(images[math.floor(0.64 * num_images) : math.floor(0.8 * num_images)])
    test_images = np.array(images[math.floor(0.8 * num_images) : ])

    train_labels = np.array(labels[ : math.floor(0.64 * num_images)])
    val_labels = np.array(labels[math.floor(0.64 * num_images) : math.floor(0.8 * num_images)])
    test_labels = np.array(labels[math.floor(0.8 * num_images) : ])

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def reshape(arr, timesteps, features, val):
    '''
    Reshapes 2D array to shape (samples, timesteps, features) for LSTM input, with constant padding if needed.
    '''
    padding = timesteps - (arr.shape[0] % timesteps)
    arr = np.pad(arr, ((0, padding), (0,0)), constant_values=(val))
    samples = int(arr.shape[0] / timesteps)
    arr = np.reshape(arr, (samples, timesteps, features))
    return arr
