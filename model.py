import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.math import *

def vgg16(fine_tuning=False, feature_extraction=False, weights=None):

    # load VGG16 with pre-trained weights
    base_model = VGG16(include_top = True, weights = 'imagenet')
    # freeze all base layers
    for layer in base_model.layers:
        layer.trainable = False
    # get second to last layer
    output = base_model.get_layer('fc2').output
    # add new dense layer to replace last layer
    output = layers.Dropout(0.5)(output)
    output = layers.Dense(3, activation = 'linear')(output)
    model = Model(base_model.input, output)

    if (fine_tuning):
        model.load_weights(weights)
        # unfreeze all layers
        model.trainable = True

    if (feature_extraction):
        model.load_weights(weights)
        # change last layer activation to tanh
        model.layers[-3].activation = tanh
        model = Model(inputs = model.input, outputs = model.layers[-3].output)

    model.summary()

    return model


def lstm(weights=None, timesteps=None, mask_value=20):

    model = tf.keras.Sequential()
    model.add(layers.Masking(mask_value, input_shape=(timesteps, 3)))
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.Dropout(0.25))
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.Dropout(0.25))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(3, activation='linear'))

    if (weights):
        model.load_weights(weights)

    model.summary()

    return model


# TODO: parameterize activation function, test that the math works correctly
def custom_loss(alpha, rho_1, rho_2):
    def call(y_true, y_pred):
        # take the mean along the first axis
        mse = tf.math.reduce_mean(tf.math.squared_difference(y_true, y_pred), 1)
        rmse = tf.math.sqrt(mse)
        rmse_sum = tf.reduce_sum(rho_1(rmse))
        grad_true = tf.math.abs(tf.math.subtract(y_true[1: :], y_true[:-1, :]))
        grad_pred = tf.math.abs(tf.math.subtract(y_pred[1:, :], y_pred[:-1, :]))
        grad_diff = tf.math.reduce_sum(tf.math.abs(tf.math.subtract(grad_true, grad_pred)), 1)
        diff_sum = tf.math.reduce_sum(rho_2(grad_diff))
        return tf.math.multiply(alpha, rmse_sum) + tf.math.multiply(1-alpha, diff_sum)
    return call

def rho(gamma, epsilon):
    def f(x):
        return tf.math.log(tf.math.pow(x, gamma) + epsilon)
    return f
