from defines import *
from model import *
from data import *
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':

	train_images, train_labels, val_images, val_labels, test_images, test_labels = load_dataset(PARAM_PATH_IMAGES, PARAM_PATH_LABELS)

	if PARAM_ACTION == 1:

		# set up model
		model = vgg16()

		# set up custom loss function
		loss_function = custom_loss(0.8, rho(2.0, 0.01), rho(1.0, 0.01))

		# compile model
		model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = PARAM_LEARNING_RATE),
					  loss="mse",
					  metrics=[PARAM_METRICS])

		# set up model checkpoint
		model_checkpoint = ModelCheckpoint(PARAM_WEIGHTS_PATH_1 ,
						    monitor = PARAM_METRICS,
						    verbose = 1,
							save_weights_only = True,
						    save_best_only = PARAM_SAVE_BEST_ONLY)

		# fit model on data
		history = model.fit(x=train_images, y=train_labels, verbose = 1, epochs=PARAM_N_EPOCHS, validation_data=(val_images, val_labels), callbacks = [model_checkpoint])

		# plot performance
		acc = history.history['accuracy']
		plt.figure(figsize=(8, 8))
		plt.plot(acc, label='Accuracy')
		plt.legend(loc='lower right')
		plt.ylabel('Accuracy')
		plt.title('Training Accuracy')
		plt.show()


		model = vgg16(PARAM_WEIGHTS_PATH_1)

		# how should we save history for evaluation?


	if PARAM_ACTION == 2:

		# instantiate vgg16 model with fine-tuned weights
		model_1 = vgg16(weights=PARAM_WEIGHTS_PATH_1, feature_extraction=True)

		# extract feature vectors
		train_features = model_1.predict(train_images, verbose=1)
		val_features = model_1.predict(val_images, verbose=1)

		# reshape inputs for lstm
		train_features = reshape(train_features, PARAM_TIMESTEPS, 4096, 20)
		val_features = reshape(val_features, PARAM_TIMESTEPS, 4096, 20)

		train_labels = reshape(train_labels, PARAM_TIMESTEPS, 3, 20)
		val_labels = reshape(val_labels, PARAM_TIMESTEPS, 3, 20)

		# initiate model
		model_2 = lstm(timesteps=PARAM_TIMESTEPS)

		# set up custom loss function
		loss_function = custom_loss(0.8, rho(2.0, 0.01), rho(1.0, 0.01))

		# compile model
		model_2.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=PARAM_LEARNING_RATE),
					  loss="mae",
					  metrics=PARAM_METRICS) # lr: 25e-4

		# set up model checkpoint
		model_checkpoint = ModelCheckpoint(PARAM_WEIGHTS_PATH_2,
						    monitor=PARAM_METRICS,
						    verbose=1,
							save_weights_only=True,
						    save_best_only=PARAM_SAVE_BEST_ONLY)

		# fit model on data
		history = model_2.fit(x=train_features, y=train_labels, verbose=1, epochs=PARAM_N_EPOCHS, validation_data=(val_features, val_labels), callbacks = [model_checkpoint])

		# plot performance
		metric = history.history['accuracy']
		plt.figure(figsize=(8, 8))
		plt.plot(metric, label='Accuracy')
		plt.legend(loc='lower right')
		plt.ylabel('Accuracy')
		plt.title('Training Accuracy')
		plt.show()


	if PARAM_ACTION == 3:

		# feature extraction and reshaping
		model_1 = vgg16(weights=PARAM_WEIGHTS_PATH_1, feature_extraction=True)
		test_features = model_1.predict(test_images, verbose=1)
		test_features = reshape(test_features, PARAM_TIMESTEPS, 4096, 20)
		test_labels = reshape(test_labels, PARAM_TIMESTEPS, 3, 20)

		# use LSTM to predict
		model_2 = lstm(weights=PARAM_WEIGHTS_PATH_2)
		results = model_2.predict(test_features, verbose = 1)

		# save results
		np.save(PARAM_PATH_TEST_NPY, results)
