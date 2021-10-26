from datetime import datetime

'''
ACTIONS:
1 - VGG16 fine-tuning
2 - VGG16 feature extraction and LSTM training
3 - VGG16 feature extraction and LSTM inference
'''
PARAM_ACTION = 1

PARAM_BATCHES = 2
PARAM_N_EPOCHS = 20
PARAM_EPOCH_STEPS = 100
PARAM_LEARNING_RATE = 1e-5

PARAM_TIMESTEPS = 64

PARAM_SAVE_BEST_ONLY = True

PARAM_HISTORY_FILE = 'CNN_(1)_1_history.csv' # format: model_history.csv

PARAM_SYSTEM_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
PARAM_PATH_IMAGES = '/Users/phuongnhi//Desktop/MHC/Summer21/research/dataset/training_images/'
PARAM_PATH_LABELS = '/Users/phuongnhi/Desktop/MHC/Summer21/research/dataset/training_labels/training_labels.npy'
PARAM_WEIGHTS_PATH_1 = 'CNN_(1)_1.hdf5' # format: CNN_(fine tuning phase)_trialnumber.hdf5
PARAM_WEIGHTS_PATH_2 = 'RNN_1.hdf5' # format: RNN_trialnumber.hdf5
PARAM_RESULTS_FILE = 'results_1.npy' # format: results_trialnumber.hdf5

PARAM_METRICS = 'accuracy'								# TODO: monitor more metrics... look up the options.
