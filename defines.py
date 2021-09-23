from datetime import datetime

'''
ACTIONS:
1 - VGG16 fine-tuning
2 - VGG16 feature extraction and LSTM training
3 - VGG16 feature extraction and LSTM inference
'''
PARAM_ACTION = 3

PARAM_BATCHES = 2
PARAM_N_EPOCHS = 20
PARAM_N_TESTS = 100
PARAM_EPOCH_STEPS = 100
PARAM_LEARNING_RATE = 1e-5

PARAM_TIMESTEPS = 64

PARAM_SAVE_BEST_ONLY = True

PARAM_SYSTEM_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
PARAM_PATH_IMAGES = '/Users/phuongnhi//Desktop/MHC/Summer21/research/dataset/training_images/'
PARAM_PATH_LABELS = '/Users/phuongnhi/Desktop/MHC/Summer21/research/dataset/training_labels/training_labels.npy'
PARAM_WEIGHTS_PATH_1 = 'CNN.hdf5'
PARAM_WEIGHTS_PATH_2 = 'RNN.hdf5'
PARAM_PATH_TEST_NPY = 'results.npy'

PARAM_METRICS = 'accuracy'								# TODO: monitor more metrics... look up the options.
