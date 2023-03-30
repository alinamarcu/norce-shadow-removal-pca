import random as rn
rn.seed(12345)

import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(1234)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from tensorflow import keras

import os
import scipy.io

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import cv2

import time

from models.mihai_nns_tf2 import map2map_generic, map2map_generic_same_initializer, map2map_generic_same_initializer_new_size
from models.losses import bce_dice_loss, dice_coeff

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler

from natsort import natsorted

from tqdm import tqdm

from sklearn.metrics import mean_squared_error

NN_TYPE = 'map2map_generic_same_initializer_new_size'

native_width = 3840
native_height = 2160

downsample_factor = 1

input_width = native_width // downsample_factor
input_height = native_height // downsample_factor

from models.mihai_nns_tf2 import map2map_generic, map2map_generic_same_initializer, map2map_generic_same_initializer_new_size

model = map2map_generic_same_initializer_new_size(input_shape=(input_height, input_width, 3), batch_norm=True, activation=None, num_classes=3)

# TODO - Add the path to NN weights
WEIGHTS_PATH = '' 

model.load_weights(filepath=WEIGHTS_PATH)
model.summary()

# Version 1
def mean_squared_error_version_1(y_actual, y_predicted):
	# sum of squared difference between the two images
	err = np.sum((y_actual.astype("float") - y_predicted.astype("float")) ** 2)
	err /= float(y_actual.shape[0] * y_actual.shape[1])
	return err

# Version 2
def mean_squared_error_version_2(y_actual, y_predicted):
	return mean_squared_error(y_actual, y_predicted)

def root_mean_squared_error(y_actual, y_predicted):
	return mean_squared_error(y_actual, y_predicted, squared=False)

def l1_error(y_actual, y_predicted):
	return np.abs(y_actual - y_predicted).mean()

BEST_EPOCH = int(os.path.basename(WEIGHTS_PATH).split('_')[1])

DATA_PATH = './data'

RES_TYPE = 'iw_' + str(input_width) + '_ih_' + str(input_height)

OUTPUT_RESULTS = os.path.join('results', 'deshadow-baseline', RES_TYPE, 'epoch_' + str(BEST_EPOCH))

os.makedirs(OUTPUT_RESULTS, exist_ok=True)

def get_files(ids=[]):
	files = []

	fds = natsorted([x for x in natsorted(os.listdir(DATA_PATH)) if os.path.isdir(os.path.join(DATA_PATH, x)) and int(x.split('_')[-1]) in ids])
	#print(fds)

	for fd in fds:
		cams = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, fd)) if os.path.isdir(os.path.join(DATA_PATH, fd, x))])
		#print(cams)

		for cam in cams:
			tmp_files = natsorted([os.path.join(fd, cam, 'with shadows', x) for x in natsorted(os.listdir(os.path.join(DATA_PATH, fd, cam, 'with shadows'))) if x.endswith('.png')])
			#print(tmp_files)
			files += tmp_files

	return files

# Train on houses: 1, 3, 4, 5 and test/valid on house: 2 - Prioritize more data for training
train_files = get_files(ids=[1,3,4,5])
valid_files = get_files(ids=[2])

test_files = train_files + valid_files

TYPE = 'pred_on_train_and_valid'
#TYPE = 'pred_on_semisup'
#TYPE = 'pred_on_test'

if TYPE == 'pred_on_semisup':

	eval_files = test_files

	results_dir = os.path.join(OUTPUT_RESULTS, TYPE + '_' + str(len(eval_files)))

	os.makedirs(results_dir, exist_ok=True)

elif TYPE == 'pred_on_test':

	eval_files = test_files

	results_dir = os.path.join(OUTPUT_RESULTS, TYPE + '_' + str(len(eval_files)))

	os.makedirs(results_dir, exist_ok=True)

elif TYPE == 'pred_on_train_and_valid':

	eval_files = test_files

	results_dir = os.path.join(OUTPUT_RESULTS, TYPE + '_' + str(len(eval_files)))

	os.makedirs(results_dir, exist_ok=True)

else:
	print('Not implemented!')

if TYPE == 'pred_on_train_and_valid' or TYPE == 'pred_on_semisup' or TYPE == 'pred_on_test':

	for fileid in tqdm(test_files):
		
		#print(fileid)

		x_batch = []
		
		img = cv2.imread(os.path.join(DATA_PATH, fileid))
		img = cv2.resize(img, (input_width, input_height)) 
		img = (img - np.min(img)) / (np.max(img) - np.min(img) + np.spacing(1))
		
		assert(np.min(img) == 0.0)
		assert(np.max(img) <= 1.0)

		#label = cv2.imread(os.path.join(DATA_PATH, os.path.dirname(os.path.dirname(fileid)), 'mean_image_with_shadows.png'))
		#label = cv2.resize(label, (input_width, input_height))
		#label = (label - np.min(label)) / (np.max(label) - np.min(label) + np.spacing(1))
				
		#assert(np.min(label) == 0.0)
		#assert(np.max(label) <= 1.0)

		x_batch.append(img)

		x_batch = np.array(x_batch, np.float32)
					
		pred = model.predict_on_batch(x_batch)[0]

		os.makedirs(os.path.join(results_dir, os.path.dirname(fileid)), exist_ok=True)

		np.savez_compressed(os.path.join(results_dir, fileid.replace('.png', '.npz')), np.array(pred, dtype=np.float16))

else:
	print('Not implemented!')
