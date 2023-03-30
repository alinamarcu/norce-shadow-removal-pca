import os
import numpy as np
from natsort import natsorted
import cv2
from tqdm import tqdm

native_width = 3840
native_height = 2160

downsample_factor = 1

input_width = native_width // downsample_factor
input_height = native_height // downsample_factor

DATA_PATH = './data'

RESULTS_PATH = './results/deshadow-baseline/iw_' + str(input_width) + '_ih_' + str(input_height)

epochs = natsorted([x for x in os.listdir(RESULTS_PATH) if os.path.isdir(os.path.join(RESULTS_PATH, x)) and 'epoch' in x])

#print(epochs)

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

train_files = get_files(ids=[1,3,4,5])
valid_files = get_files(ids=[2])
test_files = train_files + valid_files

os.makedirs(os.path.join(RESULTS_PATH, 'joined'), exist_ok=True)

for fileid in tqdm(test_files):
		
	#result_1 = np.load(os.path.join(RESULTS_PATH, epochs[0], 'pred_on_train_and_valid_506', fileid.replace('png', 'npz')))['arr_0']
	#result_1 = np.uint8(np.clip(result_1, a_min=0.0, a_max=1.0) * 255)
	
	img = cv2.imread(os.path.join(DATA_PATH, fileid))
	img = cv2.resize(img, (input_width, input_height)) 

	result_2 = np.load(os.path.join(RESULTS_PATH, epochs[0], 'pred_on_train_and_valid_506', fileid.replace('png', 'npz')))['arr_0']
	result_2 = np.uint8(np.clip(result_2, a_min=0.0, a_max=1.0) * 255)
	result_2 = cv2.resize(result_2, (input_width, input_height))

	label = cv2.imread(os.path.join(DATA_PATH, os.path.dirname(os.path.dirname(fileid)), 'mean_image_with_shadows.png'))
	label = cv2.resize(label, (input_width, input_height))
	
	#stacked = np.concatenate((result_1, result_2, label), axis=1)

	stacked = np.concatenate((img, result_2, label), axis=1)

	os.makedirs(os.path.join(RESULTS_PATH, 'joined', os.path.dirname(fileid)), exist_ok=True)

	cv2.imwrite(os.path.join(RESULTS_PATH, 'joined', fileid), stacked)
	
