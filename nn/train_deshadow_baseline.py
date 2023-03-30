import random as rn
rn.seed(12345)

import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(1234)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from tensorflow import keras

from tensorflow.keras.losses import mean_squared_error

import os
import scipy.io

import cv2

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_KERAS'] = '1' # For AdamW

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import time

from models.mihai_nns_tf2 import map2map_generic, map2map_generic_same_initializer, map2map_generic_same_initializer_new_size
from models.losses import bce_dice_loss, dice_coeff, categorical_focal_loss, tversky_loss, catce_dice_loss, weighted_categorical_crossentropy

from keras.optimizers import RMSprop, Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler

from keras_adamw import AdamW
adamw = AdamW(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)

from focal_loss import BinaryFocalLoss

from natsort import natsorted

epochs = 100
batch_size = 10 #10 - for 960x540

NN_TYPE = 'map2map_generic_same_initializer_new_size'

native_width = 3840
native_height = 2160

downsample_factor = 4

input_width = native_width // downsample_factor
input_height = native_height // downsample_factor

model = map2map_generic_same_initializer_new_size(input_shape=(input_height, input_width, 3), batch_norm=True, activation=None, num_classes=3)
model.compile(optimizer=RMSprop(lr=0.001), loss=mean_squared_error)

DATA_PATH = './data'

RES_TYPE = 'iw_' + str(input_width) + '_ih_' + str(input_height)

OUTPUT_WEIGHTS = os.path.join('weights', 'deshadow-baseline', RES_TYPE)

os.makedirs(OUTPUT_WEIGHTS, exist_ok=True)

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

ids_train_split = rn.sample(train_files, len(train_files)) 
ids_valid_split = rn.sample(valid_files, len(valid_files)) 

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def train_generator():
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]

            for fileid in ids_train_batch:

                img = cv2.imread(os.path.join(DATA_PATH, fileid))
                img = cv2.resize(img, (input_width, input_height)) 
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + np.spacing(1))

                assert(np.min(img) == 0.0)
                assert(np.max(img) <= 1.0)

                label = cv2.imread(os.path.join(DATA_PATH, os.path.dirname(os.path.dirname(fileid)), 'mean_image_with_shadows.png'))
                label = cv2.resize(label, (input_width, input_height))
                label = (label - np.min(label)) / (np.max(label) - np.min(label) + np.spacing(1))
                
                assert(np.min(label) == 0.0)
                assert(np.max(label) <= 1.0)

                img, label = randomShiftScaleRotate(img, label,
                                                shift_limit=(-0.0625, 0.0625),
                                                scale_limit=(-0.1, 0.1),
                                                rotate_limit=(-0, 0))
                
                img, label = randomHorizontalFlip(img, label)

                x_batch.append(img)
                y_batch.append(label)
                    
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]

            for fileid in ids_valid_batch:

                img = cv2.imread(os.path.join(DATA_PATH, fileid))
                img = cv2.resize(img, (input_width, input_height))
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + np.spacing(1))

                assert(np.min(img) == 0.0)
                assert(np.max(img) <= 1.0)

                label = cv2.imread(os.path.join(DATA_PATH, os.path.dirname(os.path.dirname(fileid)), 'mean_image_with_shadows.png'))
                label = cv2.resize(label, (input_width, input_height))
                label = (label - np.min(label)) / (np.max(label) - np.min(label) + np.spacing(1))

                assert(np.min(label) == 0.0)
                assert(np.max(label) <= 1.0)

                x_batch.append(img)
                y_batch.append(label)
            
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32) 
            
            yield x_batch, y_batch

#callbacks = [EarlyStopping(monitor='val_loss',
#                           patience=7,
#                           verbose=1,
#                           min_delta=1e-6),
callbacks = [ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=5,
                               verbose=1,
                               epsilon=1e-6),
             ModelCheckpoint(monitor='val_loss',
                             filepath=os.path.join(OUTPUT_WEIGHTS, 'EPOCH_{epoch:02d}_TRAIN_loss_{loss:.5f}_VALID_loss_{val_loss:.5f}.hdf5'),
                             save_best_only=False,
                             save_weights_only=True,
                             period=1),
             TensorBoard(log_dir='logs')]

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
