import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from skimage import transform
import os
from sklearn.decomposition import PCA
import pickle

import cv2

#NUM_COMPONENTS = 26

DIM_REDUCTION_MODE = 'PCA'

RESIZE_SHAPE = (270, 480)

if 'RESIZE_SHAPE' in locals():
    FRAME_WIDTH = RESIZE_SHAPE[1]
    FRAME_HEIGHT = RESIZE_SHAPE[0]
else:
    FRAME_WIDTH = 3840
    FRAME_HEIGHT = 2160

DATA_PATH = './data'

HOUSES_NAMES = natsorted([x for x in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, x))])[2:]

def compute_data(files):
    data_ch_red = []
    data_ch_green = []
    data_ch_blue = []

    for file in tqdm(files):

        img = Image.open(os.path.join(DATA_PATH, house, viewpoint, 'with shadows', file))

        img = np.array(img)

        #print(img.shape)

        img_resized = transform.resize(img, RESIZE_SHAPE, preserve_range=True)

        #print(img_resized.shape)

        data_ch_red.append(img_resized[:,:,0])
        data_ch_green.append(img_resized[:,:,1])
        data_ch_blue.append(img_resized[:,:,2])

    data_ch_red = np.array(data_ch_red)
    data_ch_red = np.reshape(data_ch_red, (data_ch_red.shape[0], data_ch_red.shape[1] * data_ch_red.shape[2]))

    data_ch_green = np.array(data_ch_green)
    data_ch_green = np.reshape(data_ch_green, (data_ch_green.shape[0], data_ch_green.shape[1] * data_ch_green.shape[2]))

    data_ch_blue = np.array(data_ch_blue)
    data_ch_blue = np.reshape(data_ch_blue, (data_ch_blue.shape[0], data_ch_blue.shape[1] * data_ch_blue.shape[2]))

    return (data_ch_red, data_ch_green, data_ch_blue) 

#filename = "my_model.pickle"
# load model
#loaded_model = pickle.load(open(filename, "rb"))

def fit_pca_per_set(save_path, data, num_components):

    pca = PCA(num_components, svd_solver='full')

    pca_fit = pca.fit(data)

    eigenhouses = pca_fit.components_#.reshape((num_components, FRAME_HEIGHT, FRAME_WIDTH))

    data_transformed = pca.transform(data)

    data_inverted = pca.inverse_transform(data_transformed)

    pickle.dump(pca, open(save_path, "wb"))

    return data_inverted, eigenhouses


def save_pca_results(save_path, red_set, green_set, blue_set, files):

    assert(len(files)==red_set.shape[0])

    for idx, file in enumerate(files):
        red_map = np.reshape(red_set[idx, :], (FRAME_HEIGHT, FRAME_WIDTH))
        green_map = np.reshape(green_set[idx, :], (FRAME_HEIGHT, FRAME_WIDTH))
        blue_map = np.reshape(blue_set[idx, :], (FRAME_HEIGHT, FRAME_WIDTH))

        img = (np.dstack((blue_map, green_map, red_map))).astype(np.uint8)

        cv2.imwrite(os.path.join(save_path, file), img)

def save_eigenhouses(save_path, input_set, filename):
    input_set = input_set.reshape((NUM_COMPONENTS, FRAME_HEIGHT, FRAME_WIDTH))

    eigen_titles = ["eigenface %d" % i for i in range(NUM_COMPONENTS)]

    n_row = 6
    n_col = 5
    plt.figure(figsize=(2.4 * n_row, 1.8 * n_col))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(NUM_COMPONENTS):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(input_set[i], cmap=plt.cm.gray)
        plt.title(eigen_titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.savefig(os.path.join(save_path, filename))

def combine_eigenhouses(method='', output_path='', set_1=[], set_2=[], set_3=[], filename='.png'):

    set_1 = set_1.reshape((NUM_COMPONENTS, FRAME_HEIGHT, FRAME_WIDTH))
    set_2 = set_2.reshape((NUM_COMPONENTS, FRAME_HEIGHT, FRAME_WIDTH))
    set_3 = set_3.reshape((NUM_COMPONENTS, FRAME_HEIGHT, FRAME_WIDTH))

    eigen_titles = ["eigenface %d" % i for i in range(NUM_COMPONENTS)]

    n_row = 6
    n_col = 5
    plt.figure(figsize=(2.4 * n_row, 1.8 * n_col))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(NUM_COMPONENTS):
        plt.subplot(n_row, n_col, i + 1)
        if method == 'color':
            combined = np.clip(np.concatenate((np.expand_dims(set_1[i], axis=2), 
                                       np.expand_dims(set_2[i], axis=2), 
                                       np.expand_dims(set_3[i], axis=2)), axis=2), 0, 1) * 255
            plt.imshow(combined)
        elif method == 'multiply':
            combined = set_1[i] * set_2[i] * set_3[i]
            plt.imshow(combined, cmap=plt.cm.gray)
        else:
            print('Not implemented!')
        
        plt.title(eigen_titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.savefig(os.path.join(output_path, filename))

if __name__ == "__main__":

    for house in tqdm(HOUSES_NAMES):

        viewpoints = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, house)) if os.path.isdir(os.path.join(DATA_PATH, house, x))])

        for viewpoint in viewpoints:

            shadows_maps_files = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, house, viewpoint, 'with shadows')) if x.endswith('.png')])
            #shadows_maps_files = shadows_maps_files[:6]

            NUM_COMPONENTS = len(shadows_maps_files)

            data = compute_data(shadows_maps_files)

            os.makedirs(os.path.join(DATA_PATH, house, viewpoint, 'pca_dump'), exist_ok=True)

            os.makedirs(os.path.join(DATA_PATH, house, viewpoint, 'pca_eigenhouses'), exist_ok=True)

            red_inverted, red_eigenhouses = fit_pca_per_set(os.path.join(DATA_PATH, house, viewpoint, 'pca_dump', 'red.pickle'), data[0], NUM_COMPONENTS)

            green_inverted, green_eigenhouses = fit_pca_per_set(os.path.join(DATA_PATH, house, viewpoint, 'pca_dump', 'green.pickle'), data[1], NUM_COMPONENTS)

            blue_inverted, blue_eigenhouses = fit_pca_per_set(os.path.join(DATA_PATH, house, viewpoint, 'pca_dump', 'blue.pickle'), data[2], NUM_COMPONENTS)

            save_pca_results(os.path.join(DATA_PATH, house, viewpoint, 'pca_dump'), red_inverted, green_inverted, blue_inverted, shadows_maps_files)

            #save_eigenhouses(os.path.join(DATA_PATH, house, viewpoint, 'pca_eigenhouses'), red_eigenhouses, green_eigenhouses, blue_eigenhouses, shadows_maps_files)
            save_eigenhouses(os.path.join(DATA_PATH, house, viewpoint, 'pca_eigenhouses'), red_eigenhouses, 'red_eigenhouses.png')
            save_eigenhouses(os.path.join(DATA_PATH, house, viewpoint, 'pca_eigenhouses'), green_eigenhouses, 'green_eigenhouses.png')
            save_eigenhouses(os.path.join(DATA_PATH, house, viewpoint, 'pca_eigenhouses'), blue_eigenhouses, 'blue_eigenhouses.png')

            combine_eigenhouses(method='color', output_path=os.path.join(DATA_PATH, house, viewpoint, 'pca_eigenhouses'), set_1=red_eigenhouses, 
                                set_2=green_eigenhouses, set_3=blue_eigenhouses, filename='combined_color.png')
            
            combine_eigenhouses(method='multiply', output_path=os.path.join(DATA_PATH, house, viewpoint, 'pca_eigenhouses'), set_1=red_eigenhouses, 
                                set_2=green_eigenhouses, set_3=blue_eigenhouses, filename='combined_multiply.png')


