import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

DATA_PATH = './data'

houses = natsorted([x for x in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, x))])

for house in tqdm(houses):
    
    print(house)

    viewpoints = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, house)) if os.path.isdir(os.path.join(DATA_PATH, house, x))])

    print('Num. viewpoints per house: ', len(viewpoints))

    for viewpoint in viewpoints:

        shadows_maps = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, house, viewpoint, 'with shadows')) if x.endswith('.png')])

        placeholder = Image.open(os.path.join(DATA_PATH, house, viewpoint, 'with shadows', shadows_maps[0]))

        placeholder = np.array(placeholder)

        mean_image = np.zeros((placeholder.shape[0], placeholder.shape[1], 3))

        for shadow_map_name in shadows_maps:

            im = Image.open(os.path.join(DATA_PATH, house, viewpoint, 'with shadows', shadow_map_name))

            im_array = np.array(im)

            mean_image += im_array[:,:,:placeholder.shape[2]-1]

        mean_image /= len(shadows_maps)

        mean_image = Image.fromarray(np.uint8(mean_image))

        mean_image.save(os.path.join(DATA_PATH, house, viewpoint, 'mean_image_with_shadows.png'))


