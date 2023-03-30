import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

import imageio

DATA_PATH = './data'

houses = natsorted([x for x in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, x))])

for house in tqdm(houses):
    
    print(house)

    viewpoints = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, house)) if os.path.isdir(os.path.join(DATA_PATH, house, x))])

    print('Num. viewpoints per house: ', len(viewpoints))

    for viewpoint in viewpoints:

        shadows_maps = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, house, viewpoint, 'with shadows')) if x.endswith('.png')])

        frames = []

        for frame in shadows_maps:
            
            image = imageio.imread(os.path.join(DATA_PATH, house, viewpoint, 'with shadows', frame))

            image = Image.fromarray(image[:,:,:3]).resize((image.shape[1] // 4, image.shape[0] // 4))

            frames.append(image)

        imageio.mimsave(os.path.join(os.path.join(DATA_PATH, house, viewpoint, 'with_shadows.gif')), frames, fps = 5)  

            
