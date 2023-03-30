import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm

DATA_PATH = './data'

houses = natsorted([x for x in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, x))])

def get_sun_coords(txt_file):
    # key: image_idx, (azimuth, elevation)
    dict = {}
    lines = open(txt_file).read().splitlines()
    for line in lines:
        splits = line.split(' ')
        dict[splits[0].replace(':', '')] = (float(splits[2]), float(splits[4]))
    
    return dict

print('Num. sample houses: ', len(houses))

for house in tqdm(houses):
    
    print(house)

    viewpoints = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, house)) if os.path.isdir(os.path.join(DATA_PATH, house, x))])

    print('Num. viewpoints per house: ', len(viewpoints))

    coords_dict = get_sun_coords(os.path.join(DATA_PATH, house, house + '_sun.txt'))
    #print(len(coords_dict))

    for viewpoint in viewpoints:

        representations = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, house, viewpoint)) if os.path.isdir(os.path.join(DATA_PATH, house, viewpoint))])

        print('Number of representations: ', len(representations)) #['depth', 'shadow mask', 'with shadows', 'without shadows']
        #print(representations)

        shadows_maps = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, house, viewpoint, 'with shadows')) if x.endswith('.png')])

        assert(len(coords_dict) == len(shadows_maps))

        #print(len(shadows_maps))

        '''
        for rep in representations:
            print(rep)

            maps = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, house, viewpoint, rep))])

            print(len(maps))
        '''
    '''
    for key in coords_dict:
        print(key)
        print(coords_dict[key])
    '''
