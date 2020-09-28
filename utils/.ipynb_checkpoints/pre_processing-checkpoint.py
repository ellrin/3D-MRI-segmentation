import numpy as np
import nibabel as nib # version=3.1.1
import random


def normalize(input_array):
    
    return (input_array - np.mean(input_array)) / (np.std(input_array - np.mean(input_array)))


def get_random(name_list, image_path, label_path,
               batch=8, steps=None, shuffle=True, img_size=128):
    
    if steps==None:
        steps=len(name_list)//batch
        
    if shuffle == True:
        seed = random.randint(0,1000)
        random.Random(seed).shuffle(name_list)             
    
    while True:
    
        for names_idx in range(0, steps):
            
            image_cube = np.zeros((batch, img_size, img_size, img_size, 1))
            label_cube = np.zeros((batch, img_size, img_size, img_size, 1))
            
            for idx, name in enumerate(name_list[names_idx:names_idx+batch]):
                
                image_array = nib.load(image_path+name).get_fdata()
                label_array = nib.load(label_path+name).get_fdata()
                
                # normalize
                image_array = normalize(image_array)
                
                image_cube[idx,:,:,:,0] = image_array
                label_cube[idx,:,:,:,0] = label_array

            yield image_cube, label_cube


