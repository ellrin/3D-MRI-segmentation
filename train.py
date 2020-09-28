#!/usr/bin/env python
# coding: utf-8
# %%
# import packages

import os
import random
import pandas as pd
import numpy as np
import nibabel as nib
import json
import time


# %%
import tensorflow.keras.backend as K
from tensorflow.keras import metrics
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam


# %%
from utils.model import Unet_3D
from utils.loss import dice_coef, dice_coef_loss
from utils.pre_processing import normalize, get_random


# %%
with open('info.json') as info:
    train_info = json.load(info)


# %%
# basic setting

train_tu_path    = train_info['train_tumor_path']
train_label_path = train_info['train_label_path']
valid_tu_path    = train_info['valid_tumor_path']
valid_label_path = train_info['valid_label_path']
batch            = train_info['batch']
learning_rate    = train_info['learning_rate']
decay_rate       = train_info['decay_rate']
image_size       = train_info['image_size']
image_channel    = train_info['image_channel']
num_epochs       = train_info['epoch']
gpu_number       = str(train_info['gpu_number'])


# %%
train_image_list = os.listdir(train_tu_path)
train_label_list = os.listdir(train_label_path)
valid_image_list = os.listdir(valid_tu_path)
valid_label_list = os.listdir(valid_label_path)
steps            = len(train_image_list)//batch


# %%
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


# %%
# resize to a square

inp_array = Input((image_size, image_size, image_size, image_channel))
model = Unet_3D(inp_array)


# %%
model.summary()


# %%
model.compile(optimizer=Adam(lr=learning_rate, decay = decay_rate), 
              loss=dice_coef_loss, 
              metrics=[dice_coef])


# %%
# make a director to save the weights and log

dir_name = '%s-%02d-%02d_%02d-%02d_train'%(time.localtime()[0], time.localtime()[1], time.localtime()[2], 
                                           time.localtime()[3], time.localtime()[4])
weight_dir = './model_weight/'+dir_name
log_dir    = './model_log/'+dir_name

try:
    os.mkdir('model_weight')
    os.mkdir('model_log')
except:
    pass

try:
    os.mkdir(weight_dir)
    os.mkdir(log_dir)
except:
    pass


# %%
# start training

val_loss = []
val_dice = []

# load the validation images
valid_img_generator = get_random(name_list=valid_image_list, image_path=valid_tu_path,
                                 label_path=valid_label_path, steps=1, batch=len(valid_image_list))

print('loading images...')
valid_images, valid_labels = next(valid_img_generator)

print('start training')
for epoch in range(num_epochs):
    
    img_generator = get_random(name_list =train_image_list, 
                               image_path=train_tu_path,
                               label_path=train_label_path,
                               batch=batch, 
                               steps=steps)
    
    print('epoch:%d'%(epoch+1))
    
    for step in range(steps):
        
        print('step %d/%d'%(step+1, steps))
        # load the generated images for training (by batch)
        train_images, train_labels = next(img_generator)
        
        # update the weights
        model.fit(x=train_images, y=train_labels, verbose=2)
        
    # evaluate the validation data
    val_acc = model.evaluate(valid_images, valid_labels, batch_size=8)
    model.save(weight_dir+'/model_weight')
    val_loss.append(val_acc[0])
    val_dice.append(val_acc[1])
    val_df = pd.DataFrame({'val_loss':val_loss, 'val_dice':val_dice})
    val_df.to_csv(log_dir+'/model_log.csv')

