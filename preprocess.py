from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Lambda, MaxPooling2D
from keras import backend as K
import os
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
import cv2
import scipy

root_dir = '/Users/naka345/Desktop/deeplearning/car_number/'
batch_size = 32
epochs = 100
num_predictions = 20
save_dir = f'{root_dir}model/'
model_name = 'plate_detect_trained_model.h5'
load_dir_path = 'output/train'

path = root_dir + load_dir_path
import glob
print(path)
x_path = glob.glob(f'{path}/20190123/*.JPG')
y_path = glob.glob(f'{path}/20190123/*.csv')
x_path.sort()
import pandas as pd
y_t_df = pd.read_csv(y_path[0],index_col=0).sort_index()

import numpy as np
x_list = [(img_to_array(load_img(x))/255.) for x in x_path]
x_train = np.array(x_list)
x_shape =  x_train.shape
print(x_shape)

label = y_t_df.columns
y_train = y_t_df.values/200
print(y_t_df.columns)
print(y_train[0:1])

print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=15,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # set range for random zoom
    zoom_range=0.1,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=False,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

test_gen = CustomedImageDataGenerator(
    # randomly rotate images in the range (deg 0 to 180)
    #rotation_range=15,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    #height_shift_range=0.1,
    # set range for random zoom
    #zoom_range=0.5,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # randomly flip images
    #horizontal_flip=True,
    # randomly flip images
    #vertical_flip=False,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0
)
xx,yy=test_gen.flow(x_train[:1],y_train[:1])
print(yy)
xx = (xx*255)
xx= xx.astype(np.int16)
yy = yy*200.
print(yy)
for i,x in enumerate(xx):
    x=x.transpose((1,2,0))
    cv2.imwrite(f"./test{i}.jpg",x)
    check_y_move(f"./test{i}.jpg", yy[i])
