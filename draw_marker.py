from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import re
import cv2
import glob
import numpy as np
import pandas as pd

class DrawMarker:
    COLIMNS = ["lu_x","lu_y","ru_x","ru_y","ld_x","ld_y","rd_x","rd_y"]
        
    def __init__():
        pass
    
    def resize_image(self, read_image, file_num, csv_df):
        img_height, img_width = read_image.shape[:2]
        print(file_num)
        print(f"img_height:{img_height}")
        print(f"img_width:{img_width}")
        width = 200
        height = 200
        size = (height, width)
        resize_image = cv2.resize(read_image, size)

        resized_df = csv_df.loc[file_num][:]

        zip_ratio_w = img_width/width
        zip_ratio_h = img_height/height

        print(f"w:{zip_ratio_w}")
        print(f"h:{zip_ratio_h}")
        resized_df.iloc[::2] /= zip_ratio_w
        resized_df.iloc[1::2] /= zip_ratio_h
        cv2.drawMarker(resize_image,(int(resized_df["lu_x"]),int(resized_df["lu_y"])),(0,0,0), markerType=cv2.MARKER_STAR,markerSize=10)
        cv2.drawMarker(resize_image,(int(resized_df["ru_x"]),int(resized_df["ru_y"])),(0,0,0) ,markerType=cv2.MARKER_STAR)
        cv2.drawMarker(resize_image,(int(resized_df["ld_x"]),int(resized_df["ld_y"])),(0,0,0) ,markerType=cv2.MARKER_STAR)
        cv2.drawMarker(resize_image,(int(resized_df["rd_x"]),int(resized_df["rd_y"])),(0,0,0) ,markerType=cv2.MARKER_STAR)
        cv2.imwrite(f"./output/test/test_mark_{file_num}.JPG", resize_image)
        print(csv_df.loc[file_num][:])
        print(resized_df)
        return resized_df

    def transform_matrix_offset_center(matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix