import re
import os
import glob
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img


class DFRead:
    COLIMNS = ["lu_x","lu_y","ru_x","ru_y","ld_x","ld_y","rd_x","rd_y"]
    
    def __init__():
        pass
    
    @staticmethod
    def zero_series_label(DFRead, columns=DFRead.COLIMNS):
        return pd.Series([0,0,0,0,0,0,0,0],index = columns)
    
    def read_csv_to_df(self, train_csv_list):
        return_dict = {}
        dir_list=[]
        df_concat = self.zero_series_label()
        for train_csv in train_csv_list:
            dir_path = train_csv.split('/')[-1].rstrip("_train.csv")
            print(train_csv)
            read_df = pd.read_csv(train_csv,index_col=0)
            df_concat = pd.concat([df_concat, read_df.T],axis=1)
            return_dict.update({dir_path:df_concat.T[1:]})
            dir_list.append(dir_path)
            df_concat = self.zero_series_label()
        return return_dict,dir_list
    
   
if __name__ == '__main__':
    current_path = os.getcwd()
    output_path = f"{current_path}/output"
    csv_path = f"{output_path}/csv"
    output_train_path = f"{output_path}/train"
    