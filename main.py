import numpy as np
import pandas as pd
import re
import glob
import os
import sys
path = os.path.join(os.path.dirname(os.path.abspath("__file__")), 'keras_yolo3/')
sys.path.append(path)

from PIL import Image
from my_yolo import MyYOLO
from utils import PlateUtils
myYolo= MyYOLO()
pu = PlateUtils()

columns = ["lu_x","lu_y","ru_x","ru_y","ld_x","ld_y","rd_x","rd_y"]
df_concat = pd.Series([0,0,0,0,0,0,0,0],index = columns)
date = 20190125
excel_df = pd.read_excel(f'./{date}/{date}list.xlsx',names=columns, index=0)
df_na = (excel_df.iloc[1:,:]).dropna()
df = df_na[1:]
img_path = f"/Users/naka345/Desktop/deeplearning/number_plate/{date}/{date}img"
output_path = "/Users/naka345/Desktop/deeplearning/number_plate/output/car/"
output_csv_path = "/Users/naka345/Desktop/deeplearning/number_plate/output/csv/"
ls = glob.glob(img_path + "/*.JPG")
c=0
for path in ls:
    file_name = path.split('/')[-1]
    file_num = re.sub(r'\D', '', file_name)
    vertex = df.loc[int(file_num)]
    print(file_name)
    image = Image.open(path)
    image = image.rotate(270, expand=True)
    image_size = image.size
    org_image = image.copy()
    image, out_boxes, out_scores, out_classes = myYolo.detect_image(image)
    image.save(output_path + '../' + file_name)
    predict_pos = pu.choice_box(vertex, out_boxes, out_scores, out_classes, image_size)
    if predict_pos is None:
        del image,org_image,vertex
        continue
    plate_npx=np.array([vertex["lu_x"],vertex["ld_x"],vertex["ru_x"],vertex["rd_x"]])
    plate_npy=np.array([vertex["lu_y"],vertex["ld_y"],vertex["ru_y"],vertex["rd_y"]])
    # one car
    one_car_img = org_image.crop((predict_pos['left'], predict_pos['top'], predict_pos['right'], predict_pos['bottom']))
    one_car_img.save(output_path + file_name)
    moved_vertex = pu.number_plate_crop(one_car_img, vertex, predict_pos, file_name)

    df_concat = pd.concat([df_concat, moved_vertex],axis=1)
    # detect_char_on_plate()
    del image,org_image,vertex

df_T=df_concat.T
df_T[1:].to_csv(f'{output_csv_path}{date}.csv')
