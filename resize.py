import os
import cv2
import matplotlib as mtl
import pandas
import keras
import glob
from my_yolo import MyYOLO
import pandas as pd
import re
import numpy as np
from PIL import Image

class Resize():
    def __init__(self):
        self.current_path = os.getcwd()
        self.image_dir_path = self.current_path + '/car_image/'
        self.csv_path = self.current_path + '/csv/'
        self.myYolo= MyYOLO()

        self.root_path = os.getcwd()
        self.output_path = f"{self.root_path}/output/"
        self.output_car_path = f"{self.output_path}car/"
        self.output_csv_path = f"{self.output_path}csv/"
        self.train_csv_path = f"{self.output_path}train/"
        self.target_dir = ""
        self.img_ext = ".jpg"

    def choice_box(self, vertex, out_boxes, out_scores, out_classes, image_size):
        predict_pos=[]
        x=[vertex["lu_x"],vertex["ld_x"],vertex["ru_x"],vertex["rd_x"]]
        y=[vertex["lu_y"],vertex["ld_y"],vertex["ru_y"],vertex["rd_y"]]

        for (out_box, out_score, out_class) in zip(out_boxes,out_scores,out_classes):
            pos_dict = self.created_boxes_vertex_dict(out_box, image_size)
            print('==========')
            print(pos_dict)
            print(x)
            x_pos = self.xpos_in_box(pos_dict, x)
            y_pos = self.ypos_in_box(pos_dict, y)

            if x_pos and y_pos:
                pos_dict.update({"score":out_score})
                predict_pos.append(pos_dict)
        if len(predict_pos) >= 2:
            max_score=0
            for target_box in predict_pos:
                if target_box["score"]>max_score:
                    max_score = target_box["score"]
                    return_box = target_box
                else:
                    return_box = None
            return return_box
        elif len(predict_pos) == 0:
            return None
        return predict_pos[0]

    def created_boxes_vertex_dict(self, out_box, size):
        top, left, bottom, right = out_box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(size[0], np.floor(right + 0.5).astype('int32'))
        box = [top, left, bottom, right]

        pos_dict={}
        pos_name = ["top", "left", "bottom", "right"]
        for pos,name in zip(box, pos_name):
            pos_dict[name] = pos
        return pos_dict

    def xpos_in_box(self, pos_dict, arr):
        print(f'pos_dict:{pos_dict}')
        print(f'arr:{arr}')
        print(type(pos_dict["left"]))
        print(pos_dict["left"] )

        for x_pos in arr:
            print(type(x_pos))
            print(x_pos)
            if (pos_dict["left"] < x_pos < pos_dict["right"]) == False:
                return False
        return True

    def ypos_in_box(self, pos_dict, arr):
        for y_pos in arr:
            if (pos_dict["top"] < y_pos < pos_dict["bottom"]) == False:
                return False
        return True

    def move_and_update(self, vertex, corner_name, position):
        move_pos = vertex[corner_name] - position
        vertex.at[corner_name] = move_pos
        return move_pos, vertex

    def number_plate_crop(self, image, vertex, predict_pos, file_name):
        import cv2
        import numpy as np
        tmp_image = np.asarray(image)
        ocv_image = tmp_image[:, :, ::-1].copy()

        move_lux, vertex = self.move_and_update(vertex, "lu_x", predict_pos["left"])
        move_ldx, vertex = self.move_and_update(vertex, "ld_x", predict_pos["left"])
        move_rux, vertex = self.move_and_update(vertex, "ru_x", predict_pos["left"])
        move_rdx, vertex = self.move_and_update(vertex, "rd_x", predict_pos["left"])
        move_luy, vertex = self.move_and_update(vertex, "lu_y", predict_pos["top"])
        move_ldy, vertex = self.move_and_update(vertex, "ld_y", predict_pos["top"])
        move_ruy, vertex = self.move_and_update(vertex, "ru_y", predict_pos["top"])
        move_rdy, vertex = self.move_and_update(vertex, "rd_y", predict_pos["top"])

        pts1 = np.float32([[move_lux,move_luy],[move_rux,move_ruy],[move_ldx,move_ldy],[move_rdx,move_rdy]])
        pts2 = np.float32([[0,0],[640,0],[0,320],[640,320]])
        #透視変換
        M = cv2.getPerspectiveTransform(pts1,pts2)
        rst = cv2.warpPerspective(ocv_image,M,(640,320))
        save_img_path = f'{self.output_path}number/{self.target_dir}/'
        save_img_name = f'{file_name}{self.img_ext}'

        os.makedirs(f'{save_img_path}', exist_ok=True)
        cv2.imwrite(f'{save_img_path}{save_img_name}', rst)

        return vertex

    def resize_image(self, read_image,file_num, csv_df):
        img_width, img_height = read_image.size
        print(f'read_image.size:{read_image.size}')

        width = 200
        height = 200
        size = (width, height)
        resize_image=read_image.resize(size, Image.LANCZOS)

        resized_df = csv_df.copy()

        zip_ratio_w = img_width/width
        zip_ratio_h = img_height/height
        print(f'zip_ratio_w:{zip_ratio_w}')
        print(f'zip_ratio_h:{zip_ratio_h}')
        print(resized_df.iloc[::2])
        print(resized_df.iloc[1::2])
        resized_df.iloc[::2] /= zip_ratio_w
        resized_df.iloc[1::2] /= zip_ratio_h
        print(resized_df <= 200)
        if (resized_df <= 200).all() == False:
            raise Exception("200!")
        print(f'resized_df:{resized_df}')
        print(self.target_dir)

        os.makedirs(f"./output/train/{self.target_dir}", exist_ok=True)
        resize_image.save(f"./output/train/{self.target_dir}/train_{file_num}{self.img_ext}")

        return resized_df

    def crop_img(self, excel_df, kv, df_concat, df_resize_concat, output):
        df_na = excel_df.dropna()

        # df = df_na[1:]
        df = df_na
        for dirr in kv.keys():
            ls = [filename for filename in os.listdir(dirr) if not filename.startswith('.')]
            count=0

            for path in ls:
                self.file_name = path.split('/')[-1]
                self.file_num = re.sub(r'\.jpeg|\.jpg|\.png|\.JPG|\.PNG', '', self.file_name)
                print("---")
                print(f"file_num:{self.file_num}")
                vertex = df.loc[self.file_num]
                if isinstance(vertex, pd.DataFrame):
                    print("!!!!!!!!!!!!!!!")
                    for i, (ind, ser) in enumerate(vertex.iterrows()):
                        self.file_name = f"{ind}-{i}"
                        self.file_num = self.file_name
                        print(self.file_name)
                        rename_ser = ser.rename(self.file_name)
                        df_concat,df_resize_concat = self.img_df_resize(rename_ser,dirr, path,df_concat, df_resize_concat, yolo_path=ind)
                else:
                    df_concat,df_resize_concat = self.img_df_resize(vertex,dirr, path,df_concat, df_resize_concat)


            df_T=df_concat.T
            df_T[1:].to_csv(f'{output}.csv')
            df_concat_T=df_resize_concat.T
            df_concat_T[1:].to_csv(f'{output}_train.csv')

    def img_df_resize(self, vertex,dirr, path,df_concat,df_resize_concat, yolo_path=None):
        image = Image.open(f'{dirr}/{path}')
        tmp = cv2.imread(f'{dirr}/{path}')
        height, width = tmp.shape[:2]
        if (height > width):
            image = image.rotate(270, expand=True)
        del tmp
        image_size = image.size
        org_image = image.copy()
        image, out_boxes, out_scores, out_classes = self.myYolo.detect_image(image)

        if yolo_path is None:
            yolo_path = self.file_name
        yolo_dir = f"{self.output_path}yolo/{self.target_dir}"
        os.makedirs(yolo_dir, exist_ok=True)
        image.save(f"{yolo_dir}/{yolo_path}{self.img_ext}")
        predict_pos = self.choice_box(vertex, out_boxes, out_scores, out_classes, image_size)
        if predict_pos is None:
            del image,org_image,vertex
            print(f"{self.file_name} is skipped")
            return df_concat,df_resize_concat
        plate_npx=np.array([vertex["lu_x"],vertex["ld_x"],vertex["ru_x"],vertex["rd_x"]])
        plate_npy=np.array([vertex["lu_y"],vertex["ld_y"],vertex["ru_y"],vertex["rd_y"]])
        # one car
        one_car_img = org_image.crop((predict_pos['left'], predict_pos['top'], predict_pos['right'], predict_pos['bottom']))
        car_img_path = f'{self.output_car_path}{self.target_dir}/{self.file_name}'
        one_car_img.save(f"{car_img_path}{self.img_ext}")
        moved_vertex = self.number_plate_crop(one_car_img, vertex, predict_pos, self.file_name)

        df_concat = pd.concat([df_concat, moved_vertex],axis=1)
        # detect_char_on_plate()
        resized_df = self.resize_image(one_car_img, self.file_num, moved_vertex)
        df_resize_concat = pd.concat([df_resize_concat, resized_df],axis=1)
        del image,org_image,vertex
        # print(df_concat)
        return df_concat,df_resize_concat
    #plate_vertex = csv_df.query()

    #car_vertex, plate_vertex = [[lu, ld, rd, ru], plate_vertex] = car_search_model(im, plate_vertex)

    #car_image = car_image_resize(im, car_vertex)
    #plate_iamge = plate_image_resize(plate_iamge, plate_vertex)

    #new_image_df.append([image_num, plate_iamge])

    #new_image_df.write_csv(current_path + '/csv/resized_plate.csv')
    #

    def get_img_ls(self, dir_path):
        jpeg_ls = glob.glob(f"{dir_path}/*.jpeg")
        jpg_ls = glob.glob(f"{dir_path}/*.jpg")
        JPG_ls = glob.glob(f"{dir_path}/*.JPG")
        png_ls = glob.glob(f"{dir_path}/*.png")
        PNG_ls = glob.glob(f"{dir_path}/*.PNG")
        img_ls = jpeg_ls + jpg_ls + JPG_ls + png_ls + PNG_ls
        return img_ls

    def get_target_csv(self, csv_path):
        read_csv = pd.read_csv(csv_path, dtype = 'object', index_col=1)
        d_type = {'lu_x': 'int32', 'lu_y': 'int32', 'ru_x': 'int32', 'ru_y': 'int32', 'ld_x': 'int32', 'ld_y': 'int32', 'rd_x': 'int32', 'rd_y': 'int32'}
        read_csv = read_csv.astype(d_type)

        csv_df = read_csv.loc[:, 'lu_x':'rd_y']
        return csv_df

if __name__ == '__main__':
    resize = Resize()

    columns = ["lu_x","lu_y","ru_x","ru_y","ld_x","ld_y","rd_x","rd_y"]
    df_concat = pd.Series([0,0,0,0,0,0,0,0],index = columns)
    df_resize_concat = pd.Series([0,0,0,0,0,0,0,0],index = columns)

    #excel_df = pd.read_excel(f'./{dir_name}/{dir_name}list.xlsx',names=columns, index=0)


    output_path = resize.output_path
    output_car_path = resize.output_car_path
    root_path = resize.root_path
    output_csv_path = resize.output_csv_path

    img_dir_path = glob.glob(f"{root_path}/data/img/*")
    all_kv={}

    # test
    #img_dir_path=['/Users/nakamura.yuta/Desktop/num_plate/car_number/data/img/001','/Users/nakamura.yuta/Desktop/num_plate/car_number/data/img/002']

    for dirr in img_dir_path:
        pattern = '.*/(.*)$'
        result = re.findall(pattern, dirr)
        os.makedirs(f"{output_car_path}{result[0]}", exist_ok=True)

        print(result)
        resize.target_dir = result[0]

        dir_path = f"{root_path}/data/img/{result[0]}"
        img_ls = resize.get_img_ls(dir_path)
        all_kv.update({dirr:img_ls})


        csv_path = f"{root_path}/data/csv/{result[0]}.csv"
        csv_df = resize.get_target_csv(csv_path)
        output = f'{output_csv_path}{result[0]}'
        resize.crop_img(csv_df, all_kv, df_concat, df_resize_concat, output)
        all_kv={}

    # image_list = glob.glob(image_dir_path + '*.JPG')
