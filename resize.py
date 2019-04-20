import os
import cv2
import matplotlib as mtl
import pandas
import keras
import glob

class Resize():
    def __init__(self):
        self.current_path = os.getcwd()
        self.image_dir_path = self.current_path + '/car_image/'
        self.csv_path = self.current_path + '/csv/'

    def resize_all_image(self, image_list):
        for image_path in image_list:
            csv_df = pd.read_csv(csv_path + '20190123img.csv')
            im = cv2.imread(image_path)
            print(image_path)

    #plate_vertex = csv_df.query()

    #car_vertex, plate_vertex = [[lu, ld, rd, ru], plate_vertex] = car_search_model(im, plate_vertex)

    #car_image = car_image_resize(im, car_vertex)
    #plate_iamge = plate_image_resize(plate_iamge, plate_vertex)

    #new_image_df.append([image_num, plate_iamge])

#new_image_df.write_csv(current_path + '/csv/resized_plate.csv')
#

if __name__ == '__main__':
    resize = Resize()
    print("ww")
    image_list = glob.glob(image_dir_path + '*.JPG')
    resize.resize_all_image(image_list)
