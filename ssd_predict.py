import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input

from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf


from ssd_keras.ssd import SSD300
from ssd_keras.ssd_utils import BBoxUtility

class SsdPredict(object):
    def __init__(self):
        np.set_printoptions(suppress=True)
        voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        self.NUM_CLASSES = len(voc_classes) + 1

    def exec_predict(self,image_path):
        input_shape=(300, 300, 3)
        model = SSD300(input_shape, num_classes=self.NUM_CLASSES)
        model.load_weights('weights_SSD300.hdf5', by_name=True)
        bbox_util = BBoxUtility(self.NUM_CLASSES)
        inputs = self.preprocess_image_per_dir(image_path)
        preds = model.predict(inputs, batch_size=1, verbose=1)
        results = bbox_util.detection_out(preds)

        return results

    def preprocess_image_per_dir(self,image_path_list):
        inputs = []
        for image_path in image_path_list:
            img = image.load_img(img_path, target_size=(300, 300))
            img = image.img_to_array(img)
            inputs.append(img)

        inputs = preprocess_input(np.array(inputs))
        return input
