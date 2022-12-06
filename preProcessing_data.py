import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
import matplotlib.pyplot as plt
from import_dataset import Dataset
import numpy as np
from tqdm import tqdm
import warnings

class preProcessing:
    @staticmethod
    def processing(dataset, pic_size):
        #Resize each image with pic_size
        image_process = []
        label_process = []
        for i in tqdm(range(1861,6435+1)):
            try:
                image, label, cropped_images = dataset[i]
            except:
                continue
            #print(i)
            if(len(cropped_images) != 0):
                if(label['NumOfAnno'] == 1):      # Check cropped image is 2d or 3d
                    image_resize = tf.image.resize(cropped_images, [pic_size, pic_size])
                    image_norm = tf.cast(image_resize, tf.float32) / 255
                    if(len(image_norm) != 256):
                        image_norm = image_norm[0]
                    image_process.append(image_norm)
                    label_process.append(label['Annotations'][0]['isProtected'])
                else:
                    for j in range(len(cropped_images)):
                        if(len(cropped_images[j]) != 0):       #Some images' crop only have 1 pixel, ignore them all
                            image_resize = tf.image.resize(cropped_images[j], [pic_size, pic_size])
                            image_norm = tf.cast(image_resize, tf.float32) / 255
                            if(len(image_norm) != 256):
                                image_norm = image_norm[0]
                            image_process.append(image_norm)
                            label_process.append(label['Annotations'][j]['isProtected'])
        return image_process, label_process

#warnings.filterwarnings(("ignore"))
dataset = Dataset()
preProcessing.processing(dataset,256)
print(1)
