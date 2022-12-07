import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
from import_dataset import Dataset
import numpy as np
from tqdm import tqdm
import warnings
from torchvision import transforms

class preProcessing:
    @staticmethod
    def processing(dataset, pic_size):

        transform = transforms.Compose([
            transforms.Resize((224,224))
            #,transforms.Normalize(mean=[0.468, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        #Resize each image with pic_size
        image_process = []
        label_process = []
        for i in tqdm(range(1801,1900)):
            try:
                image, label, cropped_images = dataset[i]
            except:
                continue
            #print(i)
            if(len(cropped_images) != 0):
                if(label['NumOfAnno'] == 1):     # Check cropped image is 2d or 3d
                    try:
                        PIL_image = Image.fromarray(cropped_images[0])
                        image_resize = transform(PIL_image)
                    except:
                        # image_process.append(123) # the cropped is not valid
                        # label_process.append(label['Annotations'][0]['isProtected'])
                        continue
                    # image_resize = tf.image.resize(cropped_images, [pic_size, pic_size])
                    # image_norm = tf.cast(image_resize, tf.float32) / 255
                    # if(len(image_norm) != 256):
                    #     image_norm = image_norm[0]
                    image_process.append(image_resize)

                    # if()

                    # label_process.append(label['Annotations'][0]['isProtected'])
                    if(label['Annotations'][0]['classname'] == 'face_with_mask'):
                        label_process.append(True)
                    else:
                        label_process.append(False)
                else:
                    for j in range(len(cropped_images)):
                        if(len(cropped_images[j]) != 0):       #Some images' crop only have 1 pixel, ignore them all

                            PIL_image = Image.fromarray(cropped_images[j])
                            image_resize = transform(PIL_image)


                            # image_resize = tf.image.resize(cropped_images[j], [pic_size, pic_size])
                            # image_norm = tf.cast(image_resize, tf.float32) / 255
                            # if(len(image_norm) != 256):
                            #     image_norm = image_norm[0]
                            image_process.append(image_resize)
                            # label_process.append(label['Annotations'][j]['isProtected'])
                            if(label['Annotations'][j]['classname'] == 'face_with_mask'):
                                label_process.append(True)
                            else:
                                label_process.append(False)
        return image_process, label_process

# #warnings.filterwarnings(("ignore"))
# dataset = Dataset()
# preProcessing.processing(dataset,256)
# print(1)
