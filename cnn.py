from preProcessing_data import preProcessing
from  import_dataset import Dataset
import tensorflow as tf
import random
import cv2 as cv


dataset = Dataset()
image_process, label_process = preProcessing.processing(dataset,256) #image's size is 256
f = open("label.txt", "w")

for i in range(image_process):
    cv.imwrite(i+'.png', image_process, [cv.IMWRITE_PNG_COMPRESSION, 0])
    f.write(label_process['Annotations'][i]['isProtected'])
f.close()

dataset = tf.data.Dataset.from_tensor_slices((image_process, label_process))
dataset = dataset.shuffle(len(image_process)).repeat()
dataset.map
