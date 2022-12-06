from preProcessing_data import preProcessing
from  import_dataset import Dataset
import tensorflow as tf
import os
import random
from tqdm import tqdm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle

image_size = 256
batch = 8
epoch = 5


if(os.path.exists("image.pickle") and os.path.exists("label.pickle")):
    with open('image.pickle', 'rb') as handle:
        image_process = pickle.load(handle)
    with open('label.pickle', 'rb') as handle:
        label_process = pickle.load(handle)
else:
    dataset = Dataset()
    image_process, label_process = preProcessing.processing(dataset,256) #image's size is 256
    with open('dataset.pickle', 'wb') as handle:
        pickle.dump(image_process, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(label_process, handle, protocol=pickle.HIGHEST_PROTOCOL)

dataset = tf.data.Dataset.from_tensor_slices((image_process, label_process))
train_dataset_num = int(len(image_process) * 0.9)
test_dataset_num = len(image_process) - train_dataset_num
train_dataset = dataset.shuffle(train_dataset_num).repeat()

test_dataset = dataset.shuffle(test_dataset_num).repeat()
print("train_dataset: " + str(train_dataset.as_numpy_iterator()))


