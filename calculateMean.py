import numpy as np
import cv2
import random
from import_dataset import Dataset
from tqdm import tqdm_notebook
import random
from preProcessing_data import preProcessing
# calculate means and std
train_txt_path = './train_val_list.txt'
CNum = 10000 # 挑选多少图片进行计算
img_h, img_w = 32, 32
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []
dataset = Dataset()
image_process, label_process = preProcessing.processing(dataset,224)
for i in tqdm_notebook(range(CNum)):
  img, _ = dataset[random.random(i)]
  img = cv2.resize(img, (img_h, img_w))
  img = img[:, :, :, np.newaxis]
  imgs = np.concatenate((imgs, img), axis=3)
  # print(i)
imgs = imgs.astype(np.float32)/255.
for i in tqdm_notebook(range(3)):
      pixels = imgs[:,:,i,:].ravel() # 拉成一行
      means.append(np.mean(pixels))
      stdevs.append(np.std(pixels))
# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse() # BGR --> RGB
stdevs.reverse()
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
