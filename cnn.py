from preProcessing_data import preProcessing
from  import_dataset import Dataset
import tensorflow as tf
import os
import random
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

# Hyper params
image_size = 224
batch = 8
num_epoch = 3
learning_rate = 0.0001
num_classes = 2

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, 2))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = AlexNet().to(device)

if(os.path.exists("image.pickle") and os.path.exists("label.pickle")):
    with open('image.pickle', 'rb') as handle:
        image_process = pickle.load(handle)
    with open('label.pickle', 'rb') as handle:
        label_process = pickle.load(handle)
else:
    dataset = Dataset()
    image_process, label_process = preProcessing.processing(dataset,224) #image's size is 224
    with open('image.pickle', 'wb') as handle:
        pickle.dump(image_process, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('label.pickle', 'wb') as handle:
        pickle.dump(label_process, handle, protocol=pickle.HIGHEST_PROTOCOL)

# dataset = tf.data.Dataset.from_tensor_slices((image_process, label_process))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.471, 0.48, 0.408], std=[0.234, 0.239, 0.242])
])


X_train, X_test, y_train, y_test = train_test_split(image_process, label_process,
                                                    test_size=0.2, shuffle = True, random_state = 8)

# Use the same function above for the validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.25, random_state= 8)

print("X_train shape: {}".format(len(X_train)))
print("X_test shape: {}".format(len(X_test)))
print("y_train shape: {}".format(len(y_train)))
print("y_test shape: {}".format(len(y_test)))
print("X_val shape: {}".format(len(X_val)))
print("y val shape: {}".format(len(y_val)))



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(X_train)

for epoch in range(num_epoch):
    print("total lenth is ", len(X_train))
    for i in tqdm(range(len(X_train))):
        # print("running with", i)
        # Convert images to tensor
        images = X_train[i]
        images = transform(images)
        # now image is tensor of image

        labels = torch.tensor(int(y_train[i]))

        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        images_t = torch.unsqueeze(images, 0)
        outputs = model(images_t)
        # _, indices = torch.sort(outputs, descending=True)
        # print(indices)
        labels_t = torch.unsqueeze(labels, 0)
        loss = criterion(outputs, labels_t)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
           .format(epoch+1, num_epoch, i+1, total_step, loss.item()))

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(len(X_val)):
            # Convert images to tensor
            images = X_val[i]
            images = transform(images)
            # now image is tensor of image

            labels = torch.tensor(int(y_val[i]))

            images = images.to(device)
            labels = labels.to(device)
            images_t = torch.unsqueeze(images, 0)
            outputs = model(images_t)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_t.size(0)
            correct += (predicted == labels_t).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))





