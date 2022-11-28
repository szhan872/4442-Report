import os
import matplotlib.pyplot as plt
import numpy as np
import json


class Dataset(object):
    def __init__(self):
        self.imgs = list(sorted(os.listdir("../Medical mask/Medical mask/images/")))
        self.labels = list(sorted(os.listdir("../Medical mask/Medical mask/images/")))

    def __getitem__(self, id: int) -> [np.array, dict, list]:
        """

        :param id: index of the image in form of 4 digits (e.g. 0123)
        :returns: [array: image array read by imread, dict: json labels, list: cropped images]
        """
        image_name = str(id) + '.jpg'
        if image_name in self.imgs:
            image_path = os.path.join("../Medical mask/Medical mask/images/", image_name)
            image = plt.imread(image_path)

            label_name = image_name + ".json"
            label_path = os.path.join("../Medical mask/Medical mask/annotations/", label_name)
            label = open(label_path)
            data = json.load(label) # data is a dictionary
            cropped_images = self.crop(image, data)


        else:
            image_name = str(id) + '.png'
            image_path = os.path.join("../Medical mask/Medical mask/images/", image_name)
            image = plt.imread(image_path)
            label_name = image_name + ".json"
            label_path = os.path.join("../Medical mask/Medical mask/annotations/", label_name)
            label = open(label_path)
            data = json.load(label) # data is a dictionary
            cropped_images = self.crop(image, data)


        return image, data, cropped_images

    def crop(self, image, label: list) -> list:
        """

        :param image: image array
        :param label: original data from annotation file
        :return: cropped images list
        """
        number = label['NumOfAnno']
        output = []
        for i in range(number):
            x, y, w, h = label['Annotations'][i]['BoundingBox']
            img_cropped = image[y:h, x:w, :]
            output.append(img_cropped)
        return output

# example of how to use the Dataset
dataset = Dataset()
image, label, cropped_images = dataset[4313]
plt.figure()
plt.imshow(image)
for i in range(len(cropped_images)):
    plt.figure()
    plt.imshow(cropped_images[i])

for items in label['Annotations']:
    print(items)
# label[NumOfAnno] is the number of annotations
# label[Annotations] is a list with dictionaries, containing 'BoundingBox' and 'isProtected'
plt.show()

# crop the image and get the faces

