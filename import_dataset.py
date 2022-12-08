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
            cropped_images,cropped_label = self.crop(image, data)


        else:
            image_name = str(id) + '.png'
            if image_name in self.imgs:
                image_path = os.path.join("../Medical mask/Medical mask/images/", image_name)
                image = plt.imread(image_path)
                label_name = image_name + ".json"
                label_path = os.path.join("../Medical mask/Medical mask/annotations/", label_name)
                label = open(label_path)
                data = json.load(label) # data is a dictionary
                cropped_images,cropped_label = self.crop(image, data)
            else:
                image_name = str(id) + '.jpeg'
                image_path = os.path.join("../Medical mask/Medical mask/images/", image_name)
                image = plt.imread(image_path)
                label_name = image_name + ".json"
                label_path = os.path.join("../Medical mask/Medical mask/annotations/", label_name)
                label = open(label_path)
                data = json.load(label) # data is a dictionary
                cropped_images,cropped_label = self.crop(image, data)

        if(isinstance(cropped_images[0][0][0][0], np.float32)):
            for i in range(len(cropped_images)):
                cropped_images[i] = (cropped_images[i] * 255).astype(int)
        # print(cropped_images[0][0][0][0])
        # print(type(cropped_images[0][0][0][0]))
        return image, cropped_label, cropped_images

    def crop(self, image, label: list) -> [list, list]:
        """

        :param image: image array
        :param label: original data from annotation file
        :return: cropped images list
        """
        number = label['NumOfAnno']
        output_image = []
        output_label = label.copy()
        output_label['Annotations'] = label['Annotations'].copy()
        output_label['Annotations'].clear()
        output_label['NumOfAnno'] = 0
        for i in range(number):
            if label['Annotations'][i]['classname'] in ['scarf_banana', 'balaclava_ski_mask', 'turban', 'helmet', 'sunglasses', 'eyeglasses', 'hair_net', 'hat', 'goggles', 'hood','mask_colorful','mask_surgical','face_other_covering']:
                continue
            x, y, w, h = label['Annotations'][i]['BoundingBox']
            img_cropped = image[y:h, x:w, :]
            output_image.append(img_cropped)
            output_label['Annotations'].append(label['Annotations'][i])
            output_label['NumOfAnno'] += 1
        return output_image,output_label

# # example of how to use the Dataset
# dataset = Dataset()
# image, label, cropped_images = dataset[1861]
# print(label)
# plt.figure()
# plt.imshow(image)
# for i in range(len(cropped_images)):
#     plt.figure()
#     plt.imshow(cropped_images[i])
#
# for items in label['Annotations']:
#     print(items)
# # label[NumOfAnno] is the number of annotations
# # label[Annotations] is a list with dictionaries, containing 'BoundingBox' and 'isProtected'
# plt.show()

# crop the image and get the faces

