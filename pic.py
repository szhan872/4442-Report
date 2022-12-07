from import_dataset import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from preProcessing_data import preProcessing
# example of how to use the Dataset
dataset = Dataset()
origini, _, _= dataset[1801]
i, l = preProcessing.processing(dataset,256)
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(origini)
plt.title("Original Image")
plt.subplot(1,2,2)
plt.imshow(i[0])
plt.title("After Processing")
plt.show()
isPro = 0
notPro = 0
for i in tqdm(range(1801,1802)):
    try:
        image, label, cropped_images = dataset[i]
    except:
        continue
    #print(i)
    if(len(cropped_images) != 0):
        if(label['NumOfAnno'] == 1):      # Check cropped image is 2d or 3d
            if(label['Annotations'][0]['classname'] == 'face_with_mask'):
                isPro += 1
            else:
                notPro += 1
                print(label['Annotations'][0]['classname'])
        else:
            for j in range(len(cropped_images)):
                if(len(cropped_images[j]) != 0):       #Some images' crop only have 1 pixel, ignore them all
                    if(label['Annotations'][j]['classname'] == 'face_with_mask'):
                        isPro += 1
                    else:
                        notPro += 1

print("Is Pro: " + str(isPro))
print("Not Pro: " + str(notPro))
plt.pie([isPro,notPro],
        labels = ["Wear Mask", "Not Wear Mask"],
        autopct='%.1f%%',
        )
plt.title("Protected Distribution on the Dataset")
plt.show()
