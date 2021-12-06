import ipdb
import numpy as np
import os
import pandas as pd
import cv2


dir_name = "data_augmentation/images"
base_dir = "data/images"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

data = pd.read_csv(
    "/home/dhagash/MS-GE-03/CudaVision/assignment-4/data/annotations/list.txt", delimiter=" ", header=None, skiprows=6)

cats = data[data[2] == 1].sample(1000)
dogs = data[data[2] == 2].sample(1000)


labels = []
for i in range(cats.shape[0]):

    img_path = os.path.join(base_dir, cats.iloc[i, 0] + '.jpg')
    img_path2 = os.path.join(base_dir, dogs.iloc[i, 0] + '.jpg')
    img = cv2.imread(img_path)
    img2 = cv2.imread(img_path2)

    cv2.imwrite(os.path.join(dir_name, cats.iloc[i, 0] + '.jpg'), img)
    cv2.imwrite(os.path.join(dir_name, dogs.iloc[i, 0] + '.jpg'), img2)

    labels.append([cats.iloc[i, 0] + '.jpg', cats.iloc[i, 2]])
    labels.append([dogs.iloc[i, 0] + '.jpg', dogs.iloc[i, 2]])


# ipdb.set_trace()

labels = np.asarray(labels)
np.savetxt("data_augmentation/trainval.txt", labels, fmt='%s')


data = data.drop(cats.index)
data = data.drop(dogs.index)

data = np.asarray(data)
np.savetxt("data_augmentation/test.txt", data, fmt='%s')


# Image Load


# Image Augmentation
