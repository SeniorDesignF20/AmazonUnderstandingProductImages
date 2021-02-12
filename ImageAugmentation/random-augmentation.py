import colorrandomization
import cutandpaste
import lighting
import saturation
from PIL import Image
import cv2
import os
import numpy as np

# Script to perform random augmentations on Amazon dataset

count = 0
benign_src_path = r'../AmazonSet/TestSet/Benign_new'
manipulated_dest_path = r'../AmazonSet/rand_Manipulated'


def rand_augment(imFile):
    global count
    image = cv2.imread(os.path.join(benign_src_path, imFile))
    flags = np.random.randint(2, size=5)
    # Cut and paste augmentation
    if flags[0] == 1:
        image = cutandpaste.cutandpaste(image)
    # Single channel augmentation
    if flags[1] == 1:
        image = colorrandomization.singleChannel(image)
    # Mutli channel augmentation
    if flags[2] == 1:
        image = colorrandomization.multipleChannels(image)
    # Spotlight augmentation
    if flags[3] == 1:
        image = lighting.add_spot_light(image)
    if flags[4] == 1:
        image = saturation.saturate(image)
    name = 'm' + '{:05d}'.format(count) + '.jpg'
    cv2.imwrite(manipulated_dest_path + '/' + name, image)
    count += 1


for filename in os.listdir(benign_src_path):
    if filename.endswith('.jpg'):
        rand_augment(filename)
