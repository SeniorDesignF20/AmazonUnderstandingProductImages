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
benign_src_path = r'AmazonSet\TestSet\Benign_new'
buffer_path = r'AmazonSet\rand_Buffer'
manipulated_dest_path = r'AmazonSet\rand_Manipulated'


def rand_augment(imFile):
    global count
    cv2im = cv2.imread(os.path.join(benign_src_path, imFile))
    pilim = Image.open(os.path.join(benign_src_path, imFile))
    # Buffer used to allow for conversion from PIL image to cv2 image, there's probably a better way of doing this
    bufname = 'buf' + '{:05d}'.format(count) + '.jpg'
    cv2.imwrite(os.path.join(buffer_path, bufname), cv2im)
    flags = np.random.randint(2, size=5)
    # Cut and paste augmentation
    if flags[0] == 1:
        cutim = Image.open(os.path.join(buffer_path, bufname))
        cutim = cutandpaste.cutandpaste(cutim)
        cutim.save(os.path.join(buffer_path, bufname))
    # Single channel augmentation
    if flags[1] == 1:
        singim = cv2.imread(os.path.join(buffer_path, bufname))
        singim = colorrandomization.singleChannel(singim)
        cv2.imwrite(os.path.join(buffer_path, bufname), singim)
    # Mutli channel augmentation
    if flags[2] == 1:
        multim = cv2.imread(os.path.join(buffer_path, bufname))
        multim = colorrandomization.multipleChannels(multim)
        cv2.imwrite(os.path.join(buffer_path, bufname), multim)
    # Spotlight augmentation
    if flags[3] == 1:
        spotim = cv2.imread(os.path.join(buffer_path, bufname))
        spotim = lighting.add_spot_light(spotim)
        cv2.imwrite(os.path.join(buffer_path, bufname), spotim)
    # if flags[4] == 1:
    #     satim = cv2.imread(os.path.join(buffer_path, bufname))
    #     satim = saturation.saturate(satim)
    #     cv2.imwrite(os.path.join(buffer_path, bufname), satim)
    retim = Image.open(os.path.join(buffer_path, bufname))
    name = 'm' + '{:05d}'.format(count) + '.jpg'
    retim.save(os.path.join(manipulated_dest_path, name))
    count += 1


for filename in os.listdir(benign_src_path):
    if filename.endswith('.jpg'):
        rand_augment(filename)
