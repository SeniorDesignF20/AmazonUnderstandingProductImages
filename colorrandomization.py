# Changes the values of the RGB channels in random locations

import numpy as np
import cv2
import os

def singleChannel(image):
	# Changes the value of one of the RGB channels in a local area

    im = cv2.imread(image)
    shape = im.shape
    
    channel = np.random.randint(3)
    
    newvalue = np.random.randint(256)
    boxsize = int(np.random.randint(max(shape)/2)/3)
    
    
    bias1 = np.random.randint(-50, 50)
    bias2 = np.random.randint(-50, 50)
    h = int(shape[0]/2) + bias1
    w = int(shape[1]/2) + bias2
    
    for i in range(h-boxsize, h+boxsize):
        for j in range(w-boxsize, w+boxsize):
            im[i,j,channel] = newvalue
    
    return im

def multipleChannels(iamge):
    # Changes the value of one of the RGB channels in a local area

    im = cv2.imread(image)
    shape = im.shape
    
    boxsize = int(np.random.randint(max(shape)/2)/3)
    
    
    bias1 = np.random.randint(-50, 50)
    bias2 = np.random.randint(-50, 50)
    h = int(shape[0]/2) + bias1
    w = int(shape[1]/2) + bias2
    
    for i in range(h-boxsize, h+boxsize):
        for j in range(w-boxsize, w+boxsize):
            channel = np.random.randint(3)
            newvalue = np.random.randint(256)
            im[i,j,channel] = newvalue
    
    return im
    


benign_directory = r'Benign\\'
manipulated_directory1 = r"SingleChannel\\"
manipulated_directory2 = r"MultipleChannels\\"
for filename in os.listdir(benign_directory):
    image = benign_directory + filename
    destination = manipulated_directory1 + filename
    manipulated_image = singleChannel(image)
    cv2.imwrite(destination, manipulated_image)
    
    destination = manipulated_directory2 + filename
    manipulated_image = multipleChannels(image)
    cv2.imwrite(destination, manipulated_image)
    
    
    