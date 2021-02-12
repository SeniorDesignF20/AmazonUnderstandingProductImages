# Changes the values of the RGB channels in random locations

import numpy as np
import cv2
import os


def singleChannel(im):
    # Changes the value of one of the RGB channels in a local area

    # im = cv2.imread(image)
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
            if i >= shape[0]:
                i = shape[0] - 1
            if j >= shape[1]:
                j = shape[1] - 1
            im[i, j, channel] = newvalue

    return im


def multipleChannels(im):
    # Changes the value of one of the RGB channels in a local area

    # im = cv2.imread(image)
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
            if i >= shape[0]:
                i = shape[0] - 1
            if j >= shape[1]:
                j = shape[1] - 1
            im[i, j, channel] = newvalue

    return im

