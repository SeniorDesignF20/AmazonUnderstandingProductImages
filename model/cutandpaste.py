from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import random


def cutandpaste(image, boundingBox=False):
    im = np.copy(image)
    ysize, xsize = im.shape[0:2]
    xbox = int(xsize/4)
    ybox = int(ysize/4)

    xrand1 = int(random.random() * (xsize - xbox))
    yrand1 = int(random.random() * (ysize - ybox))
    xrand2 = int(random.random() * (xsize - xbox))
    yrand2 = int(random.random() * (ysize - ybox))

    box = im[yrand1:(yrand1 + ybox), xrand1:(xrand1 + xbox)]
    im[yrand2:(yrand2 + ybox), xrand2:(xrand2 + xbox)] = box

    if boundingBox:
        cv2.rectangle(im,(xrand1, yrand1),(xrand1 + xbox, yrand1 + ybox), [0,255,0], 1)
        cv2.rectangle(im,(xrand2, yrand2),(xrand2 + xbox, yrand2 + ybox), [0,0,255], 1) 
    return im
