from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import random
from countPixels import countPixels

# Copies a portion of an image and pastes it somewhere else
# Use countPixels to ensure that the resulting augmented image is sufficiently different than the original
def cutandpaste(image, boundingBox=False):
    im = np.copy(image)
    ysize, xsize = im.shape[0:2]
    xbox = int(xsize/3)
    ybox = int(ysize/3)

    xrand1 = int(random.random() * (xsize - xbox))
    yrand1 = int(random.random() * (ysize - ybox))
    xrand2 = int(random.random() * (xsize - xbox))
    yrand2 = int(random.random() * (ysize - ybox))

    box = im[yrand1:(yrand1 + ybox), xrand1:(xrand1 + xbox)]
    im[yrand2:(yrand2 + ybox), xrand2:(xrand2 + xbox)] = box

    count = 20
    while countPixels(im, image) < .075 and count != 0:
        im = np.copy(image)
        xrand1 = int(random.random() * (xsize - xbox))
        yrand1 = int(random.random() * (ysize - ybox))
        xrand2 = int(random.random() * (xsize - xbox))
        yrand2 = int(random.random() * (ysize - ybox))

        box = im[yrand1:(yrand1 + ybox), xrand1:(xrand1 + xbox)]
        im[yrand2:(yrand2 + ybox), xrand2:(xrand2 + xbox)] = box
        count -= 1



    # If you want the cutandpaste image to have rectangles showing where the pasted portion came from and went to
    if boundingBox:
        cv2.rectangle(im,(xrand1, yrand1),(xrand1 + xbox, yrand1 + ybox), [0,255,0], 1)
        cv2.rectangle(im,(xrand2, yrand2),(xrand2 + xbox, yrand2 + ybox), [0,0,255], 1) 
    return im
