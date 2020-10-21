#!/usr/bin/env python
# coding: utf-8


import os
import cv2 
import numpy as np 


def translate(img):
    # store height and width
    height, width = img.shape[:2] 

    T = np.float32([ [1,0,70], [0,1,110] ])

    # warpAffine to transform the image using matrix, T 
    img_translation = cv2.warpAffine(img, T, (width + 70, height + 110))
    
    T = np.float32([ [1,0,-50], [0,1,-50] ])
    img_translation = cv2.warpAffine(img_translation, T, (width + 70 + 50, height + 110 + 50))

    return img_translation




def rotate(img, angle):
    # grab the dimensions of the image and then determine the
    # center
    (height, width) = img.shape[:2]
    (cX, cY) = (width // 2, height // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(img, M, (nW, nH))




directory = r'Benign'
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        translated_img = translate(img)
        cv2.imwrite('Translated/' + filename, translated_img)




directory = r'Benign'
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rotated_img = rotate(img, 90)
        cv2.imwrite('Rotated/' + filename, rotated_img)






