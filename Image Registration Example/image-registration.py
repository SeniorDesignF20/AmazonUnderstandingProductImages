#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
import os

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.15



def align(img1, img2):
    # convert images to gray scale
    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    height, width = img2gray.shape
    
    # detect ORB features and comput descriptors
    orb = cv2.ORB_create(MAX_FEATURES)
    kp1, d1 = orb.detectAndCompute(img1gray, None)
    kp2, d2 = orb.detectAndCompute(img2gray, None)
    
    # Match features between the two images. 
    # We create a Brute Force matcher with  
    # Hamming distance as measurement mode. 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 

    # Match the two sets of descriptors. 
    matches = matcher.match(d1, d2) 

    # Sort matches on the basis of their Hamming distance. 
    matches.sort(key = lambda x: x.distance) 

    # Take the top 90 % matches forward. 
    matches = matches[:int(len(matches)*90)] 
    no_of_matches = len(matches) 

    # Define empty matrices of shape no_of_matches * 2. 
    p1 = np.zeros((no_of_matches, 2)) 
    p2 = np.zeros((no_of_matches, 2)) 

    for i in range(len(matches)): 
        p1[i, :] = kp1[matches[i].queryIdx].pt 
        p2[i, :] = kp2[matches[i].trainIdx].pt 

    # Find the homography matrix. 
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 

    # Use this matrix to transform the 
    # colored image wrt the reference image. 
    transformed_img = cv2.warpPerspective(img1, 
                        homography, (width, height)) 
    
    return transformed_img


directory = r'Book-Benign'
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        img_ref = cv2.imread(directory + '/hpref.jpg', cv2.IMREAD_COLOR)
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        aligned_img = align(img, img_ref)
        cv2.imwrite('Book-Manipulated/' + filename, aligned_img)






