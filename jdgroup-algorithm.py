import cv2
import numpy as np
import os

# Section 3.1 of JD Group Algorithm: Image Alignment
def align(img, img_ref):
    
    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    
    # canny edge detection
    edged = cv2.Canny(img_gray, 100, 200)
    edged_ref = cv2.Canny(img_ref_gray, 100, 200)
    
    # finding contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_ref, hierarchy_ref = cv2.findContours(edged_ref, 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # SIFT keypoints detection
    sift = cv2.SIFT_create(5000)
    kp, d = sift.detectAndCompute(edged, None)
    kp_ref, d_ref = sift.detectAndCompute(edged_ref, None)
    
    # feature matching using a Brute Force matcher with  
    # Hamming distance as measurement mode
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True) 
    
    # match 2 sets of descriptors
    matches = bf.match(d, d_ref)
    matches = sorted(matches, key = lambda x:x.distance)
    
    # take the top 90 % matches forward. 
    matches = matches[:int(len(matches)*90)] 
    
    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches to compute homography matrix
    pts = np.zeros((len(matches), 2), dtype="float")
    pts_ref = np.zeros((len(matches), 2), dtype="float")
    
    # loop over top matches
    # indicate that the two keypoints in the respective images
    # map to each other
    for (i, m) in enumerate(matches):
        pts[i] = kp[m.queryIdx].pt
        pts_ref[i] = kp_ref[m.trainIdx].pt
        
    # compute homography matrix between sets of matched points
    homography, mask = cv2.findHomography(pts, pts_ref, method=cv2.RANSAC)
    
    # use homography matrix to align image
    (height, width) = img_ref.shape[:2]
    aligned = cv2.warpPerspective(img, homography, (width, height)) 
    
    
    return aligned

# Concatenates two 3-channel input images into one 6-channel image
def concat(img, img_ref):
    concatenated = np.concatenate((img, img_ref), axis=2)
    return concatenated

directory = r'DataSets/Alignment'
directory_ref = r'DataSets/Benign'
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        img_path = os.path.join(directory, filename)
        img_ref_path = os.path.join(directory_ref, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_ref = cv2.imread(img_ref_path, cv2.IMREAD_COLOR)
        aligned_img = align(img, img_ref)
        cv2.imwrite('JDGroup/' + filename, aligned_img)
