import cv2
import numpy as np

"""
Determines if the color histograms of 2 images are similar.
Computes Correlation between histograms to check similarity. 
Returns 1 if they are similar, 0 if they aren't.

image1: numpy array
image2: numpy array
"""

def checkHist(image1, image2):
	gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

	hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
	hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

	result = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

	if result > .999:
		return 1
	return 0