import cv2
import matplotlib.pyplot as plt
import numpy as np


"""
Creates one or more bounding boxes, given an image and it's GradCAM heatmap.

image: numpy array of image
heatmap: Use the following code to get the heatmap parameter
		whiteimage = np.zeros(first_image.shape, dtype=np.uint8)
		whiteimage.fill(0)
		heatmap = show_cam_on_image(whiteimage, grayscale_cam)

"""

def CreateBox(image, heatmap):

	gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
	gray_heatmap = 255 - gray_heatmap

	blur = cv2.GaussianBlur(gray_heatmap, (7,7),0)
	thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	kernel = np.ones((11,11),np.uint8)
	cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

	cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	image = image.copy()

	for c in cnts:
	    x,y,w,h = cv2.boundingRect(c)

	    count = 0
	    rgb = np.array([0,0,0])
	    for i in range(y,y+h):
	    	for j in range(x,x+w):
	    		rgb = np.add(rgb, heatmap[i,j])
	    		count += 1


	    rgb = rgb/count

	    red = rgb[0]
	    green = rgb[1]
	    blue = rgb[2]

	    box_area = h*w
	    image_area = heatmap.shape[0]*heatmap.shape[1]

	    if (red > blue) and (red > green) and box_area>.05*image_area:
	    	cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)

	return(image)
