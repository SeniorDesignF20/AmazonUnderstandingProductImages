import numpy as np 

# Counts number of pixels that are different between 2 images
# Assume image1 and image2 have same dimensions
# Return ratio of pixels that are different
def countPixels(image1, image2):

	count = 0
	for i in range(image1.shape[0]):
		for j in range(image1.shape[1]):
			for k in range(2):
				if image1[i,j,k] != image2[i,j,k]:
					count += 1
					continue

	return count/(image1.shape[0]*image1.shape[1])
