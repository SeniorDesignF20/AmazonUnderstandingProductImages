import os
import matplotlib.pyplot as plt
import numpy as np
from warp import warp
from PIL import Image


def test_warp():

	test_images_path = os.path.join(os.getcwd(), 'Test_GradCam')

	for folder in os.listdir(test_images_path):
		f = os.path.join(test_images_path, folder)

		image0 = os.listdir(f)[0]
		image1 = os.listdir(f)[1]

		image0 = os.path.join(f, image0)
		image1 = os.path.join(f, image1)

		warped_image1 = warp(image0, image1)
		
		plt.figure(1)
		plt.subplot(131)
		plt.imshow(Image.open(image0))

		plt.subplot(132)
		plt.imshow(Image.open(image1))

		plt.subplot(133)
		plt.imshow(warped_image1)
		plt.show()
		
		

test_warp()