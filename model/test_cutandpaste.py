import os
import matplotlib.pyplot as plt
import numpy as np
from cutandpaste import cutandpaste
from PIL import Image


def test_cutandpaste():

	test_images_path = os.path.join(os.getcwd(), 'Test_GradCam')

	for folder in os.listdir(test_images_path):
		f = os.path.join(test_images_path, folder)

		image = os.listdir(f)[0]
		image = np.array(Image.open(os.path.join(f, image)))

		image_cap = cutandpaste(image)
		
		plt.figure(1)
		plt.subplot(121)
		plt.imshow(image)

		plt.subplot(122)
		plt.imshow(image_cap)
		plt.show()
		

test_cutandpaste()