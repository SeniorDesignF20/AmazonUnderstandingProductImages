import os
import numpy as np
import torch
import math
import csv
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from ast import literal_eval
from modified_lenet import Modified_LeNet
from Concatenator import Concatenator
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from CreateBox import CreateBox
from invert_map import invert_map


def test_gradcam(size='small'):
	warnings.simplefilter("ignore")
	
	parameters_path = os.path.join(os.getcwd(), "parameters")
	results_path = os.path.join(os.getcwd(), size)
	results_file = os.path.join(results_path, size + '_results.csv')

	with open(results_file, 'r') as f:
	    mycsv = csv.reader(f)
	    mycsv = list(mycsv)
	    image_dim = literal_eval(mycsv[4][1])
	    batch_size = int(mycsv[5][1])

	dim = image_dim[0]
	dim = int((int(dim/2) - 2)/2) - 6
	model = Modified_LeNet(batch_size=batch_size, dim=dim)
	model.load_state_dict(torch.load(os.path.join(parameters_path, size + '.pth')))

	test_images_path = os.path.join(os.getcwd(), 'Test_GradCam')

	for folder in os.listdir(test_images_path):
		f = os.path.join(test_images_path, folder)

		image0 = os.listdir(f)[0]
		image1 = os.listdir(f)[1]

		image0 = os.path.join(f, image0)
		image1 = os.path.join(f, image1)

		concatenator = Concatenator(csvfile=None, image_dim=image_dim)
		input_tensor = concatenator.concatenate(image0, image1)
		input_tensor = torch.tensor(np.expand_dims(input_tensor, axis=0))

		target_layer = model.layer6

		cam = GradCAMPlusPlus(model=model, target_layer=target_layer)

		grayscale_cam = cam(input_tensor=input_tensor, aug_smooth=True, eigen_smooth=False)
		grayscale_cam = grayscale_cam[0,:]

		first_image = concatenator.transform_image(image0).numpy()
		first_image = np.moveaxis(first_image, 0, -1)

		second_image = concatenator.transform_image(image1).numpy()
		second_image = np.moveaxis(second_image, 0, -1)

		whiteimage = np.zeros(first_image.shape, dtype=np.uint8)
		whiteimage.fill(0)
		heatmap = show_cam_on_image(whiteimage, grayscale_cam)
		image_boxes1 = CreateBox(first_image, heatmap)
		image_boxes2 = CreateBox(second_image, heatmap)

		
		plt.figure(1)
		plt.subplot(131)
		plt.imshow(image_boxes1)

		plt.subplot(132)
		plt.imshow(image_boxes2)

		plt.subplot(133)
		plt.imshow(heatmap)
		plt.show()
		
		

test_gradcam(size='small')