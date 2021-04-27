import os
import numpy as np
import torch
import math
import csv
import matplotlib.pyplot as plt
from PIL import Image
from ast import literal_eval
from Concatenator import Concatenator
from pytorch_grad_cam import CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from modified_lenet import Modified_LeNet


def test_std(size='small'):
	parameters_path = os.path.join(os.getcwd(), "parameters")
	results_path = os.path.join(os.getcwd(), size)
	results_file = os.path.join(results_path, size + '_results.csv')
	folder_path = os.path.join(results_path, 'GradCam Results')
	test_path = os.path.join(os.getcwd(), 'New_GradCam_Test')

	if not os.path.isdir(folder_path):
		os.mkdir(folder_path)

	with open(results_file, 'r') as f:
	    mycsv = csv.reader(f)
	    mycsv = list(mycsv)
	    image_dim = literal_eval(mycsv[4][1])
	    batch_size = int(mycsv[5][1])


	dim = image_dim[0]
	dim = int((int(dim/2) - 2)/2) - 6
	model = Modified_LeNet(batch_size=batch_size, dim=dim)
	model.load_state_dict(torch.load(os.path.join(parameters_path, size + '.pth')))

	method = 'gradcam++'
	target_layer = model.layer6
	cam = CAM(model=model, target_layer=target_layer)

	concatenator = Concatenator(image_dim=image_dim)

	for f in os.listdir(test_path):
		folder = os.path.join(test_path, f)

		img1 = os.path.join(folder, os.listdir(folder)[0])
		img2 = os.path.join(folder, os.listdir(folder)[1])

		concatenated = concatenator.concatenate(img1, img2)

		input_tensor = torch.tensor(np.expand_dims(concatenated, axis=0))
		grayscale_cam = cam(input_tensor=input_tensor, method=method)

		img1 = concatenator.transform_image(img1)
		img2 = concatenator.transform_image(img2)

		first_image = img1.numpy()
		first_image = np.moveaxis(first_image, 0, -1)
		visualization0 = show_cam_on_image(first_image, grayscale_cam)

		second_image = img2.numpy()
		second_image = np.moveaxis(second_image, 0, -1)
		visualization1 = show_cam_on_image(second_image, grayscale_cam)

		fig = plt.figure()
		ax1 = fig.add_subplot(2,2,1)
		ax1.imshow(first_image)

		ax2 = fig.add_subplot(2,2,2)
		ax2.imshow(visualization0)

		ax3 = fig.add_subplot(2,2,3)
		ax3.imshow(second_image)

		ax4 = fig.add_subplot(2,2,4)
		ax4.imshow(visualization1)

		fig.savefig(os.path.join(folder_path, f'{f}.png'))

test_std(size='small')
			


