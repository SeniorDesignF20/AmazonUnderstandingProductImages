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


def create_folders(size='small'):
	parameters_path = os.path.join(os.getcwd(), "parameters")
	results_path = os.path.join(os.getcwd(), size)
	results_file = os.path.join(results_path, size + '_results.csv')
	cmlabels_file = os.path.join(results_path, 'cm_labels.csv')
	test_file = os.path.join(results_path, 'test.csv')
	folder_path = os.path.join(results_path, 'GradCam Results')
	true0 = os.path.join(folder_path, 'Different and Predicted Different')
	true1 = os.path.join(folder_path, 'Same and Predicted Same')
	false0 = os.path.join(folder_path, 'Same and Predicted Different')
	false1 = os.path.join(folder_path, 'Different and Predicted Same')

	if not os.path.isdir(folder_path):
		os.mkdir(folder_path)

	if not os.path.isdir(true0):
		os.mkdir(true0)

	if not os.path.isdir(true1):
		os.mkdir(true1)

	if not os.path.isdir(false0):
		os.mkdir(false0)

	if not os.path.isdir(false1):
		os.mkdir(false1)

	with open(results_file, 'r') as f:
	    mycsv = csv.reader(f)
	    mycsv = list(mycsv)
	    image_dim = literal_eval(mycsv[4][1])
	    batch_size = int(mycsv[5][1])


	dim = image_dim[0]
	dim = int((int(dim/2) - 2)/2) - 6
	model = Modified_LeNet(batch_size=batch_size, dim=dim)
	model.load_state_dict(torch.load(os.path.join(parameters_path, size + '.pth')))


	testing_concatenator = Concatenator(test_file, image_dim=image_dim)
	first_images = testing_concatenator.first_images
	second_images = testing_concatenator.second_images
	concatenated_images = testing_concatenator.concatenated_images

	method = 'gradcam++'
	target_layer = model.layer6
	cam = CAM(model=model, target_layer=target_layer)

	with open(cmlabels_file, 'r') as f:
		file = csv.reader(f)
		file = list(file)

		for i in range(0, len(file), 12):
			print(i)

			input_tensor = torch.tensor(np.expand_dims(concatenated_images[i], axis=0))
			grayscale_cam = cam(input_tensor=input_tensor, method=method)

			first_image = first_images[i].numpy()
			first_image = np.moveaxis(first_image, 0, -1)
			visualization0 = show_cam_on_image(first_image, grayscale_cam)

			second_image = second_images[i].numpy()
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

			if file[i][0] == 'True0':
				fig.savefig(os.path.join(true0, f'{i}.png'))
			elif file[i][0] == 'True1':
				fig.savefig(os.path.join(true1, f'{i}.png'))
			elif file[i][0] == 'False0':
				fig.savefig(os.path.join(false0, f'{i}.png'))
			elif file[i][0] == 'False1':
				fig.savefig(os.path.join(false1, f'{i}.png'))

create_folders(size='medium')
			


