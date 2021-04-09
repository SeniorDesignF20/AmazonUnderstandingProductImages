import os
import numpy as np
import torch
import csv
from ast import literal_eval
from modified_lenet import Modified_LeNet
from Concatenator import Concatenator
from pytorch_grad_cam import CAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def gradcam(image1, image2, path, size='small'):
	parameters_path = os.path.join(path, "parameters")
	results_path = os.path.join(path, size)
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

	concatenator = Concatenator(csvfile=None, image_dim=image_dim)
	input_tensor = concatenator.concatenate(image1, image2)
	input_tensor = torch.tensor(np.expand_dims(input_tensor, axis=0))
	
	method = 'gradcam++'
	target_layer = model.layer6

	cam = CAM(model=model, target_layer=target_layer)

	grayscale_cam = cam(input_tensor=input_tensor, method=method)

	first_image = concatenator.transform_image(image1).numpy()
	first_image = np.moveaxis(first_image, 0, -1)
	visualization1 = show_cam_on_image(first_image, grayscale_cam)
	
	second_image = concatenator.transform_image(image2).numpy()
	second_image = np.moveaxis(second_image, 0, -1)
	visualization2 = show_cam_on_image(second_image, grayscale_cam)

	return visualization1, visualization2