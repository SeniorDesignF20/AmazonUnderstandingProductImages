import os
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from PIL import Image
from ast import literal_eval
from pytorch_grad_cam import CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from torchvision.models import resnet50


def test_gradcam(size='small'):
	model = resnet50(pretrained=True)
	test_images_path = os.path.join(os.getcwd(), 'Test_GradCam')

	transform = transforms.Compose([
		transforms.ToPILImage(),
	    transforms.ToTensor()])

	for folder in os.listdir(test_images_path):
		f = os.path.join(test_images_path, folder)

		image0 = os.listdir(f)[0]
		image0 = os.path.join(f, image0)
		image0 =  np.asarray(Image.open(image0))
		input_tensor = transform(image0)
		input_tensor = torch.tensor(np.expand_dims(input_tensor, axis=0))

		method = 'gradcam++'

		target_layer = model.layer4[-1]

		cam = CAM(model=model, target_layer=target_layer)

		grayscale_cam = cam(input_tensor=input_tensor, method=method, target_category=1)

		first_image = transform(image0).numpy()
		first_image = np.moveaxis(first_image, 0, -1)
		visualization0 = show_cam_on_image(first_image, grayscale_cam)
		plt.imshow(visualization0)
		plt.show()


test_gradcam(size='medium')