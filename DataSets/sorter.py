import torch
import os
import shutil
import random
from PIL import Image
from torchvision.transforms import ToTensor


model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

dataset_path = r'../DataSets/multiimage_products'

for folder in os.listdir(dataset_path):
	if not os.path.isdir(os.path.join(dataset_path, folder)):
		continue

	current_path = os.path.join(dataset_path, folder)
	image = random.choice(os.listdir(current_path))
	image = os.path.join(current_path, image)

	try:
		image = Image.open(image)
		image = ToTensor()(image).unsqueeze(0)
	except:
		continue

	try:
		output = model(image)
	except RuntimeError as error:
		up = torch.nn.Upsample(scale_factor=4, mode='bilinear')
		output = model(up(image))

	probabilities = torch.nn.functional.softmax(output)
	index = torch.argmax(probabilities)

	new_folder = os.path.join(dataset_path, str(int(index)))
	if not os.path.isdir(new_folder):
		try:
			os.makedirs(new_folder)
		except OSError as error:
			pass

	shutil.move(current_path, new_folder)



			


