import torch
import os
import shutil
from PIL import Image, ImageFile
from torchvision.transforms import ToTensor


model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

dataset_path = r'../DataSets/multiimage_products'

ImageFile.LOAD_TRUNCATED_IMAGES = True

for file in os.listdir(dataset_path):
		if file.endswith(".jpg"):

			current_path = os.path.join(dataset_path, file)

			try:
				image = Image.open(current_path)
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

			folder = os.path.join(dataset_path, str(int(index)))
			if not os.path.isdir(folder):
				try:
					os.makedirs(folder)
				except OSError as error:
					pass

			os.rename(current_path, os.path.join(folder, file))



			


