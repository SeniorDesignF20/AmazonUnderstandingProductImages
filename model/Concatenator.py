import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class Concatenator(Dataset):

	def __init__(self, parent_dir, csvfile, image_dim=(56, 56)):
		self.concatenated_images = []
		self.first_images = []
		self.second_images = []
		self.labels = []

		self.transform = T.Compose([
			T.ToPILImage(),
			T.Resize(image_dim),
			T.ToTensor()])

		self.load(parent_dir, csvfile)

	def first_images(self):
		return self.first_images

	def second_images(self):
		return self.second_images

	def concatenated_images(self):
		return self.concatenated_images

	def labels(self):
		return self.labels

	def __len__(self):
		return len(self.concatenated_images)

	def __getitem__(self, index):
		item = self.concatenated_images[index]
		label = self.labels[index]
		return item, label

	def create_folder(self, path):
		if not os.path.isdir(path):
			try:
				os.makedirs(path)
				print(f"Created Folder @ {path}")

			except OSError as error:
				print("Folder cannot be created")
				print(error)
		else:
			print(f"Folder @ {path} Already Exists")
			"""
			for file in os.listdir(path):
				os.remove(os.path.join(path, file))
				"""

	def concat_name(self, dir_path, name1, name2):
		name = name1 + '+' + name2
		concat_path = os.path.join(dir_path, name)

		return concat_path

	def load(self, parent_dir, csvfile):

		first_dataset = "Benign"
		second_dataset = "Manipulated"

		path1 = os.path.join(parent_dir, first_dataset)
		path2 = os.path.join(parent_dir, second_dataset)


		new_dir = str(first_dataset + '+' + second_dataset)
		path3 = os.path.join(parent_dir, new_dir)
		self.create_folder(path3)

		csvpath = os.path.join(parent_dir, csvfile)

		df = pd.read_csv(csvpath)

		for i in df.index:

			name1 = df["image1"][i]
			image1 = np.asarray(Image.open(path1 + '/' + name1))

			name2 = df["image2"][i]

			if name2[0] == 'b':
				image2 = np.asarray(Image.open(path1 + '/' + name2))
			else:
				image2 = np.asarray(Image.open(path2 + '/' + name2))

			self.first_images.append(image1)
			self.second_images.append(image2)

			if df["label"][i] == "same":
				self.labels.append(1)
			else:
				self.labels.append(0)

			concat_name = self.concat_name(path3, name1[:-4], name2[:-4])
			if not os.path.isdir(concat_name):
				transformed1 = self.transform(image1)
				transformed2 = self.transform(image2)

				m1 = torch.mean(torch.flatten(transformed1))
				m2 = torch.mean(torch.flatten(transformed2))

				s1 = torch.std(torch.flatten(transformed1))
				s2 = torch.std(torch.flatten(transformed2))

				transformed1 = (transformed1 - m1)
				transformed2 = (transformed2 - m2)

				concatenated = torch.cat((transformed1, transformed2), 0)


				self.concatenated_images.append(concatenated)
				#np.save(concat_name, concatenated)

			else:
				self.concatenated_image.append(np.load(concat_name))