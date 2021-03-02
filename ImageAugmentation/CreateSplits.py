from PIL import Image
import cv2
import os
import re
import numpy as np
import pandas as pd
import random

def create_datasets(path1, path2, numsame=500, numdif=500):

	randDF = pd.DataFrame(columns=['image1', 'image2', 'label'])

	for i in range(numsame):
		image = random.choice(os.listdir(path1))
		product_id = image[0:-4]

		sameimage = str(product_id[0:10]) + '_' + str(random.randint(0, 10)) + '.jpg'

		count = 10
		while not os.path.exists(path2 + '/' + sameimage) and count != 0:
			sameimage = str(product_id[0:10]) + '_' + str(random.randint(0, 10)) + '.jpg'
			
			count = count - 1

		if count == 0:
			continue

		im1 = np.asarray(Image.open(path1 + '/' + image))
		im2 = np.asarray(Image.open(path1 + '/' + sameimage))

		try:
			if im1.shape[2] != 3 or im2.shape[2] !=3:
				print(":(")
				continue

		except:
			continue

		label = {'image1': image, 'image2': sameimage, 'label': "same"}
		randDF = randDF.append(label, ignore_index=True)

	for i in range(numdif):
		image = random.choice(os.listdir(path1))
		product_id = image[0:-4]

		diffimage = random.choice(os.listdir(path2))

		count = 10
		while diffimage[0:-4] == product_id and count != 0:
			diffimage = random.choice(os.listdir(path2))

			count = count - 1

		if count == 0:
			continue

		im1 = np.asarray(Image.open(path1 + '/' + image))
		im2 = np.asarray(Image.open(path1 + '/' + diffimage))

		try:
			if im1.shape[2] != 3 or im2.shape[2] !=3:
				print(":(")
				continue

		except:
			continue

		label = {'image1': image, 'image2': diffimage, 'label': "different"}
		randDF = randDF.append(label, ignore_index=True)

	return randDF

def splitDF(df, destination_path, test_ratio=.2):
	different_images = df.loc[df['label'] == 'different']
	same_images = df.loc[df['label'] == 'same']

	train_diff = different_images.sample(frac=(1-test_ratio))
	test_diff = different_images.drop(train_diff.index)

	train_same = same_images.sample(frac=(1-test_ratio))
	test_same = same_images.drop(train_same.index)

	train_frames = [train_diff, train_same]
	train = pd.concat(train_frames)

	test_frames = [test_diff, test_same]
	test = pd.concat(test_frames)

	train.to_csv(destination_path + '/train.csv', index=False)
	test.to_csv(destination_path + '/test.csv', index=False)

def clearance(path):

	for file in os.listdir(path):
		if not file.endswith(".jpg"):
			os.remove(os.path.join(path, file))

	


path1 = r'../DataSets/multiimage_products'
path2 = r'../DataSets/multiimage_products'

destination_path = r'../DataSets/multiimage_products'

clearance(path1)
df = create_datasets(path1, path2, numsame=5000, numdif=5000)
print("Datasets created")
splitDF(df, destination_path)
print("Datasets split and csvs are created")
