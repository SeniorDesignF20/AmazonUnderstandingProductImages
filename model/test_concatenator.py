import os
import matplotlib.pyplot as plt
from CreateSplits import create_datasets, splitDF
from Concatenator import Concatenator
from pathlib import Path


def test_concatenator(dataset_size='Testing', numsame=5, numdif=5):
	curr_path = os.path.dirname(os.path.abspath(__file__))
	destination_path = dataset_size
	csv_path = os.path.join(curr_path, dataset_size)
	datasets_path = str(Path(curr_path).parents[0]) + '/Datasets'

	df = create_datasets(datasets_path, numsame=numsame, numdif=numdif, size=dataset_size)
	splitDF(df, destination_path, test_ratio=0)

	concatenator = Concatenator(os.path.join(csv_path, "train.csv"), image_dim=(200,200))

	for i in range(numsame+numdif):
		image1 = concatenator.first_images_original[i]
		image2 = concatenator.second_images_original[i]

		plt.figure(1)
		plt.subplot(121)
		plt.imshow(image1)

		plt.subplot(122)
		plt.imshow(image2)
		plt.show()

test_concatenator()