# Returns a list of classes to be considered.
# If size == 'small', return 10 random classes
# If size == 'mediium', return 20 random classes
# If size == 'large', return all classes

import os
import random

def ClassSplitter(dataset_path, size='small'):

	classes = []
	exceptions = ['classifier_models', 'duplicate ids', '256_Object', '.DS_Store']

	if size == 'small':
		num_classes = 10

	elif size == 'medium':
		num_classes = 20

	elif size == 'large':
		for folder in os.listdir(dataset_path):
			if os.path.isdir(os.path.join(dataset_path, folder)) and folder not in exceptions:
				classes.append(folder)
		return classes

	while num_classes != 0:
		folder = random.choice(os.listdir(dataset_path))
		if folder not in classes and os.path.isdir(os.path.join(dataset_path, folder)) and folder not in exceptions:
			classes.append(folder)
			num_classes = num_classes - 1

	return classes



