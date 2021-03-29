import os
import shutil

def moveup(path):
	cwd = path

	for folder in os.listdir(cwd):
		if os.path.isdir(os.path.join(cwd, folder)):
			for child in os.listdir(os.path.join(cwd, folder)):
				os.rename(os.path.join(os.path.join(cwd, folder), child), os.path.join(cwd, child))

def deleteEmptyFolders(path):
	cwd = path

	for folder in os.listdir(cwd):
		if os.path.isdir(folder):
			if len(os.listdir(folder)) == 0:
				shutil.rmtree(folder)

# Bundles all images of the same product together into one folder
# E.g. A123456789_0.jpg, A123456789_4 -> A123456789

def bundle(path):

	for file in os.listdir(path):

		if file.endswith('jpg'):
			product_id = file[0:10]

			if not os.path.isdir(os.path.join(path, product_id)):
				try:
					os.makedirs(os.path.join(path, product_id))
				except OSError as error:
					pass

			os.rename(os.path.join(path, file), os.path.join(os.path.join(path, product_id), file))


cwd = os.getcwd()
exceptions = ['classifier_models', 'duplicate ids', '256_Object']

for folder in os.listdir(cwd):
	if folder not in exceptions and os.path.isdir(os.path.join(cwd, folder)):
		print(folder)
		path = os.path.join(cwd, folder)
		moveup(path)
		deleteEmptyFolders(path)
		bundle(path)