import os

def moveup(path):
	cwd = path

	for folder in os.listdir(cwd):
		if os.path.isdir(os.path.join(cwd, folder)):
			for child in os.listdir(os.path.join(cwd, folder)):
				os.rename(os.path.join(os.path.join(cwd, folder), child), os.path.join(cwd, child))