import os

def moveup():
	cwd = os.getcwd()

	for folder in os.listdir(cwd):
		if os.path.isdir(os.path.join(cwd, folder)):
			for child in os.listdir(folder):
				os.rename(os.path.join(folder, child), os.path.join(cwd, child))