import os

cwd = os.getcwd()

for folder in os.listdir(cwd):
	for file in os.listdir(folder):
		os.rename(os.path.join(folder, file), os.path.join(cwd, file))