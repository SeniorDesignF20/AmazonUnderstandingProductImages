import os
import shutil

cwd = os.getcwd()

for folder in os.listdir(cwd):
	if os.path.isdir(folder):
		if len(os.listdir(folder)) == 0:
			shutil.rmtree(folder)