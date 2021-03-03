import os

cwd = os.getcwd()

for file in os.listdir(cwd):

	if file.endswith('jpg'):
		product_id = file[0:10]

		if not os.path.isdir(product_id):
			try:
				os.makedirs(product_id)
			except OSError as error:
				pass

		os.rename(os.path.join(cwd, file), os.path.join(product_id, file))