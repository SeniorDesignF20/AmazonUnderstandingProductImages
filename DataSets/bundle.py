import os

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

path = os.path.join(os.getcwd(), 'Belts')
bundle(path)
