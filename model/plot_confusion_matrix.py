import matplotlib.pyplot as plt
import numpy as np
import warnings 


def plot_confusion_matrix(cm):

	warnings.simplefilter("ignore")

	matrix = np.empty((2,2))
	matrix[0,0] = cm[0]
	matrix[0,1] = cm[3]
	matrix[1,0] = cm[2]
	matrix[1,1] = cm[1]

	labels = ['Different', 'Same']

	fig, ax = plt.subplots(figsize=(2,2))
	cax = ax.matshow(matrix, cmap=plt.cm.Oranges)
	fig.colorbar(cax)

	ax.set_xticklabels([''] + labels)
	ax.set_yticklabels([''] + labels)
	
	for row in range(2):
		for col in range(2):
			ax.text(col, row, str(matrix[row,col]), va='center', ha='center')

	plt.title('Classifier Confusion Matrix', y=1.3)
	
	ax.set_xlabel('Classifier Prediction')
	ax.set_ylabel('Truth')

	plt.show()
	
