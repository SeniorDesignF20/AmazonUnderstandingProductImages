import matplotlib.pyplot as plt
import numpy as np 
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

# Plots confusion matrix, ROC curve, Precision, Recall, and F1 Score of the model's performance on the test set
def plot_metrics(cm, y_true, y_score):
	plot_confusion_matrix(cm)

	rocauc = round(roc_auc_score(y_true, y_score),3)
	Precision = round(cm[1]/(cm[1] + cm[3]),3)
	Recall = round(cm[1]/(cm[1] + cm[2]),3)
	F1 = round(2*Precision*Recall/(Precision + Recall),3)

	fig, ax = plt.subplots(1,1)
	data = [[Precision], [Recall], [F1]]
	row_labels = ['Precision', 'Recall', 'F1 Score']
	ax.axis('tight')
	ax.axis('off')
	ax.table(cellText=data, rowLabels=row_labels, loc='center', colWidths=[.1,.1])
	plt.show()

	_, tpr, _ = roc_curve(y_true, y_score)
	x = np.linspace(0,1,len(tpr))

	fig = plt.figure()
	plt.plot(x,tpr,'r-')
	plt.plot(x,x,'b--')
	plt.xlabel('False Positive Rate (FPR)')
	plt.ylabel('True Positive Rate (TPR)')
	plt.title(f"ROC Curve\nAUC={rocauc}")
	plt.legend(['ROC Curve of Classifier', 'TPR=FPR'], loc='lower right')
	plt.show()


