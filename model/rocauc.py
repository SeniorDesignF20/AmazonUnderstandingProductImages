import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp

def rocauc(P, y):

	num_classes = 2
	fpr = {}
	tpr = {}
	roc_auc = {}

	for i in range(num_classes):
		fpr[i], tpr[i], _ = roc_curve([1 if label == 1 else 0 for label in y], P[:,i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	y_test = np.array([[1 if label == i else 0 for label in y] for i in range(num_classes)]).ravel()
	y_preds = P.T.ravel()
	fpr['micro'], tpr['micro'], _ = roc_curve(y_test, y_preds)
	roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

	mean_tpr = np.zeros_like(all_fpr)
	for i in range(num_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
	mean_tpr /= num_classes

	fpr['macro'], tpr['macro'] = all_fpr, mean_tpr
	roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])