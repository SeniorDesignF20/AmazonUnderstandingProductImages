import os
import operator
import numpy as np
import torch
import torch.nn as nn
import csv
import sys
import time
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from modified_lenet import Modified_LeNet
from Concatenator import Concatenator
from pathlib import Path
from confusion_matrix import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from rocauc import rocauc



def test(dataset_size, image_dim, batch_size):
	curr_path = os.path.dirname(os.path.abspath(__file__))
	csv_path = os.path.join(curr_path, dataset_size)
	parameters_path = os.path.join(os.getcwd(), "parameters")

	print("Concatinating testing data")
	testing_concatenator = Concatenator(os.path.join(csv_path, "test.csv"), image_dim=image_dim)
	print("Finished")

	print("Loading testing set")
	testloader = DataLoader(testing_concatenator, batch_size, shuffle=False)

	dim = image_dim[0]
	dim = int((int(dim/2) - 2)/2) - 6

	model = Modified_LeNet(batch_size=batch_size, dim=dim)
	model.load_state_dict(torch.load(os.path.join(parameters_path, dataset_size + '.pth')))

	correct = 0
	total = 0

	cm = (0, 0, 0, 0)

	cm_labels = []

	reverse_legend = {
	        0: "True0",
	        1: "True1",
	        2: "False0",
	        3: "False1"
	    }

	results = []
	truths = []


	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        outputs = model(images)
	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()
	        probs = F.softmax(outputs.data, dim=1).cpu().detach().numpy()
	        results.append(probs)
	        truths.append(labels)

	        matrix, matrix_labels = confusion_matrix(predicted, labels)
	        for i in matrix_labels:
	            cm_labels.append([reverse_legend[i]])
	        cm = tuple(
	            map(operator.add, cm, matrix))

	P, y = np.vstack(results), np.hstack(truths)

	correct = round(100*correct/total,3)

	print(f"Accuracy over test set: {correct}%")
	print()

	print("0=Different, 1=Same")
	print(f"True 0s = {cm[0]}, {round(cm[0]*100/total)}%")
	print(f"True 1s = {cm[1]}, {round(cm[1]*100/total)}%")
	print(f"False 0s = {cm[2]}, {round(cm[2]*100/total)}%")
	print(f"False 1s = {cm[3]}, {round(cm[3]*100/total)}%")

	cm = [round(cm[i]*100/total,3) for i in range(4)]
	plot_confusion_matrix(cm)

	return cm, cm_labels, correct, testing_concatenator.__len__()

test('medium', (150,150), 64)