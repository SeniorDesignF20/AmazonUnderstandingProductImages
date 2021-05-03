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

def save_results(dataset_size, image_dim, epochs, numsame, numdif, batch_size, 
	hours, minutes, cm, cm_labels, correct, training_concatenator_length, testing_concatenator_length):
	curr_path = os.path.dirname(os.path.abspath(__file__))
	csv_path = os.path.join(curr_path, dataset_size)

	file = open(str(dataset_size) + '/cm_labels.csv', 'w+', newline='')
	with file:
	    write = csv.writer(file)
	    write.writerows(cm_labels)

	file = open(str(dataset_size) + f'/{dataset_size}_results.csv', 'w+', newline='')
	data = [('Hours', hours),
	        ('Minutes', minutes),
	        ('Dataset Size', dataset_size),
	        ('Epochs', epochs),
	        ('Image dimensions', image_dim),
	        ('Batch Size', batch_size),
	        ('Number of Training Images', training_concatenator_length),
	        ('Number of Testing Images', testing_concatenator_length),
	        ('Same to Different Ratio', numsame/numdif),
	        ('Accuracy over test set', correct),
	        ('True 0s %', cm[0]),
	        ('True 1s %', cm[1]),
	        ('False 0s %', cm[2]),
	        ('False 1s %', cm[3])]
	with file:
	    write = csv.writer(file)
	    write.writerows(data)
