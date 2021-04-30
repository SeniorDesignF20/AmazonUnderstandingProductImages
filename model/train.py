import os
import operator
import numpy as np
import torch
import torch.nn as nn
import csv
import sys
import time
import math
from CreateSplits import create_datasets, splitDF
from torch.utils.data import DataLoader
from modified_lenet import Modified_LeNet
from Concatenator import Concatenator
from pathlib import Path
from test import test
from save_results import save_results

start_time = time.time()

curr_path = os.path.dirname(os.path.abspath(__file__))
datasets_path = str(Path(curr_path).parents[0]) + '/Datasets'
print(datasets_path)

"""
 Call train.py using the following format:
 python3 train.py dataset_size image_dim epochs numsame numdif
 where

dataset_size is 'small', 'medium', or 'large'
image_dim is dimension of images when inserted into model (assumes square). Examples: 26->(26,26), 64->(64,64), 150->(150,150)
epochs is number of training epochs (default 30)
numsame is number of same examples across training and testing set (default 2000)
numdif is number of different examples across training and testing set (default 2000)

"""
arguments = sys.argv
length = len(arguments)

dataset_size = str(arguments[1]) if length > 1 else 'medium'
image_dim = (int(arguments[2]), int(arguments[2])) if length > 2 else (56,56)
epochs = int(arguments[3]) if length > 3 else 1
numsame = int(arguments[4]) if length > 4 else 50
numdif = int(arguments[5]) if length > 5 else 50



destination_path = dataset_size
csv_path = os.path.join(curr_path, dataset_size)

df = create_datasets(datasets_path, numsame=numsame, numdif=numdif, size=dataset_size)
print("Datasets created")
splitDF(df, destination_path)
print("Datasets split and csvs are created")

print("Concatinating training data")
training_concatenator = Concatenator(os.path.join(csv_path, "train.csv"), image_dim=image_dim)
print("Finished")

batch_size = 64

print("Loading training set")
trainloader = DataLoader(training_concatenator, batch_size, shuffle=True)

print("Creating Model")
dim = training_concatenator.image_dim[0]
dim = int((int(dim/2) - 2)/2) - 6

model = Modified_LeNet(batch_size=batch_size, dim=dim)

print("Training Model")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(epochs):

    running_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        print(f"Batch: {i}")

        concat_images, labels = batch
        optimizer.zero_grad()

        outputs = model(concat_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch}: Loss = {running_loss}")
    running_loss = 0

end_time = time.time()
time_elapsed = end_time - start_time
hours = math.floor(time_elapsed/3600)
minutes = math.floor((time_elapsed - 3600*hours)/60)

parameters_path = os.path.join(os.getcwd(), "parameters")
torch.save(model.state_dict(), os.path.join(parameters_path, dataset_size + '.pth'))


print("Testing Model")

training_concatenator_length = training_concatenator.__len__()
cm, cm_labels, correct, testing_concatenator_length = test(dataset_size, image_dim, batch_size)
save_results(dataset_size, image_dim, epochs, numsame, numdif, batch_size, 
	hours, minutes, cm, cm_labels, correct, training_concatenator_length, testing_concatenator_length)