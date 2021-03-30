import os
import operator
import numpy as np
import torch
import torch.nn as nn
import torchvision
import csv
import sys
import time
from CreateSplits import create_datasets, splitDF
from torch.utils.data import DataLoader
from modified_lenet import Modified_LeNet
from Concatenator import Concatenator
from pathlib import Path
from Tensor_confusion_matrix import Tensor_confusion_matrix

start_time = time.time()

curr_path = os.path.dirname(os.path.abspath(__file__))
datasets_path = str(Path(curr_path).parents[0]) + '/Datasets'
print(datasets_path)

"""
 Call binary_classifier.py using the following format:
 python3 binary_classifier.py dataset_size image_dim epochs numsame numdif
 where

dataset_size is 'small', 'medium', or 'large'
image_dim is dimension of images when inserted into model (assumes square). Examples: 26->(26,26), 64->(64,64), 150->(150,150)
epochs is number of training epochs (default 30)
numsame is number of same examples across training and testing set (default 2000)
numdif is number of different examples across training and testing set (default 2000)

"""
arguments = sys.argv
length = len(arguments)

dataset_size = str(arguments[1]) if length > 1 else 'small'
image_dim = (int(arguments[2]), int(arguments[2])) if length > 2 else (64,64)
epochs = int(arguments[3]) if length > 3 else 30
numsame = int(arguments[4]) if length > 4 else 2000
numdif = int(arguments[5]) if length > 5 else 2000



destination_path = dataset_size
csv_path = os.path.join(curr_path, dataset_size)

df = create_datasets(datasets_path, numsame=numsame, numdif=numdif, size=dataset_size)
print("Datasets created")
splitDF(df, destination_path)
print("Datasets split and csvs are created")

print("Concatinating training data")
training_concatenator = Concatenator(os.path.join(csv_path, "train.csv"), image_dim=image_dim)
print("Finished")

print("Concatinating testing data")
testing_concatenator = Concatenator(os.path.join(csv_path, "test.csv"), image_dim=image_dim)
print("Finished")

batch_size = 64

print("Loading training set")
trainloader = DataLoader(training_concatenator, batch_size, shuffle=True)

print("Loading testing set")
testloader = DataLoader(testing_concatenator, batch_size, shuffle=False)

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

print("Testing Model")

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

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        matrix, matrix_labels = Tensor_confusion_matrix(predicted, labels)
        for i in matrix_labels:
            cm_labels.append([reverse_legend[i]])
        cm = tuple(
            map(operator.add, cm, matrix))

end_time = time.time()
hours, rem = divmod(end_time-start_time, 3600)
minutes, seconds = divmod(rem, 60)
time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

print(f"Accuracy over test set: {100*correct/total}%")
print()

print("0=Different, 1=Same")
print(f"True 0s = {cm[0]}, {cm[0]*100/total}%")
print(f"True 1s = {cm[1]}, {cm[1]*100/total}%")
print(f"False 0s = {cm[2]}, {cm[2]*100/total}%")
print(f"False 1s = {cm[3]}, {cm[3]*100/total}%")

file = open(str(dataset_size) + '/cm_labels.csv', 'w+', newline='')
with file:
    write = csv.writer(file)
    write.writerows(cm_labels)

file = open(str(dataset_size) + '/results.csv', 'w+', newline='')
data = [('Time Elapsed', time_elapsed),
        ('Dataset Size', dataset_size),
        ('Epochs', epochs),
        ('Image dimensions', image_dim),
        ('Number of Training Images', training_concatenator.__len__()),
        ('Number of Testing Images', testing_concatenator.__len__()),
        ('Same to Difference Ratio', numsame/numdif),
        ('Accuracy over test set', 100*correct/total),
        ('True 0s', cm[0]*100/total),
        ('True 1s', cm[1]*100/total),
        ('False 0s', cm[2]*100/total),
        ('False 1s', cm[3]*100/total)]
with file:
    write = csv.writer(file)
    write.writerows(data)
