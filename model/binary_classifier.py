import os
import operator
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from modified_lenet import Modified_LeNet
from Concatenator import Concatenator
from pathlib import Path
from Tensor_confusion_matrix import Tensor_confusion_matrix

curr_path = os.path.dirname(os.path.abspath(__file__))
datasets_path = str(Path(curr_path).parents[0]) + "/AmazonSet"

print("Concatinating training data")
#training_concatenator = Concatenator(datasets_path, "train_expanded2.csv")
training_concatenator = Concatenator(datasets_path, "train_rand.csv")
print("Finished")

print("Concatinating testing data")
#testing_concatenator = Concatenator(datasets_path, "test_expanded2.csv")
testing_concatenator = Concatenator(datasets_path, "test_rand.csv")
print("Finished")

batch_size = 64

print("Loading training set")
trainloader = DataLoader(training_concatenator, batch_size, shuffle=True)

print("Loading testing set")
testloader = DataLoader(testing_concatenator, batch_size, shuffle=False)

print("Creating Model")
model = Modified_LeNet(batch_size=batch_size)

print("Training Model")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(60):

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

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        cm = tuple(
            map(operator.add, cm, Tensor_confusion_matrix(predicted, labels)))


print(f"Accuracy over test set: {100*correct/total}%")
print()

print("0=Different, 1=Same")
print(f"True 0s = {cm[0]}, {cm[0]*100/total}%")
print(f"True 1s = {cm[1]}, {cm[1]*100/total}%")
print(f"False 0s = {cm[2]}, {cm[2]*100/total}%")
print(f"False 1s = {cm[3]}, {cm[3]*100/total}%")
