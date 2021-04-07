import os
import operator
import numpy as np
import torch
import torch.nn as nn
import torchvision
import csv
import sys
import time
import math
import matplotlib.pyplot as plt
from CreateSplits import create_datasets, splitDF
from torch.utils.data import DataLoader
from modified_lenet import Modified_LeNet
from Concatenator import Concatenator
from pathlib import Path
from Tensor_confusion_matrix import Tensor_confusion_matrix
from pytorch_grad_cam import CAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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
image_dim = (int(arguments[2]), int(arguments[2])) if length > 2 else (128,128)
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
time_elapsed = end_time - start_time
hours = math.floor(time_elapsed/3600)
minutes = math.floor((time_elapsed - 3600*hours)/60)

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

file = open(str(dataset_size) + f'/{dataset_size}_results.csv', 'w+', newline='')
data = [('Hours', hours),
        ('Minutes', minutes),
        ('Dataset Size', dataset_size),
        ('Epochs', epochs),
        ('Image dimensions', image_dim),
        ('Number of Training Images', training_concatenator.__len__()),
        ('Number of Testing Images', testing_concatenator.__len__()),
        ('Same to Different Ratio', numsame/numdif),
        ('Accuracy over test set', 100*correct/total),
        ('True 0s', cm[0]*100/total),
        ('True 1s', cm[1]*100/total),
        ('False 0s', cm[2]*100/total),
        ('False 1s', cm[3]*100/total)]
with file:
    write = csv.writer(file)
    write.writerows(data)


torch_img = testing_concatenator.concatenated_images[0]
torch_img = torch.tensor(np.expand_dims(torch_img, axis=0))
method = 'gradcam++'
input_tensor = torch_img


target_layer3 = model.layer3
target_layer6 = model.layer6

cam3 = CAM(model=model, target_layer=target_layer3)
cam6 = CAM(model=model, target_layer=target_layer6)

grayscale_cam3 = cam3(input_tensor=input_tensor, method=method)
grayscale_cam6 = cam6(input_tensor=input_tensor, method=method)

first_image = testing_concatenator.first_images[0].numpy()
first_image = np.moveaxis(first_image, 0, -1)

second_image = testing_concatenator.second_images[0].numpy()
second_image = np.moveaxis(second_image, 0, -1)

visualization61 = show_cam_on_image(first_image, grayscale_cam6)
visualization62 = show_cam_on_image(second_image, grayscale_cam6)



plt.imshow(testing_concatenator.first_images_original[0])
plt.show()

plt.imshow(testing_concatenator.second_images_original[0])
plt.show()

plt.imshow(grayscale_cam3)
plt.show()

plt.imshow(grayscale_cam6)
plt.show()

plt.imshow(visualization61)
plt.show()

plt.imshow(visualization62)
plt.show()






