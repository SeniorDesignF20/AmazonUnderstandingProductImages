import os
import numpy as np
from Concatenator import Concatenator
from pathlib import Path
from confusion_matrix import confusion_matrix
from sklearn import svm

curr_path = os.path.dirname(os.path.abspath(__file__))
datasets_path = str(Path(curr_path).parents[0]) +  "/AmazonSet"

print("Concatinating training data")
training_concatenator = Concatenator(datasets_path, "train_expanded.csv")
print("Finished")

print("Concatinating testing data")
testing_concatenator = Concatenator(datasets_path, "test_expanded.csv")
print("Finished")

print("Creating Model")
X = training_concatenator.concatenated_images[:100]
for i, item in enumerate(X):
	X[i] = item.numpy()
print(type(X[0]))
print(X[0].shape)
y = training_concatenator.labels[:100]
model = svm.SVC()

print("Training Model")
model.fit(X, y)

print("Testing Model")

X_true = testing_concatenator.concatenated_images
y_true = testing_concatenator.labels

y_pred = model.predict(X_true)

total = len(y_true)
correct = (y_true == y_pred).sum().item()

print(f"Accuracy over test set: {100*correct/total}%")
print()

cm = confusion_matrix(y_pred, y_true)

print("0=Different, 1=Same")
print(f"True 0s = {cm[0]}, {cm[0]/2}%")
print(f"True 1s = {cm[1]}, {cm[1]/2}%")
print(f"False 0s = {cm[2]}, {cm[2]/2}%")
print(f"False 1s = {cm[3]}, {cm[3]/2}%")

