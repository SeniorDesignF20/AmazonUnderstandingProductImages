import os
import numpy as np
import torch
from ast import literal_eval
from modified_lenet import Modified_LeNet
from Concatenator import Concatenator
import csv

# image1 and image2 are expected to be paths to the images


def classify(image1, image2, path, size='small'):
    results_path = os.path.join(path, size)
    results_file = os.path.join(results_path, size + '_results.csv')
    with open(results_file, 'r') as f:
        mycsv = csv.reader(f)
        mycsv = list(mycsv)
        image_dim = literal_eval(mycsv[4][1])
        batch_size = int(mycsv[5][1])
    batch_size = 64

    dim = image_dim[0]
    dim = int((int(dim/2) - 2)/2) - 6

    concatenator = Concatenator(csvfile=None, image_dim=image_dim)
    concatenated_image = concatenator.concatenate(image1, image2)
    concatenated_image = torch.tensor(
        np.expand_dims(concatenated_image, axis=0))

    model = Modified_LeNet(batch_size=batch_size, dim=dim)

    parameters_path = os.path.join(path, "parameters")
    model.load_state_dict(torch.load(
        os.path.join(parameters_path, size + '.pth')))

    output = model(concatenated_image)
    _, predicted = torch.max(output.data, 1)
    return predicted.numpy().tolist()[0]