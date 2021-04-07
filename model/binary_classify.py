import os
import numpy as np
import torch
from modified_lenet import Modified_LeNet
from Concatenator import Concatenator

# image1 and image2 are expected to be paths to the images


def classify(image1, image2, size='small'):
    concatenator = Concatenator()
    concatenated_image = concatenator.concatenate(image1, image2)
    concatenated_image = torch.tensor(
        np.expand_dims(concatenated_image, axis=0))
    batch_size = 64
    image_dim = (56, 56)
    dim = image_dim[0]
    dim = int((int(dim/2) - 2)/2) - 6

    model = Modified_LeNet(batch_size=batch_size, dim=dim)

    parameters_path = os.path.join(os.getcwd(), "parameters")
    model.load_state_dict(torch.load(
    os.path.join(parameters_path, size + '.pth')))

    output = model(concatenated_image)
    _, predicted = torch.max(output.data, 1)
    return predicted.numpy().tolist()[0]


#Example code
"""
image1 = '/c/Users/sorou/Desktop/SeniorDesign/Datasets/Wallets/B000078QZN/B000078QZN_0.jpg'
image2 = '/c/Users/sorou/Desktop/SeniorDesign/Datasets/Wallets/B000078QZN/B000078QZN_4.jpg'

print(classify(image1, image2, size='large'))
"""

# image1 = '.\DataSets\Accessories\B000922SGS_9.jpg'
# image2 = '.\DataSets\Accessories\B000922SGS_1.jpg'

# print(classify(image1, image2, size='large'))
