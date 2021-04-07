import os
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from modified_lenet import Modified_LeNet
from Concatenator import Concatenator
from pytorch_grad_cam import CAM
from pytorch_grad_cam.utils.image import show_cam_on_image

batch_size = 64
image_dim = (56, 56)
dim = image_dim[0]
dim = int((int(dim/2) - 2)/2) - 6

model = Modified_LeNet(batch_size=batch_size, dim=dim)

parameters_path = os.path.join(os.getcwd(), "parameters")
size = 'large'
model.load_state_dict(torch.load(os.path.join(parameters_path, size + '.pth')))

torch_img = ?
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