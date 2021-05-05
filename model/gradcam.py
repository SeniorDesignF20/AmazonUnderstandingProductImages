import os
import numpy as np
import torch
import csv
from ast import literal_eval
from modified_lenet import Modified_LeNet
from Concatenator import Concatenator
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from PIL import Image
import base64
from io import BytesIO
from CreateBox import CreateBox


def gradcam(image1, image2, path, size='small'):
    parameters_path = os.path.join(path, "parameters")
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
    model = Modified_LeNet(batch_size=batch_size, dim=dim)
    model.load_state_dict(torch.load(
        os.path.join(parameters_path, size + '.pth')))

    concatenator = Concatenator(csvfile=None, image_dim=image_dim)
    input_tensor = concatenator.concatenate(image1, image2)
    input_tensor = torch.tensor(np.expand_dims(input_tensor, axis=0))

    target_layer = model.layer6

    cam = GradCAM(model=model, target_layer=target_layer)

    #grayscale_cam = cam(input_tensor=input_tensor, method=method)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]

    first_image = concatenator.transform_image(image1).numpy()
    first_image = np.moveaxis(first_image, 0, -1)
    visualization1 = show_cam_on_image(first_image, grayscale_cam)

    second_image = concatenator.transform_image(image2).numpy()
    second_image = np.moveaxis(second_image, 0, -1)
    visualization2 = show_cam_on_image(second_image, grayscale_cam)

    whiteimage = np.zeros(first_image.shape, dtype=np.uint8)
    whiteimage.fill(0)
    heatmap = show_cam_on_image(whiteimage, grayscale_cam)

    image1_boxes = CreateBox(first_image, heatmap)
    image2_boxes = CreateBox(second_image, heatmap)

    image1_boxes = (image1_boxes*255).astype('uint8')
    image2_boxes = (image2_boxes*255).astype('uint8')
    # Processing images so they can be displayed on User Interface
    visualization1 = to_image(visualization1)
    visualization2 = to_image(visualization2)
    image1_boxes = to_image(image1_boxes)
    image2_boxes = to_image(image2_boxes)
    # visualization1 = to_data_uri(visualization1)
    # visualization2 = to_data_uri(visualization2)
    # image1_boxes = to_data_uri(image1_boxes)
    # image2_boxes = to_data_uri(image2_boxes)

    return visualization1, visualization2, image1_boxes, image2_boxes


def to_image(image):
    img = Image.fromarray(image, 'RGB')
    return img


def to_data_uri(image):
    data = BytesIO()
    image.save(data, "PNG")
    return image
    #data64 = base64.b64encode(data.getvalue())
    # return u'data:img/jpeg;base64,'+data64.decode('utf-8')


# gradcam("Test_GradCam\Bag\B00BU7VG6O_0.jpg",
#         "Test_GradCam\Bag\B00BU7VG6O_2.jpg", "", "small")
