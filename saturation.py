import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import cv2

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

#import dataset
directory = r'Benign'

for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path)
        saturated = np.array(tf.image.random_saturation(
            image, 0.25, 5, seed=None
        ))
        cv2.imwrite('Saturated/' + filename, saturated)