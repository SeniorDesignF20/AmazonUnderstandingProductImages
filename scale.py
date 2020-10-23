import os
import cv2 
import numpy as np
import tensorflow as tf
import random
from PIL import Image 

def random_scale(image):
  (height, width) = image.shape[:2]
  rand_zoom = random.randint(0,1)
  if rand_zoom == 0:
    narrow = random.uniform(0.25, 0.75)
    newsize = np.array([height*narrow, width*narrow])
  else:
    amplify = random.uniform(1.5, 4)
    newsize = np.array([height*amplify, width*amplify])
  
  return tf.image.resize(
        image, newsize,  
        method=tf.image.ResizeMethod.BILINEAR, 
        preserve_aspect_ratio=False,
        antialias=False)


directory = r'drive/My Drive/Colab Notebooks/Benign'
to_directory = r'Scale/'
for filename in os.listdir(directory):
  if filename.endswith('.jpg'):
    image_path = os.path.join(directory, filename)
    image = cv2.imread(image_path)

    scaled_image = np.array(random_scale(image))
    cv2.imwrite(to_directory + filename, scaled_image)