import os
import cv2 
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from PIL import Image 
from scipy.spatial import ConvexHull


def random_scale(image, zoom_ctr):
  size = (height, width) = image.shape[:2]
  newsize = random_newsize(image, zoom_ctr)
  new_height = newsize[0]
  new_width = newsize[1]
  if new_height > height:
    left = int(round(((new_width - width)/2), 0))
    right = int(round(((new_width + width)/2), 0))
    top = int(round(((new_height - height)/2), 0))
    bottom = int(round(((new_height + height)/2), 0))
    new_im = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)
    re_image = new_im[top:bottom, left:right]
  else:
    new_im = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)
    background = np.zeros((height, width,3), np.uint8)
    background[:, :] = (255,255,255)
    x_offset = int(round(((width - new_width) / 2), 0))
    y_offset = int(round(((height - new_height) / 2),0))
    re_image = background.copy()
    re_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = new_im.copy()

  return re_image

def random_newsize(image, zoom_ctr):
  (height, width) = image.shape[:2]
  #rand_zoom = random.randint(0,1)
  if zoom_ctr == 0:
    narrow = random.uniform(0.25, 0.75)
    newsize = (int(round(height*narrow, 0)), int(round(width*narrow, 0)))
  else:
    amplify = random.uniform(1.5, 4)
    newsize = (int(round(height*amplify, 0)), int(round(width*amplify, 0)))
  
  return newsize




directory = r'DataSets/TestSet/Benign'
to_directory = r'DataSets/TestSet/Scale'
for filename in os.listdir(directory):
  if filename.endswith('.jpg'):
    image_path = os.path.join(directory, filename)
    image = cv2.imread(image_path)
    zoom_ctr = (int(filename[-5]))%2
    scaled_image = np.array(random_scale(image, zoom_ctr))
    
    cv2.imwrite(to_directory + filename, scaled_image)