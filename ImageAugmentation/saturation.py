import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random

def saturate(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    random_value = round(random.uniform(0.5, 3), 2)

    (h, s, v) = cv2.split(image)
    s = s * random_value
    s = np.clip(s , 0, 255)
    
    saturated = cv2.merge([h,s,v])
    saturated = cv2.cvtColor(saturated.astype("uint8"), cv2.COLOR_HSV2BGR)

    return saturated
