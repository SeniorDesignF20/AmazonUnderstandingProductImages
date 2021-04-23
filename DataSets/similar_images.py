import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def areTheyDifferent(img1, img2):

    img1_gray = rgb2gray(np.asarray(Image.open(img1)))
    img2_gray = rgb2gray(np.asarray(Image.open(img2)))

    img1_h, img1_w = img1_gray.shape    
    num_pixels = img1_h * img1_w

    num_differences = np.count_nonzero(np.abs(img1_gray - img2_gray))
    print(num_differences)
    print(np.abs(img1_gray - img2_gray))

    if(num_differences / num_pixels > 0.25):
        return True
    else:
        return False


def rgb2gray(rgb):
    #0.2989, 0.5870, 0.1140
    return np.dot(rgb[...,:3], [0.3333, 0.3333, 0.3333])


img1 = "../model/Test_GradCam/Dress/B00VLZQ9BC_4.jpg"
img2 = "../model/Test_GradCam/Dress/B00VLZSJH4_2.jpg"

print(areTheyDifferent(img1,img2))



