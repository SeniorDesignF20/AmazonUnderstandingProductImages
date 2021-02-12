import os
import cv2
import numpy as np
import random


def translate(img, shift_x, shift_y):
    # store height and width
    height, width = img.shape[:2]

    # transformation matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # warpAffine to transform the image using matrix, T
    translated_img = cv2.warpAffine(
        img, M, (width + shift_x, height + shift_y), borderValue=(255, 255, 255))

    return translated_img


def rotate(img, angle):

    # grab the dimensions of the image and determine the center
    (height, width) = img.shape[:2]
    (center_x, center_y) = (width // 2, height // 2)

    # rotation components of matrix
    M = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (new_width / 2) - center_x
    M[1, 2] += (new_height / 2) - center_y
    # perform the actual rotation and return the image
    return cv2.warpAffine(img, M, (new_width, new_height), borderValue=(255, 255, 255))


# directory = r'Benign'
# for filename in os.listdir(directory):
#     if filename.endswith('.jpg'):
#         img_path = os.path.join(directory, filename)
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)

#         # translate image
#         random_x = random.randint(0, 200)
#         random_y = random.randint(0, 200)
#         translated_img = translate(img, random_x, random_y)

#         # rotate translated image
#         angle = random.randint(0, 360)
#         rotated_img = rotate(translated_img, angle)

#         cv2.imwrite('Alignment/' + filename, rotated_img)
