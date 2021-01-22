from skimage.measure import compare_ssim
from cv2 import cv2
import os
import numpy as np

benign = cv2.imread(r'BoundingBox\Test1\Inputs\test1_benign.jpg')
altered = cv2.imread(r'BoundingBox\Test1\Inputs\test1_altered.jpg')

# Convert images to grayscale
benign_gray = cv2.cvtColor(benign, cv2.COLOR_BGR2GRAY)
altered_gray = cv2.cvtColor(altered, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = compare_ssim(benign_gray, altered_gray, full=True)
print("Image similarity", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] benign we can use it with OpenCV
diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(
    diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(
    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(benign.shape, dtype='uint8')

for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(benign, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(altered, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)

output = r'BoundingBox\Test1\Outputs'
cv2.imwrite(os.path.join(output, 'benignBox.jpg'), benign)
cv2.imwrite(os.path.join(output, 'alteredBox.jpg'), altered)
cv2.imwrite(os.path.join(output, 'diff.jpg'), diff)
cv2.imwrite(os.path.join(output, 'mask.jpg'), mask)

cv2.waitKey(0)
