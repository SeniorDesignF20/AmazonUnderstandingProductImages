import os
import cv2
from PIL import Image

# script to reformat Amazon mini data set 
# resizes and centers product on image with white background
# renames image files
# reformatted images will be saved in DataSet and DataSetCP 
# for original and synthesized images respectively

count = 0

def reformat(img_path, target_width, target_height):
    global count
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    image = Image.open(img_path, 'r')

    target_ratio = target_height / target_width
    image_ratio = image.height / image.width

    if target_ratio > image_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * image_ratio)
    else:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / image_ratio)

    image_resize = image.resize((resize_width, resize_height), Image.ANTIALIAS)

    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
    
    destination = 'DataSet/'
    background.paste(image_resize, offset)
    
    name = 'img' + str(count)
    background.convert('RGB').save(destination + name + '.jpg', 'JPEG')
    print(name + ".jpg - image has been resized!")

def reformat_synthesized(img_path, target_width, target_height):
    global count
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    image = Image.open(img_path, 'r')

    target_ratio = target_height / target_width
    image_ratio = image.height / image.width

    if target_ratio > image_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * image_ratio)
    else:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / image_ratio)

    image_resize = image.resize((resize_width, resize_height), Image.ANTIALIAS)

    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
    
    destination = 'DataSetCP/'
    background.paste(image_resize, offset)
    
    name = 'img' + str(count + 305)
    background.convert('RGB').save(destination + name + '.jpg', 'JPEG')
    print(name + ".jpg - c&p image has been resized!")
    count = count + 1

# change directory paths
directory = r'Benign'
cp_directory = r'CutAndPaste'

for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        # path of image files
        img_path = os.path.join(directory, filename)
        reformat(img_path, 600, 600)

        cp_img_path = os.path.join(cp_directory, filename)
        reformat_synthesized(cp_img_path, 600, 600)

