import os
import cv2
from PIL import Image

# script to reformat Amazon mini data set

count = 0


def create_folder(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
            print(f"Created Folder @ {path}")

        except OSError as error:
            print("Folder cannot be created")
            print(error)
    else:
        print(f"Folder @ {path} Already Exists")


def reformat(img_path, dest_path, target_width, target_height):
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

    background = Image.new(
        'RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - resize_width) / 2),
              round((target_height - resize_height) / 2))

    destination = os.path.dirname(
        os.path.abspath(__file__)) + '/' + dest_path + '/'
    background.paste(image_resize, offset)

    name = 'b' + '{:05d}'.format(count)
    background.convert('RGB').save(destination + name + '.jpg', 'JPEG')
    print(name + ".jpg - benign image has been resized!")


def reformat_manipulated(img_path, dest_path, target_width, target_height):
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

    background = Image.new(
        'RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - resize_width) / 2),
              round((target_height - resize_height) / 2))

    destination = os.path.dirname(
        os.path.abspath(__file__)) + '/' + dest_path + '/'
    background.paste(image_resize, offset)

    name = 'm' + '{:05d}'.format(count)
    background.convert('RGB').save(destination + name + '.jpg', 'JPEG')
    print(name + ".jpg - manipulated image has been resized!")
    count = count + 1

# change directory paths


benign_src_path = r'AmazonSet\TestSet\Benign_new'
manipulated_src_path = r'AmazonSet\TestSet\CutAndPaste_new'

benign_dest_path = r'Benign'
manipulated_dest_path = r'Manipulated'

create_folder(benign_dest_path)
create_folder(manipulated_dest_path)

for filename in os.listdir(benign_src_path):
    if filename.endswith('.jpg'):
        # path of image files
        img_path = os.path.join(benign_src_path, filename)
        reformat(img_path, benign_dest_path, 600, 600)

        img_path = os.path.join(manipulated_src_path, filename)
        reformat_manipulated(img_path, manipulated_dest_path, 600, 600)
