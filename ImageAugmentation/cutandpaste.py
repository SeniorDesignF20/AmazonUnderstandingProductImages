from PIL import Image, ImageOps
import os
import random


def cutandpaste(im):
    xsize, ysize = im.size
    xbox = int(xsize/4)
    ybox = int(ysize/4)
    xrand1 = int(random.random() * xsize)
    yrand1 = int(random.random() * ysize)
    xrand2 = int(random.random() * xsize)
    yrand2 = int(random.random() * ysize)

    if xrand1 + xbox > xsize:
        xrand1 = xsize - xbox
    if yrand1 + ybox > ysize:
        yrand1 = ysize - ybox
    if xrand2 + xbox > xsize:
        xrand2 = xsize - xbox
    if yrand2 + ybox > ysize:
        yrand2 = ysize - ybox

    box = (xrand1, yrand1, xrand1 + xbox, yrand1 + ybox)
    box1 = (xrand1 - 1, yrand1 - 1, xrand1 + xbox + 1, yrand1 + ybox + 1)
    box2 = (xrand2 - 1, yrand2 - 1, xrand2 + xbox + 1, yrand2 + ybox + 1)

    region = im.crop(box)
    region_border1 = ImageOps.expand(region, border=1, fill='blue')
    region_border2 = ImageOps.expand(region, border=1, fill='red')
    im.paste(region_border1, box1)
    im.paste(region_border2, box2)
    return im



random.seed()
directory = r'AmazonSet/TestSet/Benign_new'
for filename in os.listdir(directory):
    im = Image.open(os.path.join(directory, filename))
    im = cutandpaste(im)
    im.save(os.path.join(r'AmazonSet/TestSet/CutAndPaste_new', filename))
