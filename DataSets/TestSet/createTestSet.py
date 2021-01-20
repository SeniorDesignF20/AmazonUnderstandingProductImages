import os
import sys
import json
import gzip
import urllib.request
from PIL import Image, ImageOps
import random

sys.path.append("../../")
import cutandpaste as cp

'''
values = []
with open("../preProcessed_meta_AMAZON_FASHION.json") as data:
    values = data.read()
    values = values.replace( 'SR38,50','SR38%2050')
    values = values.split(",")


for i in range(1000):
    image_url = values[i]
    image_url = image_url.replace('"', '')
    image_url = image_url.replace('[', '')
    #image_url = image_url.replace( 'SR38,50','SR38%2050')
    filename = image_url.split("/")[-1]

    # Check if the image was retrieved successfully
    try:
        # Open the url image, set stream to True, this will return the stream content.
        r = requests.get(image_url, stream = True)
       
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True
            
            # Open a local file with wb ( write binary ) permission.
            with open(filename,'wb') as f:
                shutil.copyfileobj(r.raw, f)
                
            print('Image sucessfully Downloaded: ',filename)
        else:
            print('Image Couldn\'t be retreived')
    except: 
        pass
'''

