import os 
import json 
import gzip
import pandas as pd 
import urllib.request
import requests
import shutil
from PIL import Image, ImageOps


###load duplicate id's
#duplicates = []
#my_file = open("DataSets/duplicates.txt", "r")
#duplicates = my_file.readlines()

### load the meta data
data = []
with open('meta_AMAZON_FASHION.json') as f:
    for l in f:
        data.append(json.loads(l.strip()))

df = pd.DataFrame.from_dict(data)
df.drop(df.columns.difference(["image", "asin"]), 1, inplace=True)

df = df[df['image'].map(lambda d: len(d)) > 4]
print(len(df))

image_list = []
for index, row in df.head(1000).iterrows():
    
    image_list = row['image']
    asin = row['asin']

    for i in range(len(image_list)):
        image_url = image_list[i]
        
        if 'US40' in image_url:
            image_url = image_url.replace( 'US40','SR38%2050')
        if 'SR38,50' in image_url:
            image_url = image_url.replace( 'SR38,50','SR38%2050')
        if 'SX38_SY50_CR,0,0,38,50' in image_url:
            image_url = image_url.replace( 'SX38_SY50_CR,0,0,38,50','SR38%2050')
       
        filename = asin + "_" + str(i) + ".jpg"
        
        # Check if the image was retrieved successfully
        try:
            # Open the url image, set stream to True, this will return the stream content.
            r = requests.get(image_url, stream = True, timeout=0.1)
        
            if r.status_code == 200:
                # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                r.raw.decode_content = True
                
                # Open a local file with wb ( write binary ) permission.
                with open(filename,'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                    
                print('Image sucessfully Downloaded: ',filename)
            else:
                print('Image Couldn\'t be retreived')
        except requests.exceptions.Timeout:
            pass
        except: 
            pass



