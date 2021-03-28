import os 
import json 
import gzip
import pandas as pd 
import urllib.request
import requests
import shutil
from PIL import Image, ImageOps


#load duplicate id's
# duplicates = []
# my_file = open("duplicates.txt", "r")
# duplicates = my_file.readlines()


### load the meta data
json_path = r"meta_Clothing_Shoes_and_Jewelry.json"
data = []
print("Loading json file from path:", json_path, "...")
with open(json_path) as file:
    for line in file:
        data.append(json.loads(line.strip()))
print("Done Loading.")

# convert list into pandas dataframe
print("Converting list to pandas dataframe ...")
df = pd.DataFrame.from_dict(data)

### drop all columns except category, image, asin, and title
print("Dropping all but category, image, asin, and title ...")
df = df[["category", "image", "asin", "title"]]

num_products = 20000

### save preprocessed json
print("Saving first", num_products, "products to json ...")
output_path = "preprocessed_meta_Clothing_Shoes_and_Jewelry_" + str(num_products) + ".json"
df.head(num_products).to_json(output_path, orient="records")

print("Done. Preprocessed json for", num_products, "products saved to:", output_path, "...")

print("Saving entire preprocessed json ...")
output_path = "preprocessed_meta_Clothing_Shoes_and_Jewelry.json"
df.to_json(output_path, orient="records")

print("Done. Preprocessed json saved to:", output_path, "...")

# for line in duplicates:
#     ids = line.split()
#     for id in ids[1:]:
#         if id in df.asin:
#             df.drop(df.loc[df['asin']== id].index, inplace=True)
#             print(len(df))



