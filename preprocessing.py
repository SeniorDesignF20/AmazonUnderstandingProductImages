import os
import json
import gzip 
import pandas as pd
from urllib.request import urlopen

### load duplicate id's
duplicates = []
my_file = open("DataSets/duplicates.txt","r")
duplicates = my_file.readlines()


### load the meta data
data = []
with open('DataSets/All_Amazon_Meta.json') as f:
    for l in f:
        data.append(json.loads(l.strip()))

# convert list into pandas dataframe
df = pd.DataFrame.from_dict(data)
print(len(df))

# drop all columns from dataframe except asin, image, and title
df.drop(df.columns.difference(["asin", "image", "title"]), 1, inplace=True)

for line in duplicates:
    ids = line.split()
    for id in ids[1:]:
        if id in df.asin:
            df.drop(df.loc[df['asin']== id].index, inplace=True)
            print(len(df))


#df.drop(df.loc[df['asin']=='630456984X'].index, inplace=True)
print(len(df))
df.to_json(r'DataSets/preProcessed_All_Amazon_Meta.json')

#i = df[df["asin"] == '630456984X']
#print(i)

