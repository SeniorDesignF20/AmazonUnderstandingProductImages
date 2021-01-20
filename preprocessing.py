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
with open('../../meta_AMAZON_FASHION.json') as f:
    for l in f:
        data.append(json.loads(l.strip()))


# convert list into pandas dataframe
df = pd.DataFrame.from_dict(data)
print(len(df))


# remove duplicates as oultined in duplicates.txt
for line in duplicates:
    ids = line.split()
    for id in ids[1:]:
        if id in df.asin:
            df.drop(df.loc[df['asin']== id].index, inplace=True)
            print(len(df))

# drop all columns from dataframe except asin, image, and title
df.drop(df.columns.difference(["image"]), 1, inplace=True)

#df.drop(df.loc[df['asin']=='630456984X'].index, inplace=True)
print(len(df))

# get all images into single list
dfList = df.values.tolist()
dfList = [i for i in dfList if i != [[]]]
dfList2 = []
for i in dfList:
    for j in i:
        for k in j:
            dfList2.append(k)


# dump data into json file
with open("DataSets/preProcessed_meta_AMAZON_FASHION.json", "w") as f:
    json.dump(dfList2, f)



#i = df[df["asin"] == '630456984X']
#print(i)

