import colorrandomization
import cutandpaste
import lighting
import saturation
from PIL import Image
import cv2
import os
import numpy as np
import pandas as pd

# Script to perform random augmentations on Amazon dataset

count = 0
randDF = pd.DataFrame(columns=['image1', 'image2', 'label'])
benign_src_path = r'../AmazonSet/TestSet/Benign_new'
manipulated_dest_path = r'../AmazonSet/rand_Manipulated'


def rand_augment(imFile):
    global count, randDF
    randlabel = 'same'
    image = cv2.imread(os.path.join(benign_src_path, imFile))
    flags = np.random.randint(2, size=5)
    # Cut and paste augmentation
    if flags[0] == 1:
        image = cutandpaste.cutandpaste(image)
        randlabel = 'different'
    # Single channel augmentation
    if flags[1] == 1:
        image = colorrandomization.singleChannel(image)
    # Mutli channel augmentation
    if flags[2] == 1:
        image = colorrandomization.multipleChannels(image)
    # Spotlight augmentation
    if flags[3] == 1:
        image = lighting.add_spot_light(image)
    # Saturation augmentation
    if flags[4] == 1:
        image = saturation.saturate(image)
    mname = 'm' + '{:05d}'.format(count) + '.jpg'
    bname = 'b' + '{:05d}'.format(count) + '.jpg'
    randDF = randDF.append(
        {'image1': bname, 'image2': mname, 'label': randlabel}, ignore_index=True)
    cv2.imwrite(manipulated_dest_path + '/' + mname, image)
    count += 1


for filename in os.listdir(benign_src_path):
    if filename.endswith('.jpg'):
        rand_augment(filename)

# Train/Test CSV generator!
# Train set: 500 same (identical), 500 same (augmented), 400 different (local), 200 different (global)
# Test set: 125 same (identical), 125 same (augmented), 100 different (local), 50 different (global)

# Samples random rows for globally different image pairs
# Collisions can occur here where the sampled images should be labeled 'same', will try to fix later
train_diff_global_1 = randDF.sample(200).reset_index(drop=True)
train_diff_global_2 = randDF.sample(200).reset_index(drop=True)
train_diff_global = pd.concat(
    [train_diff_global_1['image1'], train_diff_global_2['image2']], axis=1)
train_diff_global['label'] = 'different'
test_diff_global_1 = randDF.sample(50).reset_index(drop=True)
test_diff_global_2 = randDF.sample(50).reset_index(drop=True)
test_diff_global = pd.concat(
    [test_diff_global_1['image1'], test_diff_global_2['image2']], axis=1)
test_diff_global['label'] = 'different'

# Error check, checks if an insufficient number of same/differnet pairs are generated
# randDF_same = randDF.loc[randDF['label'] == 'same'].reset_index(drop=True)
# if len(randDF_same.index < 625):
#     print("Too few same samples generated")

# Samples random rows for locally different image pairs, drops those rows when found to prevent duplicated data from appearing in training/testing set
train_diff_local = randDF.loc[randDF['label'] == 'different'].sample(400)
randDF.drop(train_diff_local.index)

test_diff_local = randDF.loc[randDF['label'] == 'different'].sample(100)
randDF.drop(test_diff_local.index)

# Samples random rows for images that have been augmented, but are still the same
train_same_m = randDF.loc[randDF['label'] == 'same'].sample(500)
randDF.drop(train_same_m.index)

test_same_m = randDF.loc[randDF['label'] == 'same'].sample(125)
randDF.drop(test_same_m.index)

# Samples random rows for identical images (benign vs benign)
train_same_b_1 = randDF.sample(500)
randDF.drop(train_same_b_1.index)
train_same_b = pd.concat(
    [train_same_b_1['image1'], train_same_b_1['image2']], axis=1, keys=['image1', 'image2'])
train_same_b['label'] = 'same'

test_same_b_1 = randDF.sample(125)
randDF.drop(test_same_b_1.index)
test_same_b = pd.concat(
    [test_same_b_1['image1'], test_same_b_1['image2']], axis=1, keys=['image1', 'image2'])
test_same_b['label'] = 'same'

train_randDF = train_diff_global.append(
    [train_diff_local, train_same_m, train_same_b]).reset_index(drop=True)
test_randDF = test_diff_global.append(
    [test_diff_local, test_same_m, test_same_b]).reset_index(drop=True)

print(train_randDF)
print(test_randDF)

train_randDF.to_csv(r'../AmazonSet/train_rand.csv', index=False)
test_randDF.to_csv(r'../AmazonSet/test_rand.csv', index=False)
