# Amazon Mini Dataset

This directory contains a subset of the Amazon Fashion dataset which will be used temporarily to train and test a difference detection baseline.

## TestSet
This directory contains the original images from the Amazon set as well as synthesized images.

## Reformatting Script
The reformat-images.py script reformats all images in the TestSet directory for consistency. All images are resized to 600 x 600 pixels. The reformatted images can be found in the Benign (regular images) and Manipulated (synthesized images) directories. 

## Image Samples

According to the JD Group paper, a sample in the dataset consists of an image pair and a label without coordinate of bounding box.

Image filenames beginning with 'b' denotes a benign image.
Image filenames beginning with 'm' denotes a manipulated (synthesized) image.

Image pairs labeled 'same' includes two benign images.
Image pairs labeled 'different' includes a benign image and a manipulated image.

## Annotation Files
The train.csv and test.csv files contains image pairs and respective labels for the training and testing set respectively. 