import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from warp import warp
import pandas as pd
import cv2
from cutandpaste import cutandpaste
from numpy.random import binomial


class Concatenator(Dataset):

    def __init__(self, csvfile=None, image_dim=(56, 56)):
        self.concatenated_images = []
        self.first_images = []
        self.second_images = []
        self.first_images_original = []
        self.second_images_original = []
        self.labels = []
        self.image_dim = image_dim

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(image_dim),
            T.ToTensor()])

        if csvfile is not None:
            self.load(csvfile)

    # image is a path to image
    def transform_image(self, image):
        image = np.asarray(Image.open(image))
        return self.transform(image)

    # image is a numpy array
    def transform_image2(self, image):
        return self.transform(image)

    # path1 and path2 are paths to image1 and image2
    def concatenate(self, path1, path2):
        image1 = np.asarray(Image.open(path1))
        image2 = np.asarray(Image.open(path2))
    

        transformed1 = self.transform(image1)
        transformed2 = self.transform(image2)

        return torch.cat((transformed1, transformed2), 0)

    # path is path to image1, image is already a numpy array
    def concatenate2(self, path, image):
        image1 = np.asarray(Image.open(path))
        

        transformed1 = self.transform(image1)
        transformed2 = self.transform(image)

        return torch.cat((transformed1, transformed2), 0)

    # Returns first_images after transformation
    def first_images(self):
        return self.first_images

    # Returns second_images after transformation
    def second_images(self):
        return self.second_images
    
    # Returns first_images before transformation
    def first_images_original(self):
        return self.first_images_original

    # Returns second_images before transformation
    def second_images_original(self):
        return self.second_images_original

    # Returns concatenated_images after both images were tranformed
    def concatenated_images(self):
        return self.concatenated_images

    def labels(self):
        return self.labels

    def image_dim(self):
        return self.image_dim

    def __len__(self):
        return len(self.concatenated_images)

    def __getitem__(self, index):
        item = self.concatenated_images[index]
        label = self.labels[index]
        return item, label

    def load(self, csvfile):

        df = pd.read_csv(csvfile)


        for i in df.index:

            print(i)
            try:
                name1 = df["image1"][i]
                image1 = np.array(Image.open(name1))
                self.first_images_original.append(image1)
                
                if df["label"][i] == "same":
                    image2 = image1
                    self.second_images_original.append(image2)
                    self.labels.append(1)
                else:
                    # Want 20% of different pairs to be different products in same class
                    if binomial(n=1,p=.2):
                        name2 = df["image2"][i]
                        image2 = np.array(Image.open(name2))
                        self.second_images_original.append(image2)
                    # Want 80% of different pairs to be cutandpaste version of each other
                    else:
                        image2 = cutandpaste(image1)
                        self.second_images_original.append(image2)

                    self.labels.append(0)
                transformed1 = self.transform(image1)
                transformed2 = self.transform(image2)

                if binomial(n=1,p=.5):
                    concatenated = torch.cat((transformed1, transformed2), 0)
                else:
                    concatenated = torch.cat((transformed2, transformed1), 0)

                self.first_images.append(transformed1)
                self.second_images.append(transformed2)
                self.concatenated_images.append(concatenated)
            except FileNotFoundError:
                continue
