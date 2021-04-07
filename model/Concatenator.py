import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import cv2


class Concatenator(Dataset):

    def __init__(self, csvfile, image_dim=(56, 56)):
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

        self.load(csvfile)

    def first_images(self):
        return self.first_images

    def second_images(self):
        return self.second_images
    
    def first_images_original(self):
        return self.first_images_original

    def second_images_oriinal(self):
        return self.second_images_original

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
        # This code is ugly lmao I'll clean it eventually

        df = pd.read_csv(csvfile)

        for i in df.index:

            print(i)

            name1 = df["image1"][i]
            image1 = np.asarray(Image.open(name1))
            self.first_images_original.append(image1)

            name2 = df["image2"][i]
            image2 = np.asarray(Image.open(name2))
            self.second_images_original.append(image2)

            if df["label"][i] == "same":
                self.labels.append(1)
            else:
                self.labels.append(0)

            
            transformed1 = self.transform(image1)
            transformed2 = self.transform(image2)

            m1 = torch.mean(torch.flatten(transformed1))
            m2 = torch.mean(torch.flatten(transformed2))

            s1 = torch.std(torch.flatten(transformed1))
            s2 = torch.std(torch.flatten(transformed2))

            transformed1 = (transformed1 - m1)
            transformed2 = (transformed2 - m2)

            concatenated = torch.cat((transformed1, transformed2), 0)

            self.first_images.append(transformed1)
            self.second_images.append(transformed2)
            self.concatenated_images.append(concatenated)
