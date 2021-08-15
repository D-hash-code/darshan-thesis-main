
from PIL import Image
import os
#import pickle
import torch
import random
import torchvision
from typing import Optional

from torchvision.datasets.vision import VisionDataset
from torchvision import datasets, transforms, utils
import numpy as np
from matplotlib import pyplot as plt


class FFHQ(VisionDataset):

    def __init__(self, root, transform, batch_size = 60, test_mode = False, return_all = False, imsize=256):

        self.root = root
        self.transform = transform
        self.return_all = return_all

        print("root:",self.root)
        all_folders = os.listdir(self.root)

        self.length = sum([len(os.listdir(os.path.join(self.root,folder))) for folder in all_folders]) # = 70000
        self.fixed_transform = transforms.Compose(
                [ transforms.Resize(imsize),
                    transforms.CenterCrop(imsize),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        self.fixed_indices = []

        for _ in range(batch_size):
            id = np.random.randint(self.length)
            self.fixed_indices.append(id)

    def __len__(self):
        return self.length


    def fixed_batch(self, random = False):
        if random == False:
            return torch.stack([self.random_batch(idx, True)[0].cuda() for idx in self.fixed_indices])
        else:
            return torch.stack([self.random_batch(np.random.randint(self.length), True)[0].cuda() for _ in range(len(self.fixed_indices))])

    def random_batch(self,index, fixed=False):

        folder = str(int(np.floor(index/1000)*1000)).zfill(5)
        file = str(index).zfill(5) + ".png"
        image_path = os.path.join(self.root, folder , file )
        img = Image.open( image_path).convert('RGB')
        if fixed:
            img = self.fixed_transform(img)
        else:
            img = self.transform(img)


        return img, torch.zeros(1).long(), image_path

    def __getitem__(self,index):

        if self.return_all:
            return self.exact_batch(index)
        else:
            return self.random_batch(index)


class Celeba(VisionDataset):

    def __init__(self, root, transform, batch_size = 60, test_mode = False, return_all = False, imsize=128):

        self.root = root
        
        self.transform = transform
        self.return_all = return_all
        self.all_files = os.listdir(self.root)
        #all_files = []
        #for i in range(1,202600): #
        #    all_files.append(str(i).zfill(6) + '.png')
        self.length = len(self.all_files)
        self.fixed_transform = transforms.Compose(
                [ transforms.Resize(imsize),
                    transforms.CenterCrop(imsize),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        self.fixed_indices = []



        for _ in range(batch_size):
            id = np.random.randint(self.length)
            self.fixed_indices.append(id)

    def __len__(self):
        return self.length


    def fixed_batch(self):
        return torch.stack([self.random_batch(idx, True)[0].cuda() for idx in self.fixed_indices])


    def random_batch(self,index, fixed=False):

        file = str(index+1).zfill(6) + '.png'
        #file = self.all_files[index+1]
        image_path = os.path.join(self.root, file )
        img = Image.open( image_path).convert('RGB')
        if fixed:
            img = self.fixed_transform(img)
        else:
            img = self.transform(img)

        return img, torch.zeros(1).long(), image_path

    def __getitem__(self,index):

        if self.return_all:
            return self.exact_batch(index)
        else:
            return self.random_batch(index)
