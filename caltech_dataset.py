from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
from typing import List, Tuple
from random import choice


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        _split_train: List[Tuple[Image, int]] = []
        _split_test: List[Tuple[Image, int]] = []

        for i, d in enumerate([item for item in os.listdir(root)
                               if os.path.isdir(os.path.join(root, item))
                               and item != "BACKGROUND_Google"]):
            path = os.path.join(root, d)
            images = [im for im in os.listdir(path) if im.endswith(".jpg")]
            n_images = len(images)

            if n_images % 2 == 0:
                n_split_train = n_split_test = int(n_images / 2)
            else:  # if odd, train is 1 more than test
                n_split_train = int(n_images / 2) + 1
                n_split_test = int(n_images / 2)

            available = list(range(1, n_images + 1))
            for _ in range(n_split_train):
                selected = choice(available)
                # add Tuple[Image, int] where int is the class
                _split_train.append(((pil_loader(os.path.join(root, d, f"image_{str(selected).zfill(4)}.jpg"))), i))
                available.remove(selected)
            for _ in range(n_split_test):
                selected = choice(available)
                _split_test.append(((pil_loader(os.path.join(root, d, f"image_{str(selected).zfill(4)}.jpg"))), i))
                available.remove(selected)

        self.split_train = _split_train
        self.split_test = _split_test


    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = ... # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = ... # Provide a way to get the length (number of elements) of the dataset
        return length
