from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
from typing import List, Tuple


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

        _dataset: List[Tuple[Image, int]] = []
        if split in ("train", "test"):
            with open(f"./{split}.txt", 'r') as f:
                classes = []
                for path in f.readlines():
                    path = path.strip("\n")
                    clazz = path.split("/")[0]
                    if clazz != "BACKGROUND_Google":
                        if clazz not in classes:
                            classes.append(clazz)
                        _dataset.append((pil_loader(os.path.join(root, path)), classes.index(clazz)))
        self.dataset = _dataset

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        if index < len(self.dataset):
            image, label = self.dataset[index]
        else:
            image, label = None, None
                           # Provide a way to access image and label via index
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
        # Provide a way to get the length (number of elements) of the dataset
        return len(self.dataset) if self.dataset else 0


if __name__ == "__main__":
    DATA_DIR = './101_ObjectCategories'
    train_dataset = Caltech(DATA_DIR, split='train', transform=None)
    for i in range(0, len(train_dataset), 100):
        print(train_dataset[i])
    test_dataset = Caltech(DATA_DIR, split='test', transform=None)
    for i in range(0, len(test_dataset), 100):
        print(test_dataset[i])
