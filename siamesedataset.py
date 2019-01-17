import torchvision
from torch.utils.data import Dataset
import numpy as np
import random
import torch
from PIL import Image
import PIL.ImageOps    

class SiameseNetworkDataset(Dataset):
    """Class in which the siamese dataset is created.

    Note:
        None.

    Args:
        imageFolderDataset (object): object containing the dataset
        transform (object): object containing all pertinent transforms for images
        shouldInvert (bool): bool to invert image or not
        randomize (bool): bool to randomize image or not

    Attributes:
        imageFolderDataset (object): object containing the dataset
        transform (object): object containing all pertinent transforms for images
        shouldInvert (bool): bool to invert image or not
        randomize (bool): bool to randomize image or not

    """
    
    def __init__(self, imageFolderDataset, transform=None, shouldInvert=True, randomize=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.shouldInvert = shouldInvert
        self.randomize = randomize
        
    def __getitem__(self, index):
        """Class methods are similar to regular functions.

        Note:
            None

        Args:
            index: the index of the images

        Returns:
            img0, img1, similarity

        """

        img0Tuple = random.choice(self.imageFolderDataset.imgs)
        getSameClass = random.randint(0,1) # ensure approx. 50% of images are in the same class
        if getSameClass:
            while True: # keep looping till the same class image is found
                img1Tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0Tuple[1]==img1Tuple[1]:
                    break
        else:
            img1Tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0Tuple[0])
        img1 = Image.open(img1Tuple[0])
        
        if self.shouldInvert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1Tuple[1]!=img0Tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)