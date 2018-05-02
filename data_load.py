import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os.path import expanduser
home = expanduser("~")

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class ReadDataset(Dataset):
    """Random distributed single emitters dataset."""

    def __init__(self, csv_file, tif_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with ground truth positions.
            tif_file (string): Path to the .tiff file stack with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.tif_file = tif_file
        self.transform = transform

    def __len__(self):
        return len(self.csv_file.iloc[:,0].unique())

    def __getitem__(self, idx):
        image = io.imread(self.tif_file)[idx]
        positions = self.csv_file[self.csv_file.iloc[:,0]==(idx+1)].iloc[:,1:3].as_matrix()
        positions = positions.reshape(-1, 2)
        sample = {'image': image, 'positions': positions}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image. """

    def __init__(self, scale_factor, norm=True):
        """
        Args:
            scale_factor (int): scale fact in folds.
            norm (bool): normalize the input image.
        """
        self.scale_factor = scale_factor
        self.norm = norm

    def __call__(self, sample):

        image, positions = sample['image'], sample['positions']
        img = transform.resize(image, (sample['image'].shape[0]*self.scale_factor, sample['image'].shape[1]*self.scale_factor))
        if self.norm:
            img = (img - img.mean())/img.std() # Normalization
        pos = sample['positions']*self.scale_factor

        return {'image': img, 'positions': pos}

class PlotLabels(object):
    """Plot ground truth positions in a image. """

    def __init__(self, pixel_size):
        """
        Args:
            pixel_size (float): pixel size in nanometers.
        """
        self.pixel_size = pixel_size

    def __call__(self, sample):
        image, positions = sample['image'], sample['positions']
        pos = sample['positions']/self.pixel_size
        img_label = np.zeros((sample['image'].shape[0], sample['image'].shape[1]))
        for x, y in pos:
            img_label[int(x), int(y)] = 1.

        return {'image': image, 'positions': img_label.T}

class ToTensor(object):
    """Convert ndarrays in sample to PyTorch Tensors."""

    def __call__(self, sample):
        image, positions = sample['image'], sample['positions']
        image = image.reshape(-1, image.shape[0], image.shape[1])
        positions = positions.reshape(-1, positions.shape[0], positions.shape[1])
        return {'image': torch.FloatTensor(image.astype('float')),
                'positions': torch.FloatTensor(positions.astype('float')).contiguous()}
