#!/usr/bin/env python3

import glob
import logging
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def make_flat_dataset(dir, extensions):
    images = []

    for root, _, fnames in sorted(os.walk(os.path.expanduser(dir))):
        for fname in sorted(fnames):
            _, ext = os.path.splitext(fname)
            if ext in extensions:
                path = os.path.join(root, fname)
                images.append((path, 0))

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Images(torch.utils.data.Dataset):
    """
    Args:
        glob_files (string): glob pointing to the files.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, glob_files, transform=transforms.Compose([transforms.ToTensor()]), cache=True):
        super().__init__()

        samples = glob.glob(glob_files)
        assert len(samples) > 0, "Found 0 files matching {}".format(glob_files)

        self.loader = pil_loader
        self.samples = sorted(samples)
        self.transform = transform
        # In-memory cache of transformed data:
        # Set to None to prevent caching. 
        self.cache = [None for _ in samples] if cache else None

    def _get_from_cache(self, index):
        if self.cache is None or self.cache[index] is None:
            path_a = self.samples[index]
            sample_a = self.loader(path_a)
            if self.transform is not None:
                sample_a = self.transform(sample_a)
            if self.cache is not None:
                self.cache[index] = sample_a
        else:
            sample_a = self.cache[index]

        return sample_a

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        rv = self._get_from_cache(index)
        return (rv, rv)

    def __len__(self):
        return len(self.samples)

def build(props):
    return Images(props["files"], cache="nocache" not in props)
