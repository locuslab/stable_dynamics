#!/usr/bin/env python3

from . import images
import torch

class SeqPairs(torch.utils.data.Dataset):
    def __init__(self, image_dataset):
        super().__init__()
        self.image_dataset = image_dataset

    def __getitem__(self, index):
        q1, _ = self.image_dataset[index]
        q2, _ = self.image_dataset[index + 1]
        return ((q1, q2), (q1, q2))

    def __len__(self):
        return len(self.image_dataset) - 1

def build(props):
    return SeqPairs(images.build(props))
