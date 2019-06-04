#!/usr/bin/env python3

import argparse
import datetime
import glob
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from util import DynamicLoad, setup_logging, to_variable, latest_file

logger = setup_logging(os.path.basename(__file__))

def main(args):
    model = args.model.model
    model.load_state_dict(torch.load(args.weight))
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    vaemod = model.vae

    if args.random is not None:
        X = torch.tensor(np.random.normal(loc=0.0, scale=args.random, size=[args.batch_size, 320]).astype(np.float32))
        X = to_variable(X, torch.cuda.is_available())
        output = vaemod.decode(X)

    else:
        test_dataloader = DataLoader(args.dataset, batch_size=args.batch_size, shuffle=False)

        for batch_idx, data in enumerate(test_dataloader):
            X, Yactual = data
            X = to_variable(X, torch.cuda.is_available())
            output = vaemod(X)
            break

    if isinstance(output, tuple):
        output = output[0]
    save_image(output, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render the images passed through the model')
    parser.add_argument('dataset', type=DynamicLoad("datasets"), help='dataset to train on')
    parser.add_argument('model', type=DynamicLoad("models"), help='model to train with')
    parser.add_argument('weight', type=latest_file, help='save model weight')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--save', type=str, help="save renders as a single image")
    parser.add_argument('--random', type=float, help="sample the space randomly, with zero mean and provided variance")

    try:
        args = parser.parse_args()
        main(args)
    except:
        raise
