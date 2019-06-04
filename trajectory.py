#!/usr/bin/env python3

import argparse
import datetime
import glob
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from util import DynamicLoad, latest_file, setup_logging, to_variable

logger = setup_logging(os.path.basename(__file__))

def main(args):
    model = args.model.model
    model.load_state_dict(torch.load(args.weight))

    dataset = args.dataset
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # TODO: Flag to disable CUDA
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    trajectory = []
    for batch_idx, data in enumerate(test_dataloader):
        img, _ = data
        img = to_variable(img, cuda=torch.cuda.torch.cuda.is_available())

        # z = model(img)
        # Break the abstraction here:
        Y_a, mu_a, logvar_a, z_a, Y_b, z_b = model(img)
        if args.mu:
            z = mu_a
        else:
            z = z_a

        trajectory.append(z.cpu().data.numpy())

    trajectory = np.concatenate(trajectory)
    h5f = h5py.File(args.output, 'w')
    h5f.create_dataset('seq', data=[trajectory])
    h5f.create_dataset('param', data=[])
    h5f.close()

    mn = np.mean(trajectory, axis=0)
    linf = np.max(np.abs(trajectory - mn))

    print(f"Data: {trajectory.shape}; Mean: {mn.shape}; Linf: {linf}")

    np.savetxt(Path(args.output).with_suffix(".mean"), mn)
    Path(args.output).with_suffix(".linf").write_text(str(linf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE on a set of videos, fine-tune it on a single video, and generate the decoder.')
    parser.set_defaults(func=lambda *a: parser.print_help())

    parser.add_argument('dataset', type=DynamicLoad("datasets"), help='dataset to train on')
    parser.add_argument('model', type=DynamicLoad("models"), help='model to train with')
    parser.add_argument('weight', type=latest_file, help='save model weights')
    parser.add_argument('output', type=str, help='trajectory file')
    parser.add_argument('--test-with', type=DynamicLoad("datasets"), help='dataset to test with instead of the training data')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--mu', action="store_true", help='sample with zero variance')
    parser.set_defaults(func=main)

    try:
        args = parser.parse_args()
        main(args)
    except:
        raise
