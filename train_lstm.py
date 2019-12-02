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
from datasets import pendulum
from torch import nn, optim, tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from util import DynamicLoad, setup_logging, to_variable

logger = setup_logging(os.path.basename(__file__))


def trajectory_physics(number, n, steps):
    physics = pendulum.pendulum_gradient(n)

    h = 0.01 # Timestep

    cache_path = Path("pendulum-cache") / f"p-lstmtraj-{n}.npy"
    if not cache_path.exists():
        logger.info(f"Generating trajectories for {cache_path}")
        # Initialize args.number initial positions:
        X_init = np.zeros((number, 2*n)).astype(np.float32)
        X_init[:,:] = (np.random.rand(number, 2*n).astype(np.float32) - 0.5) * np.pi/2 # Pick values in range [-pi/4, pi/4] radians, radians/sec

        X_phy = np.zeros((steps, *X_init.shape), dtype=np.float32)
        X_grad = np.zeros((steps, *X_init.shape), dtype=np.float32)
        X_phy[0,...] = X_init
        X_grad[0,...] = physics(X_init)
        for i in range(1, steps):
            logger.info(f"Timestep {i}")
            k1 = h * physics(X_phy[i-1,...])
            k2 = h * physics(X_phy[i-1,...] + k1/2)
            k3 = h * physics(X_phy[i-1,...] + k2/2)
            k4 = h * physics(X_phy[i-1,...] + k3)
            X_phy[i,...] = X_phy[i-1,...] + 1/6*(k1 + 2*k2 + 2*k3 + k4)
            X_grad[i,...] = physics(X_phy[i,...])
            assert not np.any(np.isnan(X_phy[i,...]))

        np.save(cache_path, (X_phy, X_grad))
        logger.info(f"Done generating trajectories for {cache_path}")

    else:
        X_phy, X_grad = np.load(cache_path).astype(np.float32)
        logger.info(f"Loaded trajectories from {cache_path}. {X_phy.shape}, {X_grad.shape}")

    return X_phy, X_grad

N = 8
STEPS = 50
TRAJECTORIES = 32

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=116):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.enc = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.dec = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))

    def forward(self, x, hidden):
        x = self.enc(x)
        x, hidden = self.lstm(x, hidden)
        return self.dec(x), hidden


def main(args):
    hidden_dim = 116
    N = args.links
    model = LSTMModel(2*N, hidden_dim)

    X_phy, X_grad = trajectory_physics(TRAJECTORIES, N, STEPS)

    X_phy = tensor(X_phy)
    X_grad = tensor(X_grad)

    # TODO: Flag to disable CUDA
    if torch.cuda.is_available():
        model.cuda()
        X_phy = X_phy.cuda()
        X_grad = X_grad.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_parts = []

        hiddens = None
        for timestep in range(X_phy.shape[0]):
            optimizer.zero_grad()

            y, new_hiddens = model(X_phy[timestep,...].unsqueeze(0), hiddens)
            hiddens = tuple(d.detach() for d in new_hiddens)
            loss = torch.sum((y - X_grad[timestep,...])**2)
            loss_parts.append(loss.cpu().item())

            loss.backward()
            optimizer.step()

        epoch_loss = sum(loss_parts) / X_phy.shape[0]
        print(f"TRAIN\t{epoch}\t{epoch_loss}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), args.weights.format(epoch=epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE on a set of videos, fine-tune it on a single video, and generate the decoder.')
    parser.set_defaults(func=lambda *a: parser.print_help())

    parser.add_argument('weights', type=str, help='save model weights')

    parser.add_argument('--links', type=int, default=8, help="number of links")
    parser.add_argument('--log-to', type=str, help='log destination within runs/')
    parser.add_argument('--learning-rate', type=float, default=5 e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to run')
    parser.add_argument('--save-every', type=int, default=100, help='save after this many epochs')
    parser.set_defaults(func=main)

    try:
        args = parser.parse_args()
        main(args)
    except:
        raise
