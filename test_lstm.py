#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import pendulum_energy
from plot_data import plot_data, plot_data_args
from scipy.integrate import odeint
from torchvision.utils import save_image
from train_lstm import STEPS, TRAJECTORIES, LSTMModel, N
from util import (DynamicLoad, latest_file, loadDataFile, setup_logging,
                  to_variable)

logger = setup_logging(os.path.basename(__file__))


def main(args):
    hidden_dim = 116
    model = LSTMModel(2*N, hidden_dim)

    model.load_state_dict(torch.load(args.weight))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    physics = args.data._pendulum_gen
    n = args.data._n
    redim = args.data._redim
    h = args.timestep

    logger.info(f"Loaded physics simulator for {n}-link pendulum")

    cache_path = Path("pendulum-cache") / f"p-physics-{n}.npy"

    # Energy functions
    energy = pendulum_energy.pendulum_energy(n)

    if cache_path.exists():
        X_phy = np.load(cache_path).astype(np.float32)
        logger.info(f"Loaded trajectories from {cache_path}")
    else:
        raise Exception(f"No trajectories for {cache_path}")

    X_nn = to_variable(torch.tensor(X_phy[0,:,:]), cuda=torch.cuda.is_available())
    errors = np.zeros((args.steps,))
    X_nn.requires_grad = True
    X_nn = X_nn.unsqueeze(0)

    hiddens = None
    for i in range(1, args.steps):
        k1, new_hiddens = model(X_nn, hiddens)
        k1 = h * k1.detach()
        k2, _ = model(X_nn + k1/2, hiddens)
        k2 = h * k2.detach()
        k3, _ = model(X_nn + k2/2, hiddens)
        k3 = h * k3.detach()
        k4, _ = model(X_nn + k3, hiddens)
        k4 = h * k4.detach()
        X_nn = X_nn + 1/6*(k1 + 2*k2 + 2*k3 + k4)

        # Detach
        X_nn = X_nn.detach()
        hiddens = tuple(d.detach() for d in new_hiddens)

        logger.info(f"Timestep {i}")

        y = X_nn.cpu().numpy()
        vel_error = np.sum((X_phy[i,:,n:] - y[0,:,n:])**2)
        ang_error = (X_phy[i,:,:n] - y[0,:,:n])
        while np.any(ang_error >= np.pi):
            ang_error[ang_error >= np.pi] -= 2*np.pi
        while np.any(ang_error < -np.pi):
            ang_error[ang_error < -np.pi] += 2*np.pi

        ang_error = np.sum(ang_error**2)
        errors[i] = (vel_error + ang_error)

    for i in range(args.steps):
        print(f"{i}\t{np.sum(errors[0:i])}\t{errors[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Error of .')
    parser.add_argument('--links', type=int, default=8, help="number of links")
    parser.add_argument('--number', type=int, default=1000, help="number of starting positions to evaluate from")
    parser.add_argument('--timestep', type=float, default=0.01, help="duration of each timestep")

    parser.add_argument('data', type=DynamicLoad("datasets"), help='the pendulum dataset to load the simulator from')
    parser.add_argument('weight', type=latest_file, help='model weight to load')
    parser.add_argument('steps', type=int, help="number of steps to evaluate over")

    main(parser.parse_args())
