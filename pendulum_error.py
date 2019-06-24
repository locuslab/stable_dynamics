#!/usr/bin/env python3

import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from plot_data import plot_data, plot_data_args
from torchvision.utils import save_image
from util import (DynamicLoad, latest_file, loadDataFile, setup_logging,
                  to_variable)

from pathlib import Path

logger = setup_logging(os.path.basename(__file__))


def main(args):
    model = args.model.model
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    physics = args.data._pendulum_gen
    n = args.data._n
    redim = args.data._redim

    logger.info(f"Loaded physics simulator for {n}-link pendulum")

    cache_path = Path("pendulum-cache") / f"p-physics-{n}.npy"

    if not cache_path.exists():
        logger.info(f"Generating trajectories for {cache_path}")
        # Initialize args.number initial positions:
        X_gen = np.zeros((args.number, 2 * n)).astype(np.float32)
        X_gen[:,:] = (np.random.rand(X_gen.shape[0], 2*n).astype(np.float32) - 0.5) * np.pi # Pick values in range [-pi/2, pi/2] radians, radians/sec

        X_phy = [X_gen]

        for i in range(max(args.steps)):
            logger.info(f"Iteration {i}")
            X_gen = redim(X_gen + args.timestep * physics(X_gen))
            assert not np.any(np.isnan(X_gen))
            X_phy.append(X_gen)

        X_phy = np.array(X_phy)
        np.save(cache_path, X_phy)
        logger.info(f"Done generating trajectories for {cache_path}")

    else:
        X_phy = np.load(cache_path).astype(np.float32)
        logger.info(f"Loaded trajectories from {cache_path}")

    # Copy the initial states for NN:
    X_nn = X_phy[0,...]

    errors = []
    for i in range(max(args.steps)):
        if i % 100 == 0:
            logger.info(f"Iteration {i}")

        # Gradient predicted by NN:
        grad_nn = model(to_variable(torch.tensor(X_nn), cuda=torch.cuda.is_available())).detach().cpu().numpy()
        # Take the step:
        X_nn = redim(X_nn + args.timestep * grad_nn)

        # TODO: Update error calculation
        vel_error = np.sum((X_phy[i,:,n:] - X_nn[:,n:])**2)
        ang_error = (X_phy[i,:,:n] - X_nn[:,:n])

        while np.any(ang_error >= np.pi):
            ang_error[ang_error >= np.pi] -= 2*np.pi
        while np.any(ang_error < -np.pi):
            ang_error[ang_error < -np.pi] += 2*np.pi

        ang_error = np.sum(ang_error**2)

        errors.append((vel_error + ang_error)**0.5)

        i += 1
        if i in args.steps:
            print(f"{i}\t{np.sum(errors)}\t{errors[-1]}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Error of .')
    parser.add_argument('--number', type=int, default=2000, help="number of starting positions to evaluate from")
    parser.add_argument('--timestep', type=float, default=0.01, help="duration of each timestep")

    parser.add_argument('data', type=DynamicLoad("datasets"), help='the pendulum dataset to load the simulator from')
    parser.add_argument('model', type=DynamicLoad("models"), help='model to load')
    parser.add_argument('weight', type=latest_file, help='model weight to load')
    parser.add_argument('steps', type=int, nargs="+", help="number of steps to evaluate over")

    main(parser.parse_args())
