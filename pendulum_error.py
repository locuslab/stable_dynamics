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
from scipy.integrate import odeint

from models import pendulum_energy

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
    times = args.timestep * np.arange(0, max(args.steps)).astype(np.float32)

    logger.info(f"Loaded physics simulator for {n}-link pendulum")

    cache_path = Path("pendulum-cache") / f"p-physics-{n}.npy"

    # Energy functions
    energy = pendulum_energy.pendulum_energy(n)

    if not cache_path.exists():
        logger.info(f"Generating trajectories for {cache_path}")
        # Initialize args.number initial positions:
        X_init = np.zeros((args.number, 2 * n)).astype(np.float32)
        X_init[:,:] = (np.random.rand(args.number, 2*n).astype(np.float32) - 0.5) * np.pi/2 # Pick values in range [-pi/4, pi/4] radians, radians/sec

        X_phy = np.zeros((len(times), *X_init.shape), dtype=np.float32)
        for i in range(args.number):
            logger.info(f"Trajectory {i}")
            traj = odeint(physics, X_init[i,...], times)
            X_phy[:,i,:] = traj
            assert not np.any(np.isnan(traj))

        np.save(cache_path, X_phy)
        logger.info(f"Done generating trajectories for {cache_path}")

    else:
        X_phy = np.load(cache_path).astype(np.float32)
        logger.info(f"Loaded trajectories from {cache_path}")

    # Copy the initial states for NN:
    X_nn = np.zeros((len(times), args.number, 2 * n), np.float32)
    def model_wrapped(inp, *a, **kw):
        inp = np.reshape(inp, (1, -1))
        rv = model(to_variable(torch.tensor(inp, dtype=torch.float32), cuda=torch.cuda.is_available())).detach().cpu().numpy()
        return rv.flatten()

    errors = np.zeros((len(times),))
    for i in range(args.number):
        logger.info(f"Trajectory {i}")

        traj = odeint(model_wrapped, X_phy[0,i,:].astype(np.float32), times)
        X_nn[:,i,:] = traj

        # TODO: Update error calculation
        vel_error = np.sum((X_phy[:,i,n:] - X_nn[:,i,n:])**2, axis=1)
        ang_error = (X_phy[:,i,:n] - X_nn[:,i,:n])
        while np.any(ang_error >= np.pi):
            ang_error[ang_error >= np.pi] -= 2*np.pi
        while np.any(ang_error < -np.pi):
            ang_error[ang_error < -np.pi] += 2*np.pi

        ang_error = np.sum(ang_error**2, axis=1)
        errors[:] += (vel_error + ang_error)**0.5

    for i in args.steps:
        print(f"{i}\t{np.sum(errors[0:i-1])}\t{errors[i-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Error of .')
    parser.add_argument('--number', type=int, default=1000, help="number of starting positions to evaluate from")
    parser.add_argument('--timestep', type=float, default=0.01, help="duration of each timestep")

    parser.add_argument('data', type=DynamicLoad("datasets"), help='the pendulum dataset to load the simulator from')
    parser.add_argument('model', type=DynamicLoad("models"), help='model to load')
    parser.add_argument('weight', type=latest_file, help='model weight to load')
    parser.add_argument('steps', type=int, nargs="+", help="number of steps to evaluate over")

    main(parser.parse_args())
