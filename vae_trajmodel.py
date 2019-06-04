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

logger = setup_logging(os.path.basename(__file__))

def main(args):
    model = args.model.model
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    dyn_model = model.dyn
    vae_model = model.vae

    # Load sequence:
    seq, param = args.data
    original_seq = seq[args.select,:,:]
    trajectory = [original_seq[args.start_step:(args.start_step+1),:]]
    renders = []

    get_image = lambda r: torch.squeeze(vae_model.decode(r).detach().cpu(), dim=0)

    # Start with the first index:
    X = to_variable(torch.tensor(trajectory[0]), cuda=torch.cuda.is_available())
    X.requires_grad_()
    if args.act == "render":
        renders.append(get_image(X))

    for i in range(args.steps):
        X = to_variable((X + dyn_model(X)).data, cuda=torch.cuda.is_available())
        X.requires_grad_()

        trajectory.append(X.detach().cpu().numpy())
        if args.act == "render":
            renders.append(get_image(X))

    trajectory = np.squeeze(np.stack(trajectory), axis=1)

    if args.act == "plot":
        plot_data(args, trajectory, original_seq=original_seq)
    elif args.act == "render":
        if args.save:
            save_image(renders, args.save)
        if args.save_frames:
            for i, im in enumerate(renders):
                save_image(im, args.save_frames.format(i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data files.')
    parser.add_argument('--select', type=int, default=0, help="select the starting position from this index")
    parser.add_argument('--start-step', type=int, default=0, help="number of steps to render")
    parser.add_argument('--steps', type=int, default=240, help="number of steps to render")

    parser.add_argument('data', type=loadDataFile, help='the data file to load the start position from')
    parser.add_argument('model', type=DynamicLoad("models"), help='model to load')
    parser.add_argument('weight', type=latest_file, help='model weight to load')

    subparsers = parser.add_subparsers(help='output to produce')

    parser_plot = plot_data_args(subparsers.add_parser('plot', help='plot trajectories in latent space'))
    parser_plot.set_defaults(act="plot")

    parser_render = subparsers.add_parser('render', help='create images from the latent space')
    parser_render.set_defaults(act="render")
    parser_render.add_argument('--save', type=str, help="save renders as a single image")
    parser_render.add_argument('--save-frames', type=str, help="save frames, pass a format string")

    parser_field = subparsers.add_parser('field', help='create a field plot of the latent space')
    parser_field.set_defaults(act="field")
    parser_field.add_argument('--save', type=str, help="save field plot as an image")

    main(parser.parse_args())
