#!/usr/bin/env python3

import argparse

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from util import loadDataFile
import h5py

def main(args):
    seq, param = args.data
    plot_data(args, seq[args.select,:,:])

def plot_data(args, seq, original_seq = None):
    if args.delta:
        plt.figure()
        dist = np.sum((seq[1:,...] - seq[:-1,...]) ** 2, axis=1) ** 0.5
        plt.hist(dist)
        if args.save:
            plt.savefig(args.save, dpi=120)
        else:
            plt.show()
        return

    if args.pca:
        pca = PCA(n_components=args.pca)
        if original_seq is None:
            seq = pca.fit_transform(seq)
        else:
            original_seq = pca.fit_transform(original_seq)
            seq = pca.transform(seq)

    if args.tsne:
        tsne = TSNE(n_components=2, perplexity=30.0, n_iter=2000, verbose=2)
        if original_seq is None:
            seq = tsne.fit_transform(seq)
        else:
            tsne.fit(original_seq)
            seq = tsne.transform(seq)

    if seq.shape[1] == 2:
        plt.figure()
        x, y = zip(*seq[:,:])
        color_list = cm.get_cmap(name="viridis")
        if args.strip:
            n, m = tuple(args.strip)
            for i in range(0, seq.shape[0] - 1, m):
                plt.plot(x[i:(i+n)], y[i:(i+n)], '-', color=color_list(i/(seq.shape[0]-1)))
        else:
            for i in range(seq.shape[0] - 1):
                plt.plot(x[i:(i+2)], y[i:(i+2)], '.', color=color_list(i/(seq.shape[0]-1)))
        plt.axis('equal')
        if args.save:
            plt.savefig(args.save, dpi=120)
        else:
            plt.show()
    else:
        print("Cannot plot sequence: data is of size {}".format(seq.shape))

def plot_data_args(parser):
    parser.add_argument('--pca', type=int, help="preprocess trajectories down to these many dimensions using PCA; happens before TSNE")
    parser.add_argument('--tsne', action="store_true", help="use tsne to preprocess trajectories down to two dimensions")
    parser.add_argument('--save', type=str, help="save image")
    parser.add_argument('--strip', type=int, nargs=2, help="draw the first n elements in blocks of size m")
    parser.add_argument('--delta', action="store_true", help="provide a histogram of the distribution of distances between points along the trajectory")
    return parser

if __name__ == "__main__":
    parser = plot_data_args(argparse.ArgumentParser(description='Plot data files.'))
    parser.add_argument('data', type=loadDataFile, help='the data file to load')
    parser.add_argument('--select', type=int, default=0, help="draw the trajectory at this index")
    
    main(parser.parse_args())
