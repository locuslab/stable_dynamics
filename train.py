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
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from util import DynamicLoad, setup_logging, to_variable

logger = setup_logging(os.path.basename(__file__))

def runbatch(args, model, loss, batch):
    X, Yactual = batch
    X = to_variable(X, cuda=torch.cuda.is_available())
    Yactual = to_variable(Yactual, cuda=torch.cuda.is_available())

    Ypred = model(X)
    return loss(Ypred, Yactual, X), Ypred

def test_model(args, model, test_dataloader, epoch=None, summarywriter=None):
    loss_parts = []
    model.eval()
    for batch_idx, data in enumerate(test_dataloader):
        loss, Ypred = runbatch(args, model, args.model.loss, data)
        loss_parts.append(np.array([l.cpu().item() for l in args.model.loss_flatten(loss)]))

        # Add parts to the summary if needed.
        args.model.summary(epoch, summarywriter, Ypred, data[0])

    return sum(loss_parts) / len(test_dataloader.dataset)

global _first_printed
_first_printed = False
def print_update(args, train_test, epoch, loss_elements):
    logger_progress = logger.getChild("progress")
    global _first_printed
    if not _first_printed:
        _first_printed = True
        loss_parts = "\t".join(args.model.loss_labels())
        s = f"TIMESTAMP\tTRAIN/TEST\tepoch\t{loss_parts}"
        logger_progress.info(s)
        print(s)

    now = datetime.datetime.now()
    loss_parts = "\t".join(map(str, loss_elements))
    s = f"{now}\t{train_test}\t{epoch}\t{loss_parts}"
    print(s)
    logger_progress.info(s)

def main(args):
    writer = SummaryWriter(logdir=args.log_to)
    model = args.model.model
    dataset = args.dataset

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    if args.test_with:
        test_dataloader = DataLoader(args.test_with, batch_size=args.batch_size, shuffle=False)
    else:
        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # TODO: Flag to disable CUDA
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # TODO: Resume training support
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_parts = []

        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss, _ = runbatch(args, model, args.model.loss, data)
            loss_parts.append(np.array([l.cpu().item() for l in args.model.loss_flatten(loss)]))

            optim_loss = loss[0] if isinstance(loss, (tuple, list)) else loss
            optim_loss.backward()
            optimizer.step()

        epoch_loss = sum(loss_parts) / len(dataset)
        print_update(args, "TRAIN", epoch, epoch_loss)
        for lbl, val in zip(args.model.loss_labels(), epoch_loss):
            writer.add_scalar(f'train_loss/{lbl}', val, epoch)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), args.weights.format(epoch=epoch))
            test_loss = test_model(args, model, test_dataloader, epoch=epoch, summarywriter=writer)
            print_update(args, "TEST", epoch, test_loss)
            for lbl, val in zip(args.model.loss_labels(), test_loss):
                writer.add_scalar(f'test_loss/{lbl}', val, epoch)

    # Ensure the writer is completed.
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE on a set of videos, fine-tune it on a single video, and generate the decoder.')
    parser.set_defaults(func=lambda *a: parser.print_help())

    parser.add_argument('dataset', type=DynamicLoad("datasets"), help='dataset to train on')
    parser.add_argument('model', type=DynamicLoad("models"), help='model to train with')
    parser.add_argument('weights', type=str, help='save model weights')
    parser.add_argument('--log-to', type=str, help='log destination within runs/')
    parser.add_argument('--test-with', type=DynamicLoad("datasets"), default=None, help='dataset to test with instead of the training data')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs to run')
    parser.add_argument('--save-every', type=int, default=100, help='save after this many epochs')
    parser.set_defaults(func=main)

    try:
        args = parser.parse_args()
        main(args)
    except:
        raise
