#!/usr/bin/env python3

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

global SIZE_A, SIZE_B, model
SIZE_A = 320
SIZE_B = 320

model = None

loss_ = nn.MSELoss()
loss = lambda Ypred, Yactual, X, **kw: loss_(Ypred, Yactual)

def loss_flatten(l):
    return [l]

def loss_labels():
    return ["loss"]

def summary(*a, **kw):
    pass

def configure(props):
    global SIZE_A, SIZE_B, model
    if "a" in props:
        SIZE_A = int(props["a"])

    if "b" in props:
        SIZE_B = int(props["b"])

    model = nn.Sequential(
        nn.Linear(SIZE_A, SIZE_B), nn.ReLU(),
        nn.Linear(SIZE_B, SIZE_B), nn.ReLU(),
        nn.Linear(SIZE_B, SIZE_A))

    logger.info(f"Set layer sizes to {SIZE_A} -> {SIZE_B} -> {SIZE_B} -> {SIZE_A}")

