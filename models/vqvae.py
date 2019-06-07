#!/usr/bin/env python3

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from .ritheshkumar95_vqvae.modules import VQEmbedding, weights_init

# Implementation from https://github.com/ritheshkumar95/pytorch-vqvae

logger = logging.getLogger(__name__)

class VQVAE(nn.Module):
    def __init__(self, LATENT_SPACE_DIM=320, K=512):
        super().__init__()
        self.codebook = VQEmbedding(K, LATENT_SPACE_DIM)

        self.fc_e1 = nn.Conv2d( 3,   8, 9, stride=2)
        self.fc_e2 = nn.Conv2d( 8,  16, 9, stride=2)
        self.fc_e3 = nn.Conv2d(16,  32, 5, stride=2)
        self.fc_e4 = nn.Conv2d(32,  64, 5, stride=2)
        self.fc_e5 = nn.Conv2d(64, 128, 3, stride=1)
        self.fc_e6 = nn.Linear(9*14*128, LATENT_SPACE_DIM)

        self.fc_d1 = nn.Linear(LATENT_SPACE_DIM, 9*14*128)
        self.fc_d2 = nn.ConvTranspose2d(128, 64, 3, stride=1)
        self.fc_d3 = nn.ConvTranspose2d(64, 32, 5, stride=2)
        self.fc_d4 = nn.ConvTranspose2d(32, 16, 5, stride=2)
        self.fc_d5 = nn.ConvTranspose2d(16,  8, 9, stride=2)
        self.fc_d6 = nn.ConvTranspose2d( 8,  3, 9, stride=2)

        self.apply(weights_init)

    def encode(self, x):
        assert list(x.size())[1:] == [3, 240, 320]
        x = F.relu(self.fc_e1(x))
        x = F.relu(self.fc_e2(x))
        x = F.relu(self.fc_e3(x))
        x = F.relu(self.fc_e4(x))
        x = F.relu(self.fc_e5(x))
        # x = x.view([x.size()[0], -1])
        # x = self.fc_e6(x)

        logger.info(f"Shape {x.shape}")
        latents = self.codebook(x)
        return latents

    def decode(self, latents):
        z = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)

        nb = z.size()[0]
        # z = self.fc_d1(z)
        z = z.view([nb, 128, 9, 14])
        z = F.relu(self.fc_d2(z, output_size=[nb, 32, 11, 16]))
        z = F.relu(self.fc_d3(z, output_size=[nb, 32, 25, 35]))
        z = F.relu(self.fc_d4(z, output_size=[nb, 16, 54, 74]))
        z = F.relu(self.fc_d5(z, output_size=[nb,  8, 116, 156]))
        z = torch.sigmoid(F.relu(self.fc_d6(z, output_size=[nb,  3, 240, 320])))
        return z

    def forward(self, x):
        z_e_x = self.encode(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decode(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

# model is a torch.nn.Module that contains the model definition.
global model, BETA
model = None
BETA = 1.0

# Use MSE loss as distance from input to output:
reconstruction_function = lambda recon_x, x: torch.sum((recon_x - x) ** 2)
def loss(Ypred, Yactual, X):
    """loss function for learning problem

    Arguments:
        Ypred {Model output type} -- predicted output
        Yactual {Data output type} -- output from data
        X {torch.Variable[input]} -- input

    Returns:
        Tuple[nn.Variable] -- Parts of the loss function; the first element is passed to the optimizer
        nn.Variable -- the loss to optimize
    """
    reconstructed, z_e_x, z_q_x = Ypred
    # Reconstruction loss
    loss_recons = F.mse_loss(x_tilde, images)
    # Vector quantization objective
    loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
    # Commitment objective
    loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

    return (loss_recons + loss_vq + BETA * loss_commit, loss_recons, loss_vq, loss_commit)

def loss_labels():
    return ("loss", "reconstruction", "vector_quantization", "commitment")

def configure(props):
    global model

    # K must be the same as the number of channels in the image.
    K = props["codebook_size"] if "codebook_size" in props else 128
    lsd = props["latent_space_dim"] if "latent_space_dim" in props else 128
    model = VQVAE(lsd, K)

    logger.info(f"Latent space is 1x{lsd}, codebook has {K} entries.")

    try:
        global BETA
        BETA = float(props["beta"])
    except KeyError:
        pass
    logger.info(f"BETA (commiment loss multiplier) {BETA}.")
