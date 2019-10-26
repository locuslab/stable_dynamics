#!/usr/bin/env python3

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from .aaronvanderood_vqvae.vqvaeimpl import VectorQuantizer, VectorQuantizerEMA

# Implementation from https://github.com/ritheshkumar95/pytorch-vqvae

logger = logging.getLogger(__name__)

class VQVAE(nn.Module):
    def __init__(self, LATENT_SPACE_DIM=320, K=512, decay=0.0):
        super().__init__()
        if decay:
            self.codebook = VectorQuantizerEMA(K, 128, decay)
        else:
            self.codebook = VectorQuantizer(K, 128)

        self.fc_e1 = nn.Conv2d( 3,   8, 9, stride=2)
        self.fc_e2 = nn.Conv2d( 8,  16, 9, stride=2)
        self.fc_e3 = nn.Conv2d(16,  32, 5, stride=2)
        self.fc_e4 = nn.Conv2d(32,  64, 5, stride=2)
        self.fc_e5 = nn.Conv2d(64, 128, 3, stride=1)
#       self.fc_e6 = nn.Linear(9*14*128, LATENT_SPACE_DIM)

#       self.fc_d1 = nn.Linear(LATENT_SPACE_DIM, 9*14*128)
        self.fc_d2 = nn.ConvTranspose2d(128, 64, 3, stride=1)
        self.fc_d3 = nn.ConvTranspose2d(64, 32, 5, stride=2)
        self.fc_d4 = nn.ConvTranspose2d(32, 16, 5, stride=2)
        self.fc_d5 = nn.ConvTranspose2d(16,  8, 9, stride=2)
        self.fc_d6 = nn.ConvTranspose2d( 8,  3, 9, stride=2)

    def encode(self, x):
        assert list(x.size())[1:] == [3, 240, 320]
        x = F.relu(self.fc_e1(x))
        x = F.relu(self.fc_e2(x))
        x = F.relu(self.fc_e3(x))
        x = F.relu(self.fc_e4(x))
        x = F.relu(self.fc_e5(x))
        # x = x.view([x.size()[0], -1])
        # x = self.fc_e6(x)
        return x

    def decode(self, z):
        nb = z.size()[0]
        # z = self.fc_d1(z)
        # z = z.view([nb, 128, 9, 14])
        z = F.relu(self.fc_d2(z, output_size=[nb, 32, 11, 16]))
        z = F.relu(self.fc_d3(z, output_size=[nb, 32, 25, 35]))
        z = F.relu(self.fc_d4(z, output_size=[nb, 16, 54, 74]))
        z = F.relu(self.fc_d5(z, output_size=[nb,  8, 116, 156]))
        z = torch.sigmoid(self.fc_d6(z, output_size=[nb,  3, 240, 320]))
        return z

    def forward(self, x):
        z = self.encode(x)
        quantized, perplexity = self.codebook(z)
        x_recon = self.decode(quantized)
        return x_recon, z, quantized, perplexity

# model is a torch.nn.Module that contains the model definition.
global model, BETA, COMMITMENT_COST
model = None
BETA = 1.0
COMMITMENT_COST = 1.0
DATA_VARIANCE = 0.00025 # This is the general range for our datasets

# Use MSE loss as distance from input to output:
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

    x_recon, z, quantized, perplexity = Ypred

    #data_variance = (torch.var(X) / X.size(0)).detach()
    #logger.info(f"Batch variance {data_variance}")

    # Loss
    e_latent_loss = torch.mean((quantized.detach() - z)**2)
    q_latent_loss = torch.mean((quantized - z.detach())**2)
    recon_loss = torch.mean((x_recon - Yactual)**2)/DATA_VARIANCE

    loss = recon_loss + q_latent_loss + COMMITMENT_COST * e_latent_loss

    return (loss, recon_loss, e_latent_loss, q_latent_loss, perplexity)

def loss_flatten(x):
    return x

def loss_labels():
    return ("loss", "reconstruction", "eloss", "qloss", "perplexity")

def summary(epoch, summarywriter, Ypred, X):
    x_recon, z, quantized, perplexity = Ypred
    # summarywriter.add_embedding(z.data, label_img=X.data, global_step=epoch, tag="learned_embedding")
    summarywriter.add_images("reconstructed", x_recon, global_step=epoch)

    Xrandstd = X.std().detach().item()
    Xrand = torch.normal(mean=torch.zeros_like(X), std=Xrandstd)
    Ypredrand = model(Xrand.to(Ypred[0]))
    summarywriter.add_images("random", Ypredrand[0], global_step=epoch)

def configure(props):
    global model, COMMITMENT_COST

    # K must be the same as the number of channels in the image.
    K = props["codebook_size"] if "codebook_size" in props else 512
    lsd = props["latent_space_dim"] if "latent_space_dim" in props else 128
    decay = props["ema_decay"] if "ema_decay" in props else 0
    COMMITMENT_COST = props["commitment_cost"] if "commitment_cost" in props else 1.0
    model = VQVAE(lsd, K, decay=decay)

    logger.info(f"Latent space is 1x{lsd}, codebook has {K} entries.")

    try:
        global BETA
        BETA = float(props["beta"])
    except KeyError:
        pass
    logger.info(f"BETA (commiment loss multiplier) {BETA}.")
