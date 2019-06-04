#!/usr/bin/env python3

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

logger = logging.getLogger(__name__)

class VAE(nn.Module):
    def __init__(self, LATENT_SPACE_DIM=320):
        super().__init__()

        self.fc_e1 = nn.Conv2d(3, 8, 9, stride=2)
        self.fc_e2 = nn.Conv2d(8, 16, 9, stride=2)
        self.fc_e3 = nn.Conv2d(16, 32, 5, stride=2)
        self.fc_e4 = nn.Conv2d(32, 64, 5, stride=2)
        self.fc_e5 = nn.Conv2d(64, 128, 3, stride=1)
        self.fc_e61 = nn.Linear(9*14*128, LATENT_SPACE_DIM)
        self.fc_e62 = nn.Linear(9*14*128, LATENT_SPACE_DIM)

        self.fc_d1 = nn.Linear(LATENT_SPACE_DIM, 9*14*128)
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
        x = x.view([x.size()[0], -1])
        return self.fc_e61(x), self.fc_e62(x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        nb = z.size()[0]
        z = torch.sigmoid(self.fc_d1(z))
        z = z.view([nb, 128, 9, 14])
        z = F.relu(self.fc_d2(z, output_size=[nb, 32, 11, 16]))
        z = F.relu(self.fc_d3(z, output_size=[nb, 32, 25, 35]))
        z = F.relu(self.fc_d4(z, output_size=[nb, 16, 54, 74]))
        z = F.relu(self.fc_d5(z, output_size=[nb,  8, 116, 156]))
        z = F.relu(self.fc_d6(z, output_size=[nb,  3, 240, 320]))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, z

# model is a torch.nn.Module that contains the model definition.
global model, VAE_BCE_LOSS
model = VAE()
VAE_BCE_LOSS = 1.

# Use MSE loss as distance from input to output:
reconstruction_function = lambda recon_x, x: torch.sum((recon_x - x) ** 2)
def vae_loss_fn(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = torch.sum(reconstruction_function(recon_x, x))
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return VAE_BCE_LOSS * BCE + KLD, BCE, KLD

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
    reconstructed, mu, logvar, z = Ypred
    return vae_loss_fn(reconstructed, Yactual, mu, logvar)

def loss_labels():
    """Keys corresponding to the components of the loss
    
    Returns:
        Tuple[str] -- Tuple of loss 
    """
    return ("vae_loss", "reconstruction_loss", "prior_loss")

def configure(props):
    global model

    try:
        lsd = props["latent_space_dim"]
        model = VAE(lsd)
    except KeyError:
        pass
    logger.info(f"Latent space is 1x{lsd}.")

    try:
        global VAE_BCE_LOSS
        VAE_BCE_LOSS = 10**float(props["bce"])
    except KeyError:
        pass
    logger.info(f"BCE to KLD ratio {VAE_BCE_LOSS}.")
