#!/usr/bin/env python3

import logging

import torch
from torch import nn

from . import simple as dynmod_simple
from . import stabledynamics as dynmod_stable
from . import vae as vaemod

logger = logging.getLogger(__name__)

class TrajectoryVAE(nn.Module):
    def __init__(self, vae, dyn):
        super().__init__()
        self.vae = vae
        self.dyn = dyn

    def forward(self, X):
        X_a, X_b = X
        Y_a, mu_a, logvar_a, z_a = self.vae(X_a)
        z_b = z_a + self.dyn(z_a)
        Y_b = self.vae.decode(z_b)

        return (Y_a, mu_a, logvar_a, z_a, Y_b, z_b)

global model, WEIGHT_NEXT
model = None
WEIGHT_NEXT = None

def loss(Ypred, Yactual, X):
    Y_a, mu_a, logvar_a, z_a, Y_b, z_b = Ypred
    X_a, X_b = X
    YA_a, YA_b = Yactual

    ls_a, ls_a_bce, ls_a_kld = vaemod.loss((Y_a, mu_a, logvar_a, z_a), X_a, X_a)
    ls_b_bce = vaemod.reconstruction_function(Y_b, YA_b)

    return (ls_a + WEIGHT_NEXT * ls_b_bce, ls_a, ls_a_bce, ls_a_kld, ls_b_bce, torch.sum((z_a - z_b)**2))

def summary(epoch, summarywriter, Ypred, X):
    Y_a, mu_a, logvar_a, z_a, Y_b, z_b = Ypred
    X_now, X_next = X
    summarywriter.add_embedding(z_a.data, label_img=X_now.data, global_step=epoch, tag="learned_embedding")
    summarywriter.add_images("current_reconstructed", Y_a.clamp(max=1.0), global_step=epoch)
    summarywriter.add_images("next_reconstructed", Y_b.clamp(max=1.0), global_step=epoch)

def loss_flatten(l):
    return l

def loss_labels():
    return ["traj", "vae_a", "vae_a_bce", "vae_a_kld", "vae_b_bce", "z_dist"]

def configure(props):
    dynmod = dynmod_stable if "stable" in props else dynmod_simple
    logger.info(f"Latent space dynamics {dynmod.__file__}")
    logger.info(f"Latent space dynamics {vaemod.__file__}")

    lsd = int(props["latent_space_dim"]) if "latent_space_dim" in props else 320
    logger.info(f"Set latent space dim to {lsd}")
    vaemod.configure({ **props, "latent_space_dim": lsd })
    dynmod.configure({ **props, "latent_space_dim": lsd })

    if "w" in props:
        global model, WEIGHT_NEXT
        WEIGHT_NEXT = 10**float(props["w"])
        logger.info(f"Set weight of next step reconstruction to {WEIGHT_NEXT}")
    else:
        logger.warn("No weight of next step reconstruction set; loss function will not run")

    model = TrajectoryVAE(vaemod.model, dynmod.model)
