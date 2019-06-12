#!/usr/bin/env python3

import logging

import torch
from torch import nn

from . import simple as dynmod_simple
from . import stabledynamics as dynmod_stable
from . import vqvae as vaemod

logger = logging.getLogger(__name__)

global model, WEIGHT_NEXT, LATENT_SPACE_DIM
model = None
WEIGHT_NEXT = None
LATENT_SPACE_DIM = None

class TrajectoryVQVAE(nn.Module):
    def __init__(self, vae, dyn):
        super().__init__()
        self.vae = vae
        self.dyn = dyn

        self.to_latent = nn.Linear(9*14*128, LATENT_SPACE_DIM)
        self.from_latent = nn.Linear(LATENT_SPACE_DIM, 9*14*128)

    def forward(self, X):
        X_a, X_b = X
        Code_a = self.vae.encode(X_a)
        Quantized_a, Perplexity_a = self.vae.codebook(Code_a)

        inp_shape = Quantized_a.size()
        Latent_a = self.to_latent(Quantized_a.view([inp_shape[0], -1]))
        Latent_b = Latent_a + self.dyn(Latent_a)
        Quantized_b = self.from_latent(Latent_b).view(inp_shape)

        Y_a = self.vae.decode(self.from_latent(Latent_a).view(inp_shape))
        Y_b = self.vae.decode(Quantized_b)

        return (Y_a, Code_a, Quantized_a, Perplexity_a, Latent_a, Y_b, Quantized_b, Latent_b)


def loss(Ypred, Yactual, X):
    (Y_a, Code_a, Quantized_a, Perplexity_a, Latent_a, Y_b, Quantized_b, Latent_b) = Ypred
    X_a, X_b = X
    YA_a, YA_b = Yactual

    ls_a, ls_a_recon, ls_a_elatent, ls_a_qlatent, ls_a_perplexity = vaemod.loss((Y_a, Code_a, Quantized_a, Perplexity_a), Y_a, X_a)
    ls_b_recon = torch.mean((Y_b - YA_b)**2)/vaemod.DATA_VARIANCE

    return (ls_a + WEIGHT_NEXT * ls_b_recon, ls_a, ls_a_recon, ls_a_elatent, ls_a_qlatent, ls_a_perplexity, ls_b_recon, torch.mean((Quantized_a - Quantized_b)**2))

def summary(epoch, summarywriter, Ypred, X):
    (Y_a, Code_a, Quantized_a, Perplexity_a, Latent_a, Y_b, Quantized_b, Latent_b) = Ypred
    X_now, X_next = X
    summarywriter.add_embedding(Latent_a.data, label_img=X_now.data, global_step=epoch, tag="learned_embedding")
    summarywriter.add_images("current_reconstructed", Y_a.clamp(max=1.0), global_step=epoch)
    summarywriter.add_images("next_reconstructed", Y_b.clamp(max=1.0), global_step=epoch)

def loss_flatten(l):
    return l

def loss_labels():
    return ["loss", "vqvae_a", "vqvae_a_recon", "vqvae_a_e", "vqvae_a_q", "vqvae_a_perplexity", "vqvae_b_recon", "z_dist"]

def configure(props):
    dynmod = dynmod_stable if "stable" in props else dynmod_simple
    logger.info(f"Latent space dynamics {dynmod.__file__}")
    logger.info(f"Latent space dynamics {vaemod.__file__}")

    global model, LATENT_SPACE_DIM, WEIGHT_NEXT

    LATENT_SPACE_DIM = int(props["latent_space_dim"]) if "latent_space_dim" in props else 320
    logger.info(f"Set latent space dim to {LATENT_SPACE_DIM}")
    vaemod.configure({ **props, "latent_space_dim": LATENT_SPACE_DIM })
    dynmod.configure({ **props, "latent_space_dim": LATENT_SPACE_DIM })

    if "w" in props:
        WEIGHT_NEXT = 10**float(props["w"])
        logger.info(f"Set weight of next step reconstruction to {WEIGHT_NEXT}")
    else:
        logger.warn("No weight of next step reconstruction set; loss function will not run")

    model = TrajectoryVQVAE(vaemod.model, dynmod.model)
