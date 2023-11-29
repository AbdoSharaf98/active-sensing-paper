import distributions
import torch
from torch import nn
from distributions import Gaussian, GaussianMixture, Concrete
from torch.distributions import Categorical
from torch.nn import functional as F
from nets import create_ff_network


class LowerVAE(nn.Module):

    def __init__(self, obs_dim: int,
                 loc_dim: int,
                 latent_dim: int,
                 summarizer: nn.Module,
                 layers=None):
        super(LowerVAE, self).__init__()

        if layers is None:
            layers = [32, 64, 128]

        # summarizer
        self.summarizer = summarizer

        # build the encoder network
        enc_layers = [obs_dim + loc_dim] + layers + [latent_dim]
        self.mu_encoder = create_ff_network(enc_layers, h_activation='relu')
        self.log_std_encoder = create_ff_network(enc_layers, h_activation='relu')

        # build the decoder network
        dec_layers = [latent_dim] + list(reversed(layers)) + [obs_dim]
        self.decoder = create_ff_network(dec_layers, h_activation='relu')

    def encode(self, obs: torch.Tensor, loc: torch.Tensor):
        # summarize observations and locations
        vae_input = self.summarizer(obs, loc)

        return Gaussian(self.mu_encoder(vae_input), torch.exp(self.log_std_encoder(vae_input)))

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, obs: torch.Tensor, loc: torch.Tensor):
        q_z = self.encode(obs, loc)
        z = q_z.sample()
        x_hat = self.decode(z)

        return x_hat, z, q_z
