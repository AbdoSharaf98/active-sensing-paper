from types import MappingProxyType

import torch.optim
from torch import nn
from pytorch_lightning import LightningModule
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from typing_extensions import Required

from models.vae_1d import VAE, RecurrentVAE, MultiObsVAE
from utils import training

DEFAULT_VAE1_PARAMS = MappingProxyType({

    'type': 'recurrent',  # OPTIONS: 'recurrent', 'mlp'
    'recurrent_params': {
        'rnn_hidden_size': 256,
        'rnn_num_layers': 1
    },
    'vq_params': {
        'num_embeddings': 8
    },
    'layers': [256]
})

DEFAULT_VAE2_PARAMS = MappingProxyType({
    'layers': [256, 256],
    'rnn_hidden_size': 256,
    'rnn_num_layers': 1
})


class PerceptionModel(LightningModule):

    def __init__(self,
                 z_dim,
                 s_dim,
                 action_dim,
                 obs_dim,
                 vae1_params=DEFAULT_VAE1_PARAMS,
                 vae2_params=DEFAULT_VAE2_PARAMS,
                 vae1=None,
                 vae2=None,
                 use_latents=True,  # whether to use the observation directly to infer s or map them through z
                 encode_loc=False,
                 lr=0.001):

        super().__init__()

        self.save_hyperparameters()

        self.z_dim = z_dim
        self.s_dim = s_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.lr = lr
        self.use_latents = use_latents

        self.automatic_optimization = False

        # ------------------------ building the vae models --------------------------- #
        """ VAE 1 """
        if vae1 is None:

            # default is a ff vae
            vae_class = VAE
            vae_class_params = {}

            if vae1_params['type'] == 'recurrent':
                vae_class = RecurrentVAE
                vae_class_params = vae1_params['recurrent_params']

            self.vae1 = vae_class(input_dim=obs_dim + action_dim,
                                  latent_dim=z_dim,
                                  layers=vae1_params['layers'],
                                  **vae_class_params)
        else:
            self.vae1 = vae1

        """ VAE 2: this is always an instance of MultiObsVAE """
        if vae2 is None:

            self.vae2 = MultiObsVAE(input_dim=z_dim if use_latents else obs_dim + action_dim,
                                    loc_dim=action_dim, latent_dim=s_dim,
                                    **vae2_params, decoder_output='dist')

        else:
            assert isinstance(vae2, MultiObsVAE), 'Second VAE must be an instance of MultiObsVAE'
            self.vae2 = vae2

        # -------------------------- location encoder --------------------------- #
        if encode_loc:
            self.loc_encoder = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, action_dim))
        else:
            self.loc_encoder = nn.Identity()

        # construct a manual optimizer (in case of manual training)
        self.manual_optimizer = self.configure_optimizers()

        # the trainer attribute should be a PerceptionModelTrainer
        self._trainer: Required["training.PerceptionModelTrainer"] = None

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer

    def parameter_list(self):
        return list(self.vae2.parameters()) + int(self.use_latents) * list(self.vae1.parameters())

    def configure_optimizers(self, lr=None):
        if lr is None:
            lr = self.lr
        return torch.optim.Adam(self.parameter_list(), lr=lr)

    def forward(self, obs, locations):

        # if un-batched, add a dummy batch dimension
        if len(obs.shape) == 1:
            obs = obs.reshape(1, *obs.shape)

        if len(locations.shape) == 1:
            locations = locations.reshape(1, *locations.shape)

        # encode locations
        locations = self.loc_encoder(locations)

        # pass through the first vae
        vae2_input = torch.cat([obs[..., :-2], locations], dim=-1)
        z_posterior = None
        z_latents = None
        if self.use_latents:
            _, z_latents, z_posterior = self.vae1(vae2_input)
            vae2_input = z_latents

        # pass through the second vae
        z_prior, s_latent, s_posterior, h = self.vae2(vae2_input, locations)

        # compute the reconstructions of the z latents
        z_recons = None
        if self.use_latents:
            z_recons = z_prior.mu
            recons = self.vae1.decode(z_prior.sample()).squeeze()
        else:
            recons = z_prior.mu

        return recons, z_recons, z_latents, z_posterior, h, z_prior, s_posterior

    def _compute_losses(self, obs, locations, beta=1.0, rec_loss_scale=1.0):
        # TODO: in computing the KL losses (or expected log prob diffs) we can try drawing multiple samples and
        # TODO: averaging over them

        # forward the model
        recons, z_recons, z_latents, z_posterior, h, z_prior, s_posterior = self.forward(obs, locations)

        # compute the reconstruction loss
        x_rec_loss = mse_loss(recons, obs, reduction='none').sum(dim=[-2, -1]).mean()
        z_rec_loss = 0.0
        if self.use_latents:
            z_rec_loss = mse_loss(z_recons, z_latents, reduction='none').sum(dim=[-2, -1]).mean()
        rec_loss = x_rec_loss  # + z_rec_loss #TODO: what happens if we don't include the z rec loss

        # compute the kl loss between the z posterior and prior
        if self.use_latents:
            z_kl_loss = (z_posterior.torch_dist.log_prob(z_latents)
                         - z_prior.torch_dist.log_prob(z_latents)).sum(-1).mean()
        else:
            z_kl_loss = torch.tensor(0.0)

        # compute the kl loss between the s posterior and a standard gaussian
        s_mu, s_logvar = s_posterior.mu, torch.log(s_posterior.sigma ** 2)
        s_kl_loss = torch.mean(
            -0.5 * torch.sum(1 + s_logvar - s_mu ** 2 - s_logvar.exp(), dim=-1)
        )

        # total loss
        total_loss = rec_loss_scale * rec_loss + beta * s_kl_loss + 0.05 * z_kl_loss

        return total_loss, rec_loss, z_kl_loss, s_kl_loss

    def _training_step(self, obs, locations, beta=1.0, rec_loss_scale=1.0):

        total_loss, rec_loss, z_kl_loss, s_kl_loss = self._compute_losses(obs, locations, beta, rec_loss_scale)

        # step the optimizer
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), rec_loss.detach(), z_kl_loss.detach(), s_kl_loss.detach()

    def training_step(self, batch, batch_idx=None):

        obs, locations = batch

        beta = self.trainer.beta_schedule[self.trainer.current_epoch - 1]

        total_loss, rec_loss, z_kl_loss, s_kl_loss = self._compute_losses(obs, locations, beta,
                                                                          self.trainer.rec_loss_scale)

        optimizer = self.optimizers()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        results_dict = {
            'total_loss': total_loss,
            'rec_loss': rec_loss,
            'z_kl_loss': z_kl_loss,
            's_kl_loss': s_kl_loss
        }

        self.log_dict(results_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx=None, in_training=True):

        obs, locations = batch
        total_loss, rec_loss, z_kl_loss, s_kl_loss = self._compute_losses(obs, locations)

        results_dict = {
            'total_loss': total_loss,
            'rec_loss': rec_loss,
            'z_kl_loss': z_kl_loss,
            's_kl_loss': s_kl_loss
        }

        if in_training:
            self.log_dict(results_dict, prog_bar=True)

        return results_dict

    def train_dataloader(self):

        return DataLoader(self.trainer.datamodule.train_data, self.trainer.datamodule.batch_size)

    def val_dataloader(self):

        return DataLoader(self.trainer.datamodule.val_data, self.trainer.datamodule.batch_size)

    def test_dataloader(self):

        return DataLoader(self.trainer.datamodule.test_data, self.trainer.datamodule.batch_size)

    """
    def reset_rnn_states(self, batch_size, lower_state=None, higher_state=None):
        lower_state = self.vae1.reset_rnn_state(batch_size, device=self.device, state=lower_state)
        higher_state = self.vae2.reset_rnn_state(batch_size, device=self.device, state=higher_state)

        return lower_state, higher_state
    """

    def reset_rnn_states(self):
        self.vae1.reset_rnn_state()
        self.vae2.reset_rnn_state()

    def get_rnn_states(self):
        return self.vae1.get_rnn_state(), self.vae2.get_rnn_state()