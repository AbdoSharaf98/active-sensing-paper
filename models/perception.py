from types import MappingProxyType

import torch.optim
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from typing_extensions import Required

from models.vaes.lower_vae import LowerVAE
from models.vaes.higher_vae import HigherVAE
from utils import training

DEFAULT_PARAMS = MappingProxyType({
    'summarization_method': 'mlp',
    'lower_vae': {
        'layers': [256, 256]
    },
    'higher_vae': {
        'layers': [256, 256],
        'integration_method': 'sum',
        'rnn_hidden_size': 512,
        'rnn_num_layers': 1
    }
})


class MlpSummarizer(nn.Module):

    def __init__(self, obs_dim: int, loc_dim: int):
        super(MlpSummarizer, self).__init__()

        self.fc_obs = nn.Linear(obs_dim, obs_dim + loc_dim)
        self.fc_loc = nn.Linear(loc_dim, obs_dim + loc_dim)
        self.output_dim = obs_dim + loc_dim

    def forward(self, obs: torch.Tensor, loc: torch.Tensor):
        return F.relu(self.fc_obs(obs) + self.fc_loc(loc))


class CatSummarizer(nn.Module):

    @staticmethod
    def forward(obs: torch.Tensor, loc: torch.Tensor):
        return torch.cat([obs, loc], dim=-1)


class PerceptionModel(LightningModule):

    def __init__(self,
                 z_dim,
                 s_dim,
                 loc_dim,
                 obs_dim,
                 vae_params: dict = DEFAULT_PARAMS,
                 vae1=None,
                 vae2=None,
                 encode_loc=False,
                 lr=0.001,
                 ):

        super(PerceptionModel, self).__init__()

        self.save_hyperparameters()

        self.z_dim = z_dim
        self.s_dim = s_dim
        self.loc_dim = loc_dim
        self.obs_dim = obs_dim
        self.lr = lr
        self.automatic_optimization = False

        # build the summarizer
        self.lower_summarizer = MlpSummarizer(obs_dim, loc_dim) if \
            vae_params['summarization_method'] == 'mlp' else CatSummarizer()

        self.higher_summarizer = MlpSummarizer(z_dim, loc_dim) if \
            vae_params['summarization_method'] == 'mlp' else CatSummarizer()

        # build the location encoder
        if encode_loc:
            self.location_encoder = nn.Sequential(nn.Linear(2, loc_dim), nn.ReLU())
        else:
            self.location_encoder = nn.Identity()

        # build the lower vae
        if vae1 is None:
            self.vae1 = LowerVAE(self.obs_dim, self.loc_dim, self.z_dim, summarizer=self.lower_summarizer,
                                 layers=vae_params['lower_vae']['layers'])
        else:
            self.vae1 = vae1

        # build the higher vae
        if vae2 is None:
            self.vae2 = HigherVAE(self.z_dim, self.loc_dim, self.s_dim,
                                  summarizer=self.higher_summarizer,
                                  **vae_params['higher_vae'])
        else:
            self.vae2 = vae2

        # construct a manual optimizer (in case of manual training)
        self.manual_optimizer = self.configure_optimizers()

        # the trainer attribute should be a PerceptionModelTrainer
        self._trainer: Required["training.PerceptionModelTrainer"] = None

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: training.PerceptionModelTrainer):
        self._trainer = trainer

    def parameter_list(self):
        return list(self.vae2.parameters()) + list(self.vae1.parameters())

    def configure_optimizers(self, lr: float = None):
        if lr is None:
            lr = self.lr
        return torch.optim.Adam(self.parameter_list(), lr=lr)

    def forward(self, obs: torch.Tensor, locs: torch.Tensor):

        # if un-batched, add a dummy batch dimension
        if len(obs.shape) == 1:
            obs = obs.reshape(1, *obs.shape)
        if len(locs.shape) == 1:
            locs = locs.reshape(1, *locs.shape)

        # encode locations
        locs = self.location_encoder(locs)

        # pass through the first vae
        _, z_latents, z_posterior = self.vae1(obs, locs)
        vae2_input = z_latents

        # pass through the second vae
        z_prior, s_latent, s_posterior, h = self.vae2(vae2_input, locs)

        # compute the reconstructions of the z latents
        recs = self.vae1.decode(z_prior.sample())

        return recs, z_latents, z_posterior, h, z_prior, s_posterior

    def _compute_losses(self, obs, locations, beta=1.0, rec_loss_scale=1.0):
        # TODO: in computing the KL losses (or expected log prob diffs) we can try drawing multiple samples and
        # TODO: averaging over them

        # forward the model
        recs, z_latents, z_posterior, h, z_prior, s_posterior = self.forward(obs, locations)

        # compute the reconstruction loss
        rec_loss = mse_loss(recs, obs, reduction='none').sum(dim=[-2, -1]).mean()

        # compute the kl loss between the z posterior and prior
        z_kl_loss = (z_posterior.torch_dist.log_prob(z_latents)
                     - z_prior.torch_dist.log_prob(z_latents)).sum(-1).mean()

        # compute the kl loss between the s posterior and a standard gaussian
        s_mu, s_logvar = s_posterior.mu, torch.log(s_posterior.sigma ** 2)
        s_kl_loss = torch.mean(
            -0.5 * torch.sum(1 + s_logvar - s_mu ** 2 - s_logvar.exp(), dim=-1)
        )

        # total loss
        total_loss = rec_loss_scale * rec_loss + beta * s_kl_loss + 0.001 * z_kl_loss

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

    def reset_rnn_states(self):
        self.vae2.reset_rnn_state()

    def get_rnn_states(self):
        return self.vae2.get_rnn_state()
