import distributions
import torch
from torch import nn
from distributions import Gaussian, GaussianMixture, Concrete
from torch.distributions import Categorical
from torch.nn import functional as F
from nets import create_ff_network

EPS = 1e-10


def sum_sequence(zs: torch.Tensor):
    return torch.sum(zs, dim=1)


def concat_sequence(zs: torch.Tensor):
    return zs.flatten(start_dim=-2)


class RNNIntegrator(nn.Module):

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int = 1):
        super(RNNIntegrator, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.rnn_state = None

    def forward(self, zs: torch.Tensor):

        if zs.shape[1] == 1:  # only one element in the sequence
            h_init = self.rnn_state if self.rnn_state is not None else self.init_state(zs.shape[0], zs.device)
        else:
            h_init = self.init_state(zs.shape[0], zs.device)

        # pass through the encoder rnn
        z_out, (hn, cn) = self.lstm(zs, h_init)

        # record the rnn state if not training
        self.rnn_state = (hn, cn)

        return z_out[:, -1, :]

    def init_state(self, batch_size, device='cuda', state=None):
        if state is not None:
            init_state = state
        else:
            init_state = (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size,
                                      device=device,
                                      dtype=torch.float,
                                      requires_grad=True),
                          torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size,
                                      device=device,
                                      dtype=torch.float,
                                      requires_grad=True))
        return init_state


class HigherVAE(nn.Module):

    def __init__(self,
                 obs_dim: int,
                 loc_dim: int,
                 latent_dim: int,
                 summarizer: nn.Module,
                 layers: list[int] = None,
                 integration_method: str = 'sum',  # OPTIONS are ['sum', 'rnn', 'cat']
                 rnn_hidden_size: int = None,
                 rnn_num_layers: int = 1,
                 seq_len: int = None):
        super(HigherVAE, self).__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        # summarizer will summarize past observation and locations
        self.summarizer = summarizer

        # integration method will specify how to integrate past information
        if integration_method == 'sum':
            self.integrator = sum_sequence
            enc_input = obs_dim + loc_dim
        elif integration_method == 'rnn':
            assert rnn_hidden_size is not None, "rnn hidden size must be provided if using rnn for integration"
            self.integrator = RNNIntegrator(obs_dim + loc_dim, rnn_hidden_size, rnn_num_layers)
            enc_input = rnn_hidden_size
        else:   # integration method is concatenation
            assert seq_len is not None, "seq_len must be provided if using concat integration method"
            self.integrator = concat_sequence
            enc_input = (obs_dim + loc_dim) * seq_len

        if layers is None:
            layers = [256]

        self.layers = layers

        # build the encoder network
        enc_layers = [enc_input] + layers + [latent_dim]
        self.mu_encoder = create_ff_network(enc_layers, h_activation='relu')
        self.log_std_encoder = create_ff_network(enc_layers, h_activation='relu')

        # build the decoder network
        dec_layers = [latent_dim + loc_dim] + list(reversed(layers)) + [obs_dim]
        self.mu_decoder = create_ff_network(dec_layers, h_activation='relu')
        self.log_std_decoder = create_ff_network(dec_layers, h_activation='relu')

    def encode(self, obs: torch.Tensor, loc: torch.Tensor):

        # if the input is un-batched, batch it
        if len(obs.shape) == 1:  # one data point, seq_len=1, batch_size=1
            obs = obs.reshape(1, 1, *obs.shape)
        elif len(obs.shape) == 2:  # batch of points, seq_len=1, batch_size is first dimension
            obs = obs.unsqueeze(1)

        # make sure loc has the same shape
        loc = loc.reshape((obs.shape[:-1]) + (-1,))

        # summarize
        summary = self.summarizer(obs, loc)

        # integrate
        aggregate = self.integrator(summary)

        # encode
        return Gaussian(self.mu_encoder(aggregate),
                        torch.exp(self.log_std_encoder(aggregate)) + EPS), aggregate

    def decode(self, s: torch.Tensor, loc: torch.Tensor):

        cats = torch.cat([s, loc], dim=-1)

        return Gaussian(self.mu_decoder(cats), torch.exp(self.log_std_decoder(cats)))

    def forward(self, obs: torch.Tensor, loc: torch.Tensor):

        # encode
        s_dist, agg = self.encode(obs, loc)
        s = s_dist.sample()

        # repeat s to match the shape of loc
        s = s.unsqueeze(1).repeat(1, loc.shape[1], 1)

        # feed through the decoder
        z_hat_dist = self.decode(s, loc)

        return z_hat_dist, s[:, 0, :], s_dist, agg

    def reset_rnn_state(self):
        if isinstance(self.integrator, RNNIntegrator):
            self.integrator.rnn_state = None

    def get_rnn_state(self):
        if isinstance(self.integrator, RNNIntegrator):
            return self.integrator.rnn_state
