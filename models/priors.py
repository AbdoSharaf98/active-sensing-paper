import warnings
import distributions
import torch
from torch import nn
from nets import GaussianMDN
from models.vae_1d import create_ff_network
from distributions import Gaussian


def compute_nll_loss(dist, y):
    """
    computes the negative log likelihood loss of data in y under mixture_dist
    """

    # determine what distribution we have
    if isinstance(dist, distributions.Gaussian) or isinstance(dist, distributions.GaussianMixture) or isinstance(dist,
                                                                                                                 torch.distributions.Categorical):
        nll = -dist.log_prob(y)
        return torch.mean(nll)
    else:
        raise ValueError("dist must be either a Gaussian, a GaussianMixture, or a Categorical distribution")


class FFPrior(nn.Module):
    """
    """

    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim

        # encoder network to be specified by the subclass
        self.prior_encoder = None

    def forward(self, x):
        return self.prior_encoder(x)

    def compute_loss(self, x, y):
        """
        computes the negative log likelihood loss of y under the output of the prior encoder
        """
        out_dist = self.forward(x)
        loss = -torch.log(out_dist.probability(y))

        return torch.mean(loss)

    def get_parameters(self):
        """
        should be called only by subclass instances
        """
        if self.prior_encoder is None:
            raise NotImplementedError

        if isinstance(self.prior_encoder, GaussianMDN):
            return self.prior_encoder.get_parameters()
        else:
            return self.prior_encoder.parameters()


class MixtureDensityFF(FFPrior):

    def __init__(self, input_dim, output_dim, encoder_layers=None, n_gaussians=4):
        super().__init__(input_dim)

        if encoder_layers is None:
            encoder_layers = [256, 256]
        self.prior_encoder = GaussianMDN(input_dim, output_dim, n_gaussians, layers=encoder_layers)


class GaussianFF(FFPrior):

    def __init__(self, input_dim, output_dim, encoder_layers=None):
        """
        """
        super().__init__(input_dim)

        self.output_dim = output_dim

        # construct the gaussian network
        if encoder_layers is None:
            encoder_layers = [256, 256]
        self.prior_encoder = create_ff_network([input_dim] + encoder_layers + [2 * output_dim])

    def forward(self, x):
        mu, logvar = torch.split(self.prior_encoder(x), self.output_dim, -1)
        return Gaussian(mu, torch.exp(0.5 * logvar))


class RNNPrior(nn.Module):
    """
    """

    def __init__(self, input_dim, hidden_size=64, num_layers=1):
        """
        """
        super(RNNPrior, self).__init__()

        self.input_dim = input_dim

        # construct the RNN
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)

        # the encoder network will be specified by the subclasses
        self.prior_encoder = None

    def forward(self, x, h, full_output=False):
        """
        should be called only by subclass instances
        :param x: Tensor (BxTxL) where B is batch size, T is sequence/rollout length, and L is the input size
        :param h: (tuple) initial hidden state of the network
        :param full_output: (bool) flag to return final rnn state
        :return:
        """
        if self.prior_encoder is None:
            raise NotImplementedError

        out, (hn, cn) = self.lstm(x, h)

        if full_output:
            return self.prior_encoder(out), (hn, cn)
        else:
            return self.prior_encoder(out)

    def compute_loss(self, x, h, y):
        """
        computes the negative log likelihood loss of y under the output of the mdn
        """
        out_dist = self.forward(x, h)
        loss = -torch.log(out_dist.probability(y))

        return torch.mean(loss)

    def init_lstm(self, batch_size, device='cpu'):
        """
        """
        h_init = (torch.randn(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device,
                              requires_grad=False),
                  torch.randn(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device,
                              requires_grad=False))
        return h_init

    def get_parameters(self):
        """
        should be called only by subclass instances
        """
        if self.prior_encoder is None:
            raise NotImplementedError

        return list(self.lstm.parameters()) + self.prior_encoder.get_parameters()


class MixtureDensityRNN(RNNPrior):
    """
    """

    def __init__(self, input_dim, output_dim, hidden_size=64, num_layers=1, n_gaussians=8,
                 encoder_layers=None):
        """
        """
        super(MixtureDensityRNN, self).__init__(input_dim, hidden_size=hidden_size,
                                                num_layers=num_layers)

        # construct the mixture density network
        if encoder_layers is None:
            encoder_layers = [256]
        self.prior_encoder = GaussianMDN(hidden_size, output_dim, n_gaussians, layers=encoder_layers)


class GaussianRNN(RNNPrior):
    """
    """

    def __init__(self, input_dim, output_dim, hidden_size=64, num_layers=1, encoder_layers=None):
        """

        :param input_dim:
        :param output_dim:
        :param hidden_size:
        :param num_layers:
        :param encoder_layers:
        """
        super(GaussianRNN, self).__init__(input_dim, hidden_size=hidden_size,
                                          num_layers=num_layers)

        self.output_dim = output_dim

        # construct the gaussian network
        if encoder_layers is None:
            encoder_layers = []
        layers = [hidden_size] + encoder_layers + [2 * output_dim]
        self.prior_encoder = create_ff_network(layers)

    def forward(self, x, h, full_output=False):
        out, (hn, cn) = self.lstm(x, h)
        mu, logvar = torch.split(self.prior_encoder(out), self.output_dim, -1)

        if full_output:
            return Gaussian(mu, torch.exp(0.5 * logvar)), (hn, cn)

        return Gaussian(mu, torch.exp(0.5 * logvar))