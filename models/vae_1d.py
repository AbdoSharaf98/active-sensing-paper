import distributions
import torch
from torch import nn
from distributions import Gaussian, GaussianMixture, Concrete
from torch.distributions import Categorical
from torch.nn import functional as F
from nets import create_ff_network

EPS = 1e-10


def compute_loss_standard_prior(x,
                                x_rec,
                                q_dist: Gaussian,
                                prior_dist=None,  # for consistency
                                beta=1.0,
                                rec_loss_scale=1.0):
    """
    computes -ELBO objective with a standard normal as the prior on the latents
    Inputs:
        x (BxD): input
        x_rec (BxD): reconstructed input
        q_dist (Gaussian): approximate posterior
        beta (float): kl loss regularization weight
        rec_loss_scale (float): reconstruction loss scaling factor
    Outputs:
        total_loss (float): rec_loss_scale * rec_loss + beta * kl_loss
        rec_loss (float): reconstruction loss
        kl_loss (float): KL loss
    """

    rec_loss = F.mse_loss(x, x_rec, reduction='none').sum(dim=[-2, -1]).mean()

    mu, logvar = q_dist.mu, torch.log(q_dist.sigma ** 2)

    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1)
    )

    total_loss = rec_loss_scale * rec_loss + beta * kl_loss

    return total_loss, rec_loss.detach(), kl_loss.detach()


def compute_loss_mog_prior(x,
                           x_rec,
                           q_dist: Gaussian,
                           prior_dist: GaussianMixture,
                           beta=0.1,
                           rec_loss_scale=1.0):
    """
    computes the -ELBO loss when the prior on the latents is a mixture of gaussians given by prior_dist
    in this case, the distribution params for q_dist and prior_dist must have a batch shape of BxSEQ_LEN where
    B is the batch size and SEQ_LEN is the sequence length
    """

    # reconstruction loss
    rec_loss = F.mse_loss(x, x_rec, reduction='sum')

    # KL loss

    # 1. draw n samples from the approximate posterior (more samples make the estimate more accurate)
    n_samples = 100
    z = q_dist.sample_n(n_samples)  # shape should be (n_samples x B x seq_len x D)

    # 2. compute the probability of all samples under the respective priors
    # pz = prior_dist.probability(z, multiple_samples=True)
    # pz += torch.ones_like(pz) * EPS  # for numerical stability
    log_pz = prior_dist.log_prob(z).view(n_samples, -1)

    # 3. compute the probability of all samples under the approximate posterior
    # qz = q_dist.probability(z, multiple_samples=True)
    log_qz = q_dist.log_prob(z).view(n_samples, -1)

    # 4. now compute the kl loss
    kl_loss = torch.mean(torch.mean(log_qz - log_pz, dim=-1))

    total_loss = rec_loss_scale * rec_loss + beta * kl_loss

    return total_loss, rec_loss.detach(), kl_loss.detach()


def compute_loss_gaussian_prior(x,
                                x_rec,
                                q_dist: Gaussian,
                                prior_dist: Gaussian,
                                beta=0.1,
                                rec_loss_scale=1.0):
    """
    computes the -ELBO loss when the prior on the latents is a mixture of gaussians given by prior_dist
    in this case, the distribution params for q_dist and prior_dist must have a batch shape of BxSEQ_LEN where
    B is the batch size and SEQ_LEN is the sequence length
    """

    # reconstruction loss
    rec_loss = F.mse_loss(x, x_rec, reduction='sum')

    # KL loss
    kl_loss = torch.mean(torch.distributions.kl.kl_divergence(q_dist.torch_dist, prior_dist.torch_dist))

    total_loss = rec_loss_scale * rec_loss + beta * kl_loss

    return total_loss, rec_loss.detach(), kl_loss.detach()


def compute_loss_discrete_prior(x,
                                x_rec,
                                latent_distances,
                                prior: distributions.Categorical = None,
                                beta=0.1,
                                rec_loss_scale=1.0):
    """
    #TODO: add description
    """

    # reconstruction loss
    rec_loss = F.mse_loss(x, x_rec, reduction='sum')

    # cross entropy loss
    # in this case, we are optimizing the cross entropy based on the distance scores resulting from the encoder output
    # we can't directly compute the kl loss since we can't propagate the gradients through the probabilities, so we
    # are using cross entropy as a surrogate - optimizing the cross entropy is equivalent to optimizing the KL loss

    if prior is None:  # default is a uniform prior, return only the reconstruction loss
        kl_loss = None
        total_loss = rec_loss_scale * rec_loss

        return total_loss, rec_loss, kl_loss
    else:
        neg_distance_scores = torch.softmax(-latent_distances, dim=-1)
        encoder_crosse_loss = F.cross_entropy(neg_distance_scores, prior.probs, reduction='mean')

        total_loss = rec_loss_scale * rec_loss + beta * encoder_crosse_loss

        return total_loss, rec_loss, encoder_crosse_loss


class VAE(nn.Module):

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 layers=None,
                 **kwargs):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        if layers is None:
            layers = [32, 64, 128]

        # build the encoder network
        enc_layers = [input_dim] + layers + [2 * latent_dim]
        self.encoder = create_ff_network(enc_layers)

        # build the decoder
        dec_layers = [latent_dim] + list(reversed(layers)) + [input_dim]
        self.decoder = create_ff_network(dec_layers)

    def encode(self, x) -> Gaussian:
        """
        encodes input x
        Inputs:
            x (BxD): B is batch size and D is input dimensionality
        Outputs:
            Gaussian distribution
        """

        mu, logvar = torch.split(self.encoder(x), self.latent_dim, dim=-1)

        return Gaussian(mu, torch.exp(0.5 * logvar))

    def decode(self, z):
        """
        decodes a latent variable z
        Inputs:
            z (BxL): B is batch size and L is latent dim
        Outputs:
            x (BxD): decoded output
        """

        return self.decoder(z)

    def forward(self, x):
        """
        forward method
        Inputs:
            x (BxD): B is batch size and D is input dim
        Outputs:
            X_rec (BxD): reconstructions
            z (BxL): drawn samples - L is latent dim
            Gaussian dist
        """

        dist = self.encode(x)
        z = dist.sample()

        return self.decode(z), z, dist

    def reset_rnn_state(self):
        # dummy function for consistency
        pass


class RecurrentVAE(nn.Module):

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 layers=None,
                 rnn_hidden_size=64,
                 rnn_num_layers=1):
        super(RecurrentVAE, self).__init__()

        self.latent_dim = latent_dim

        # create the encoder RNN
        self.encoder_rnn = nn.LSTM(input_size=input_dim,
                                   hidden_size=rnn_hidden_size,
                                   num_layers=rnn_num_layers,
                                   batch_first=True)

        if layers is None:
            layers = [32, 64, 128]

        self.layers = layers

        # build the encoder network
        enc_layers = [rnn_hidden_size] + layers + [2 * latent_dim]
        self.encoder = create_ff_network(enc_layers)

        # build the decoder
        # 1. two fc layers that initialize the decoder rnn
        self.decoder_h0 = nn.Linear(latent_dim, rnn_num_layers * rnn_hidden_size)
        self.decoder_c0 = nn.Linear(latent_dim, rnn_num_layers * rnn_hidden_size)

        # 2. decoder rnn
        self.decoder_rnn = nn.LSTM(input_size=latent_dim, hidden_size=rnn_hidden_size,
                                   num_layers=rnn_num_layers, batch_first=True)

        # 3. final decoder mlp
        dec_layers = [rnn_hidden_size] + list(reversed(layers)) + [input_dim]
        self.decoder = create_ff_network(dec_layers)

        # state of the rnn (used in evaluation mode)
        self.rnn_state = None

    def _encode(self, x):
        """
        helper function for encode, meant to be used within classes
        Args:
            x:
        Returns:
        """
        # if the input is unbatched, batch it
        if len(x.shape) == 1:  # one data point, seq_len=1, batch_size=1
            x = x.reshape(1, 1, *x.shape)
        elif len(x.shape) == 2:  # batch of points, seq_len=1, batch_size is first dimension
            x = x.unsqueeze(1)

        # initialize the encoder rnn
        enc_h_init = (torch.zeros(self.encoder_rnn.num_layers, x.shape[0], self.encoder_rnn.hidden_size,
                                  device=x.device,
                                  requires_grad=False),
                      torch.zeros(self.encoder_rnn.num_layers, x.shape[0], self.encoder_rnn.hidden_size,
                                  device=x.device,
                                  requires_grad=False))

        # if not self.training and (x.shape[1] == 1):        # TODO
        if x.shape[1] == 1:
            if self.rnn_state is None:
                self.rnn_state = enc_h_init
            else:
                enc_h_init = self.rnn_state

        # pass through the encoder rnn
        enc_rnn_out, (hn, cn) = self.encoder_rnn(x, enc_h_init)

        # record the rnn state if not training
        self.rnn_state = (hn, cn)

        # pass through the encoder ff net
        return self.encoder(enc_rnn_out)

    def encode(self, x):
        """
        encodes input x
        Inputs:
            x (BxD): B is batch size and D is input dimensionality
        Outputs:
            Gaussian distribution
        """

        # pass through the encoder
        mu, logvar = torch.split(self._encode(x), self.latent_dim, dim=-1)

        return Gaussian(mu, torch.exp(0.5 * logvar))

    def decode(self, z):
        """
        decodes a latent variable z
        Inputs:
            z (BxL): B is batch size and L is latent dim
        Outputs:
            x (BxD): decoded output
        """

        if len(z.shape) == 1:  # if unbatched, batch it
            z = z.reshape(1, 1, *z.shape)
        elif len(z.shape) == 2:  # if we have a batch but not a sequence
            z = z.unsqueeze(1)

        # initialize the decoder rnn
        batch_size = z.shape[0]

        dec_h_init = (self.decoder_h0(z[:, 0, :]).reshape(self.decoder_rnn.num_layers, batch_size,
                                                          self.decoder_rnn.hidden_size),
                      self.decoder_c0(z[:, 0, :]).reshape(self.decoder_rnn.num_layers, batch_size,
                                                          self.decoder_rnn.hidden_size))

        dec_rnn_out, _ = self.decoder_rnn(z, dec_h_init)

        return self.decoder(dec_rnn_out)

    def forward(self, x):
        """
        forward method
        Inputs:
            x (BxD): B is batch size and D is input dim
        Outputs:
            X_rec (BxD): reconstructions
            z (BxL): drawn samples - L is latent dim
            Gaussian dist
        """
        dist = self.encode(x)
        z = dist.sample()

        return self.decode(z), z, dist

    def reset_rnn_state(self):
        self.rnn_state = None


class MultiObsVAE(nn.Module):

    def __init__(self,
                 input_dim: int,
                 loc_dim: int,
                 latent_dim: int,
                 layers=None,
                 rnn_hidden_size=64,
                 rnn_num_layers=1,
                 decoder_output='rec'):

        super(MultiObsVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # create the encoder RNN
        self.encoder_rnn = nn.LSTM(input_size=input_dim + loc_dim,
                                   hidden_size=rnn_hidden_size,
                                   num_layers=rnn_num_layers,
                                   batch_first=True)

        if layers is None:
            layers = [256]

        self.layers = layers

        # build the encoder network
        enc_layers = [rnn_hidden_size] + layers + [2 * latent_dim]
        self.encoder = create_ff_network(enc_layers)

        # build the decoder network
        dec_out = input_dim if decoder_output == 'rec' else 2 * input_dim
        dec_layers = [latent_dim + loc_dim] + list(reversed(layers)) + [dec_out]
        self.decoder = create_ff_network(dec_layers)
        self.decoder_output = decoder_output

        self.rnn_state = None

    def encode(self, x, loc):

        # if the input is unbatched, batch it
        if len(x.shape) == 1:  # one data point, seq_len=1, batch_size=1
            x = x.reshape(1, 1, *x.shape)
        elif len(x.shape) == 2:  # batch of points, seq_len=1, batch_size is first dimension
            x = x.unsqueeze(1)

        # make sure loc has the same shape
        loc = loc.reshape((x.shape[:-1]) + (-1,))

        # input to the encoder rnn
        x_in = torch.cat([x, loc], dim=-1)

        # initialize the encoder rnn
        enc_h_init = (torch.zeros(self.encoder_rnn.num_layers, x_in.shape[0], self.encoder_rnn.hidden_size,
                                  device=x_in.device,
                                  requires_grad=False),
                      torch.zeros(self.encoder_rnn.num_layers, x_in.shape[0], self.encoder_rnn.hidden_size,
                                  device=x_in.device,
                                  requires_grad=False))

        if x.shape[1] == 1:
            if self.rnn_state is None:
                self.rnn_state = enc_h_init
            else:
                enc_h_init = self.rnn_state

        # pass through the encoder rnn
        enc_rnn_out, (hn, cn) = self.encoder_rnn(x_in, enc_h_init)

        # record the rnn state if not training
        self.rnn_state = (hn, cn)

        # pass through the encoder ff net
        mu, logvar = torch.split(self.encoder(enc_rnn_out[:, -1, :]), self.latent_dim, dim=-1)

        return Gaussian(mu, torch.exp(0.5 * logvar))

    def decode(self, s, loc):

        dec_out = self.decoder(torch.cat([s, loc], dim=-1))

        if self.decoder_output == 'rec':
            return dec_out

        mu, logvar = torch.split(dec_out, self.input_dim, dim=-1)

        return Gaussian(mu, torch.exp(0.5 * logvar))

    def forward(self, x, loc):

        # if the input is unbatched, batch it
        if len(x.shape) == 1:  # one data point, seq_len=1, batch_size=1
            x = x.reshape(1, 1, *x.shape)
        elif len(x.shape) == 2:  # batch of points, seq_len=1, batch_size is first dimension
            x = x.unsqueeze(1)

        # make sure loc has the same shape
        loc = loc.reshape((x.shape[:-1]) + (-1,))

        # latent variable s
        s_dist = self.encode(x, loc)
        s = s_dist.sample()

        # repeat s to match the shape of loc
        s = s.unsqueeze(1).repeat(1, loc.shape[1], 1)

        # feed through the decoder
        x_hats = self.decode(s, loc)

        return x_hats, s[:, 0, :], s_dist

    def reset_rnn_state(self):
        self.rnn_state = None


class TwoLatentMultiObsVAE(MultiObsVAE):

    def __init__(self,
                 input_dim: int,
                 loc_dim: int,
                 latent_dim: int,
                 num_locs: int,
                 loc_table: torch.Tensor,
                 layers=None,
                 rnn_hidden_size=64,
                 rnn_num_layers=1,
                 decoder_output='rec'):

        super(TwoLatentMultiObsVAE, self).__init__(input_dim, loc_dim, latent_dim,
                                                   layers=layers,
                                                   rnn_hidden_size=rnn_hidden_size,
                                                   rnn_num_layers=rnn_num_layers,
                                                   decoder_output=decoder_output)

        # adjustments to account for the two latents
        # the second latent is assumed to be the global location so it has the same dimensionality as loc_dim

        self.num_locs = num_locs

        # build a second encoder network for the global location
        enc_layers = [rnn_hidden_size + latent_dim] + self.layers + [self.num_locs]
        self.global_loc_encoder = create_ff_network(enc_layers, h_activation='relu', out_activation='softmax')

        # location table
        self.loc_table = loc_table

        # adjust the decoder network
        dec_out = input_dim if decoder_output == 'rec' else 2 * input_dim
        dec_layers = [latent_dim + loc_dim + loc_dim] + list(reversed(self.layers)) + [dec_out]
        self.decoder = create_ff_network(dec_layers)

    def encode(self, x, loc):
        """ this will give two posterior distributions,
         the first is a gaussian over the abstract state s
         the second is a concrete dist over the global locations
         """

        # if the input is unbatched, batch it
        if len(x.shape) == 1:  # one data point, seq_len=1, batch_size=1
            x = x.reshape(1, 1, *x.shape)
        elif len(x.shape) == 2:  # batch of points, seq_len=1, batch_size is first dimension
            x = x.unsqueeze(1)

        # make sure loc has the same shape
        loc = loc.reshape((x.shape[:-1]) + (-1,))

        # input to the encoder rnn
        x_in = torch.cat([x, loc], dim=-1)

        # initialize the encoder rnn
        enc_h_init = (torch.zeros(self.encoder_rnn.num_layers, x_in.shape[0], self.encoder_rnn.hidden_size,
                                  device=x_in.device,
                                  requires_grad=False),
                      torch.zeros(self.encoder_rnn.num_layers, x_in.shape[0], self.encoder_rnn.hidden_size,
                                  device=x_in.device,
                                  requires_grad=False))

        if x.shape[1] == 1:
            if self.rnn_state is None:
                self.rnn_state = enc_h_init
            else:
                enc_h_init = self.rnn_state

        # pass through the encoder rnn
        enc_rnn_out, (hn, cn) = self.encoder_rnn(x_in, enc_h_init)

        # record the rnn state if not training
        self.rnn_state = (hn, cn)

        # pass through the first encoder net to get the distribution over s
        mu, logvar = torch.split(self.encoder(enc_rnn_out[:, -1, :]), self.latent_dim, dim=-1)

        # pass through the second encoder net to get the distribution over L
        loc_enc_input = torch.cat([enc_rnn_out[:, -1, :], mu], dim=-1)
        L_probs = self.global_loc_encoder(loc_enc_input)

        return Gaussian(mu, torch.exp(0.5 * logvar)), Concrete(probs=L_probs, table=self.loc_table)

    def decode(self, s, L, loc):

        dec_out = self.decoder(torch.cat([s, L, loc], dim=-1))

        if self.decoder_output == 'rec':
            return dec_out

        mu, logvar = torch.split(dec_out, self.input_dim, dim=-1)

        return Gaussian(mu, torch.exp(0.5 * logvar))

    def forward(self, x, loc):

        # if the input is unbatched, batch it
        if len(x.shape) == 1:  # one data point, seq_len=1, batch_size=1
            x = x.reshape(1, 1, *x.shape)
        elif len(x.shape) == 2:  # batch of points, seq_len=1, batch_size is first dimension
            x = x.unsqueeze(1)

        # make sure loc has the same shape
        loc = loc.reshape((x.shape[:-1]) + (-1,))

        # latent variable s
        s_dist, L_dist = self.encode(x, loc)
        s = s_dist.sample()
        L = L_dist.sample()

        # repeat s and L to match the shape of loc
        s = s.unsqueeze(1).repeat(1, loc.shape[1], 1)
        L = L.unsqueeze(1).repeat(1, loc.shape[1], 1)

        # feed through the decoder
        x_hats = self.decode(s, L, loc)

        return x_hats, s[:, 0, :], s_dist, L[:, 0, :], L_dist