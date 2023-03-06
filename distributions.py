"""
collection of useful distributions
the methods defined here should work with
unspecified batch shapes (so we can work readily with sequence data)
"""

from torch.distributions import Categorical, MixtureSameFamily, Normal, Independent
import math
import torch

# define some constants
CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

# will be used for numerical stability
EPS = 1e-20


class Gaussian:
    """
    a simple gaussian distribution
    """

    def __init__(self, mu, sigma):
        """
        Inputs:
            mu (BxD): means - B is batch size and D is dim of the variable
            var (BxD): standard deviations
        """

        self.mu = mu
        self.sigma = sigma

        self.torch_dist = Independent(Normal(mu, sigma), reinterpreted_batch_ndims=1)

    def probability(self, target, multiple_samples=False):
        """
        calculates the probability of target under the distribution
        Inputs:
            target (BxD)
            multiple_samples (boolean): whether probability to be computed for multiple samples, in this case the
                                        number of samples is expected to be the first dimension
        Outputs:
            probs (Bx1)
        """

        if not multiple_samples:
            sigma = self.sigma
            mu = self.mu
        else:
            n_samples = target.shape[0]
            sigma = self.sigma.reshape(-1, *self.sigma.shape).repeat_interleave(n_samples, dim=0)
            mu = self.mu.reshape(-1, *self.mu.shape).repeat_interleave(n_samples, dim=0)

        # compute the individual probabilities of target under each gaussian
        gaussian_probs = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(
            -0.5 * ((target - mu) / sigma) ** 2) / sigma

        # add a small value for numerical stability
        gaussian_probs += torch.ones_like(gaussian_probs) * EPS

        # multiply uni-variate probabilities to get multivariate probability
        return torch.prod(gaussian_probs, -1)

    def log_prob(self, target):
        return self.torch_dist.log_prob(target)

    def sample(self):
        """
        draw a (reparameterized) sample
        """

        return torch.randn_like(self.sigma) * self.sigma + self.mu

    def sample_n(self, n):
        """
        draw n (reparameterized) samples
        """

        # reshape and repeat across the number of sample to vectorize operations
        sigma = self.sigma.reshape(-1, *self.sigma.shape).repeat_interleave(n, dim=0)
        mu = self.mu.reshape(-1, *self.mu.shape).repeat_interleave(n, dim=0)

        return torch.randn_like(sigma) * sigma + mu

    def detach(self):

        self.mu = self.mu.detach()
        self.sigma = self.sigma.detach()

        self.torch_dist = Independent(Normal(self.mu, self.sigma), reinterpreted_batch_ndims=1)

        return self

    def reshape(self, shape):

        self.mu = torch.reshape(self.mu, shape)
        self.sigma = torch.reshape(self.sigma, shape)
        self.torch_dist = Independent(Normal(self.mu, self.sigma), reinterpreted_batch_ndims=1)
        return self

    @property
    def shape(self):
        return self.mu.shape

    def get_argmax(self):
        return self.mu

    def params(self):
        return torch.cat([self.mu, self.sigma], dim=-1)


class GaussianMixture:
    """
    a mixture of gaussian distributions
    """

    def __init__(self, pi, sigma, mu):
        """
        Supports batch input.
        Here B is batch size, G is number of Gaussians, and D is the dimensionality of the variable
        Inputs:
            pi (BxG): weights of the mixture Gaussians
            sigma (BxGxD): standard deviations (for now, the Gaussians are all diagonal)
            mu (BxGxD): means
        """

        self.pi = pi
        self.sigma = sigma
        self.mu = mu
        self.n_gaussians = pi.shape[-1]

        # torch distribution
        mix = Categorical(pi)
        comp = Independent(Normal(mu, sigma), reinterpreted_batch_ndims=1)
        self.torch_dist = MixtureSameFamily(mix, comp)

    def probability(self, target, multiple_samples=False):
        """
        compute the probability of target under the distribution
        Inputs:
            target (BxD): target vector
            multiple_samples (boolean): whether probability to be computed for multiple samples, in this case the
                                        number of samples is expected to be the first dimension
        Outputs:
            probs (Bx1): probability under each distribution in the batch
        """
        if not multiple_samples:
            sigma = self.sigma
            mu = self.mu
            pi = self.pi
        else:
            n_samples = target.shape[0]
            sigma = self.sigma.reshape(-1, *self.sigma.shape).repeat_interleave(n_samples, dim=0)
            mu = self.mu.reshape(-1, *self.mu.shape).repeat_interleave(n_samples, dim=0)
            pi = self.pi.reshape(-1, *self.pi.shape).repeat_interleave(n_samples, dim=0)

        # replicate the target for each Gaussian in the mixture
        target = target.unsqueeze(-2).repeat_interleave(self.n_gaussians, dim=-2)

        # compute the individual probabilities of target under each gaussian
        gaussian_probs = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(
            -0.5 * ((target - mu) / sigma) ** 2) / sigma

        # add a small value for numerical stability
        gaussian_probs += torch.ones_like(gaussian_probs) * EPS

        # multiply univariate probabilities to get multivariate probability
        gaussian_probs = torch.prod(gaussian_probs, -1)

        # compute the weighted mixture probability
        return torch.sum(pi * gaussian_probs, dim=-1)

    def sample(self):
        """
        draw a (non-reparameterized) sample from the mixture distribution
        """

        # sample from a categorical first to decide which individual gaussian to sample from
        pi_dist = Categorical(self.pi)
        sample_pies = pi_dist.sample().view(*list(pi_dist.batch_shape), 1, 1)

        # replicate and reshape so we can vector operate with the individual distribution params
        sample_pies = torch.repeat_interleave(sample_pies, self.sigma.shape[-1], dim=-1)

        # get the means and variances of the distributions corresponding to the sampled weights
        stds = self.sigma.detach().gather(-2, sample_pies).squeeze()
        means = self.mu.detach().gather(-2, sample_pies).squeeze()

        # standard normal noise
        normal_noise = torch.randn(
            stds.shape, requires_grad=False
        ).to(stds.device)

        return normal_noise * stds + means

    def log_prob(self, target):
        return self.torch_dist.log_prob(target)

    def detach(self):

        self.sigma = self.sigma.detach()
        self.mu = self.mu.detach()
        self.pi = self.pi.detach()

        # detached torch distribution
        mix = Categorical(self.pi)
        comp = Independent(Normal(self.mu, self.sigma), reinterpreted_batch_ndims=1)
        self.torch_dist = MixtureSameFamily(mix, comp)

        return self

    def reshape(self, shape):

        new_shape = shape[:-1] + (self.n_gaussians, shape[-1])
        self.sigma = self.sigma.reshape(new_shape)
        self.mu = self.mu.reshape(new_shape)
        self.pi = self.pi.reshape(shape[:-1] + (self.n_gaussians,))

        mix = Categorical(self.pi)
        comp = Independent(Normal(self.mu, self.sigma), reinterpreted_batch_ndims=1)
        self.torch_dist = MixtureSameFamily(mix, comp)

        return self

    @property
    def shape(self):
        return self.mu.shape[:-2] + (self.mu.shape[-1],)


class Concrete:

    def __init__(self, probs, table):
        self.probs = probs
        self.logits = torch.log(probs + EPS)    # for numerical stability
        self.num_classes = self.probs.shape[-1]

        # torch categorical dist
        self.torch_dist = Categorical(probs=probs)

        # table to index locations when sampling
        table_shape = tuple([1 for _ in self.torch_dist.batch_shape]) + table.shape
        reps = tuple(self.torch_dist.batch_shape) + tuple([1 for _ in table.shape])
        self.table = table.reshape(table_shape).repeat(*reps).to(probs.device)
        self.class_dim = self.table.shape[-1]

    def probability(self, sample):
        """ sample can be (B,) or (B,N) with N = num_classes in the case of one-hot vectors """
        if sample.shape == self.torch_dist.batch_shape:
            # convert to one hot
            sample = torch.nn.functional.one_hot(sample, self.num_classes)

        return (sample * self.probs).sum(dim=-1)

    def log_prob(self, target):
        return torch.log(self.probability(target))

    def sample(self, tau=0.5, hard=True, one_hot=False):
        """
        draw a (reparameterized) sample using the gumbel softmax function
        """

        one_hot_sample = torch.nn.functional.gumbel_softmax(self.logits, tau, hard=hard)

        if one_hot:
            return one_hot_sample

        one_hot_sample = one_hot_sample.unsqueeze(-1).repeat_interleave(self.class_dim, dim=-1)

        return (one_hot_sample * self.table).sum(-2)

    def get_argmax(self):

        return self.table[0][torch.argmax(self.probs, dim=-1), :]

    def detach(self):
        probs = self.probs.detach()
        return self.__init__(probs, self.table)

    @property
    def shape(self):
        return self.probs.shape

    def kl_from_uniform(self):

        return torch.log(torch.tensor(self.num_classes)) - self.torch_dist.entropy()

    def kl_div(self, target_probs):

        return torch.sum(self.probs * torch.log(self.probs/target_probs), dim=-1)