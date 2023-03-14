import torch
from torch import nn
from distributions import GaussianMixture, Gaussian, Concrete


def create_ff_network(layer_dims, h_activation='tanh', out_activation='none'):
    layers = []

    if h_activation == 'sigmoid':
        h_activation_fxn = nn.Sigmoid
    elif h_activation == 'relu':
        h_activation_fxn = nn.ReLU
    elif h_activation == 'none':
        h_activation_fxn = None
    else:
        h_activation_fxn = nn.Tanh

    for h in range(len(layer_dims) - 2):
        if h_activation_fxn is not None:
            layers.append(
                nn.Sequential(
                    nn.Linear(layer_dims[h], layer_dims[h + 1]),
                    h_activation_fxn()
                ))
        else:
            layers.append(nn.Linear(layer_dims[h], layer_dims[h + 1]))

    if out_activation == 'tanh':
        layers.append(
            nn.Sequential(nn.Linear(layer_dims[-2], layer_dims[-1]), nn.Tanh())
        )
    elif out_activation == 'sigmoid':
        layers.append(
            nn.Sequential(nn.Linear(layer_dims[-2], layer_dims[-1]), nn.Sigmoid())
        )
    elif out_activation == 'softmax':
        layers.append(
            nn.Sequential(nn.Linear(layer_dims[-2], layer_dims[-1]), nn.Softmax(dim=-1))
        )
    else:
        layers.append(
            nn.Linear(layer_dims[-2], layer_dims[-1])
        )

    return nn.Sequential(*layers)


class ActionNetwork(nn.Module):
    """
    An action network for active sensing
    """

    def __init__(self, input_dim, action_dim, layers=None, out_dist='gaussian',
                 num_actions=None, action_table=None, lr=0.001, action_std=0.05):
        """
        :param input_dim: dimensionality of the input to the network
        :param action_dim: dimensionality of the action
        :param out_dist: output distributions. Can be continuous -> 'gaussian' or discrete -> 'concrete'
        :param num_actions: in case out_dist='concrete', this corresponds to the number of actions
        :param action_table: in case out_dist='concrete', this is the action table over which probs are calculated
        """

        super().__init__()

        self.std = action_std

        # construct the base network
        if layers is None:
            layers = [256]
        base_layers = [input_dim] + layers
        self.base_net = create_ff_network(base_layers, h_activation='none', out_activation='none')

        # construct the distribution network
        self.out_dist = out_dist
        self.action_table = action_table
        if self.out_dist == 'concrete':
            assert num_actions is not None, 'number of actions must be specified if output dist = concrete'
            assert action_table is not None, 'action table must be specified if output dist = concrete'
            self.dist_net = nn.Sequential(
                nn.Linear(base_layers[-1], num_actions),
                nn.Softmax()
            )

        else:  # if anything other than 'concrete', default to 'gaussian'
            # this will output the mean vector, the variance is fixed to the identity
            self.dist_net = nn.Sequential(
                nn.Linear(base_layers[-1], action_dim),
                nn.Tanh()
            )

        # construct the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):

        # forward the base net
        y = self.base_net(x)

        # forward the dist net and return the appropriate distribution
        if self.out_dist == 'concrete':
            action_probs = self.dist_net(y)
            action_dist = Concrete(action_probs, self.action_table)
        else:
            action_mu = self.dist_net(y)
            action_dist = Gaussian(action_mu, self.std * torch.ones_like(action_mu).to(action_mu.device))

        return action_dist


class DecisionNetwork(nn.Module):
    """
    A decision network base class for active sensing
    """

    def __init__(self, input_size, layers, num_classes):
        super().__init__()

        # feedforward layer
        self.ff = create_ff_network([input_size] + layers + [num_classes], h_activation='relu',
                                    out_activation='softmax')

    def forward(self, x):
        return self.ff(x)


class RNNDecisionNetwork(DecisionNetwork):

    def __init__(self, input_size, layers, num_classes, hidden_size, lr=0.001):

        # lstm first
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        # create super instance l
        super().__init__(hidden_size, layers, num_classes)

        # initialize the rnn state
        self.rnn_state = None

        # initialize the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):

        # if necessary, adjust dimensions to match conventions
        if len(x.shape) == 1:  # one data point, seq_len=1, batch_size=1
            x = x.reshape(1, 1, *x.shape)
        elif len(x.shape) == 2:  # batch of points, seq_len=1, batch_size is first dimension
            x = x.unsqueeze(1)

        # initial state of the RNN
        h_init = (torch.zeros(self.lstm.num_layers, x.shape[0], self.lstm.hidden_size,
                              device=x.device,
                              requires_grad=False),
                  torch.zeros(self.lstm.num_layers, x.shape[0], self.lstm.hidden_size,
                              device=x.device,
                              requires_grad=False))

        # if not in training mode and there's only one element in the sequence, use the current rnn state
        if not self.training and (x.shape[1] == 1):
            if self.rnn_state is None:
                self.rnn_state = h_init
            else:
                h_init = self.rnn_state

        h_out, (hn, cn) = self.lstm(x, h_init)

        # update the state
        if not self.training:
            self.rnn_state = (hn, cn)

        # return the decision distribution 
        return super().forward(h_out)

    def reset_rnn_state(self):
        self.rnn_state = None


class FFDecisionNetwork(DecisionNetwork):

    def __init__(self, *args, lr=0.001):

        super().__init__(*args)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):

        # if necessary, adjust dimensions to match conventions
        if len(x.shape) == 1:  # one data point, seq_len=1, batch_size=1
            x = x.reshape(1, 1, *x.shape)
        elif len(x.shape) == 2:  # batch of points, seq_len=1, batch_size is first dimension
            x = x.unsqueeze(1)

        return super().forward(x)


class GaussianMDN(nn.Module):
    """
    A gaussian mixture density network
    """

    def __init__(self, input_dim, output_dim, n_gaussians, layers=None):
        """
        input_dim (int): dimensionality of the input
        output_dim (int): dimensionality of the output
        n_gaussians (int): number of gaussians in the output mixture
        layers (list): a list of ints indicating the number of neurons in each hidden layer
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gaussians = n_gaussians

        if layers is None:
            layers = []

        pi_layers = [input_dim] + layers + [n_gaussians]
        mu_sigma_layers = [input_dim] + layers + [output_dim * n_gaussians]

        # construct the networks
        self.pi = create_ff_network(pi_layers, h_activation='tanh', out_activation='softmax')
        self.mu = create_ff_network(mu_sigma_layers)
        self.sigma = create_ff_network(mu_sigma_layers)

    def forward(self, x):
        """
        forward method - returns a gaussian mixture distribution
        """

        pi = self.pi(x)
        new_shape = pi.shape + (self.output_dim,)
        sigma = torch.exp(self.sigma(x)).view(*new_shape)
        mu = self.mu(x).view(*new_shape)

        return GaussianMixture(pi, sigma, mu)
