import math
import numpy as np
from BAS import ActionGrid, score_action
import torch
from nets import ActionNetwork


class ActionStrategy:
    """
    an abstract class for action selection strategy with Bayesian Action Selection
    """

    def __init__(self, perception_model, action_grid_size):
        self.perception_model = perception_model
        self.action_grid = ActionGrid(action_grid_size)

    def quantize_action(self, action):
        """
        maps a continuous action to the closest action in the (discrete) action grid
        :param action: (Bx2) batch of actions to be quantized
        :return:
        """
        batch_size = action.shape[0]
        num_actions = self.action_grid.num_actions

        actions_rep = action.unsqueeze(1).repeat_interleave(num_actions, dim=1)
        action_table_full = self.action_grid.table.unsqueeze(0).repeat_interleave(batch_size,
                                                                                  dim=0).to(action.device)
        # compute the mean square distances
        msds = ((actions_rep - action_table_full) ** 2).sum(dim=-1)
        inds = torch.argmin(msds, dim=-1)

        return self.action_grid.table.to(action.device)[inds]

    def train(self, mode: bool = True):     # relevant only for trainable strategies
        pass

    def select_action(self, states, actions):  # needs to be implemented by subclasses
        raise NotImplementedError

    def state_dict(self):     # for checkpointing purposes
        raise NotImplementedError


class ActionNetworkStrategy(ActionStrategy):
    """
    BAS strategy with a trainable action network
    """

    def __init__(self, *args, layers=None, lr=0.001, out_dist='gaussian'):

        super().__init__(*args)

        # construct the action network
        self.action_net = ActionNetwork(input_dim=self.perception_model.s_dim,
                                        action_dim=self.perception_model.action_dim,
                                        layers=layers,
                                        out_dist=out_dist,
                                        num_actions=self.action_grid.num_actions,
                                        action_table=self.action_grid.table,
                                        lr=lr).to(self.perception_model.device)

        # strategy is initialized into training mode
        self._training = True

    def train(self, mode: bool):
        """ puts the strategy in either training or evaluation mode """
        self._training = mode
        self.action_net = self.action_net.train(mode)

    def _select_action(self, states, actions):
        """ helper for select_action below """

        # reset the perception model
        self.perception_model.reset_rnn_states()
        # get the input to the action network
        action_input = self.perception_model(states, actions)[-1].mu.detach()
        # get the action distribution
        action_dist = self.action_net.forward(action_input)

        # select the appropriate action from the distribution
        if self.action_net.out_dist == 'concrete':
            action = action_dist.sample()
        else:
            # in the gaussian case, the continuous action is the mean, and then it's quantized to one of
            # the actions on the grid
            cont_action = action_dist.mu
            discrete_action = self.quantize_action(cont_action).to(actions.device)

            # copy the gradient to the discrete actions
            action = (discrete_action - cont_action).detach() + cont_action

        return action

    def select_action(self, states, actions):
        """ select an action given states and actions collected so far """

        if self._training:
            # select action
            action = self._select_action(states, actions)

            # # action evaluation
            # reset the perception model
            self.perception_model.reset_rnn_states()
            loss = torch.sum(-score_action(self.perception_model,
                                           states, actions,
                                           action, n_samples=1))

            # step the optimizer
            self.action_net.optimizer.zero_grad()
            loss.backward()
            self.action_net.optimizer.step()

            return action.detach()

        return self._select_action(states, actions).detach()

    def state_dict(self):
        return self.action_net.state_dict()


class DirectEvaluationStrategy(ActionStrategy):
    """ selects action by directly evaluating the BAS score for every action on the grid """

    def __init__(self, *args, entropy_samples=20, eval_frac=1.0):
        """

        :param args: arguments to the super class
        :param entropy_samples: (int) number of MC samples to use for approximating the info gain score
        :param eval_frac: (float) fraction of actions to evaluate. if less than 1, will randomly select
                                  floor(eval_frac * num_actions) actions and evaluate them
        """
        super().__init__(*args)

        self.entropy_samples = entropy_samples
        self.eval_frac = eval_frac

    def select_action(self, states, actions):

        num_actions = self.action_grid.num_actions
        batch_size = states.shape[0]

        # get all candidate actions
        action_table = self.action_grid.table.to(self.perception_model.device)
        candidate_actions = action_table.unsqueeze(1).repeat_interleave(batch_size, dim=1)

        # tensor that will hold the scores
        scores = torch.zeros((batch_size, num_actions)) - torch.inf

        # find the actions near the inferred global location
        num_eval = math.floor(self.eval_frac * num_actions)

        # loop through the actions and compute the scores
        rnd_sample = np.random.choice(np.arange(num_actions), size=(num_eval,), replace=False)
        for a in rnd_sample:
            self.perception_model.reset_rnn_states()  # reset the perception model
            scores[:, a] = score_action(self.perception_model,
                                        states, actions, candidate_actions[a],
                                        n_samples=self.entropy_samples)

        # get the actions with the highest scores
        best_action_inds = torch.argmax(scores, dim=-1)
        best_acts = torch.zeros((batch_size, 2))
        for i in range(batch_size):
            best_acts[i, :] = candidate_actions[best_action_inds[i], i, :]

        return best_acts

    def state_dict(self):
        return "DirectEvaluationStrategy"


class RandomActionStrategy(ActionStrategy):
    """ selects a random action from the action grid """
    def select_action(self, states, actions=None):
        action_inds = np.random.randint(0, high=self.action_grid.num_actions, size=(states.shape[0],))
        return torch.tensor(self.action_grid.get_action(action_inds)).float().to(states.device)

    def state_dict(self):
        return "RandomActionStrategy"



