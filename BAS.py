import numpy as np
import torch


class ActionGrid:
    """ a class for a discrete set (grid) of normalized actions with some convenience methods """

    def __init__(self, grid_size):

        # make sure grid sizes are odd so actions are equally spaced
        if grid_size[0] % 2 == 0:
            grid_size[0] -= 1
        if grid_size[1] % 2 == 0:
            grid_size[1] -= 1

        self.grid_size = grid_size
        self.num_actions = grid_size[0] * grid_size[1]

        # get the individual action coordinates
        x_vals = np.append(np.linspace(-1.0, 0, int((grid_size[0] - 1) / 2), endpoint=False),
                           np.linspace(0, 1.0, int((grid_size[0] + 1) / 2), endpoint=True))
        y_vals = np.append(np.linspace(-1.0, 0, int((grid_size[1] - 1) / 2), endpoint=False),
                           np.linspace(0, 1.0, int((grid_size[1] + 1) / 2), endpoint=True))

        # construct the grid
        self.grid = np.meshgrid(x_vals, y_vals)

        # construct the action table (not in grid format) as a torch tensor
        self.table = torch.tensor(self.get_action(np.arange(self.num_actions))).float()

    def get_action(self, action_inds):
        """
        return the action on a 2D grid corresponding to linear indices in action_inds
        Args:
            action_inds: ndarray (Bx1) or (B,) where B = batch_size
        Returns:
        """

        grid_shape = self.grid[0].shape
        x = self.grid[0][np.unravel_index(action_inds, grid_shape)]
        y = self.grid[1][np.unravel_index(action_inds, grid_shape)]

        actions = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=-1)

        return actions


def score_action(perception_model, states, actions, candidate, n_samples=1):
    """
    returns a monte carlo estimate of the uncertainty minimization (=info gain)
    score associate for candidate action.
    Tensor dims are denoted as follows:
    B = batch size, S = sequence length, D_s = state_dim, D_a = action_dim
    Args:
        perception_model: perception model with respect to which entropies are calculated
        states: (BxSxD_s) states collected so far
        actions: (BxSxD_a) actions performed so far
        candidate: (BxD_a) action to be scored
        n_samples: (int) number of MC samples to use for approximation
    Returns: H(s) - H(s|action)
    """

    B, S, D_s = states.shape
    D_a = actions.shape[-1]

    # compute the current entropy
    curr_s_dist = perception_model(states, actions)[-1]
    H_s = curr_s_dist.torch_dist.entropy()

    # use the current estimate of s and the candidate location to predict the next observation
    curr_s = curr_s_dist.mu
    z_dist = perception_model.vae2.decode(curr_s, perception_model.location_encoder(candidate))

    z_samples = z_dist.sample_n(n_samples)
    x_samples = perception_model.vae1.decode(z_samples)

    # some rearranging of dimensions for batch computation
    x_samples = x_samples.reshape((n_samples, B, 1, D_s))
    candidate_pop = candidate.reshape((1, B, 1, D_a)).repeat(n_samples, 1, 1, 1)
    states = states.reshape((1, *states.shape)).repeat(n_samples, 1, 1, 1)
    actions = actions.reshape((1, *actions.shape)).repeat(n_samples, 1, 1, 1)

    # concatenate
    states = torch.cat([states, x_samples], dim=-2).flatten(end_dim=1)
    actions = torch.cat([actions, candidate_pop], dim=-2).flatten(end_dim=1)

    # calculate the entropies of the resulting distribution and average across samples
    new_z_post, new_z_prior, post_s_dist = perception_model(states, actions)[-3:]
    H_s_post = post_s_dist.torch_dist.entropy().reshape((n_samples, B)).mean(dim=0)

    # information gain
    info_gain_score = H_s - H_s_post

    return info_gain_score
