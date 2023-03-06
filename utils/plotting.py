import torch


def plot_patches(env, states, locs, h=28, w=28):
    """ states and locs should have shapes (N_patches x obs_dim) and (N_patches x action_dim) respectively """

    # full patch size
    max_sz = env.sample_dim * (env.fovea_scale ** (env.num_foveated_patches - 1))

    # create the base image
    base = torch.ones((h + max_sz, w + max_sz))

    # convert normalized locations to image coordinates
    start_idx = (0.5 * ((locs + 1) * h)).long()
    end_idx = start_idx + max_sz

    # get the fovs
    fovs = env.get_fov_from_obs(states)  # fovs is (N_patches x max_sz x max_sz)

    # go through the patches and put them in the base image
    for p in range(fovs.shape[0]):
        base[start_idx[p, 1]:end_idx[p, 1], start_idx[p, 0]:end_idx[p, 0]] = fovs[p]

    return base
