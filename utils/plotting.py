import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_patches(env, locs, states=None, patches=None, h=28, w=28):
    """ states and locs should have shapes (N_patches x obs_dim) and (N_patches x action_dim) respectively """

    # full patch size
    max_sz = env.sample_dim * (env.fovea_scale ** (env.num_foveated_patches - 1))

    # create the base image
    base = torch.ones((h + max_sz, w + max_sz))

    # convert normalized locations to image coordinates
    start_idx = (0.5 * ((locs + 1) * h)).long()
    end_idx = start_idx + max_sz

    # get the patches
    if patches is None:
        assert states is not None, "Either patches or states must be provided"
        patches = env.get_fov_from_obs(states)  # patches is (N_patches x max_sz x max_sz)

    # go through the patches and put them in the base image
    for p in range(patches.shape[0]):
        base[start_idx[p, 1]:end_idx[p, 1], start_idx[p, 0]:end_idx[p, 0]] = patches[p]

    return base


def plot_fixation_trajectory(image, locs, patch_dim, w=28, h=28):
    # convert the locations to image coordinates
    top_left = (0.5 * ((locs + 1) * h)).long()
    centers = top_left + (patch_dim // 2) * torch.ones_like(top_left)
    centers = centers.cpu().numpy()
    # plot the image
    plt.figure()
    plt.imshow(image, cmap='gray')
    # draw the arrows
    for b in range(locs.shape[0] - 1):
        x, y = centers[b][0], centers[b][1]
        dx, dy = centers[b + 1][0] - x, centers[b + 1][1] - y
        hw = 0 if b < locs.shape[0] - 2 else 2.0
        hl = 0 if b < locs.shape[0] - 2 else 2.5
        if b < locs.shape[0] - 2:
            style = f"-"
            kw = dict(arrowstyle=style, color="firebrick", linewidth=5.5)
        else:
            style = f"simple, tail_width=5.5, head_width=9.5, head_length=10.0"
            kw = dict(arrowstyle=style, color="firebrick")
        arrow = patches.FancyArrowPatch((x, y), (x + dx, y + dy), **kw)
        plt.scatter(x, y, s=60, c='firebrick')
        plt.gca().add_patch(arrow)
