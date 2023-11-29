import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


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


def plot_missing_info(bas_data, random_data, boltz_data, delta=10, labels=None):
    plt.style.use(['science', 'ieee'])
    plt.rcParams['text.usetex'] = False
    plt.rcParams.update({'font.size': 5})

    if labels is None:
        labels = ['BAS', 'Random', 'Boltzmann']

    bas_avg, bas_sem = np.mean(bas_data, 0), np.std(bas_data, 0) / np.sqrt(bas_data.shape[0])
    rnd_avg, rnd_sem = np.mean(random_data, 0), np.std(random_data, 0) / np.sqrt(random_data.shape[0])
    boltz_avg, boltz_sem = np.mean(boltz_data, 0), np.std(boltz_data, 0) / np.sqrt(boltz_data.shape[0])

    plt.plot(np.arange(len(bas_avg)) * delta, bas_avg, 'firebrick', label=labels[0])
    plt.plot(np.arange(len(rnd_avg)) * delta, rnd_avg, '-', color='navy', label=labels[1])
    plt.plot(np.arange(len(boltz_avg)) * delta, boltz_avg, '-', color='forestgreen', label=labels[2])
    plt.fill_between(np.arange(len(rnd_avg)) * delta, rnd_avg - rnd_sem, rnd_avg + rnd_sem, alpha=0.4, color='navy')
    plt.fill_between(np.arange(len(bas_avg)) * delta, bas_avg - bas_sem, bas_avg + bas_sem, alpha=0.4,
                     color='firebrick')
    plt.fill_between(np.arange(len(boltz_avg)) * delta, boltz_avg - boltz_sem, boltz_avg + boltz_sem, alpha=0.4,
                     color='forestgreen')
    plt.xlim([0, len(bas_avg) * delta])
    plt.xlabel('Step #')
    plt.ylabel('Missing Information (bits)')
    plt.ylim([0, max(max(np.max(bas_avg + bas_sem), np.max(rnd_avg + rnd_sem)), np.max(boltz_avg + boltz_sem)) + 20])
    plt.legend()


def prob_diff_heatmap(true_dist, learned_dist_bas, learned_dist_rand, science_style=False):
    if science_style:
        plt.style.use(['science', 'ieee'])
        plt.rcParams['text.usetex'] = False
        plt.rcParams.update({'font.size': 4})

    diff_dist_bas = np.abs(true_dist - learned_dist_bas)
    diff_dist_rand = np.abs(true_dist - learned_dist_rand)

    fig, axs = plt.subplots(2, 4, sharey=True, sharex=True)
    for a in range(true_dist.shape[1]):
        dist_im_bas = diff_dist_bas[:, a, :]
        dist_im_rand = diff_dist_rand[:, a, :]
        dist_im_bas = (dist_im_bas - np.min(diff_dist_bas[:])) / (np.max(diff_dist_bas[:]) - np.min(diff_dist_bas[:]))
        dist_im_rand = (dist_im_rand - np.min(diff_dist_rand[:])) / (
                np.max(diff_dist_rand[:]) - np.min(diff_dist_rand[:]))

        axs[0][a].imshow(dist_im_bas, cmap='Blues', interpolation='none', vmin=0, vmax=1)
        axs[0][a].set_title(f"{np.sum(dist_im_bas):0.2f}")  # axs[0][a].set_title(f"a = {a+1}")
        axs[1][a].set_title(f"{np.sum(dist_im_rand):0.2f}")
        im = axs[1][a].imshow(dist_im_rand, cmap='Blues', interpolation='none', vmin=0, vmax=1)
        axs[1][a].set_xlabel('s')

    axs[0][0].set_ylabel("s'")
    axs[1][0].set_ylabel("s'")

    plt.gcf().subplots_adjust(right=0.8)
    plt.gcf().colorbar(im, cax=plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7]))


def maze_heat_map(maze_array, visit_frequency, science_style=False):
    if science_style:
        plt.style.use(['science', 'ieee'])
        plt.rcParams['text.usetex'] = False

    # convert the maze array to a float array
    maze_array[maze_array == 'w'] = 0.1
    maze_array[maze_array == '.'] = 0.001
    maze_array = np.array(maze_array, dtype=np.float)

    # populate the maze array
    for st in range(len(visit_frequency)):
        maze_array[maze_array == st] = visit_frequency[st] / np.max(visit_frequency)

    plt.figure()
    plt.imshow(maze_array, cmap='hot', interpolation='kaiser')
    plt.colorbar(label='Visitation Frequency')

    return maze_array
