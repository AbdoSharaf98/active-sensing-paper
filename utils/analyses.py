from sklearn.feature_selection import mutual_info_classif
import numpy as np
import torch
from agents.active_sensor import BayesianActiveSensor
import matplotlib.pyplot as plt
from utils.plotting import plot_patches
from BAS import ActionGrid


def mutual_info_map(dataset):
    """
    estimates the mutual information maps for each class. That is, for each class, the mutual information between
    different locations on images (belonging to that class) and class label is estimated. This essentially measures
    how much individual parts of an image tells us about the class of the image
    :param dataset:
    :return:
    """
    # get the images and their labels
    b, w, h = dataset.data.shape
    x, y = dataset.data.flatten(start_dim=1), dataset.targets

    # starting getting the mutual information maps for the different images
    mi_map = np.zeros((len(dataset.classes), w, h))

    for cidx in dataset.class_to_idx.values():
        mi_map[cidx] = mutual_info_classif(x.numpy(), (y == cidx).int()).reshape(w, h)

    return mi_map


def get_model_actions_by_class(active_sensor: BayesianActiveSensor):
    test_batch = (active_sensor.env.dataset[1].data,
                  active_sensor.env.dataset[1].targets)

    states, actions = active_sensor.sensing_loop(
        testing=True,
        with_batch=test_batch
    )

    # group by class
    actions_by_class = dict.fromkeys(active_sensor.env.dataset[1].class_to_idx.values())
    for cidx in actions_by_class.keys():
        # get only the active fixations performed when this class was presented
        actions_by_class[cidx] = actions[test_batch[1] == cidx, 1:, :]

    return actions_by_class


def latent_space_pca(bas_agent, random_agent, env):
    all_states, all_locs, labels = [], [], []
    all_states_rnd, all_locs_rnd, labels_rnd = [], [], []

    env.train_iterator = iter(env.train_loader)
    for _ in range(1):
        batch = next(env.train_iterator)
        states, locs = bas_agent.sensing_loop(testing=True, with_batch=batch)
        all_states.append(states)
        all_locs.append(locs)

        states_rnd, locs_rnd = random_agent.sensing_loop(testing=True, with_batch=batch, random_action=True)
        all_states_rnd.append(states_rnd)
        all_locs_rnd.append(locs_rnd)
        labels.append(batch[1])

    all_states = torch.cat(all_states, 0)
    all_locs = torch.cat(all_locs, 0)
    labels = torch.cat(labels, 0)

    all_states_rnd = torch.cat(all_states_rnd, 0)
    all_locs_rnd = torch.cat(all_locs_rnd, 0)

    s_dist = bas_agent.perception_model(all_states, all_locs)[-1]
    s_dist_rnd = random_agent.perception_model(all_states_rnd, all_locs_rnd)[-1]

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    pca = PCA(n_components=3)
    tsne = TSNE(n_components=3)

    mu_bas = s_dist.mu.detach().cpu().numpy()
    mu_rnd = s_dist_rnd.mu.detach().cpu().numpy()

    return pca.fit_transform(mu_bas), tsne.fit_transform(mu_bas), pca.fit_transform(mu_rnd), tsne.fit_transform(
        mu_rnd), labels


def generative_model_test(env, perception_model, obs, locs, gen_grid_size=None):
    if gen_grid_size is None:
        gen_grid_size = (7, 7)

    loc_table = ActionGrid(gen_grid_size).table
    val_recs, _, _, _, _, s_post = perception_model(obs, locs)
    s_sample = s_post.sample().repeat(len(loc_table), 1)
    gens = perception_model.vae1.decode(perception_model.vae2.decode(s_sample, loc_table).mu)
    original = plot_patches(env, obs, locs)
    rec = plot_patches(env, val_recs, locs)
    gen = plot_patches(env, gens, loc_table)

    # plotting time
    plt.imshow(original.numpy(), cmap='gray')
    plt.title('Original')
    plt.figure()
    plt.imshow(rec.numpy(), cmap='gray')
    plt.title('Reconstructed')
    plt.figure()
    plt.imshow(gen.numpy(), cmap='gray')
    plt.title('Generated')
    plt.show()
