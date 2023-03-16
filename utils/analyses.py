from sklearn.feature_selection import mutual_info_classif
import numpy as np
import torch
from agents.active_sensor import BayesianActiveSensor


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























