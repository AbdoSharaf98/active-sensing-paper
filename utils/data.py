import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader, SubsetRandomSampler
from pytorch_lightning import LightningDataModule


class ActiveSensingDataset(TensorDataset):

    def __init__(self, data_dir=None, data_dict=None):

        if (data_dir is None) and (data_dict is None):
            raise Exception("either data_dir or data_dict mus be provided")

        if data_dict is not None:
            self.data_dict = data_dict
        else:
            self.data_dict = torch.load(data_dir)

        vals = (data_dict['obs'], data_dict['locations'])

        super(ActiveSensingDataset, self).__init__(*vals)


class GymDataModule(LightningDataModule):
    """
    Lightning data module for working Gym environment data
    """

    def __init__(self, data_dir=None, data_dict=None, batch_size=32, train_size=None, test_size=None,
                 val_frac=0.1, dataset_class=ActiveSensingDataset):
        """
        Inputs:
            - data_dir (str): directory of the data in torch-save format
            - batch_size (int): number of sequences in one batch
            - train_size (int):
            - test_size (int):
            - val_frac (float): fraction of training data to use as a validation set
        """

        super(GymDataModule, self).__init__()

        if (data_dir is None) and (data_dict is None):
            raise Exception("either data_dir or data_dict mus be provided")

        # get data set
        if data_dict is not None:
            self.full_data = dataset_class(data_dict=data_dict)
        else:
            self.full_data = dataset_class(data_dir=data_dir)

        self.batch_size = batch_size
        data_sz = len(self.full_data)

        # figure out the training and test sizes
        if (train_size is None) and (test_size is None):
            train_size = np.ceil(0.7 * data_sz)
            test_size = data_sz - train_size
        elif train_size is None:
            train_size = data_sz - test_size
        elif test_size is None:
            test_size = data_sz - train_size

        # validation set
        self.val_size = int(val_frac * train_size)
        self.train_size = train_size - self.val_size
        self.test_size = test_size

        # initialize the training and testing sets
        self.train_data, self.val_data, self.test_data = random_split(self.full_data, [self.train_size,
                                                                                       self.val_size,
                                                                                       self.test_size])

    def train_dataloader(self):

        return DataLoader(self.train_data, batch_size=self.batch_size)

    def test_dataloader(self):

        return DataLoader(self.test_data, batch_size=self.batch_size)

    def val_dataloader(self):

        return DataLoader(self.val_data, batch_size=self.batch_size)


def get_mnist_data(data_dir=None,
                   transform=True,
                   data_version='centered',
                   translate_size=60):
    class Translate(object):
        def __init__(self, output_size, vmin):
            super(Translate, self).__init__()
            self.output_size = output_size
            self.vmin = vmin

        def __call__(self, x):
            C, H, W = x.size()
            x_t = -torch.ones(C, self.output_size, self.output_size)  # background of MNIST is mapped to -1
            torch.fill_(x_t, self.vmin)
            loch = np.random.randint(0, 33)
            locw = np.random.randint(0, 33)
            x_t[:, loch:loch + H, locw:locw + W] = x
            return x_t

    if data_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_dir, 'data', 'mnist')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    data_transformer = transforms.Compose([transforms.ToTensor(), normalize])

    if transform:
        dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=data_transformer)
    else:
        dataset = datasets.MNIST(root=data_dir, train=True, download=True)

    t = torch.cat([x for (x, y) in dataset])
    v_min = torch.min(t)

    if data_version == 'translated':
        translator = Translate(translate_size, v_min)
        data_transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            translator
        ])
        dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=data_transformer)

    return dataset


def train_test_split(data, train_size=None, test_size=None):
    data_sz = data['obs'].shape[0]

    # if train and test sizes are not given, split with 0.7-0.3 ratio by default
    if (train_size is None) and (test_size is None):
        train_size = np.ceil(0.7 * data_sz)
        test_size = data_sz - train_size
    elif train_size is None:
        train_size = data_sz - test_size
    elif test_size is None:
        test_size = data_sz - train_size

    # shuffle and then split the data
    shuffle_inds = torch.randperm(data_sz)
    train_inds = shuffle_inds[0:train_size]
    test_inds = shuffle_inds[train_size: train_size + test_size]

    training_data = {k: v[train_inds] for k, v in data.items()}
    testing_data = {k: v[test_inds] for k, v in data.items()}

    return training_data, testing_data
