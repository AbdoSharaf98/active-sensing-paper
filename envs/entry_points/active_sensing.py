import warnings
from abc import ABC
import gym
from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
from gym.spaces import Box
import torch.nn.functional as F


class ActiveSensingEnv(gym.Env, ABC):

    def __init__(self, n_samples: Optional[int],
                 batch_size: Optional[int],
                 sample_dim: Optional[int],
                 dataset: Dataset,
                 num_classes: int,
                 num_foveated_patches: int = 1,
                 fovea_scale=2,
                 min_pixel_value=-np.inf,
                 max_pixel_value=np.inf,
                 num_channels=1,
                 valid_frac=0.1,
                 val_batch_size=None,
                 num_workers=4):
        """
        required params:
        - n_samples: (int)
        - batch_size: (int)
        - sample_dims: (tuple of ints)
        - data_loader: (torch DataLoader)
        """
        super(ActiveSensingEnv, self).__init__()

        self.n_samples = n_samples
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.sample_dim = sample_dim
        self.num_classes = num_classes
        self.num_foveated_patches = num_foveated_patches
        self.fovea_scale = fovea_scale
        self.num_channels = num_channels

        if self.batch_size == -1:  # -1 indicates use all training data
            self.batch_size = len(dataset) - int(np.floor(valid_frac * len(dataset)))

        self.dataset = dataset
        self._set_data_loaders(valid_frac, num_workers)
        self.train_iterator = iter(self.train_loader)
        self.valid_iterator = iter(self.valid_loader)

        # define the observation space
        obs_shape = (self.batch_size, self.num_foveated_patches * num_channels * sample_dim ** 2 + 2)
        self.observation_space = Box(low=min_pixel_value,
                                     high=max_pixel_value,
                                     shape=obs_shape)

        # define the action space (normalized coordinates)
        self.action_space = Box(low=-1.0, high=1.0,
                                shape=(self.batch_size, 2))

        # variables active when the environment is run
        self.current_batch = None
        self.current_step = None
        self.done = None
        self.current_sample = None

    def _set_data_loaders(self, valid_frac=0.1, num_workers=4):

        dsize = len(self.dataset)
        indices = list(range(dsize))
        split = int(np.floor(valid_frac * dsize))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        if self.val_batch_size is None:
            self.val_batch_size = len(valid_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=num_workers
        )

        self.valid_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.val_batch_size,
            sampler=valid_sampler,
            num_workers=num_workers
        )

    def get_next_batch(self, validation=False):

        if not validation:
            try:
                batch = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                batch = next(self.train_iterator)
        else:
            try:
                batch = next(self.valid_iterator)
            except StopIteration:
                self.valid_iterator = iter(self.valid_loader)
                batch = next(self.valid_iterator)

        return batch

    @staticmethod
    def get_patch(images, locs, sz, flatten=True):
        b, c, h, w = images.shape

        start_idx = (0.5 * ((locs + 1.0) * h)).long()
        end_idx = start_idx + sz

        # pad with zeros to avoid exceeding image boundaries
        images_padded = F.pad(images, (sz // 2, sz // 2, sz // 2, sz // 2))

        # loop through and extract
        samples = []
        for i in range(b):
            samples.append(images_padded[i, :, start_idx[i, 1]:end_idx[i, 1], start_idx[i, 0]:end_idx[i, 0]])

        return torch.flatten(torch.stack(samples), start_dim=-3, end_dim=-1) if flatten else torch.stack(samples)

    def sense(self, images, locs):
        # non-foveated sensing
        if self.num_foveated_patches == 1:
            return self.get_patch(images, locs, self.sample_dim)

        # foveated sensing
        patches = []
        sz = self.sample_dim
        # extract patches
        for i in range(self.num_foveated_patches):
            patches.append(self.get_patch(images, locs, sz, flatten=False))
            sz = int(self.fovea_scale * sz)

        # resize the patches to sample_dim
        for i in range(1, len(patches)):
            k = patches[i].shape[-1] // self.sample_dim
            patches[i] = F.avg_pool2d(patches[i], k)

        # concatenate, flatten and return
        patches = torch.cat(patches, 1)

        return patches.view(patches.shape[0], -1)

    def reset(self, validation=False, with_batch=None):

        # get the samples
        self.current_batch = self.get_next_batch(validation)

        bsz = len(self.current_batch[0])
        loc = torch.FloatTensor(bsz, 2).uniform_(-1, 1)

        samples = self.sense(self.current_batch[0], loc).numpy()
        samples = np.concatenate((samples, loc), axis=-1)
        self.current_sample = samples

        self.current_step = 0
        self.done = False

        return samples

    def step(self, action: Optional[np.ndarray]):

        # handle the case where the environment is being stepped after it's done
        if self.done:
            raise Exception('You are stepping the environment after it is done.')

        if (self.current_batch is None) or (self.current_step is None):
            warnings.warn('You must reset the environment before stepping it.')
            raise

        done = False
        if self.current_step == self.n_samples:
            # make sure a decision is provided if the allowable number of steps has been taken
            if len(action.shape) > 1:
                warnings.warn(f"Expected a decision after {self.n_samples} steps but none was given. The "
                              f"reward will be calculated as if all decision were wrong.")

                reward = 0.0
                info = {'remaining_steps': self.n_samples - self.current_step}
            else:
                outcomes = (action == self.current_batch[1].cpu().numpy())
                reward = np.sum(outcomes) / len(outcomes)

                # episode is done only when an actual decision has been made
                done = True
                self.done = True

                # additional possibly helpful info?
                info = {'remaining_steps': self.n_samples - self.current_step,
                        'outcomes': torch.tensor(outcomes).float(),
                        'true_labels': self.current_batch[1],
                        'one_hot_labels': F.one_hot(self.current_batch[1], self.num_classes)}

            return self.current_sample, reward, done, info

        else:
            next_sample = self.sense(self.current_batch[0], torch.tensor(action)).detach().numpy()
            next_sample = np.concatenate((next_sample, action), axis=-1)
            self.current_step += 1
            self.current_sample = next_sample

            # additional possibly helpful info?
            info = {'remaining_steps': self.n_samples - self.current_step}

            return next_sample, 0.0, done, info

    def get_fov_from_obs(self, obs):

        # remove locations and convert to tensor
        obs = torch.tensor(obs[..., :-2])

        # reshape
        batch_shape = obs.shape[:-1]
        new_sample_shape = (self.num_foveated_patches, self.sample_dim, self.sample_dim)
        obs = obs.reshape(batch_shape + new_sample_shape)

        # create the base image
        max_sz = self.sample_dim * (self.fovea_scale ** (self.num_foveated_patches - 1))
        base = torch.zeros(batch_shape + (max_sz, max_sz))

        # get individual patches and interpolate
        patches = [obs[..., 0, :, :]]
        for i in range(1, self.num_foveated_patches):
            patch = torch.nn.functional.interpolate(obs[..., [i], :, :], scale_factor=self.fovea_scale ** i)
            patches.append(patch.squeeze())

        # start adding the patches to the base image in reverse
        for j in range(self.num_foveated_patches - 1, -1, -1):
            start = int(max_sz / 2) - int(self.sample_dim / 2) * (self.fovea_scale ** j)
            end = start + self.sample_dim * (self.fovea_scale ** j)
            base[..., start:end, start:end] = patches[j]

        return base

    def relinquish_views(self):
        self.current_step = self.n_samples
