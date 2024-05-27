import torch
from einops import repeat

from vwm.util import default, instantiate_from_config


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2, num_frames=25):
        self.p_mean = p_mean
        self.p_std = p_std
        self.num_frames = num_frames

    def __call__(self, n_samples, rand=None):
        bs = n_samples // self.num_frames
        rand_init = torch.randn((bs,))[..., None]
        rand_init = repeat(rand_init, "b 1 -> (b t)", t=self.num_frames)
        rand = default(rand, rand_init)
        log_sigma = self.p_mean + self.p_std * rand
        return log_sigma.exp()


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True, num_frames=25):
        self.num_idx = num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        self.num_frames = num_frames

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        bs = n_samples // self.num_frames
        rand_init = torch.randint(0, self.num_idx, (bs,))[..., None]
        rand_init = repeat(rand_init, "b 1 -> (b t)", t=self.num_frames)
        idx = default(rand, rand_init)
        return self.idx_to_sigma(idx)
