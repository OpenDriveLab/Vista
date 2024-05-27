from typing import Dict, Union

import torch
import torch.nn as nn

from vwm.util import append_dims, instantiate_from_config
from .denoiser_scaling import DenoiserScaling


class Denoiser(nn.Module):
    def __init__(self, scaling_config: Dict, num_frames: int = 25):
        super().__init__()
        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)
        self.num_frames = num_frames

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
            self,
            network: nn.Module,
            noised_input: torch.Tensor,
            sigma: torch.Tensor,
            cond: Dict,
            cond_mask: torch.Tensor
    ):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, noised_input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return (network(noised_input * c_in, c_noise, cond, cond_mask, self.num_frames) * c_out + noised_input * c_skip)


class DiscreteDenoiser(Denoiser):
    def __init__(
            self,
            scaling_config: Dict,
            num_idx: int,
            discretization_config: Dict,
            do_append_zero: bool = False,
            quantize_c_noise: bool = True,
            flip: bool = True
    ):
        super().__init__(scaling_config)
        sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
