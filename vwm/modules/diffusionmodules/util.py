"""
Partially adopted from
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
and
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
and
https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
"""

import math
from typing import Iterable

import torch
import torch.fft as fft  # differentiable
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def fourier_filter(x, scale, d_s=0.25):
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    for h in range(H):
        for w in range(W):
            d_square = (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
            if d_square <= 2 * d_s:
                mask[..., h, w] = scale

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    x_filtered = x_filtered.type(dtype)
    return x_filtered


def fourier_filter_3d(x, scale, num_frames, d_s=0.25, d_t=0.25):
    dtype = x.dtype
    x = x.type(torch.float32)
    x_ = rearrange(x, "(b t) c h w -> b c t h w", t=num_frames)

    # FFT
    x_freq = fft.fftn(x_, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))

    B, C, T, H, W = x_freq.shape
    mask = torch.ones((B, C, T, H, W)).cuda()

    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (d_s / d_t * (2 * t / T - 1)) ** 2 + (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
                if d_square <= 2 * d_s:
                    mask[..., t, h, w] = scale

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-3, -2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-3, -2, -1)).real

    x_filtered = rearrange(x_filtered, "b c t h w -> (b t) c h w")
    x_filtered = x_filtered.type(dtype)
    return x_filtered


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
    if schedule == "linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
        return betas.numpy()
    if schedule == "scaled_linear":
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        return betas.numpy()
    else:
        raise NotImplementedError(f"Unknown schedule: {schedule}")


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """

    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled()
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_params = [x.requires_grad_(True) for x in ctx.input_params]
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # fixes a bug where the first op in run_function modifies the Tensor storage in place,
            # which is not allowed for detach()'d Tensors
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.

    :return: an [N x dim] Tensor of positional embeddings.
    """

    if repeat_only:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    else:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """

    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """

    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """

    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.

    :return: nn.Module for normalization.
    """

    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class CausalConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

        # make causal padding
        assert isinstance(kernel_size, Iterable) and len(kernel_size) == 3 and kernel_size[-1] == kernel_size[-2]
        temporal_padding = [kernel_size[0] - 1, 0]  # causal padding on temporal dimension
        spatial_padding = [kernel_size[-1] // 2] * 4  # keep padding on spatial dimension
        causal_padding = tuple(spatial_padding + temporal_padding)  # starting from the last dimension
        self.causal_padding = causal_padding

    def forward(self, x):
        x = F.pad(x, self.causal_padding)
        x = super().forward(x)
        return x


def conv_nd(dims, *args, causal=False, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """

    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        if causal:
            return CausalConv3d(*args, **kwargs)
        else:
            return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """

    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """

    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
            self,
            alpha: float,
            merge_strategy: str,
            rearrange_pattern: str
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert merge_strategy in self.strategies, f"merge_strategy needs to be in {self.strategies}"

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_alpha(self) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            alpha = rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1")
            alpha = rearrange(alpha, self.rearrange_pattern)
        else:
            raise NotImplementedError
        return alpha

    def forward(
            self,
            x_spatial: torch.Tensor,
            x_temporal: torch.Tensor
    ) -> torch.Tensor:
        alpha = self.get_alpha()
        x = alpha.to(x_spatial.dtype) * x_spatial + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        return x
