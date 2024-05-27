from typing import Callable, Iterable, Union

import torch
from einops import rearrange

from vwm.modules.diffusionmodules.model import Decoder, ResnetBlock
from vwm.modules.diffusionmodules.openaimodel import ResBlock
from vwm.util import partialclass


class VideoResBlock(ResnetBlock):
    def __init__(
            self,
            out_channels,
            *args,
            dropout=0.0,
            video_kernel_size=3,
            alpha=0.0,
            merge_strategy="learned",
            **kwargs
    ):
        super().__init__(out_channels=out_channels, dropout=dropout, *args, **kwargs)
        if video_kernel_size is None:
            video_kernel_size = [3, 1, 1]
        self.time_stack = ResBlock(
            channels=out_channels,
            emb_channels=0,
            dropout=dropout,
            dims=3,
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=False,
            skip_t_emb=True
        )

        self.merge_strategy = merge_strategy
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_alpha(self):
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError

    def forward(self, x, temb, skip_video=False, timesteps=None):
        if timesteps is None:
            timesteps = self.timesteps

        x = super().forward(x, temb)

        if not skip_video:
            x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            x = self.time_stack(x, temb)

            alpha = self.get_alpha()
            x = alpha * x + (1.0 - alpha) * x_mix

            x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class AE3DConv(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, video_kernel_size=3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        if isinstance(video_kernel_size, Iterable):
            padding = [int(k // 2) for k in video_kernel_size]
        else:
            padding = int(video_kernel_size // 2)

        self.time_mix_conv = torch.nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=video_kernel_size,
            padding=padding
        )

    def forward(self, input, timesteps, skip_video=False):
        x = super().forward(input)
        if skip_video:
            return x
        else:
            x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
            x = self.time_mix_conv(x)
            return rearrange(x, "b c t h w -> (b t) c h w")


class Conv2DWrapper(torch.nn.Conv2d):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class VideoDecoder(Decoder):
    available_time_modes = ["all", "conv-only", "attn-only"]

    def __init__(
            self,
            *args,
            video_kernel_size: Union[int, list] = 3,
            alpha: float = 0.0,
            merge_strategy: str = "learned",
            time_mode: str = "conv-only",
            **kwargs
    ):
        self.video_kernel_size = video_kernel_size
        self.alpha = alpha
        self.merge_strategy = merge_strategy
        self.time_mode = time_mode
        assert (
                self.time_mode in self.available_time_modes
        ), f"time_mode parameter has to be in {self.available_time_modes}"
        super().__init__(*args, **kwargs)

    def get_last_layer(self, skip_time_mix=False, **kwargs):
        if self.time_mode == "attn-only":
            raise NotImplementedError
        else:
            return (
                self.conv_out.time_mix_conv.weight
                if not skip_time_mix
                else self.conv_out.weight
            )

    def _make_conv(self) -> Callable:
        if self.time_mode != "attn-only":
            return partialclass(AE3DConv, video_kernel_size=self.video_kernel_size)
        else:
            return Conv2DWrapper

    def _make_resblock(self) -> Callable:
        if self.time_mode not in ["attn-only", "only-last-conv"]:
            return partialclass(
                VideoResBlock,
                video_kernel_size=self.video_kernel_size,
                alpha=self.alpha,
                merge_strategy=self.merge_strategy
            )
        else:
            return super()._make_resblock()
