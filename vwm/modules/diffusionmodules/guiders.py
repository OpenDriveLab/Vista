from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

import torch
from einops import rearrange, repeat

from vwm.util import append_dims, default


class Guider(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        pass

    def prepare_inputs(self, x, s, c, cond_mask, uc):
        pass


class VanillaCFG(Guider):
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        x_pred = x_u + self.scale * (x_c - x_u)
        return x_pred

    def prepare_inputs(self, x, s, c, cond_mask, uc):
        c_out = dict()
        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out, torch.cat([cond_mask] * 2)


class IdentityGuider(Guider):
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x

    def prepare_inputs(self, x, s, c, cond_mask, uc):
        c_out = dict()
        for k in c:
            c_out[k] = c[k]
        return x, s, c_out, cond_mask


class LinearPredictionGuider(Guider):
    def __init__(
            self,
            num_frames: int = 25,
            max_scale: float = 2.5,
            min_scale: float = 1.0,
            additional_cond_keys: Optional[Union[List[str], str]] = None
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_frames = num_frames
        self.scale = torch.linspace(min_scale, max_scale, num_frames).unsqueeze(0)

        additional_cond_keys = default(additional_cond_keys, list())
        if isinstance(additional_cond_keys, str):
            additional_cond_keys = [additional_cond_keys]
        self.additional_cond_keys = additional_cond_keys

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        x_u = rearrange(x_u, "(b t) ... -> b t ...", t=self.num_frames)
        x_c = rearrange(x_c, "(b t) ... -> b t ...", t=self.num_frames)
        scale = repeat(self.scale, "1 t -> b t", b=x_u.shape[0])
        scale = append_dims(scale, x_u.ndim).to(x_u.device)
        return rearrange(x_u + scale * (x_c - x_u), "b t ... -> (b t) ...")

    def prepare_inputs(self, x, s, c, cond_mask, uc):
        c_out = dict()
        for k in c:
            if k in ["vector", "crossattn", "concat"] + self.additional_cond_keys:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out, torch.cat([cond_mask] * 2)


class TrianglePredictionGuider(LinearPredictionGuider):
    def __init__(
            self,
            num_frames: int = 25,
            max_scale: float = 2.5,
            min_scale: float = 1.0,
            period: float = 1.0,
            period_fusing: Literal["mean", "multiply", "max"] = "max",
            additional_cond_keys: Optional[Union[List[str], str]] = None
    ):
        super().__init__(num_frames, max_scale, min_scale, additional_cond_keys)
        values = torch.linspace(0, 1, num_frames)
        # constructs a triangle wave
        if isinstance(period, float):
            period = [period]

        scales = list()
        for p in period:
            scales.append(self.triangle_wave(values, p))

        if period_fusing == "mean":
            scale = sum(scales) / len(period)
        elif period_fusing == "multiply":
            scale = torch.prod(torch.stack(scales), dim=0)
        elif period_fusing == "max":
            scale = torch.max(torch.stack(scales), dim=0).values
        else:
            raise NotImplementedError
        self.scale = (scale * (max_scale - min_scale) + min_scale).unsqueeze(0)

    def triangle_wave(self, values: torch.Tensor, period) -> torch.Tensor:
        return 2 * (values / period - torch.floor(values / period + 0.5)).abs()
