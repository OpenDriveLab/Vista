import math
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import kornia
import numpy as np
import open_clip
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import ListConfig
from transformers import CLIPTextModel, CLIPTokenizer

from vwm.modules.diffusionmodules.openaimodel import Timestep
from vwm.util import (
    append_dims,
    autocast,
    count_params,
    default,
    disabled_train,
    expand_dims_like,
    instantiate_from_config
)


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: Union[List, ListConfig]):
        super().__init__()
        embedders = list()
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"Embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(f"Need either `input_key` or `input_keys` for embedder {embedder.__class__.__name__}")

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def forward(self, batch: Dict, force_zero_embeddings: Optional[List] = None) -> Dict:
        output = dict()
        force_zero_embeddings = default(force_zero_embeddings, list())
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and embedder.input_key is not None:
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    if embedder.input_key in batch:
                        emb_out = embedder(batch[embedder.input_key])
                    elif embedder.add_sequence_dim:  # concatenation
                        emb_dim = embedder.num_features * embedder.outdim
                        emb_out = torch.zeros((batch["cond_aug"].shape[0], 1, emb_dim), device=batch["cond_aug"].device)
                    else:  # addition
                        continue
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"Encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = (
                            expand_dims_like(
                                torch.bernoulli(
                                    (1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device)
                                ),
                                emb
                            )
                            * emb
                    )
                if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    if emb.shape[-1] == 768 and out_key == "vector":
                        output[out_key] += emb
                    else:
                        output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
                else:
                    output[out_key] = emb
        return output

    def get_unconditional_conditioning(
            self,
            batch_c: Dict,
            batch_uc: Optional[Dict] = None,
            force_cond_zero_embeddings: Optional[List[str]] = None,
            force_uc_zero_embeddings: Optional[List[str]] = None
    ):
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0

        c = self(batch_c, force_cond_zero_embeddings)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class FrozenCLIPEmbedder(AbstractEmbModel):
    """
    Uses the CLIP transformer encoder for text (from huggingface).
    """

    def __init__(
            self,
            # version="path_to/openai/clip-vit-large-patch14/pytorch_model.bin",
            version="openai/clip-vit-large-patch14",
            device="cuda",
            max_length=77,
            freeze=True,
            layer="last",
            layer_idx=None,
            always_return_pooled=False
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in ["last", "pooled", "hidden"]
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt"
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(
            input_ids=tokens,
            output_hidden_states=self.layer == "hidden"
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None]
        else:
            z = outputs.hidden_states[self.layer_idx]
        if self.return_pooled:
            return z, outputs.pooler_output
        else:
            return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEmbModel):
    """
    Uses the OpenCLIP vision transformer encoder for images.
    """

    def __init__(
            self,
            arch="ViT-H-14",
            # version="path_to/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
            version="laion2b_s32b_b79k",
            device="cuda",
            max_length=77,
            freeze=True,
            antialias=True,
            ucg_rate=0.0,
            unsqueeze_dim=False,
            repeat_to_max_len=False,
            num_image_crops=0,
            output_tokens=False,
            init_device=None
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device(default(init_device, "cpu")),
            pretrained=version
        )
        del model.transformer
        self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.antialias = antialias

        self.register_buffer("mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer("std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        tokens = None
        if self.output_tokens:
            z, tokens = z[0], z[1]
        z = z.to(image.dtype)
        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            z = (
                    torch.bernoulli(
                        (1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device)
                    )[:, None]
                    * z
            )
            if tokens is not None:
                tokens = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - self.ucg_rate) * torch.ones(tokens.shape[0], device=tokens.device)
                            ),
                            tokens
                        )
                        * tokens
                )
        if self.unsqueeze_dim:
            z = z[:, None]
        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            return tokens, z
        elif self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None]
            else:
                z_ = z
            return repeat(z_, "b 1 d -> b n d", n=self.max_length), z
        elif self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat(
                (
                    z,
                    torch.zeros(z.shape[0], self.max_length - z.shape[1], z.shape[2], device=z.device)
                ),
                1
            )
            return z_pad, z_pad[:, 0, ...]
        else:
            return z

    def encode_with_vision_transformer(self, img):
        if img.dim() == 5:
            assert self.max_crops == img.shape[1]
            img = rearrange(img, "b n c h w -> (b n) c h w")
        img = self.preprocess(img)
        if self.output_tokens:
            assert self.model.visual.output_tokens
            x, tokens = self.model.visual(img)
        else:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        if self.max_crops > 0:
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)
            # drop out between 0 and all along the sequence axis
            x = (
                    torch.bernoulli(
                        (1.0 - self.ucg_rate) * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
                    )
                    * x
            )
            if tokens is not None:
                tokens = rearrange(tokens, "(b n) t d -> b t (n d)", n=self.max_crops)
                print(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message"
                )
        if self.output_tokens:
            return x, tokens
        else:
            return x

    def encode(self, text):
        return self(text)


class ConcatTimestepEmbedderND(AbstractEmbModel):
    """
    Embeds each dimension independently and concatenates them.
    """

    def __init__(self, outdim, num_features=None, add_sequence_dim=False):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim
        self.num_features = num_features
        self.add_sequence_dim = add_sequence_dim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        assert dims == self.num_features or self.num_features is None
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        if self.add_sequence_dim:
            emb = emb[:, None]
        return emb


class VideoPredictionEmbedderWithEncoder(AbstractEmbModel):
    def __init__(
            self,
            n_cond_frames: int,
            n_copies: int,
            encoder_config: dict,
            sigma_sampler_config: Optional[dict] = None,
            sigma_cond_config: Optional[dict] = None,
            is_ae: bool = False,
            scale_factor: float = 1.0,
            disable_encoder_autocast: bool = False,
            en_and_decode_n_samples_a_time: Optional[int] = None
    ):
        super().__init__()
        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.encoder = instantiate_from_config(encoder_config)
        self.sigma_sampler = (
            instantiate_from_config(sigma_sampler_config)
            if sigma_sampler_config is not None
            else None
        )
        self.sigma_cond = (
            instantiate_from_config(sigma_cond_config)
            if sigma_cond_config is not None
            else None
        )
        self.is_ae = is_ae
        self.scale_factor = scale_factor
        self.disable_encoder_autocast = disable_encoder_autocast
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.skip_encode = False

    def forward(
            self, vid: torch.Tensor
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, dict],
        Tuple[Tuple[torch.Tensor, torch.Tensor], dict]
    ]:
        if self.skip_encode:
            return vid
        else:
            if self.sigma_sampler is not None:
                bs = vid.shape[0] // self.n_cond_frames
                sigmas = self.sigma_sampler(bs).to(vid.device)
                if self.sigma_cond is not None:
                    sigma_cond = self.sigma_cond(sigmas)
                    sigma_cond = repeat(sigma_cond, "b d -> (b t) d", t=self.n_copies)
                sigmas = repeat(sigmas, "b -> (b t)", t=self.n_cond_frames)
                noise = torch.randn_like(vid)
                vid = vid + noise * append_dims(sigmas, vid.ndim)

            with torch.autocast("cuda", enabled=not self.disable_encoder_autocast):
                n_samples = default(self.en_and_decode_n_samples_a_time, vid.shape[0])
                n_rounds = math.ceil(vid.shape[0] / n_samples)
                all_out = list()
                for n in range(n_rounds):
                    if self.is_ae:
                        out = self.encoder.encode(vid[n * n_samples: (n + 1) * n_samples])
                    else:
                        out = self.encoder(vid[n * n_samples: (n + 1) * n_samples])
                    all_out.append(out)

            vid = torch.cat(all_out, dim=0)
            vid *= self.scale_factor

            vid = rearrange(vid, "(b t) c h w -> b () (t c) h w", t=self.n_cond_frames)
            vid = repeat(vid, "b 1 c h w -> (b t) c h w", t=self.n_copies)

            if self.sigma_cond is not None:
                return vid, sigma_cond
            else:
                return vid


class FrozenOpenCLIPImagePredictionEmbedder(AbstractEmbModel):
    def __init__(self, open_clip_embedding_config: Dict, n_cond_frames: int, n_copies: int):
        super().__init__()
        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.open_clip = instantiate_from_config(open_clip_embedding_config)

    def forward(self, vid):
        vid = self.open_clip(vid)
        vid = rearrange(vid, "(b t) d -> b t d", t=self.n_cond_frames)
        vid = repeat(vid, "b t d -> (b s) t d", s=self.n_copies)
        return vid
