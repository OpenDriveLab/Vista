import math
import re
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from packaging import version
from pytorch_lightning import LightningModule

from vwm.modules.autoencoding.regularizers import AbstractRegularizer
from vwm.modules.ema import LitEma
from vwm.util import default, get_obj_from_str, instantiate_from_config


class AbstractAutoencoder(LightningModule):
    """
    This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
    unCLIP models, etc. Hence, it is fairly general, and specific features
    (e.g. discriminator training, encoding, decoding) must be implemented in subclasses.
    """

    def __init__(
            self,
            ema_decay: Union[None, float] = None,
            monitor: Union[None, str] = None,
            input_key: str = "img"
    ):
        super().__init__()
        self.input_key = input_key
        self.use_ema = ema_decay is not None
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")

        if version.parse(pl.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False

    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        if ckpt is None:
            return
        elif isinstance(ckpt, str):
            ckpt = {
                "target": "vwm.modules.checkpoint.CheckpointEngine",
                "params": {"ckpt_path": ckpt}
            }
        engine = instantiate_from_config(ckpt)
        engine(self)

    @abstractmethod
    def get_input(self, batch) -> Any:
        raise NotImplementedError

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @abstractmethod
    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("encode()-method of abstract base class called")

    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("decode()-method of abstract base class called")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        print(f"Loading >>> {cfg['target']} <<< optimizer from config")
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self) -> Any:
        raise NotImplementedError


class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    def __init__(
            self,
            *args,
            encoder_config: Dict,
            decoder_config: Dict,
            loss_config: Dict,
            regularizer_config: Dict,
            optimizer_config: Union[Dict, None] = None,
            lr_g_factor: float = 1.0,
            trainable_ae_params: Optional[List[List[str]]] = None,
            ae_optimizer_args: Optional[List[dict]] = None,
            trainable_disc_params: Optional[List[List[str]]] = None,
            disc_optimizer_args: Optional[List[dict]] = None,
            disc_start_iter: int = 0,
            diff_boost_factor: float = 3.0,
            ckpt_engine: Union[None, str, dict] = None,
            ckpt_path: Optional[str] = None,
            additional_decode_keys: Optional[List[str]] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if version.parse(pl.__version__) >= version.parse("2.0.0"):  # pytorch lightning
            self.automatic_optimization = False

        self.encoder: nn.Module = instantiate_from_config(encoder_config)
        self.decoder: nn.Module = instantiate_from_config(decoder_config)
        self.loss: nn.Module = instantiate_from_config(loss_config)
        self.regularization: AbstractRegularizer = instantiate_from_config(regularizer_config)
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        self.diff_boost_factor = diff_boost_factor
        self.disc_start_iter = disc_start_iter
        self.lr_g_factor = lr_g_factor
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [dict() for _ in range(len(self.trainable_ae_params))]
            )
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [dict()]  # makes type consistent

        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [dict() for _ in range(len(self.trainable_disc_params))]
            )
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [dict()]  # makes type consistent

        if ckpt_path is not None:
            assert ckpt_engine is None, "Cannot set ckpt_engine and ckpt_path"
            print("Checkpoint path is deprecated, use `checkpoint_engine` instead")
        self.apply_ckpt(default(ckpt_path, ckpt_engine))
        self.additional_decode_keys = set(default(additional_decode_keys, list()))

    def get_input(self, batch: Dict) -> torch.Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in channels-first format (e.g., bchw instead if bhwc)
        return batch[self.input_key]

    def get_autoencoder_params(self) -> list:
        params = list()
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            params += list(self.loss.get_trainable_autoencoder_parameters())
        if hasattr(self.regularization, "get_trainable_parameters"):
            params += list(self.regularization.get_trainable_parameters())
        params = params + list(self.encoder.parameters())
        params = params + list(self.decoder.parameters())
        return params

    def get_discriminator_params(self) -> list:
        if hasattr(self.loss, "get_trainable_parameters"):
            params = list(self.loss.get_trainable_parameters())  # e.g., discriminator
        else:
            params = list()
        return params

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def encode(
            self,
            x: torch.Tensor,
            return_reg_log: bool = False,
            unregularized: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x)
        if unregularized:
            return z, dict()
        else:
            z, reg_log = self.regularization(z)
            if return_reg_log:
                return z, reg_log
            else:
                return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)
        return x

    def forward(
            self, x: torch.Tensor, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)
        dec = self.decode(z, **additional_decode_kwargs)
        return z, dec, reg_log

    def inner_training_step(
            self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        x = self.get_input(batch)
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, x_reconstruct, regularization_log = self(x, **additional_decode_kwargs)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
                "autoencoder": self
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if optimizer_idx == 0:
            # autoencoder
            out_loss = self.loss(x, x_reconstruct, **extra_info)
            if isinstance(out_loss, tuple):
                ae_loss, log_dict_ae = out_loss
            else:
                # simple loss function
                ae_loss = out_loss
                log_dict_ae = {"train/loss/rec": ae_loss.detach()}

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False
            )
            self.log(
                "loss",
                ae_loss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_epoch=False,
                on_step=True
            )
            return ae_loss
        elif optimizer_idx == 1:
            # discriminator
            disc_loss, log_dict_disc = self.loss(x, x_reconstruct, **extra_info)
            # discriminator always needs to return a tuple
            self.log_dict(
                log_dict_disc,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True
            )
            return disc_loss
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")

    def training_step(self, batch: dict, batch_idx: int):
        opts = self.optimizers()
        if not isinstance(opts, list):
            # non-adversarial case
            opts = [opts]
        optimizer_idx = batch_idx % len(opts)
        if self.global_step < self.disc_start_iter:
            optimizer_idx = 0
        opt = opts[optimizer_idx]
        opt.zero_grad()
        with opt.toggle_model():
            loss = self.inner_training_step(batch, batch_idx, optimizer_idx=optimizer_idx)
            self.manual_backward(loss)
        opt.step()

    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)
        return log_dict

    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        x = self.get_input(batch)

        z, x_reconstruct, regularization_log = self(x)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": 0,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": regularization_log,
                "autoencoder": self
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()
        out_loss = self.loss(x, x_reconstruct, **extra_info)
        if isinstance(out_loss, tuple):
            ae_loss, log_dict_ae = out_loss
        else:
            # simple loss function
            ae_loss = out_loss
            log_dict_ae = {f"val{postfix}/loss/rec": ae_loss.detach()}
        full_log_dict = log_dict_ae

        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1
            disc_loss, log_dict_disc = self.loss(x, x_reconstruct, **extra_info)
            full_log_dict.update(log_dict_disc)
        self.log(
            f"val{postfix}/loss/rec",
            log_dict_ae[f"val{postfix}/loss/rec"],
            sync_dist=True
        )
        self.log_dict(
            full_log_dict,
            sync_dist=True
        )
        return full_log_dict

    def get_param_groups(
            self, parameter_names: List[List[str]], optimizer_args: List[dict]
    ) -> Tuple[List[Dict[str, Any]], int]:
        groups = list()
        num_params = 0
        for names, args in zip(parameter_names, optimizer_args):
            params = list()
            for pattern_ in names:
                pattern_params = list()
                pattern = re.compile(pattern_)
                for p_name, param in self.named_parameters():
                    if re.match(pattern, p_name):
                        pattern_params.append(param)
                        num_params += param.numel()
                if len(pattern_params) == 0:
                    print(f"Did not find parameters for pattern {pattern_}")
                params.extend(pattern_params)
            groups.append({"params": params, **args})
        return groups, num_params

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        if self.trainable_ae_params is None:
            ae_params = self.get_autoencoder_params()
        else:
            ae_params, num_ae_params = self.get_param_groups(
                self.trainable_ae_params, self.ae_optimizer_args
            )
            print(f"Number of trainable autoencoder parameters: {num_ae_params:,}")
        if self.trainable_disc_params is None:
            disc_params = self.get_discriminator_params()
        else:
            disc_params, num_disc_params = self.get_param_groups(
                self.trainable_disc_params, self.disc_optimizer_args
            )
            print(f"Number of trainable discriminator parameters: {num_disc_params:,}")
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config
        )
        opts = [opt_ae]
        if len(disc_params) > 0:
            opt_disc = self.instantiate_optimizer_from_config(
                disc_params,
                self.learning_rate,
                self.optimizer_config
            )
            opts.append(opt_disc)
        return opts

    @torch.no_grad()
    def log_images(
            self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        log = dict()
        additional_decode_kwargs = dict()
        x = self.get_input(batch)
        additional_decode_kwargs.update(
            {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
        )

        _, x_reconstruct, _ = self(x, **additional_decode_kwargs)
        log["inputs"] = x
        log["reconstructions"] = x_reconstruct
        diff = 0.5 * torch.abs(torch.clamp(x_reconstruct, -1.0, 1.0) - x)
        diff.clamp_(0, 1.0)
        log["diff"] = 2.0 * diff - 1.0
        # diff_boost shows location of small errors, by boosting their brightness
        log["diff_boost"] = 2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        if hasattr(self.loss, "log_images"):
            log.update(self.loss.log_images(x, x_reconstruct))
        with self.ema_scope():
            _, x_reconstruct_ema, _ = self(x, **additional_decode_kwargs)
            log["reconstructions_ema"] = x_reconstruct_ema
            diff_ema = 0.5 * torch.abs(torch.clamp(x_reconstruct_ema, -1.0, 1.0) - x)
            diff_ema.clamp_(0, 1.0)
            log["diff_ema"] = 2.0 * diff_ema - 1.0
            log["diff_boost_ema"] = 2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1
        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            _, x_reconstruct_add, _ = self(x, **additional_decode_kwargs)
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            log[log_str] = x_reconstruct_add
        return log


class AutoencodingEngineLegacy(AutoencodingEngine):
    def __init__(self, embed_dim: int, **kwargs):
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        ddconfig = kwargs.pop("ddconfig")
        ckpt_path = kwargs.pop("ckpt_path", None)
        ckpt_engine = kwargs.pop("ckpt_engine", None)
        super().__init__(
            encoder_config={
                "target": "vwm.modules.diffusionmodules.model.Encoder",
                "params": ddconfig
            },
            decoder_config={
                "target": "vwm.modules.diffusionmodules.model.Decoder",
                "params": ddconfig
            },
            **kwargs
        )
        self.quant_conv = nn.Conv2d(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1
        )
        self.post_quant_conv = nn.Conv2d(
            embed_dim,
            ddconfig["z_channels"],
            1
        )
        self.embed_dim = embed_dim

        self.apply_ckpt(default(ckpt_path, ckpt_engine))

    def get_autoencoder_params(self) -> list:
        params = super().get_autoencoder_params()
        return params

    def encode(
            self, x: torch.Tensor, return_reg_log: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if self.max_batch_size is None:
            z = self.encoder(x)
            z = self.quant_conv(z)
        else:
            N = x.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            z = list()
            for i_batch in range(n_batches):
                z_batch = self.encoder(x[i_batch * bs: (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)
                z.append(z_batch)
            z = torch.cat(z, 0)

        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        else:
            return z

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        if self.max_batch_size is None:
            dec = self.post_quant_conv(z)
            dec = self.decoder(dec, **decoder_kwargs)
        else:
            N = z.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            dec = list()
            for i_batch in range(n_batches):
                dec_batch = self.post_quant_conv(z[i_batch * bs: (i_batch + 1) * bs])
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)
                dec.append(dec_batch)
            dec = torch.cat(dec, 0)
        return dec


class AutoencoderKL(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        super().__init__(
            regularizer_config={
                "target": (
                    "vwm.modules.autoencoding.regularizers.DiagonalGaussianRegularizer"
                )
            },
            **kwargs
        )


class AutoencoderKLModeOnly(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        super().__init__(
            regularizer_config={
                "target": (
                    "vwm.modules.autoencoding.regularizers.DiagonalGaussianRegularizer"
                ),
                "params": {"sample": False},
            },
            **kwargs
        )
