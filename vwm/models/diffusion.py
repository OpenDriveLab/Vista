import math
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union

import torch
from einops import rearrange
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning import LightningModule
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR

from vwm.modules import UNCONDITIONAL_CONFIG
from vwm.modules.autoencoding.temporal_ae import VideoDecoder
from vwm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from vwm.modules.ema import LitEma
from vwm.util import default, disabled_train, get_obj_from_str, instantiate_from_config


class DiffusionEngine(LightningModule):
    def __init__(
            self,
            network_config,
            denoiser_config,
            first_stage_config,
            conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            network_wrapper: Union[None, str] = None,
            ckpt_path: Union[None, str] = None,
            use_ema: bool = False,
            ema_decay_rate: float = 0.9999,
            scale_factor: float = 1.0,
            disable_first_stage_autocast=False,
            input_key: str = "img",
            log_keys: Union[List, None] = None,
            no_cond_log: bool = False,
            compile_model: bool = False,
            en_and_decode_n_samples_a_time: int = 14,
            num_frames: int = 25,
            slow_spatial_layers: bool = False,
            train_peft_adapters: bool = False,
            replace_cond_frames: bool = False,
            fixed_cond_frames: Union[List, None] = None
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(
            default(network_wrapper, OPENAIUNETWRAPPER)
        )(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        # if slow_spatial_layers:
        #     for n, p in self.model.named_parameters():
        #         if "time_stack" not in n:
        #             p.requires_grad = False
        # elif train_peft_adapters:
        #     for n, p in self.model.named_parameters():
        #         if "adapter" not in n and p.requires_grad:
        #             p.requires_grad = False

        self.use_ema = use_ema
        self.ema_decay_rate = ema_decay_rate
        if use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.num_frames = num_frames
        self.slow_spatial_layers = slow_spatial_layers
        self.train_peft_adapters = train_peft_adapters
        self.replace_cond_frames = replace_cond_frames
        self.fixed_cond_frames = fixed_cond_frames

    def reinit_ema(self):
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=self.ema_decay_rate)
            print(f"Reinitializing EMAs of {len(list(self.model_ema.buffers()))}")

    def init_from_ckpt(self, path: str) -> None:
        if path.endswith("ckpt"):
            svd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("bin"):  # for deepspeed merged checkpoints
            svd = torch.load(path, map_location="cpu")
            for k in list(svd.keys()):  # remove the prefix
                if "_forward_module" in k:
                    svd[k.replace("_forward_module.", "")] = svd[k]
                del svd[k]
        elif path.endswith("safetensors"):
            svd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(svd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict
        # image tensors should be scaled to -1 ... 1 and in bchw format
        input_shape = batch[self.input_key].shape
        if len(input_shape) != 4:  # is an image sequence
            assert input_shape[1] == self.num_frames
            batch[self.input_key] = rearrange(batch[self.input_key], "b t c h w -> (b t) c h w")
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z, overlap=3):
        z = z / self.scale_factor
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        all_out = list()
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            if overlap < n_samples:
                previous_z = z[:overlap]
                for current_z in z[overlap:].split(n_samples - overlap, dim=0):
                    if isinstance(self.first_stage_model.decoder, VideoDecoder):
                        kwargs = {"timesteps": current_z.shape[0] + overlap}
                    else:
                        kwargs = dict()
                    context_z = torch.cat((previous_z, current_z), dim=0)
                    previous_z = current_z[-overlap:]
                    out = self.first_stage_model.decode(context_z, **kwargs)

                    if not all_out:
                        all_out.append(out)
                    else:
                        all_out[-1][-overlap:] = (all_out[-1][-overlap:] + out[:overlap]) / 2
                        all_out.append(out[overlap:])
            else:
                for current_z in z.split(n_samples, dim=0):
                    if isinstance(self.first_stage_model.decoder, VideoDecoder):
                        kwargs = {"timesteps": current_z.shape[0]}
                    else:
                        kwargs = dict()
                    out = self.first_stage_model.decode(current_z, **kwargs)
                    all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = list()
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples: (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = z * self.scale_factor
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)  # go to StandardDiffusionLoss
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx):
    #     loss, loss_dict = self.shared_step(batch)
    #     self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.slow_spatial_layers:
            param_dicts = [
                {
                    "params": [p for n, p in self.model.named_parameters() if "time_stack" in n]
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if "time_stack" not in n],
                    "lr": lr * 0.1
                }
            ]
        elif self.train_peft_adapters:
            param_dicts = [
                {
                    "params": [p for n, p in self.model.named_parameters() if "adapter" in n]
                }
            ]
        else:
            param_dicts = [
                {
                    "params": list(self.model.parameters())
                }
            ]
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                param_dicts.append(
                    {
                        "params": list(embedder.parameters())
                    }
                )
        opt = self.instantiate_optimizer_from_config(param_dicts, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1
                }
            ]
            return [opt], scheduler
        else:
            return opt

    @torch.no_grad()
    def sample(
            self,
            cond: Dict,
            cond_frame=None,
            uc: Union[Dict, None] = None,
            N: int = 25,
            shape: Union[None, Tuple, List] = None,
            **kwargs
    ):
        randn = torch.randn(N, *shape).to(self.device)
        cond_mask = torch.zeros(N).to(self.device)
        if self.replace_cond_frames:
            assert self.fixed_cond_frames
            cond_indices = self.fixed_cond_frames
            cond_mask = rearrange(cond_mask, "(b t) -> b t", t=self.num_frames)
            cond_mask[:, cond_indices] = 1
            cond_mask = rearrange(cond_mask, "b t -> (b t)")

        denoiser = lambda input, sigma, c, cond_mask: self.denoiser(self.model, input, sigma, c, cond_mask, **kwargs)
        samples = self.sampler(  # go to EulerEDMSampler
            denoiser, randn, cond, uc=uc, cond_frame=cond_frame, cond_mask=cond_mask
        )
        return samples

    @torch.no_grad()
    def log_images(
            self,
            batch: Dict,
            N: int = 25,
            sample: bool = True,
            ucg_keys: List[str] = None,
            **kwargs
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders if e.ucg_rate > 0.0]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys, "
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else list()
        )

        sampling_kwargs = dict()

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]

        z = self.encode_first_stage(x)
        x_reconstruct = self.decode_first_stage(z)

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))
                if c[k].shape[0] < N:
                    c[k] = c[k][[0]]
                if uc[k].shape[0] < N:
                    uc[k] = uc[k][[0]]

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, cond_frame=z, shape=z.shape[1:], uc=uc, N=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = log["samples_mp4"] = samples

        log["inputs"] = log["inputs_mp4"] = x
        log["targets"] = log["targets_mp4"] = x_reconstruct
        return log
