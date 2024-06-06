import math
import os
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from einops import rearrange, repeat
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch import autocast

from train import save_img_seq_to_video
from vwm.modules.diffusionmodules.sampling import EulerEDMSampler
from vwm.util import default, instantiate_from_config


def init_model(version_dict, load_ckpt=True):
    config = OmegaConf.load(version_dict["config"])
    model = load_model_from_config(config, version_dict["ckpt"] if load_ckpt else None)
    return model


lowvram_mode = False


def set_lowvram_mode(mode):
    global lowvram_mode
    lowvram_mode = mode


def initial_model_load(model):
    global lowvram_mode
    if lowvram_mode:
        model.model.half()
    else:
        model.cuda()
    return model


def load_model(model):
    model.cuda()


def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()


def load_model_from_config(config, ckpt=None):
    model = instantiate_from_config(config.model)

    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        if ckpt.endswith("ckpt"):
            pl_svd = torch.load(ckpt, map_location="cpu")
            # dict contains:
            # "epoch", "global_step", "pytorch-lightning_version",
            # "state_dict", "loops", "callbacks", "optimizer_states", "lr_schedulers"
            if "global_step" in pl_svd:
                print(f"Global step: {pl_svd['global_step']}")
            svd = pl_svd["state_dict"]
        elif ckpt.endswith("safetensors"):
            svd = load_safetensors(ckpt)
        else:
            raise NotImplementedError("Please convert the checkpoint to safetensors first")

        missing, unexpected = model.load_state_dict(svd, strict=False)
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    model = initial_model_load(model)
    model.eval()
    return model


def init_embedder_options(keys):
    # hardcoded demo settings, might undergo some changes in the future
    value_dict = dict()
    for key in keys:
        if key in ["fps_id", "fps"]:
            fps = 10
            value_dict["fps"] = fps
            value_dict["fps_id"] = fps - 1
        elif key == "motion_bucket_id":
            value_dict["motion_bucket_id"] = 127  # [0, 511]
    return value_dict


def perform_save_locally(save_path, samples, mode, dataset_name, sample_index):
    assert mode in ["images", "grids", "videos"]
    merged_path = os.path.join(save_path, mode)
    os.makedirs(merged_path, exist_ok=True)
    samples = samples.cpu()

    if mode == "images":
        frame_count = 0
        for sample in samples:
            sample = rearrange(sample.numpy(), "c h w -> h w c")
            if "real" in save_path:
                sample = 255.0 * (sample + 1.0) / 2.0
            else:
                sample = 255.0 * sample
            image_save_path = os.path.join(merged_path, f"{dataset_name}_{sample_index:06}_{frame_count:04}.png")
            # if os.path.exists(image_save_path):
            #     return
            Image.fromarray(sample.astype(np.uint8)).save(image_save_path)
            frame_count += 1
    elif mode == "grids":
        grid = torchvision.utils.make_grid(samples, nrow=int(samples.shape[0] ** 0.5))
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
        if "real" in save_path:
            grid = 255.0 * (grid + 1.0) / 2.0
        else:
            grid = 255.0 * grid
        grid_save_path = os.path.join(merged_path, f"{dataset_name}_{sample_index:06}.png")
        # if os.path.exists(grid_save_path):
        #     return
        Image.fromarray(grid.astype(np.uint8)).save(grid_save_path)
    elif mode == "videos":
        img_seq = rearrange(samples.numpy(), "t c h w -> t h w c")
        if "real" in save_path:
            img_seq = 255.0 * (img_seq + 1.0) / 2.0
        else:
            img_seq = 255.0 * img_seq
        video_save_path = os.path.join(merged_path, f"{dataset_name}_{sample_index:06}.mp4")
        # if os.path.exists(video_save_path):
        #     return
        save_img_seq_to_video(video_save_path, img_seq.astype(np.uint8), 10)
    else:
        raise NotImplementedError


def init_sampling(sampler="EulerEDMSampler", guider="VanillaCFG", discretization="EDMDiscretization",
                  steps=50, cfg_scale=2.5, num_frames=25):
    discretization_config = get_discretization(discretization)
    guider_config = get_guider(guider, cfg_scale, num_frames)
    sampler = get_sampler(sampler, steps, discretization_config, guider_config)
    return sampler


def get_discretization(discretization):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "vwm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        }
    elif discretization == "EDMDiscretization":
        discretization_config = {
            "target": "vwm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0
            }
        }
    else:
        raise NotImplementedError
    return discretization_config


def get_guider(guider="LinearPredictionGuider", cfg_scale=2.5, num_frames=25):
    if guider == "IdentityGuider":
        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif guider == "VanillaCFG":
        scale = cfg_scale

        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {
                "scale": scale
            }
        }
    elif guider == "LinearPredictionGuider":
        max_scale = cfg_scale
        min_scale = 1.0

        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.LinearPredictionGuider",
            "params": {
                "max_scale": max_scale,
                "min_scale": min_scale,
                "num_frames": num_frames
            }
        }
    elif guider == "TrianglePredictionGuider":
        max_scale = cfg_scale
        min_scale = 1.0

        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.TrianglePredictionGuider",
            "params": {
                "max_scale": max_scale,
                "min_scale": min_scale,
                "num_frames": num_frames
            }
        }
    else:
        raise NotImplementedError
    return guider_config


def get_sampler(sampler, steps, discretization_config, guider_config):
    if sampler == "EulerEDMSampler":
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = 999.0
        s_noise = 1.0

        sampler = EulerEDMSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown sampler {sampler}")
    return sampler


def get_batch(keys, value_dict, N: Union[List, ListConfig], device="cuda"):
    # hardcoded demo setups, might undergo some changes in the future
    batch = dict()
    batch_uc = dict()

    for key in keys:
        if key in value_dict:
            if key in ["fps", "fps_id", "motion_bucket_id", "cond_aug"]:
                batch[key] = repeat(torch.tensor([value_dict[key]]).to(device), "1 -> b", b=math.prod(N))
            elif key in ["command", "trajectory", "speed", "angle", "goal"]:
                batch[key] = repeat(value_dict[key][None].to(device), "1 ... -> b ...", b=N[0])
            elif key in ["cond_frames", "cond_frames_without_noise"]:
                batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
            else:
                # batch[key] = value_dict[key]
                raise NotImplementedError

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_condition(model, value_dict, num_samples, force_uc_zero_embeddings, device):
    load_model(model.conditioner)
    batch, batch_uc = get_batch(
        list(set([x.input_key for x in model.conditioner.embedders])),
        value_dict,
        [num_samples]
    )
    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=force_uc_zero_embeddings
    )
    unload_model(model.conditioner)

    for k in c:
        if isinstance(c[k], torch.Tensor):
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))
            if c[k].shape[0] < num_samples:
                c[k] = c[k][[0]]
            if uc[k].shape[0] < num_samples:
                uc[k] = uc[k][[0]]
    return c, uc


def fill_latent(cond, length, cond_indices, device):
    latent = torch.zeros(length, *cond.shape[1:]).to(device)
    latent[cond_indices] = cond
    return latent


@torch.no_grad()
def do_sample(
        images,
        model,
        sampler,
        value_dict,
        num_frames,
        ensemble_size: int = 5,
        force_uc_zero_embeddings: Optional[List] = None,
        initial_cond_indices: Optional[List] = None,
        device="cuda"
):
    if initial_cond_indices is None:
        initial_cond_indices = [0]

    force_uc_zero_embeddings = default(force_uc_zero_embeddings, list())
    precision_scope = autocast

    with torch.no_grad(), precision_scope(device), model.ema_scope("Sampling"):
        load_model(model.first_stage_model)
        z = model.encode_first_stage(images)
        unload_model(model.first_stage_model)

        def denoiser(x, sigma, cond, cond_mask):
            return model.denoiser(model.model, x, sigma, cond, cond_mask)

        load_model(model.denoiser)
        load_model(model.model)

        initial_cond_mask = torch.zeros(num_frames).to(device)
        initial_cond_mask[initial_cond_indices] = 1

        c, uc = get_condition(model, value_dict, num_frames, force_uc_zero_embeddings, device)

        sample_ensemble = list()
        for _ in range(ensemble_size):
            noise = torch.randn_like(z)
            sample = sampler(
                denoiser,
                noise,
                cond=c,
                uc=uc,
                cond_frame=z,  # cond_frame will be rescaled when calling the sampler
                cond_mask=initial_cond_mask
            )
            sample[0] = z[0]
            sample_ensemble.append(sample)

        u = torch.mean(torch.stack(sample_ensemble), 0)
        diff = torch.zeros_like(sample)
        for each_sample in sample_ensemble:
            diff.add_((each_sample - u) ** 2)
        variance = diff / (ensemble_size - 1)
        reward = torch.exp(-variance.mean()).cpu()

        unload_model(model.model)
        unload_model(model.denoiser)
        return images, reward
