import os

import torch
from safetensors.torch import save_file

ckpt = "path_to/pytorch_model.bin"

vista_bin = torch.load(ckpt, map_location="cpu")  # only contains model weights

for k in list(vista_bin.keys()):  # merge LoRA weights (if exist) for inference
    if "adapter_down" in k:
        if "q_adapter_down" in k:
            up_k = k.replace("q_adapter_down", "q_adapter_up")
            pretrain_k = k.replace("q_adapter_down", "to_q")
        elif "k_adapter_down" in k:
            up_k = k.replace("k_adapter_down", "k_adapter_up")
            pretrain_k = k.replace("k_adapter_down", "to_k")
        elif "v_adapter_down" in k:
            up_k = k.replace("v_adapter_down", "v_adapter_up")
            pretrain_k = k.replace("v_adapter_down", "to_v")
        else:
            up_k = k.replace("out_adapter_down", "out_adapter_up")
            if "model_ema" in k:
                pretrain_k = k.replace("out_adapter_down", "to_out0")
            else:
                pretrain_k = k.replace("out_adapter_down", "to_out.0")

        lora_weights = vista_bin[up_k] @ vista_bin[k]
        del vista_bin[k]
        del vista_bin[up_k]
        vista_bin[pretrain_k] = vista_bin[pretrain_k] + lora_weights

for k in list(vista_bin.keys()):  # remove the prefix
    if "_forward_module" in k and "decay" not in k and "num_updates" not in k:
        vista_bin[k.replace("_forward_module.", "")] = vista_bin[k]
    del vista_bin[k]

for k in list(vista_bin.keys()):  # combine EMA weights
    if "model_ema" in k:
        orig_k = None
        for kk in list(vista_bin.keys()):
            if "model_ema" not in kk and k[10:] == kk[6:].replace(".", ""):
                orig_k = kk
        assert orig_k is not None
        vista_bin[orig_k] = vista_bin[k]
        del vista_bin[k]
        print("Replace", orig_k, "with", k)

vista_st = dict()
for k in list(vista_bin.keys()):
    vista_st[k] = vista_bin[k]

os.makedirs("ckpts", exist_ok=True)
save_file(vista_st, "ckpts/vista.safetensors")
