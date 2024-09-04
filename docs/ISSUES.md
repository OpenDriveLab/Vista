## Trouble Shooting

1. #### Out of memory during sampling.

   - Possible reason:
     - Too many high-resolution frames for parallel decoding. The default setting will request ca. 66 GB peak VARM.

   - Try this:
     - Reduce the number of jointly decoded frames *en_and_decode_n_samples_a_time* in `inference/vista.yaml`.

2. #### Get stuck at loading FrozenCLIPEmbedder or FrozenOpenCLIPImageEmbedder.

   - Possible reason:
     - A network failure.

   - Try this:
     1. Download [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14/tree/main) and [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main) in advance.
     2. Set *version* of FrozenCLIPEmbedder and FrozenOpenCLIPImageEmbedder in `vwm/modules/encoders/modules.py` to the new paths of `pytorch_model.bin`/`open_clip_pytorch_model.bin`.

3. #### Datasets not yet available during training.

   - Possible reason:

     - The installed [sdata](https://github.com/Stability-AI/datapipelines) is not detected.

   - Try this:

     - Reinstall in the current project directory.

       ````shell
       pip3 install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
       ````

4. #### The shapes of linear layers cannot be multiplied at the cross-attention layers.

   - Possible reason:
     - The dimension of cross-attention is not expended while the action conditions are injected, resulting in a mismatch.

   - Try this:
     - Enable `action_control: True` in the YAML config file.

---

<= Previous: [[Sampling](https://github.com/OpenDriveLab/Vista/blob/main/docs/SAMPLING.md)]
