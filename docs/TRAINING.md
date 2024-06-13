## Training

- ### Requirement

  Nvidia GPUs with **80 GB** VRAM are required for training, but you can train low-resolution variants on smaller GPUs.

- ### Preparation

  Download the pretrained `svd_xt.safetensors` from [Hugging Face](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/svd_xt.safetensors) and place the checkpoint into `ckpts`.

- ### Training (example)

  - We take **nuScenes** dataset as an example for training. After finishing the setups in [INSTALL.md](https://github.com/OpenDriveLab/Vista/blob/main/docs/INSTALL.md), remember to edit *data_root* in `vwm/data/subsets/nuscenes.py` to the proper path of nuScenes.

  - We use DeepSpeed ZeRO stage 2 to improve data parallelism and lower memory footprint during training. The training can be launched as: 

    - Distributed training (suppose you train with 2 nodes, and each node has 8 GPUs).

      ```shell
      torchrun \
          --nnodes=2 \
          --nproc_per_node=8 \
          train.py \
          --base configs/example/nusc_train.yaml \
          --num_nodes 2 \
          --n_devices 8
      ```

    - Single GPU debugging (too slow, not recommended for training).

      ```shell
      python train.py --base configs/example/nusc_train.yaml --num_nodes 1 --n_devices 1
      ```

    > The training logs, including visualization samples and model checkpoints, will be saved in the project directory by default. Given that the size of checkpoints could be very large, you can set another directory to save these logs by providing an available path to `--logdir`.
    >
    > You can disable `--no_test` to test a bunch of samples for every checkpoint, but we recommend evaluating them offline for flexible comparison and uninterrupted training.

  - After training, switch to the log directory with the model checkpoint. You should find a Python script named `zero_to_fp32.py` and a `checkpoint` folder that contains all partitioned checkpoints. The final checkpoint can be obtained by:

    1. [*if you only want to resume training*] Merge the partitioned checkpoints as `pytorch_model.bin` using `zero_to_fp32.py`.
    
       ```shell
       python zero_to_fp32.py . pytorch_model.bin
       ```
    
    2. [*if you also want to do inference*] Navigate into the project root, and use `bin_to_st.py` to convert the resulting `path_to/pytorch_model.bin` to `ckpts/vista.safetensors`.

- ### Training of Vista

  - Download **OpenDV-YouTube** dataset (or a part of it) from [DriveAGI](https://github.com/OpenDriveLab/DriveAGI#genad-dataset-opendv-youtube). You can refer to the structure in `vwm/data/subsets/youtube.py` to organize the dataset.
  
  - #### Phase 1: learning high-fidelity future prediction
  
    - This phase uses unlabeled OpenDV-YouTube for training.
  
    - The model is trained at a resolution of 576x1024 on 128 GPUs for 20K iterations with gradient accumulation.
  
      ```shell
      torchrun \
          --nnodes=16 \
          --nproc_per_node=8 \
          train.py \
          --base configs/training/vista_phase1.yaml \
          --num_nodes 16 \
          --n_devices 8
      ```
  
    - We pause the training after the effect of dynamics priors can be witnessed. The last checkpoint is merged for the training of next phase.
  
  - #### Phase 2:  learning versatile action controllability
  
    - This phase uses OpenDV-YouTube and nuScenes for collaborative training.
  
    - ##### Stage 1: low-resolution training
  
      - The model is finetuned at a resolution of 320x576 on 8 GPUs for 120K iterations.
  
        ```shell
        torchrun \
            --nnodes=1 \
            --nproc_per_node=8 \
            train.py \
            --base configs/training/vista_phase2_stage1.yaml \
            --finetune ${PATH_TO_PHASE1_CKPT}/pytorch_model.bin \
            --num_nodes 1 \
            --n_devices 8
        ```
  
      - We pause the training after the controllability can be clearly witnessed. The last checkpoint is merged for the training of next stage.
  
    - ##### Stage 2: high-resolution training
  
      - The model is finetuned at a resolution of 576x1024 on 8 GPUs for another 10K iterations.
  
        ```shell
        torchrun \
            --nnodes=1 \
            --nproc_per_node=8 \
            train.py \
            --base configs/training/vista_phase2_stage2.yaml \
            --finetune ${PATH_TO_STAGE1_CKPT}/pytorch_model.bin \
            --num_nodes 1 \
            --n_devices 8
        ```
  
      - We pause the training after the model adapt to the desired resolution. The last checkpoint is merged for application.

---

<= Previous: [[Installation](https://github.com/OpenDriveLab/Vista/blob/main/docs/INSTALL.md)]

=> Next: [[Sampling](https://github.com/OpenDriveLab/Vista/blob/main/docs/SAMPLING.md)]
