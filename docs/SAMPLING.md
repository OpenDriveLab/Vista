## Sampling

- ### Requirement

  Currently, we suggest using Nvidia GPUs with a minimum of **48 GB** VRAM for sampling.

- ### Preparation

  Make sure you have prepared (or trained) the checkpoint of Vista. Move (or link) `pytorch_model.bin` into `ckpts`.

- ### Future Prediction

  - We provide a sampling example for nuScenes. Make sure to prepare the dataset as [INSTALL.md](https://github.com/OpenDriveLab/Vista/blob/main/docs/INSTALL.md) and replace the correct *data_root* in `sample.py`.

    - Short-term action-free prediction.

      ```shell
      python sample.py
      ```


    - Long-term rollout.

      ```shell
      python sample.py --n_rounds 6
      ```

    - Action-conditioned simulation (take trajectory as an example).

      ```
      python sample.py --action traj
      ```

  - Important arguments:

    - `--dataset`: You can also customize the scenes by providing your own driving views within a folder of images. They will be used as the initial frames for prediction when you set `--dataset` to "IMG".
    - `--action`: The mode of control inputs. By default, we perform action-free prediction. You can try different actions using "traj", "cmd", "steer", or "goal". It will import ground truth actions (if available), but you can enforce any actions by making slight modifications to the code.
    - `--n_rounds`: The number of sampling rounds, which determines the duration to predict. You can increase it to perform autoregressive long-horizon rollout. Each additional round extends the prediction by about 2.3 seconds.
    - `--n_steps`: The number of DDIM sampling steps, which can be reduced for efficiency.
    - `--rand_gen`: Whether to generate diverse samples randomly selected from the whole dataset or go through all samples one by one. 
    - `--low_vram`: Enable the low VRAM mode if you are using a GPU with less than 80 GB VRAM.

- ### Reward Estimation

  - We provide a simplified example to estimate the rewards on nuScenes. Make sure to replace the correct *data_root* in `reward.py`.

    ```shell
    python reward.py
    ```

  - Important arguments:
  
    - `--ens_size`: The number of samples to generate per case (initial frame and action condition).

---

<= Previous: [[Training](https://github.com/OpenDriveLab/Vista/blob/main/docs/TRAINING.md)]

=> Next: [[Trouble Shooting](https://github.com/OpenDriveLab/Vista/blob/main/docs/ISSUES.md)]
