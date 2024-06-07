## Installation

- ### Requirement

  Our experiments are conducted with **PyTorch 2.0.1**, **CUDA 11.7**, **Ubuntu 22.04**, and **NVIDIA Tesla A100** (80 GB). For other requirements, please check [TRAINING.md](https://github.com/OpenDriveLab/Vista/blob/main/docs/TRAINING.md) and [SAMPLING.md](https://github.com/OpenDriveLab/Vista/blob/main/docs/SAMPLING.md).

- ### Preparation

  Clone the repository to your local directory.

  ```shell
  git clone https://github.com/OpenDriveLab/Vista.git
  ```

  We provide an example on nuScenes dataset for training and sampling. Before you start, make sure you have:
  
  - Downloaded the translated action annotations from [here](https://drive.google.com/drive/folders/1JpZObdR0OXagCbnPZfMSI8vhGLom5pht?usp=sharing) and put the JSON files into `annos`.
    
  - Downloaded all splits of **Trainval** in **Full dataset (v1.0)** to your device following [official instructions](https://www.nuscenes.org/download). After downloading, it should contain:

    ```
    $<your-nusc-data-root>
    ├── samples
    ├── sweeps
    ├── ...
    └── v1.0-trainval
    ```

- ### Installation

  - We use conda to manage the environment.

    ```shell
    conda create -n vista python=3.9 -y
    conda activate vista
    ```
  
  - Install dependencies.
  
    ```shell
    conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
    pip3 install -r requirements.txt
    pip3 install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
    ```

---

=> Next: [[Training](https://github.com/OpenDriveLab/Vista/blob/main/docs/TRAINING.md)]
