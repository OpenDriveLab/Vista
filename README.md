# Vista

The official implementation of the paper:

**Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability**

>  [Shenyuan Gao](https://github.com/Little-Podi), [Jiazhi Yang](https://scholar.google.com/citations?user=Ju7nGX8AAAAJ&hl=en), [Li Chen](https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=en), [Kashyap Chitta](https://kashyap7x.github.io/), [Yihang Qiu](https://scholar.google.com/citations?user=qgRUOdIAAAAJ&hl=en), [Andreas Geiger](https://www.cvlibs.net/), [Jun Zhang](https://eejzhang.people.ust.hk/), [Hongyang Li](https://lihongyang.info/)
>
> 📜 [[technical report](https://arxiv.org/abs/2405.17398)], 🎬 [[video demos](https://vista-demo.github.io/)], 🤗 [[model weights](https://huggingface.co/OpenDriveLab/Vista)], 🗃️ [[OpenDV dataset](https://github.com/OpenDriveLab/DriveAGI?tab=readme-ov-file#opendv)]

<div id="top" align="center">
<p align="center">
<img src="assets/teaser.gif" width="1000px" >
</p>
</div>

> Simulated futures in a wide range of driving scenarios by [Vista](https://arxiv.org/abs/2405.17398). Best viewed on [demo page](https://vista-demo.github.io/).

## 🔥 Highlights

**Vista** is a generalizable driving world model that can:

- *Predict high-fidelity futures in various scenarios*.
- *Extend its predictions to continuous and long horizons*.
- *Execute multi-modal actions (steering angles, speeds, commands, trajectories, goal points).*
- *Provide rewards for different actions without accessing ground truth actions.*

<div id="top" align="center">
<p align="center">
<img src="assets/overview.png" width="1000px" >
</p>
</div>

## 📢 News

> [!IMPORTANT]
> There is an error in merging the EMA weights of the previously uploaded model. Please download the latest model below.

- **[2024/06/06]** We released the model weights v1.0 at [Hugging Face](https://huggingface.co/OpenDriveLab/Vista/blob/main/vista.safetensors) and [Google Drive](https://drive.google.com/file/d/1bCM7XLDquRqnnpauQAK5j1jP-n0y1ama/view).
- **[2024/06/04]** We released the installation, training, and sampling scripts.
- **[2024/05/28]** We released the implementation of our model.
- **[2024/05/28]** We released our [paper](https://arxiv.org/abs/2405.17398) on arXiv.

## 📋 TODO List

- [ ] New model weights trained with a larger batch size ane more iterations.
- [ ] Memory efficient training and sampling.
- [ ] Online demo for interaction.

## 🕹️ Getting Started

- [Installation](https://github.com/OpenDriveLab/Vista/blob/main/docs/INSTALL.md)

- [Training](https://github.com/OpenDriveLab/Vista/blob/main/docs/TRAINING.md)

- [Sampling](https://github.com/OpenDriveLab/Vista/blob/main/docs/SAMPLING.md)

- [Trouble Shooting](https://github.com/OpenDriveLab/Vista/blob/main/docs/ISSUES.md)

## ❤️ Acknowledgement

Our implementation is based on [generative-models](https://github.com/Stability-AI/generative-models) from Stability AI. Thanks for their great open-source work!

## ⭐ Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```bibtex
@article{gao2024vista,
 title={Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability}, 
 author={Shenyuan Gao and Jiazhi Yang and Li Chen and Kashyap Chitta and Yihang Qiu and Andreas Geiger and Jun Zhang and Hongyang Li},
 journal={arXiv preprint arXiv:2405.17398},
 year={2024}
}

@inproceedings{yang2024genad,
  title={Generalized Predictive Model for Autonomous Driving},
  author={Jiazhi Yang and Shenyuan Gao and Yihang Qiu and Li Chen and Tianyu Li and Bo Dai and Kashyap Chitta and Penghao Wu and Jia Zeng and Ping Luo and Jun Zhang and Andreas Geiger and Yu Qiao and Hongyang Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## ⚖️ License

All content in this repository are under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
