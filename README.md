# Vista

The official implementation of the paper:

**Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability**

[Shenyuan Gao](https://github.com/Little-Podi), [Jiazhi Yang](https://scholar.google.com/citations?user=Ju7nGX8AAAAJ&hl=en), [Li Chen](https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=en), [Kashyap Chitta](https://kashyap7x.github.io/), [Yihang Qiu](https://scholar.google.com/citations?user=qgRUOdIAAAAJ&hl=en), [Andreas Geiger](https://www.cvlibs.net/), [Jun Zhang](https://eejzhang.people.ust.hk/), [Hongyang Li](https://lihongyang.info/)

[[arXiv](https://arxiv.org/abs/2405.17398)] [[video demos](https://vista-demo.github.io/)]

<div id="top" align="center">
<p align="center">
<img src="assets/teaser.gif" width="1000px" >
</p>
</div>

## Highlight

:bookmark: ​**Vista** is a generalizable driving world model that can:

- Predict high-fidelity futures in various scenarios.
- Extend its predictions to continuous and long horizons.
- Execute multi-modal actions (steering angles, speeds, commands, trajectories, goal points).
- Provide rewards for different actions without accessing ground truth actions.

<div id="top" align="center">
<p align="center">
<img src="assets/overview.png" width="1000px" >
</p>
</div>

## News

- **[2024/05/28]** We released the implementation of our model.
- **[2024/05/28]** We released our [paper](https://arxiv.org/abs/2405.17398) on arXiv.

## TODO List

- [ ] Installation, training, and sampling scripts (**within one week**).
- [ ] Model weights release.
- [ ] More detailed instructions.
- [ ] Online demo for interaction.

## Acknowledgement

Our implementation is based on [generative-models](https://github.com/Stability-AI/generative-models) from Stability AI. Thanks for their open-source work! :heart::heart::heart:

## Citation

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
