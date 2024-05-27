# Vista

The official implementation of our new paper:

**Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability**

[Shenyuan Gao](https://github.com/Little-Podi), [Jiazhi Yang](https://scholar.google.com/citations?user=Ju7nGX8AAAAJ&hl=en), [Li Chen](https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=en), [Kashyap Chitta](https://kashyap7x.github.io/), [Yihang Qiu](https://scholar.google.com/citations?user=qgRUOdIAAAAJ&hl=en), [Andreas Geiger](https://www.cvlibs.net/), [Jun Zhang](https://eejzhang.people.ust.hk/), [Hongyang Li](https://lihongyang.info/)

[[arXiv](https://arxiv.org/abs/2405.xxxxx)] [[video demos](https://vista-demo.github.io/)]

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
- Provide rewards for different actions without accessing the ground truth actions.

<div id="top" align="center">
<p align="center">
<img src="assets/overview.png" width="1000px" >
</p>
</div>

## News

- **[2024/05/28]** Release the implemetation of our model.
- **[2024/05/28]** Share our [paper](https://arxiv.org/abs/2405.xxxxx) on arXiv.
- **[2024/05/27]** Repository initialization.

## TODO List

- [ ] Release the installation, training and sampling scripts (**within one week**).
- [ ] Release the model weights.
- [ ] Optimize the configuration of model settings.
- [ ] Provide more detailed instructions.
- [ ] Build an online demo for interaction.

## Acknowledgement

Our implementation is based on [generative-models](https://github.com/Stability-AI/generative-models) from Stability AI. Thanks for their open-source work.

## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```bibtex
@article{gao2024vista,
 title={Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability}, 
 author={Shenyuan Gao and Jiazhi Yang and Li Chen and Kashyap Chitta and Yihang Qiu and Andreas Geiger and Jun Zhang and Hongyang Li},
 journal={arXiv preprint arXiv:2405.xxxxx},
 year={2024}
}
```
