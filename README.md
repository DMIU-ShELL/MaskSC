# Lifelong Context-Aware Retrieval and Composition for Scalable Knowledge Reuse
Code for paper [Lifelong Context-Aware Retrieval and Composition for Scalable Knowledge Reuse](https://arxiv.org/abs/2212.11110).
Implementation of modulatory mask combined with PPO. The repository contains MASK RI/LC/BLC implementations. Please see EWC branch for implemenation of PPO and Online EWC.
Implementation of Mask Selective Combination (Mask-SC) with PPO which introduces task similarity-based retrieval of prior policies for composition to improve the scalability of forward transfer over long task sequences in lifelong reinforcement learning.

The code was developed on top of the [Mask-LRL](https://github.com/dlpbc/mask-lrl) repository, extending the Mask-LC algorithm.

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2506.05577)

![MaskSC Diagram](assets/MaskSC.png)
> Figure 1: High-level illustration of Mask Selective Combination.

## Evaluation environments
- [CT-graph](https://github.com/soltoggio/CT-graph)
- [Minigrid](https://github.com/Farama-Foundation/gym-minigrid)
- [MiniHack](https://github.com/samvelyan/minihack)
- [Continual World](https://github.com/awarelab/continual_world) (see note below)

## Requirements
- See requirements.txt file
- See [CT-graph](https://github.com/soltoggio/CT-graph) requirements.
- See [Minigrid](https://github.com/Farama-Foundation/gym-minigrid) requirements.
- See [MiniHack](https://github.com/samvelyan/minihack) requirements.
- See [Continual World](https://github.com/dlpbc/continual_world) requirements and how to install. Note, MuJoCo (now freely available) is required to run Continual World

## Usage
Example commands below using [CT-graph](https://github.com/soltoggio/CT-graph) environment.
To run agents in the CT-graph CT28 curriculum defined in the paper, use the command below:

```
# baseline PPO agent.
python train_ctgraph.py baseline --seed 86

# selective retrieval + composition (MASK SC) agent.
python train_ctgraph.py ll_supermask --new_task_mask linear_comb --seed 86
```

Full experiments were run using the commands:
```
python launcher.py --env ctgraph_sc --exp ct28_mask_sc/
```
which will produce the path `./log/ct28_mask_sc/` containing all seed runs.

The `launcher.py` script can also be used to run the single-task experts experiments which can be used to 

Note: 
- the command to run a balanced linear combination (MASK BLC) agent is the same as the MASK LC command above, but should be run in the `exp_maskblc` git branch.
- the full list of commands to run experiments in the paper can be found in the `paper_experiments.txt` file.
- sample commands and the full list of commands for `ewc` experiments in the paper can be found in the `exp_ewc` git branch. 
- sample commands and the full list of commands for setting up the single task expert (STE) experiments can be found in the `exp_ste` git branch.
- In the continualworld curriculum (CW10), the random initialization mask agent implemented in this branch is the MASK RI\_C (continuous values mask). The sample command to run MASK RI_\D in CW10 can be found in the `exp_maskri_discrete_mask_cw10` git branch.

## Analysis
The analysis pipeline for Mask-SC can be executed using the `eval_XXX.py` and `ft_auc_analysis.py` scripts. Single-task experts can be run via the 


#### BibTex
To cite this work, please use the information below. Thanks.
```
@article{esbn2022masklrl,
  title={Lifelong Reinforcement Learning with Modulating Masks},
  author={Ben-Iwhiwhu, Eseoghene and Nath, Saptarshi and Pilly, Praveen K and Kolouri, Soheil and Soltoggio, Andrea},
  journal={arXiv preprint arXiv:2212.11110},
  year={2022}
}
```

## Maintainers
This repository is currently maintained by researchers from Loughborough University.

## Bug Reporting
If you encounter any bugs using the code, please raise an issue in this repository on Github.

## Note on Continual World
The Continual World benchmark was built on top of the [Meta-World](https://github.com/rlworkgroup/metaworld) benchmark, which comprise of a number of simulated robotics tasks. The originally released Continual World employed the use of version 1 (v1) Meta-World environments. However, the Meta-World v1 environments contained some issues in the reward function (discussed [here](https://github.com/rlworkgroup/metaworld/issues/226) and [here](https://github.com/awarelab/continual_world/issues/2)) which was fixed in the updated v2 environments. Therefore, the experiments in the paper employed the use of the v2 environment for each task in the Continual World. The modification can be downloaded from the forked repository [here](https://github.com/dlpbc/continual_world).

## Acknowledgements
TBD
