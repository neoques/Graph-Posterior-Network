# Improvements on Uncertainty Quantification for Node Classification via Distance-Based Regularization

This repository presents the experiments of the paper: 

Improvements on Uncertainty Quantification for Node Classification via Distance-Based Regularization<br>
Russell Alan Hart, Linlin Yu, Yifei Lou, Feng Chen <br>
Conference on Neural Information Processing Systems (NeurIPS), 2020. 

[paper] [video] coming soon

## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
conda activate gpn2
conda env list
```

## Training & Evaluation

To train the model(s) in the paper, run `train_and_eval.py`

## Cite
Please cite our paper if you use the model or this code in your own work:
```
@inproceedings{
anonymous2023improvements,
title={Improvements on Uncertainty Quantification for Node Classification via Distance Based Regularization},
author={Anonymous},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=MUzdCW2hC6}
}
```
Our code is mostly adapted from :
[Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification](https://arxiv.org/pdf/2110.14012.pdf)<br>
Maximilian Stadler, Bertrand Charpentier, Simon Geisler, Daniel Zügner, Stephan Günnemann<br>
Conference on Neural Information Processing Systems (NeurIPS) 2021.

Please also cite if you use the model or this code in your own work.

```
@incollection{graph-postnet,
title={Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification},
author={Stadler, Maximilian and Charpentier, Bertrand and Geisler, Simon and Z{\"u}gner, Daniel and G{\"u}nnemann, Stephan},
booktitle = {Advances in Neural Information Processing Systems},
volume = {34},
publisher = {Curran Associates, Inc.},
year = {2021}
}
```
