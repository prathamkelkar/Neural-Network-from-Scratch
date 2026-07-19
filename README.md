# Neural Network from Scratch

A fully-connected (dense) neural network implemented from scratch in NumPy/CuPy — no TensorFlow, no PyTorch — trained and evaluated on the ATLAS Higgs Boson Challenge dataset to classify particle collision events as Higgs boson signal or background noise.

This project was built to understand the mathematics of neural networks at the level of derivation, not just usage: forward propagation, backpropagation via the chain rule, vectorization, gradient descent, mini-batch training, and GPU acceleration are all implemented directly, with the full derivation written up alongside the code.

## What's in this repo

- `nn_helper.py` — the core neural network implementation: forward/backward propagation, cost computation, parameter initialization and updates, mini-batch generation, and training loops (both full-batch and epoch/mini-batch based)
- `report/` — the full written report (PDF and LaTeX source), covering the mathematical derivation from single-neuron logistic regression through to an arbitrary-depth deep network, along with all empirical results
- Supporting notebooks/scripts for data preprocessing and experiments

## Dataset

[ATLAS Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson) (originally released by CERN, hosted via Kaggle) — 250,000 labeled collision events, 30 kinematic and physics-derived features, binary classification (signal vs. background).

## Key results

- Implemented logistic regression through deep, arbitrary-depth fully-connected networks entirely from first principles, including a full mathematical derivation of backpropagation for both the single-neuron and vectorized multi-layer cases
- Best model achieved **~84–85% test accuracy**, compared to a ~65.7% majority-class baseline and ~75% from plain logistic regression on the same features — demonstrating that the classification problem has genuine nonlinear structure a linear model can't fully capture
- Compared 8 network architectures ranging from 944 to 188,417 parameters, revealing a clear bias-variance tradeoff: training accuracy improved steadily with capacity, while test accuracy plateaued and, at the largest scales, became numerically unstable
- Migrated from full-batch to mini-batch gradient descent and added GPU acceleration via `cupy`, cutting training time for a full architecture sweep from several hours to roughly 75 minutes
- Benchmarked results against a prior deep learning implementation on the same dataset ([Ahmad, 2015](https://arxiv.org/abs/1510.02674)), achieving comparable or slightly higher accuracy despite using a substantially smaller, single (non-ensembled) network with far less compute

## Report

Full writeup with derivations, methodology, and results: [Building a Neural Network from Scratch (PDF)](report/Building_a_Neural_Network_from_Scratch.pdf)

## Author

Pratham Kelkar
