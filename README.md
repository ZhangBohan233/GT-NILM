# GT-NILM
*A Generative, Transferable Non-Intrusive Load Monitoring System*

## Introduction
This repository contains the official implementation of the paper:

> **GT-NILM: A Generative, Transferable Non-Intrusive Load Monitoring System Based on Conditional Diffusion Models and Convolutional Neural Networks** [1]

GT-NILM introduces a novel NILM framework that combines a **conditional diffusion model (DM)** and a **CNN-based gating mechanism** to improve both generation quality and transferability across domains.

This implementation is built with **PyTorch** [2], and extends the architecture and functionality of __NeuralNILM_Pytorch__ [3] that based on the NILM toolkit __nilmtk__ [4].

## Setup

### Environment Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (GPU version recommended)
- Common packages: `numpy`, `pandas`, `matplotlib`, etc.

The environments we used are listed in the file `environment.yml`. If you use `conda`, you can use `conda env create -f environment.yml` to set up the environment.

## Algorithms Included
This repository includes:
* Baseline methods from __NeuralNILM_Pytorch__ [3].
* Our proposed methods from:
  * __U-Net-DM__: Conditional diffusion model (DM) based on a U-Net as denoising network
  * __GT-NILM__<sup>GM</sup>: U-Net-DM enhanced with a CNN-based filter that filters out windows without active appliance usage

### Core Code Components
* `nilmtk/disaggregate/dm_gated2_pytorch.py`: Main implementation of U-Net-DM and GT-NILM
* `nilmtk/disaggregate/dm/`: Modules related to the conditional diffusion model
* `nilmtk/disaggregate/gater_cnn_pytorch.py`: CNN filter used in GT-NILM

## Example Usage
### Training and Testing
* `train_dm.py`: For U-Net-DM
* `train_dm_gated.py`: For GT-NILM<sup>GM</sup>

### Fine-Tuning
* `ft_dm.ipynb`: Fine-tune the diffusion model
* `ft_cnn.ipynb`: Fine-tune the CNN filter

### Data Preparation
Before training, data must be converted to HDF5 (.h5) format and placed under `mnt/` folder.
The __NeuralNILM_Pytorch__ has provided conversion scripts for the commonly used NILM datasets in `nilmtk/dataset_converters/`,
including REDD [5] and UK-DALE [6] we used.

## References
[1] B. Zhang, F. Luo, Y. He and G. Ranzi, "GT-NILM: A Generative, Transferable Non-Intrusive Load Monitoring System 
Based on Conditional Diffusion Models and Convolutional Neural Networks", 2025

[2] https://pytorch.org/

[3] https://github.com/Ming-er/NeuralNILM_Pytorch/ 

[4] https://github.com/nilmtk/nilmtk

[5] J.Z. Kolter and M.J. Johnson, "REDD: A public data set for energy disaggregation research," in _Workshop on data mining applications in sustainability (SIGKDD), San Diego, CA_, 2011.

[6] J. Kelly and W. Knottenbelt, "The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes," _Scientific Data_, vol. 2, pp. 150007, 2015.
