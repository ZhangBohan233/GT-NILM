# GT-NILM
###### *A generative, transferable Non-Intrusive Load Monitoring system.*

### Introduction
This is the official implementation of the NILM system 
introduced in paper *__GT-NILM: A Generative, Transferable Non-Intrusive Load Monitoring System 
Based on Conditional Diffusion Models and Convolutional Neural Networks__* [1]. \
This implementation uses pytorch [2] and is based on __NeuralNILM_Pytorch__ [3].

### Set up
* Create your own virtual environment with Python > 3.8
* Install necessary dependencies such as `numpy`, `pandas`, `matplotlib`, etc.
* Configure deep learning environment with pytorch (GPU edition) ≥ 2.0

The environments we used are listed in the file `environment.yml`. 
If you use `conda`, you can use `conda env create -f environment.yml` to set up the environment.

### Algorithms
This repository includes the original implemented algorithms in __NeuralNILM_Pytorch__ [2]. 
A new algorithm is added, which is our implementation of __GT-NILM__. \
The core part of the two algorithms introduced in [1], referring to the __U-Net-DM__ and __GT-NILM__, 
are implemented in `nilmtk/disaggregate/dm_gated2_pytorch.py`. The conditional diffusion model (DM) 
related components are in `nilmtk/disaggregate/dm/`. The CNN filter is implemented in
`nilmtk/disaggregate/gater_cnn_pytorch.py`.

### Tutorial
Examples of testing these methods are provided in `train_dm.py` and `train_dm_gated.py`, corresponding
to the training and testing of __U-Net-DM__ and __GT-NILM__ in [1], respectively. \
Examples of fine-tuning the DM and the filter CNN are provided in `ft_dm.ipynb` and 
`ft_cnn.ipynb`, respectively.

### References
[1] B. Zhang, F. Luo, Y. He and G. Ranzi, "GT-NILM: A Generative, Transferable Non-Intrusive Load Monitoring System 
Based on Conditional Diffusion Models and Convolutional Neural Networks" \
[2] https://pytorch.org/ \
[3] https://github.com/Ming-er/NeuralNILM_Pytorch/ 
