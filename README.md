# ConvMamba

## Overview

<details open>
<summary>Code Structures</summary>



- `core/` core training plugins and metrics.
- `methods/` contains training methods for various methods.
- `Model/` contains the control files for various methods and is used to store the model and training results.
- `models/` contains the main network architectures of various methods.
- `modules/` contains network modules and layers.
- `runfiles/` contains various startup scripts.
- `utils/` contains a variety of utilities for model training, model modules, parsers, etc.
- `DataDefine.py` is used to get the dataset and make a dataloader.
- `modelbuild.py` is used to build and initialize the model.
- `modeltrain.py` is used to train, validate, test and inference about model.
- `main.py` is the main function that runs the program.
- `inference.py` is used for model inference.
- `test.py` is used for model test.
</details>


## Overall methodology and relevant experiments and discussions

<p align="center" width="100%">
<img src=".\fig\Overall.svg" width="60%" />
</p>

## Matrices B and C in traditional SSM and selective SSM

<p align="center" width="100%">
<img src=".\fig\selective SSM.svg" width="60%" />
</p>

## Structure of Mamba block

<p align="center" width="100%">
<img src=".\fig\Mamba block.svg" width="60%" />
</p>