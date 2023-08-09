# Score-based Denoising

This is a wrapper for *OVITO* around the "Score-based denoising for atomic structure identification" presented in the [graphite repo](https://github.com/LLNL/graphite/) by Lawrence Livermore National Lab. Further information and the official citation is [here](https://doi.org/10.48550/arXiv.2212.02421).

## Description
[[Full description]]

## Parameters 
- `steps` / "Number of denoising steps": Number of denoising interations taken. More iterations require more time. You can check the mean displacement per iteration graph to assess convergence.
- `scale` / "Nearest neighbor distance": Estimation of the nearest neighbor distance used to scale the coordinates before they are input into the model. If this is `None` OVITO will try to estimate the correct nearest neighbor distance. 
- `structure` / "Crystal structure / material system": Allows you to select one of: "FCC", "BCC", "HCP", or "SiO2", depending on your input structure. Note, that an SiO2 structure requires a type named "Si". 
- `device` / "Device": Allows you to select your computing device from: "cpu", "cuda", "mps". Only available devices will be shown. Please read the "Installation" section for additional information.

## Example
[[Usage example]]

## Installation
- *OVITO PRO* built-in Python interpreter
```
ovitos -m pip install --user git+https://github.com/nnn911/ScoreBasedDenoising.git
``` 
- Standalone Python package or Conda environment
```
pip install --user git+https://github.com/nnn911/ScoreBasedDenoising.git
```
- Please note that the `--user` tag is recommended but optional and depends on your Python installation.

By default this will install the CPU version of [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io). 

On Mac, the `mps` backend will also be presented. This is mostly for future proofing since currently not all required PyTorch and PyG methods have been ported to `mps`.

On other platforms you can install the CUDA accelelerated versions of PyTorch and PyG yourself. At this point, you should be able to select `CUDA` in the modifier device selection to run model inference on GPU.

## Technical information / dependencies
Tested on:
    - OVITO == 3.9.1
    - torch == 1.11.0 | 2.0.1
    - torch-geometric == 2.0.4 | 2.3.1

## Contact
Daniel Utt utt@ovito.org