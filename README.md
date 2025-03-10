This repository contains code for reproducing the experiments in "InfoGain: Furthering the Design of Diffusion Wavelets for Graph-Structured Data" (Johnson et. al).

## Dependencies

1. This project requires Python>=3.11.
2. It also requires PyTorch. To install the correct version of PyTorch for your system, see [PyTorch's "Getting Started" guide](https://pytorch.org/get-started/locally/).
3. To use the Jupyter notebooks in this repo, first [install Jupyter](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).
4. LEGS requires the `scatter_add` method from the `torch_scatter` package. This project also makes use of several other python packages, which can be installed with pip:
```
pip3 install \
numpy \
pandas \
scipy \
scikit-learn \
matplotlib \
matplotlib-inline \
accelerate \
torch_geometric \
torch-scatter \
torchmetrics
```
## Running experiments

A Jupyter notebook for reproducing the experiments is provided in `infogain_testing/consolidated_experiments.ipynb`.
> Note that you must have Python>=3.11 installed and available as an ipykernel to Jupyter to use this notebook.

## Data availability

Datasets are publicly available and downloadable through pytorch-geometric's datasets library. Note that the provided scripts will auto-download the datasets 'under the hood,' so long as an internet connection is available.

## Notes

- We recommend a project folder structure of:
```
../infogain
    |_code [clone this repo's files into here]
    |_data
    |_models
    |_results
```

- If your system does not support relative paths with "../", it may be helpful to add the path to the project's `code` folder to the `PYTHONPATH` (for importing files as modules), e.g.:
```
export PYTHONPATH="<path/to/mfcn/code>":$PYTHONPATH
```

- Set general experiment arguments in `args_template.py`. Note that the `__post_init__` method in this file is how the project directories on your machine(s) are set. Modify this method with the keys and paths of your choice. (This provides a convenient way to pass this key as a command line argument in the data processing and experiment execution scripts, and set directories correctly when running the scripts on different systems.) Note that many of the arguments in `args_template.py` store default values which are overridden in the task-specific args files.

- Set dataset-specific arguments (which will override those in `args_template.py`) in the respective files in the `infogain_testing` folder.
