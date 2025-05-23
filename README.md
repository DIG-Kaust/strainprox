# StrainProx

This repository contains reproducible material for: **StrainProx: High resolution seismic time-strain estimation**

## Overview

StrainProx is a Python package for high-resolution seismic time-strain estimation using total variation regularized inversion and joint inversion and segmentation methods, using as building block proximal operators.

## Project Structure

The repository is organized as follows:

* :open_file_folder: **strainprox**: A Python library that includes routines for the inversion algorithms and plotting functions.
* :open_file_folder: **data**: A folder containing the test data.
* :open_file_folder: **notebooks**: A collection of Jupyter notebooks that demonstrate StrainProx for time-strain inversion with the Hess model, and benckmark time-strain inversion methods.

## Installation

### Option 1: Using the installation script

```bash
./install_env.sh
```

The installation takes some time. If you see `Done!` in your terminal, the setup is complete.

### Option 2: Manual installation

1. Create a conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate strainprox
```

3. Install the package in development mode:

```bash
pip install -e .
```

## Usage

Always activate the environment before running any code:

```bash
conda activate strainprox
```

Then, you can explore the example notebooks in the `notebooks` directory to understand how to use the package.


## Citation

If you use this code in your research, please cite:

```
@article{strainprox2023,
  title={High resolution seismic time-strain estimation},
  author={[Romero, J., Heidrich, W., Casasanta, L., Akcelik, V., Ravasi, M.]},
  journal={[86th EAGE Annual Conference & Exhibition, Toulouse, France, June 2025.]},
  year={2025}
}



