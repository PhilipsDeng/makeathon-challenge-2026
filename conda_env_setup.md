# Conda Environment Setup Guide

This guide provides instructions on how to set up a Conda environment with all the necessary dependencies for the Makeathon Challenge 2026 and the AlphaEarth workshop, including a GPU-enabled version of PyTorch.

## 1. Create a New Conda Environment

First, create a new conda environment (let's call it `mkt`) with Python 3.10 (or your preferred version):

```bash
conda create -n mkt python=3.10 -y
conda activate mkt
```

## 2. Install GPU-enabled PyTorch

To install the GPU version of PyTorch, you can use the following command. This example installs PyTorch with CUDA 11.8 support (ensure this matches your system's installed CUDA drivers):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

*(Note: If you are using a different CUDA version, please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for the exact installation command.)*

## 3. Install Required Packages

Next, install the remaining dependencies required by `challenge.ipynb` and `alphaearth_workshop.ipynb`. We recommend installing geospatial and data science packages via `conda` from the `conda-forge` channel to avoid C-dependency conflicts:

```bash
conda install -c conda-forge geopandas numpy matplotlib pandas rasterio shapely boto3 botocore scikit-learn folium leafmap plotly tqdm xarray ipykernel nbformat ipywidgets umap-learn -y
```

Install any remaining packages via `pip`:

```bash
pip install localtileserver gdown
```

## 4. Add the Environment to Jupyter

To ensure your new conda environment is available as a kernel in your Jupyter Notebooks:

```bash
python -m ipykernel install --user --name=mkt --display-name "Python (mkt)"
```

## 5. Verify the Installation

You can verify that PyTorch is installed correctly and can detect your GPU by running the following snippet in Python:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
```

If the output is `True`, your environment is successfully set up and ready to go!
