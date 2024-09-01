## Evidential Storm Forecasting

## Introduction

## Installation

We recommend managing the environment through Anaconda. 

First, create a new conda environment:

```bash
conda create -n edl-st python=3.9
conda activate edl-st
```

Make sure that `pip < 24.1`. If it isn't, run:

```bash
conda install pip=24.0
```

Lastly, install dependencies. SciencePlots requires Latex on your machine, instructions can be found [here](https://github.com/garrettj403/SciencePlots/wiki/FAQ#installing-latex).

```bash
python3 -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install pytorch_lightning==1.6.4
python3 -m pip install xarray netcdf4 opencv-python earthnet==0.3.9
python3 -m pip install omegaconf, matplotlib, SciencePlots
python3 -m pip install torchinfo, h5py, thop
cd ROOT_DIR/edl-stormcast
```

## Dataset
