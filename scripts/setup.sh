#!/bin/bash
conda create -n RobustBind python==3.9
conda activate RobustBind
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda install cartopy
conda install -c conda-forge libstdcxx-ng
pip install -r requirements.txt