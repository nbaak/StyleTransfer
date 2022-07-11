#!/bin/bash

# install dependencies for this tool
# apt packages
# https://stackoverflow.com/questions/66977227/could-not-load-dynamic-library-libcudnn-so-8-when-running-tensorflow-on-ubun
sudo apt install -y nvidia-cuda-toolkit python3-pip python3-opencv swig nvidia-cudnn

# python modules
python3 -m pip install numpy
python3 -m pip install matplotlib
python3 -m pip install tensorflow tensorboard
