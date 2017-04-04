# Implementation of Freeze-Thaw Bayesian Optimization and Data-Freeze-Thaw Bayesian Optimization

## Setup instructions ##
### Step 1:  Install the prerequisites: ###
    sudo apt-get install libeigen3-dev
    cd RoBO/
    for req in $(cat requirements.txt); do pip install $req; done
### Step 2: install our modified version of RoBO: ###
    python setup.py install
### Step 3: install the CUDA toolkit and CUDA SDK ###
### Step 4: follow the instructions at https://github.com/akrizhevsky/cuda-convnet2/blob/wiki/Compiling.md to compile the codes in cuda-convnet2 and cuda-convnet2-vu ###

## Run FreezeThawBO example on the cifar10 dataset ##
### Step 1: Download the cifar10 dataset from http://www.cs.toronto.edu/~kriz/cifar-10-py-colmajor.tar.gz and uncompress into folder data ###
### Step 2: Change the constants ROOT_ML_DIR and EXAMPLES_LINK in RoBO/robo/task/ml/convnet_cifar_freeze.py with the repository_root_folder and the repository_root_folder/RoBO/examples ###
    python RoBO/examples/FreezeThawBO.py

## Run DataFreezeThawBO example on the cifar10 dataset ##
### Step 1: Download the cifar10 dataset from http://www.cs.toronto.edu/~kriz/cifar-10-py-colmajor.tar.gz and uncompress into folder data ###
### Step 2: ###
    python data/genData.py
### Step 3: Change the constants ROOT_ML_DIR and EXAMPLES_LINK in RoBO/robo/task/ml/var_size_data_freeze_convnet_cifar_fixed.py with the repository_root_folder and the repository_root_folder/RoBO/examples ###
### Step 4: ###
    python RoBO/examples/DataFreezeThawBO.py
