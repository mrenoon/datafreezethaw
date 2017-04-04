# Implementation of Freeze-Thaw Bayesian Optimization and Data-Freeze-Thaw Bayesian Optimization

## Setup ##
### Step 1:  Install the prerequisites: ###
    sudo apt-get install libeigen3-dev
    cd RoBO/
    for req in $(cat requirements.txt); do pip install $req; done'''
### Step 2: install our modified version of RoBO: ###
    python setup.py install
