#!/bin/bash
# Verify Conda installation: (https://conda.io/docs/user-guide/install/index.html)
# return 1 if global command line program installed, else 0
# example
function program_is_installed {
  # set to 1 initially
  local return_=1
  # set to 0 if not found
  type $1 >/dev/null 2>&1 || { local return_=0; }
  # return value
  echo "$return_"
}

#check if conda is installed, if not install it with brew
if [ $(program_is_installed conda) -eq 1 ]
then
  echo "conda installed"
  echo 'source ~/.bash_functions' >> ~/.bashrc
else
    if [ $(program_is_installed brew) != 1 ]
    then
        ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    fi

    brew install wget

    #install miniconda
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bash_functions 
fi

conda create -y -n deepdiva_test python=3.5

echo 'Created environment....'

source activate deepdiva_test

#set pythonpath
echo 'export PYTHONPATH=$PWD:$PYTHONPATH' >> ~/.bash_functions
export PYTHONPATH=$PWD:$PYTHONPATH

conda install -y pytorch torchvision -c pytorch
conda install -y numpy=1.14.1 scipy=1.0.0 scikit-learn=0.19.1 matplotlib=2.1.2 x264=20131218 seaborn=0.8.1 colorlog=3.1.2 pillow
conda install -y -c conda-forge opencv=3.3.1

pip install tensorboard==1.6.0 tensorboard-logger==0.1.0 tensorboardx==1.1 tensorflow==1.6.0 tensorflow-tensorboard==1.5.1 sigopt==3.0.0 tqdm==4.19.6


# Congratulate user on success
echo "You're the best! Everything worked!"
