## Instructions for installing CASCADE on Macbook (including Apple Silicon)

Always perform the following steps on your Macbook terminal

### 1. Install Rosetta and Miniforge3

Because Macbooks with Apple Silicon use the arm64 architecture, Python 3.7 and relevant packages are no longer available. Homebrew has also disabled Python 3.7. Therefore, we need to install Rosetta 2 to build an x86-64 architecture and virtual environment (Macbooks with Intel chips can skip this step)

#### 1.1 Install 'Rosetta 2'
`softwareupdate --install-rosetta`

#### 1.2 Install the Intel (x86_64) version of Miniforge3

Remove the existing arm64 Miniforge if you have it:

`rm -rf ~/miniforge3`

Download Intel version

`curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh -o ~/Miniforge3.sh`

Make it executable

`chmod +x ~/Miniforge3.sh`

Run conda commands with Rosetta 2

`arch -x86_64 bash ~/Miniforge3.sh`

Initialize conda for zsh

`arch -x86_64 ~/miniforge3/bin/conda init zsh`



## 2. Install CASCADE

Close and reopen your terminal, create the virtual environment

`arch -x86_64 conda create -n cascade_env python=3.8`

Activate the virtual environment

`conda activate cascade_env`

Switch to your own path

`cd /your/own/path/CASCADE`

Install CASCADE (comment out the line `python_requires=">=3.7, <3.8"` in `setup.py`)

`pip install -e .`



## 3. Make sure everything is ready (inside the cascade_env)

Python version, should be python3.8.x

`python --version`

Confirm which architecture Python is currently using, should be x86

`python -c "import platform; print(platform.machine())"`

Confirm it has all relevant packages and cascade2p

`conda list `

You can use 'conda deactivate' to exit current virtual environment and 'conda cascade_env' to enter again.


## 4. Fixing potential problems

If you meet 'illegal Instruction Error' when running CASCADE, try with following steps.

### 4.1  Illegal instructions encountered by Tensorflow

When TensorFlow encounters illegal instructions that aren't supported by the CPU, remove the old environment.

`conda env remove -n cascade_env`

Make sure your terminal is selected to open using Rosetta (click 'get info' and select)

Create a fresh environment

`CONDA_SUBDIR=osx-64 conda create -n cascade_env python=3.8`

Open the virtual environment and install cascade again but modifying 'setup.py' Keras==2.4.3 tensorflow==2.2.1

`pip install -e .`

### 4.2 Illegal instructions encountered in Sypder

When Spyder encounters illegal instructions install a more stable version of Spyder for TensorFlow and x86 architecture

`conda uninstall spyder`

`conda install spyder=4.2.5`

 or use other IDE such as 'VS Code'.





