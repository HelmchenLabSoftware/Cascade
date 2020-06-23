## CASCADE: Calibrated spike inference of spiking from calcium imaging data

![Concept of supervised inference of spiking activity from calcium imaging data using deep networks](https://github.com/HelmchenLabSoftware/Calibrated-inference-of-spiking/blob/master/Figure%20concept.png)

*Cascade* translates calcium imaging ΔF/F traces into spiking probabilities or discrete spikes.

*Cascade* is described in detail in this preprint [link].

*Cascade's* toolbox consists of

- A large ground truth dataset spanning brain regions, calcium indicators, species
- A deep network that learns a new model (calcium -> spike probability) for each condition and noise level
- A large set of pre-trained deep networks for various conditions
- *(optional)* Tools to quantify the out-of-dataset generalization for a given model and noise level
- *(optional)* A tool to transform spike probabilities into discrete spikes



## Gettings started

#### Without installation

If you want to test the algorithm, just open this online Notebook [link]. It will  also allow you to apply pre-trained models to your own data. No installation will be required since the entire algorithm runs in the cloud (Colaboratory Notebook hosted by Google servers). The entire Notebook is designed to be easily accessible also to researchers with little background in Python, but it is also the best starting point for experienced programmers. The Notebook also includes a comprehensive FAQ section. Try it out - within less than 10 minutes, you will have used the algorithm for the first time!

#### With local installation

If you want to modify the code, integrate the algorithm into your existing pipeline (e.g., with CaImAn or Suite2P), an installation on your local machines is necessary. Important: Although *Cascade* is based on deep networks, GPU-support is not strictly necessary. Therefore, the installation process is much easier than for deep learning-based toolboxes like DeepLabCut that require GPU-based processing.

We recommend the following installation procedure, but many other options are possible as well:

- clone / download repository
- Anaconda installation
- Keras with Tensorflow, yaml.ruaml, etc. pp.

For developers who want to train their own models or who want to systematically study the algorithm, we recommend GPU-based deep learning frameworks. Provide a link to a tutorial on how to set this up (e.g., DeepLabCut tutorial).


## Typical work flow

Describe data format for inputs and functions that read the data. 

Describe output data format and how it can be interpreted correctly. 

Describe the main functions used for predictions, together with the identifier for each model.

## How it works ...

Describe ground truth datasets, how they are resampled. 

Provide a lot of references to key figures and sections in the paper, how predictions generalize; single-spike precision; etc.

## Under the hood

Describe the main methods, how they use functions in the utils-folders and what they do.

Describe where the neural network is defined, etc. pp.

## Frequently asked questions

copy from the Colab Notebook


