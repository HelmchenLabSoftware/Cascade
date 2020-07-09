## Cascade: Calibrated spike inference from calcium imaging data

![Concept of supervised inference of spiking activity from calcium imaging data using deep networks](https://github.com/HelmchenLabSoftware/Calibrated-inference-of-spiking/blob/master/Figure%20concept.png)

*Cascade* translates calcium imaging ΔF/F traces into spiking probabilities or discrete spikes.

*Cascade* is described in detail in **this preprint [link will be posted when available!]**.

*Cascade's* toolbox consists of

- A large ground truth database spanning brain regions, calcium indicators, species
- A deep network that is trained to predict spiking activity from calcium data
- Procedures to resample the training ground truth such that noise levels and frame rates of calcium recordings are matched
- A large set of pre-trained deep networks for various conditions
- Tools to quantify the out-of-dataset generalization for a given model and noise level
- A tool to transform spike probabilities into discrete spikes



## Getting started

#### Without installation

If you want to try out the algorithm, just open **[this online Notebook](https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb)**. With the Notebook, you can apply the algorithm to existing test datasets, or you can apply **pre-trained models** to **your own data**. No installation will be required since the entire algorithm runs in the cloud (Colaboratory Notebook hosted by Google servers; a Google account is required). The entire Notebook is designed to be easily accessible for researchers with little background in Python, but it is also the best starting point for experienced programmers. The Notebook includes a comprehensive FAQ section. Try it out - within a couple of minutes, you can start using the algorithm!

#### With a local installation

If you want to modify the code, if you want to integrate the algorithm into your existing pipeline (e.g., with CaImAn or Suite2P), or if you want to train your own networks, an installation on your local machine is necessary. Important: Although *Cascade* is based on deep networks, **GPU-support is not necessary**, it runs smoothly without (of course, GPUs speed up the processing). Therefore, the installation is much easier than for typical deep learning-based toolboxes that require GPU-based processing.

We recommend the following installation procedure, but many other options are possible as well:

1. Download / clone the repository to your local computer
2. Install the Python environment Anaconda with Python 3 (https://www.anaconda.com/distribution/)
3. Use the Anaconda prompt (Windows) or the console to navigate to the main folder where you downloaded *Cascade*
4. Create a new Anaconda environment with the required packages: ``conda create -n Cascade python=3.6 tensorflow==2.1.0 keras==2.3.1 numpy scipy matplotlib seaborn numpy ruamel.yaml spyder``. Other versions of python will work as well, we have mainly worked with Python 3.6 and 3.7.
5. Activate the new environment using ``conda activate Cascade`` in Ubuntu and ``activate Cascade`` on Windows
6. Use your editor of choice (e.g., Spyder or PyCharm) to get started with the demo files: type ``spyder`` in the console after activating the environment.<br> If you want to use the Jupyter demo Notebooks, you have to install ipython via ``pip install ipython ipykernel`` and make it visible in the new environment via ``python -m ipykernel install --user --name Cascade``. Then start the Jupyter notebook in the browser from the activated environment via ``jupyter notebook``, and do not forget to select the environment in the menu (Kernel -> Change Kernel -> Cascade). If you encounter problems, the internet is your friend (for example,e [here](https://stackoverflow.com/questions/58068818/how-to-use-jupyter-notebooks-in-a-conda-environment))
7. Now you're ready to process your data on your local computer!

If you have an existing Python environment, you can also try simply installing the missing dependencies. If you are interested in training models from scratch and speed up processing in general, you should use a dedicated GPU and install a GPU-based version of the deep learning framework (for the extensive analyses in the paper, we used a GeForce RTX 2080 Ti). This procedure can be challenging for beginners. You will find instructions for that via Google search, but a good starting point is the tutorial provided by [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/installation.md).


## Typical work flow

The average user will only use pretrained models to produce predictions for his/her own data, and the [Colaboratory Notebook](https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb) should in most cases be sufficient. The description of the complete work flow (below) is, however, helpful to understand what the algorithm does.


**Train a model with ground truth (optional)**

This section can be reproduced with the ``Demo_train.py`` file.

The user specifies the properties of the model: the sampling rate of the calcium imaging data, the ground truth datasets used for training, the range of noise levels, the smoothing of the ground truth and whether a causal kernel is used. For an explanation what the last two adjustments mean, please read the FAQ below.

Then, a folder is created, where the configuration parameters of the model and the trained deep networks are stored.

Finally, the real training procedure is started with ``cascade.train_model( model_name )``. For each noise level, the algorithm resamples the indicated ground truth datasets such that the resampled ground truth matches the target noise level and imaging rate. The resampled ground truth is then used to train the deep network, and the trained network is saved to disk. For each noise level, several models are trained to create more robust predictions (ensemble approach).

**Make predictions with your data**

This section can be reproduced with the ``Demo_predict.py`` file.

First, a function is defined that loads the calcium traces. In the default configuration, the input data should be a matrix (number of neurons x time points) that has been saved as a \*.-mat-file in Matlab or as a \*.npy-file in Python. Usually, we name the variable ``dF_traces``. However, we also give instructions on how to easily adapt the function to your requirements.

Next, the user indicates the path of the file that should be processed and the frame rate. We recommend to plot example traces to see whether everything went well.

Now, the user indicates the model of the (already trained) model and performs the spike inference with the command ``spike_rates = cascade.predict( model_name, traces )``. The input (``traces``) is processed by the model (``model_name``) to produce the spiking probabilities as output (``spike_rates``). The spiking probabilities are given at the same sampling rate as the input calcium recording.

Finally, the predictions are saved to disk.

**Convert to discrete spikes (optional)**

This section can be reproduced with the ``Demo_discrete_spikes.py`` file.

In this section, the output from the previous step (``spike_rates``) is loaded from disk. Single spikes are fitted into the smooth probabilities such that the most likely spike sequence is recovered. For optimal fitting, the parameters of the model used for spike predictions has to be loaded as well (``model_name``). The result of the procedure are spike times. They are given with the same temporal precision as the sampling rate of the calcium recording.

We do not recommend discrete spike predictions except for outstanding high-quality recordings and refer to the FAQ and the paper ([link will be posted when available!], Fig. S19XX) for a discussion.

**Quantify expected performance of the model (optional)**

This section can be reproduced with the ``Demo_benchmark_model.py`` file.

To understand how good predictions are, it is important to quantify the performance of a given trained model. As discussed in depth in the paper [link will be posted when available!], this is best measured by quantifying the performance when training the deep network on all except one ground truth dataset and test it on the held-out dataset.

To do this systematically, a lot of training and testing needs to performed, and we do not recommend this procedure for CPU-only installations.

The input of this step is the model (``model_name``), while the output is a set of metrics (correlation, error, bias; see the paper for discussion and details) that quantify the expected performance of the algorithm when applied to unseen datasets.

## Under the hood

If you want to understand how the code works, you will be surprised how simple the code is.

All main functions are described in the ``cascade2p/cascade.py`` file, including the functions ``cascade.train()`` and ``cascade.predict()``.

Some helper functions to load the ground truth data for training (which is a bit more challenging due to the initial diversity of ground truth datasets) and to plot results are contained in the ``cascade2p/utils.py`` file. In addition, this file also contains the definition of the deep network ``define_model()``, which is only a few lines. If you want to use a different architecture for training (see Fig. S16 in the paper [link will be posted when available!]), it is very simple to modify or replace.

Functions used to convert spiking probabilities into discrete spikes can be found in the file ``cascade/utils_discrete_spikes.py``. 

The ``cascade/config.py`` contains the default model parameters. Fig. S15 ([link will be posted when available!]) shows that changing those parameters does not greatly affect the prediction quality, such that the user does not need to change any of the hyper-parameters.

The folder ``Ground_truth`` contains all ground truth datasets. The folder also contains a Matlab script and a Python script which can be used to explore the ground truth data. Highly recommended, it's very interesting!

The folder ``Example_datasets`` contains population calcium imaging datasets that can be used to test the algorithm if no own data are available.

The folder ``Pretrained_models`` contains pre-trained models.

Any more questions? Probably you will find the answer below!


## Frequently asked questions

#### What does the output of the algorithm mean?

>The output is the estimated probability of action potentials, at the same resolution as the original calcium recording. If you sum over the trace in time, you will get the estimated **number of action potentials**. If you multiply the trace with the frame rate, you will get an estimate of the instantaneous **firing rate**.

#### How large would a single spike be?
>This depends on your frame rate (Hz) and on the smoothing (standard deviation, milliseconds) of your model. Use the following script to compute the spike probability shape for given parameters.
> 
```python
from scipy.ndimage.filters import gaussian_filter
import numpy as np

sampling_rate = 30
smoothing = 50

single_spike = np.zeros(1001,)
single_spike[501] = 1
single_spike_smoothed = gaussian_filter(single_spike.astype(float), sigma=smoothing/1e3*sampling_rate)

gaussian_amplitude = np.round(np.max(single_spike_smoothed)*1000)/1000
gaussian_width = np.round(2*np.sqrt(2*np.log(2))*smoothing/1e3*100)/100
```

#### How precise and good are my predictions? 

>This depends mainly on the **shot noise level** of your dataset. If you want to compute how good the chosen model generalizes to unseen data for a given noise level, check out the Github repository and use [the demo script](https://github.com/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Demo_benchmark_model.py) which computes the performance of a given model.
>
>If you want to get a good idea about the quality of predictions, check out **Figures 3 and 4**, as well as the corresponding supplementary figures in the paper/preprint [link will be made available when online!].

#### Why is the output of the algorithm a probability, why not discrete spikes?

>Good question! We think that providing spike times instead of spiking probabilities is misleading, since it suggests a false precision and certainty of the spiking estimates. In addition, we found (**Fig. SXX** in the preprint) that single-spike precision could not achieved with any of the ground truth datasets.
>
>However, for some cases, discrete spikes still might be a good approach. We provide a Python function that converts the spiking probability into discrete spikes (**[demo](https://github.com/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Demo_discrete_spikes.py)** on Github).


#### Why are the first and last datapoints of the predictions NaNs?

>The deep network uses a window that looks at the calcium trace around the current time point to better understand the context of the current time point. For the first and last points in time, the network is unable to look into the environment and therefore gives back NaNs. If the window size of the network is 64 datapoints (which is the default), the first and last 32 time points will be NaNs.


#### I get a certain noise level for my recordings. What is good or bad?

>For an illustration of different noise levels, check out Fig. SXX in the preprint. To give an example, the Allen Brain Observatory Visual Coding dataset is of very high imaging quality, with noise levels around **1, which is very good** (unit is percent / sqrt(seconds)). A noise level of **3-4 is still decent**, especially for population imaging with many neurons. Noise levels **above 5 indicates rather poor signal** levels. For a definition of the noise level, check out the Methods section of the preprint.
>
>However, even for excellent shot noise levels, the recording quality can be bad due to bad imaging resolution, **neuropil contamination** and, most importantly, **movement artifacts**. See Fig. S10 in the preprint and the associated text as well as the Discussion for more details .


#### How do I select an appropriate model for my data?

> Each model is trained on a resampled ground truth dataset. The training dataset is resampled at the desired frame rate and at multiple noise levels. The model automatically chooses the model with matching noise-levels for each neuron. You only have to select the correct frame rate (which is indicated in the model name).
>
>If you do not have a specific ground truth for your dataset, it is typically best (see Fig. 4 and the associated discussion in the paper) to use a model that has been trained on all available datasets (called 'Universal Model').
>
>There are two additional model specifications that you can choose, "causal" kernels and "smoothing". The choice of these specifications does not make a model better or worse, but better or less well suited for your needs. See the following two questions!

#### What does the "smoothing" for some of the models mean?

> The ground truth which has been used to train the model has been slightly smoothed with a Gaussian kernel. This is a processing step which helps the deep network to learn quicker and more reliably. However, this also means that the predictions will be smoothed in a similar fashion. How to choose these parameters optimally?
>
> From our experience, at a frame rate of 7.5 Hz, a smoothing kernel with standard deviation of 200 ms is appropriate. At 30 Hz, a smoothing kernel of 50 ms works well. If the calcium imaging quality is not ideal, it can make sense to increase the smoothing kernel standard deviation. In the end, it is always a trade-off between reliability and optimal learning (more smoothing) and temporal precision (less smoothing of the ground truth).
>
> If you use our suggested default specifications, you should be good!


#### What does the "causal" mean for some of the models?

> By default, the ground truth is smoothed symmetrically in time. This means, also the predicted spike probabilities are symetrically distributed in time around the true time point. In some cases, this can be a problem because this predicts non-zero neuronal spiking probability before the calcium event had even started. Especially when you want to analyze stimulus-triggered activity patterns, this is an important issue and a common problem for all deconvolution algorithms.
> 
> However, if the ground truth is smoothed not with a symmetric Gaussian but with a smooth causal kernel, this limitation can be circumvented (discussed in detail in Fig. SXX in the preprint), and spiking activity is almost exclusively assigned to time points after the calcium event started. It must be noted that this reliable causal re-assignment of activity works well for high-quality datasets, but in case of higher noise levels, any deconvolution algorithm will assign activity to non-causal time points. Good to keep in mind when you interpret your results!


#### None of the models is good for me. What can I do?

> First of all, is this really true? For example, if you have recorded at 30.5 Hz, you can also use a model trained at 30 Hz imaging rates. A deviation by less than 5\% of the imaging rate is totally okay in our experience!
>
> If however you want to use an entirely different model, for example a model trained at a sampling rate of 2 Hz, or a model only trained with a specific ground truth dataset, you have two options. 1) You go to the [Github page](https://github.com/HelmchenLabSoftware/Cascade) and follow the instructions on how to train you own model. This can be done even without GPU-support, but it will take some time (on the other hand, you only have to do this once). 2) You contact us via [e-Mail](p.t.r.rupprecht+cascade@gmail.com) and tell us what kind of model you would like to have. We will train it for you and upload it to our repository. Not only you, but everybody will then be able to use it further on.



#### I have my own ground truth dataset. How can I use it?

> You have two options.
>
> Either you process the data yourself. You can inspect the ground truth datasets, which consist of Matlab structs saved as a file for each neuron from the [ground truth](https://github.com/HelmchenLabSoftware/Cascade/tree/master/Ground_truth). If you process your ground truth recordings into the same format, you can use it as a training set and train the model yourself. All instructions are indicated at the Github repository.
>
> Or you can contact us, and we help to process your dataset if it meets certain quality standards. We can process raw calcium and ephys recordings, but of course extracted dF/F traces and spike times would be even better. Yes, we will do the work for you. But only under the condition that the processed dataset will then be integrated into the published set of ground truth datasets, where it is openly accessible to everybody. Please get in touch with us to discuss options on how to get credit for the recording of the dataset, which we will discussed case by case.

#### I want to use my own ground truth dataset, but I don't want to share it.

> As mentioned, you can process the ground truth dataset yourself. However, we will only help you with the dataset is made public afterwards.


#### Can I use the algorithm also locally, *e.g.*, within [CaImAn](https://github.com/flatironinstitute/CaImAn), or in my own pipeline?

> Sure! We have done this ourselves with CaImAn and our custom analysis pipelines. Your starting point to do this will not be this Colaboratory Notebook, but rather the [Github repository](https://github.com/HelmchenLabSoftware/Cascade). Check out the demo scripts. They are very easy to understand and will show you which functions you have to use and how. If you have successfully used this Colaboratory Notebook, understanding the demo scripts will be a piece of cake.

#### I would like to look at the ground truth data.

> We actually recommend this to anybody who is doing calcium imaging at cellular resolution. Looking at the ground truth data of simultaneous calcium and juxtacellular recording is very enlightening. In the [Github repository](https://github.com/HelmchenLabSoftware/Cascade), you will find scripts both for Python and Matlab to conveniently visualize ground truth recordings. Just download the Github repository and use the scripts in Matlab or Python (**to be done**).




#### Which reference should I cite?

> Please cite the preprint [link will be posted when the preprint is online!] as a reference.

