[![DOI](https://zenodo.org/badge/241174650.svg)](https://zenodo.org/badge/latestdoi/241174650)
[![License](https://img.shields.io/badge/License-GPL--3.0-brightgreen)](https://github.com/HelmchenLabSoftware/Cascade/blob/master/LICENSE)
[![Size](https://img.shields.io/github/repo-size/HelmchenLabSoftware/Cascade?style=plastic)](https://img.shields.io/github/repo-size/HelmchenLabSoftware/Cascade?style=plastic)
[![Language](https://img.shields.io/github/languages/top/HelmchenLabSoftware/Cascade?style=plastic)](https://github.com/HelmchenLabSoftware/Cascade)

## Cascade: Calibrated spike inference from calcium imaging data

<!---![Concept of supervised inference of spiking activity from calcium imaging data using deep networks](https://github.com/HelmchenLabSoftware/Calibrated-inference-of-spiking/blob/master/etc/Figure%20concept.png)--->
<p align="center"><img src="https://github.com/HelmchenLabSoftware/Calibrated-inference-of-spiking/blob/master/etc/CA1_deconvolution_CASCADE.gif "  width="85%"></p>

*Cascade* translates calcium imaging ΔF/F traces into spiking probabilities or discrete spikes.

*Cascade* is described in detail in **[the main paper](https://www.nature.com/articles/s41593-021-00895-5)**. There are follow-up papers which describe the application of Cascade to **[spinal cord data](https://www.biorxiv.org/content/10.1101/2024.07.17.603957)** and the application of Cascade to **[GCaMP8](https://www.biorxiv.org/content/10.1101/2025.03.03.641129)**.

*Cascade's* toolbox consists of

- A large and continuously updated ground truth database spanning brain regions, calcium indicators, species
- A deep network that is trained to predict spike rates from calcium data
- Procedures to resample the training ground truth such that noise levels and frame rates of calcium recordings are matched
- A large set of pre-trained deep networks for various conditions
- Tools to quantify the out-of-dataset generalization for a given model and noise level
- A tool to transform inferred spike rates into discrete spikes

Get started quickly with the following two *Colaboratory Notebooks*:

## [Spike inference from calcium data](https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb)

Upload your calcium data, use Cascade to process the data, download the inferred spike rates.

Spike inference with Cascade improves the temporal resolution, denoises the recording and provides an absolute spike rate estimate.

No parameter tuning, no installation required.

You will get started within few minutes.

[Spike inference from calcium data](https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb)

<p align="center">
<a href="https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb" rel="Spike inference from calcium data, showing activations of intermediate network layers"><img src="https://github.com/HelmchenLabSoftware/Calibrated-inference-of-spiking/blob/master/etc/Network_activations_output.gif "  width="85%"></a>
</p>

## Updates to ground truth datasets and pretrained models:

***2025-05-01*** - Our study of spike inference for **calcium imaging data from spinal cord** is now published in the Journal of Neuroscience. Check out the [openly accessible manuscript](https://doi.org/10.1523/JNEUROSCI.1187-24.2025), which also includes the reviewer reports and rebuttal letters. Cascade models pretrained on spinal cord ground truth are already available, and the ground truth with both excitatory and inhibitory spinal cord neurons is already part of this repository's ground truth database (datasets #40 and #41).

***2025-04-25*** - New [blog post](https://gcamp6f.com/2025/04/25/accuaretly-computing-noise-levels-for-calcium-imaging-data/) about the standardized noise level $\nu$ that can be used to compare shot noise levels across calcium imaging recordings. The blog post investigas the effect of true noise levels, imaging rates and true neuronal spike rates on the measured noise metric.

***2025-04-10*** - First models for spike inference with interneurons, primarily based on GCaMP8 data, are now available. The models are described in our recent [preprint](https://www.biorxiv.org/content/10.1101/2025.03.03.641129v1) in Figure 6. Please check out this [FAQ section](https://github.com/HelmchenLabSoftware/Cascade?tab=readme-ov-file#can-i-apply-cascade-to-data-recorded-with-interneurons) for more details about these models and their potential limitations.

***2025-03-24*** - Models for online spike inference are now uploaded and available. The models are characterized in Figure 5 in our [preprint](https://www.biorxiv.org/content/10.1101/2025.03.03.641129v1), and the model selection is described in more practical terms in this [blog post](https://gcamp6f.com/2025/03/24/online-spike-inference-with-gcamp8/). If you are into online spike inference, definitely check it out!

***2025-03-21*** - New [blog post](https://gcamp6f.com/2025/03/21/detecting-single-spikes-from-calcium-imaging/) on the detection of isolated single spikes with CASCADE for GCaMP8 vs. other indicators such as GCaMP6, GCaMP7 and X-CaMP-Gf.

***2025-03-14*** - New [blog post](https://gcamp6f.com/2025/03/14/non-linearity-of-calcium-indicators-history-dependence-of-spike-reporting/) on the non-linearity of calcium indicators (GCaMP6 vs. GCaMP8) and how this feature affects spike inference.

***2025-03-10*** - **[New preprint](https://www.biorxiv.org/content/10.1101/2025.03.03.641129)** on spike inference with **GCaMP8**. The paper studies spike inference with specifically GCaMP8-trained models, for the algorithms Cascade, MLSpike and OASIS. The analyses also provide insights into the consequences of the non-linearity of GCaMP8 and GCaMP6 variants, and the potential for single-action potential-detection with GCaMP8 vs. other indicators.

***2025-01-02*** - Updated **spinal cord ground truth data**. The ground truth data for spinal cord (describedy in this [preprint](https://www.biorxiv.org/content/10.1101/2024.07.17.603957v1)) are now updated and also contain the field *stim*, which indicates timepoints of electrical dorsal root stimulation.

***2024-08-22*** - New **models pretrained with GCaMP8** ground truth are now available for Cascade. They are briefly described in this **[blog post](https://gcamp6f.com/2024/08/22/spike-inference-with-gcamp8-new-pretrained-models-available/)** with a coarse comparison of the model with previous Cascade models. A more detailed analysis of these models and their application to GCaMP8 data will follow in a few months!

***2024-07-23*** - **[A new preprint](https://www.biorxiv.org/content/10.1101/2024.07.17.603957v1)** about Cascade, where it is applied to **calcium imaging data from spinal cord**. Cascade models pretrained on spinal cord ground truth are already available, and the ground truth with both excitatory and inhibitory spinal cord neurons is already part of this repository's ground truth database (datasets #40 and #41).

***2024-06-26*** - Peter Rupprecht presents a poster at the FENS conference in Vienna about ongoing work on spike inference with **GCaMP8**, and about spike inference in **spinal cord** in mice.

***2024-06-02*** - Models and ground truth datasets for **GCaMP6s in spinal cord** in mice (**excitatory/inhibitory transgenic**) are added (datasets #40 and #41), trained for imaging rates of 2.5, 3 and 30 Hz. Additional models for spinal cord datasets are trained upon request. A preprint on the datasets and models will be released within the next months.

***2024-02-08*** - Models and ground truth datasets for **GCaMP7f** and **GCaMP8f/m/s** will be added in a few months. See issue [#43](https://github.com/HelmchenLabSoftware/Cascade/issues/43) for a preliminary discussion on GCaMP7f.

***2021-12-01*** - Spike times for dataset #1 were found to be misaligned (see issue [#28](https://github.com/HelmchenLabSoftware/Cascade/issues/28)). The corrected dataset was uploaded and replaced the previous dataset #1.

***2021-12-01*** - Some neurons in dataset #15 exhibited a systematic delay with calcium signals with respect to spike times. This delay was corrected based on inspection of spike-triggered averages. Dataset #15 was replaced with the corrected dataset.

***2021-12-11*** - Neuron #8 in dataset #20 was removed since calcium signals were found to be unrelated to simultaneously recorded spike patterns. Most likely, calcium imaging and electrophysiology were performed from two different neurons.



## [Exploration of the ground truth database](https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Explore_ground_truth_datasets.ipynb)

Explore the 35 ground truth data sets and browse through >400 neurons.

Zoom into single events and observe calcium responses (or lack thereof) to single spikes.

Indicators: GCaMP8s, GCaMP8m, GCaMP8f, GCaMP7f, GCaMP6f, GCaMP6s, R-CaMP, jRCaMP, jRGECO, GCaMP5k, OGB-1, Cal-520.

Mouse pyramidal cells in visual and somatosensory cortices, interneurons, hippocampal principal cells; zebrafish forebrain and olfactory bulb.

[Exploration of the ground truth database](https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Explore_ground_truth_datasets.ipynb)

Just click on the link or the images!

<p align="center">
<a href="https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Explore_ground_truth_datasets.ipynb" rel="Tool for the exploration of the ground truth database"><img src="https://github.com/HelmchenLabSoftware/Calibrated-inference-of-spiking/blob/master/etc/Exploration%20tool.gif"  width="85%"></a>
</p>


## Getting started

#### Without installation

If you want to try out the algorithm, just open **[this online Colaboratory Notebook](https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb)**, as advertised above. With the Notebook, you can apply the algorithm to existing test datasets, or you can apply **pre-trained models** to **your own data**. No installation will be required since the entire algorithm runs in the cloud (Colaboratory Notebook hosted by Google servers; a Google account is required). The entire Notebook is designed to be easily accessible for researchers with little background in Python, but it is also the best starting point for experienced programmers. The Notebook includes a comprehensive FAQ section. Try it out - within a couple of minutes, you can start using the algorithm!

#### With a local installation (Ubuntu/Windows)

If you want to modify the code, if you want to integrate the algorithm into your existing pipeline (e.g., with CaImAn or Suite2P), or if you want to train your own networks, an installation on your local machine is necessary. Important: Although *Cascade* is based on deep networks, **GPU-support is not necessary**, it runs smoothly without (of course, GPUs speed up the processing). Therefore, the installation is much easier than for typical deep learning-based toolboxes that require GPU-based processing.

We recommend the following installation procedure, but many other options are possible as well:

1. Download / clone the repository to your local computer
2. Install the Python environment Anaconda with Python 3 (https://www.anaconda.com/distribution/)
3. Use the Anaconda prompt (Windows) or the console to navigate to the main folder where you downloaded *Cascade*
4. Create a new Anaconda environment with the required packages:

    * For a CPU installation (slower, recommended if you will not train a network):

         ``conda create -n Cascade python=3.7 tensorflow==2.3 keras==2.3.1 h5py numpy scipy matplotlib seaborn ruamel.yaml spyder``. 
    * For a GPU installation (faster, recommended if you will train networks): 
        
        ``conda create -n Cascade python=3.7 tensorflow-gpu==2.4.1 keras h5py numpy scipy matplotlib seaborn ruamel.yaml spyder`` (Linux)

        ``conda create -n Cascade python=3.7 tensorflow-gpu==2.3.0 keras h5py numpy scipy matplotlib seaborn ruamel.yaml spyder`` (Windows)

    Conda environments with Python 3.8 seem to work equally well. Earlier versions of Cascade (pre-2022) were based on Tensorflow 2.1. Installations with Tensorflow 2.1 can still work, but are deprecated, since there might arise problems when using newer pretrained models.

5. Activate the new environment using ``conda activate Cascade`` in Ubuntu and ``activate Cascade`` on Windows
6. Use your editor of choice (e.g., Spyder or PyCharm) to get started with the demo files: type ``spyder`` in the console after activating the environment.<br> If you want to use the Jupyter demo Notebooks, you have to install ipython via ``pip install ipython ipykernel`` and make it visible in the new environment via ``python -m ipykernel install --user --name Cascade``. Then start the Jupyter notebook in the browser from the activated environment via ``jupyter notebook``, and do not forget to select the environment in the menu (Kernel -> Change Kernel -> Cascade). If you encounter problems, the internet is your friend (for example, [here](https://stackoverflow.com/questions/58068818/how-to-use-jupyter-notebooks-in-a-conda-environment))
7. Now you're ready to process your data on your local computer!

If you have an existing Python environment, you can also try simply installing the missing dependencies. Optionally, a Docker file together with instructions can be found in the ``etc`` folder of the repository. If you are interested in training models from scratch and speed up processing in general, you should use a dedicated GPU and install a GPU-based version of the deep learning framework (for the extensive analyses in the paper, we used a GeForce RTX 2080 Ti). This procedure can be challenging for beginners. You will find instructions for that via Google search, but a good starting point is the tutorial provided by [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/installation.md).

#### With a local installation (macOS)

For more recent Macbook versions, the Apple Silicon chip makes it challenging to install the packages recommended for Cascade (however, CASCADE user [Matúš Halák](https://github.com/matushalak) has recently found a nice and simple solution, [check it out!](https://github.com/HelmchenLabSoftware/Cascade/issues/46#issuecomment-2765456770)). There is a [related issue](https://github.com/HelmchenLabSoftware/Cascade/issues/46) on this topic. We provide instructions to install Cacade with Rosetta that is compatible with Apple Silicon chips: [Link to instructions](https://github.com/HelmchenLabSoftware/Cascade/blob/master/etc/Instructions_Mac2024.md).


For older macOS versions, we provide instructions on how to run Cascade with the old (Intel) chips:

On macOS there is an issue with the Tensorflow build provided by Conda (see https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial ). There are several workarounds, but it seems to be most reliable to install Tensorflow and Keras using `pip`. To do so, download/clone the Github repository to your local computer, navigate to the main folder of the repository, and follow these steps:
1. Create a new Anaconda environment with required base packages: `conda env create -f etc/environment_mac.yml`
2. Activate the environment: `conda activate Cascade`
3. Install Tensorflow / Keras: `pip install -r etc/requirements_mac.txt`
4. If you want to use the Jupyter demo Notebooks, make the new environment visible in Jupyter: `ipython kernel install --user --name=Cascade`

This recipe has been tested on macOS 10.15 (Catalina).





## Typical work flow

The average user will only use pretrained models to produce predictions for his/her own data, and the [Colaboratory Notebook](https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb) should in most cases be sufficient. The description of the complete work flow (below) is not necessary but helpful to understand what the algorithm does.


**Train a model with ground truth (optional)**

This section can be reproduced with the ``Demo_train.py`` file.

The user specifies the properties of the model: the sampling rate of the calcium imaging data, the ground truth datasets used for training, the range of noise levels, the smoothing of the ground truth and whether a causal kernel is used. For an explanation what the last two adjustments mean, please read the FAQ below.

Then, a folder is created, where the configuration parameters of the model and the trained deep networks are stored.

Finally, the real training procedure is started with ``cascade.train_model( model_name )``. For each noise level, the algorithm resamples the indicated ground truth datasets such that the resampled ground truth matches the target noise level and imaging rate. The resampled ground truth is then used to train the deep network, and the trained network is saved to disk. For each noise level, several models are trained to create more robust predictions (ensemble approach).

**Make predictions with your data**

This section can be reproduced with the ``Demo_predict.py`` file.

First, a function is defined that loads the calcium traces. In the default configuration, the input data should be a matrix (number of neurons x time points) that has been saved as a \*.-mat-file in Matlab or as a \*.npy-file in Python. Usually, we name the variable ``dF_traces``. However, we also give instructions on how to easily adapt the function to your requirements.

Next, the user indicates the path of the file that should be processed and the frame rate. We recommend to plot example traces to see whether everything went well.

Now, the user indicates the model of the (already trained) model and performs the spike inference with the command ``spike_prob = cascade.predict( model_name, traces )``. The input (``traces``) is processed by the model (``model_name``) to produce the spiking probabilities as output (``spike_prob``). The spiking probabilities are given at the same sampling rate as the input calcium recording.

Finally, the predictions are saved to disk.

**Convert to discrete spikes (optional)**

This section can be reproduced with the ``Demo_discrete_spikes.py`` file.

In this section, the output from the previous step (``spike_prob``) is loaded from disk. Single spikes are fitted into the smooth probabilities such that the most likely spike sequence is recovered. For optimal fitting, the parameters of the model used for spike predictions has to be loaded as well (``model_name``). The result of the procedure are spike times. They are given with the same temporal precision as the sampling rate of the calcium recording.

We do not recommend discrete spike predictions except for outstanding high-quality recordings and refer to the FAQ and the paper ([link](https://www.nature.com/articles/s41593-021-00895-5), Fig. S7 and Supplementary Note 3) for a discussion.

**Quantify expected performance of the model (optional)**

This section can be reproduced with the ``Demo_benchmark_model.py`` file.

To understand how good predictions are, it is important to quantify the performance of a given trained model. As discussed in depth in the [paper](https://www.nature.com/articles/s41593-021-00895-5), this is best measured by quantifying the performance when training the deep network on all except one ground truth dataset and test it on the held-out dataset.

To do this systematically, a lot of training and testing needs to performed, and we do not recommend this procedure for CPU-only installations.

The input of this step is the model (``model_name``), while the output is a set of metrics (correlation, error, bias; see the paper for discussion and details) that quantify the expected performance of the algorithm when applied to unseen datasets.

## Under the hood

If you want to understand how the code works, you will be surprised how simple the code is.

All main functions are described in the ``cascade2p/cascade.py`` file, including the functions ``cascade.train()`` and ``cascade.predict()``.

Some helper functions to load the ground truth data for training (which is a bit more challenging due to the initial diversity of ground truth datasets) and to plot results are contained in the ``cascade2p/utils.py`` file. In addition, this file also contains the definition of the deep network ``define_model()``, which is only a few lines. If you want to use a different architecture for training (see Fig. S8 in the [paper](https://www.nature.com/articles/s41593-021-00895-5), it is very simple to modify or replace.

Functions used to convert spiking probabilities into discrete spikes can be found in the file ``cascade/utils_discrete_spikes.py``. 

The ``cascade/config.py`` contains the default model parameters. Fig. S4 in the [paper](https://www.nature.com/articles/s41593-021-00895-5) shows that changing those parameters does not greatly affect the prediction quality, such that the user does not need to change any of the hyper-parameters.

The folder ``Ground_truth`` contains all ground truth datasets. The folder also contains a Matlab script and a Python script which can be used to explore the ground truth data. Highly recommended, it's very interesting!

The folder ``Example_datasets`` contains population calcium imaging datasets that can be used to test the algorithm if no own data are available.

The folder ``Pretrained_models`` contains pre-trained models.

Any more questions? Probably you will find the answer below!


## Frequently asked questions

#### What does the output of the algorithm mean?

>The output **spike_prob** is the _expected number of spikes_ in this time bin, at the same resolution as the original calcium recording. This metric is also called _spike probability_ for brevity in the paper and elsewhere. If you sum over the trace in time, you will get the estimated **number of spikes**. If you multiply the trace with the frame rate, you will get an estimate of the instantaneous **spike rate**. Spike probability and spike rates can therefore be converted by multiplication with the frame rate.

#### Can **spike_prob** be larger than 1?

>Yes. As described above ("What does the output of the algorithm mean?"), the output of the algorithm is strictly speaking not a probability and therefore not restricted to values between 0 and 1. A value >1 indicates that the estimated number of spikes in the time bin is larger than 1. 

#### How large would a single spike be?
>This depends on your frame rate (Hz) and on the smoothing (standard deviation, milliseconds) of your model. Use the following script to compute the spike probability shape for given parameters.
> 
> **Smoothing** is the standard deviation of the Gaussian used to smooth the ground truth spike rate before it is used for training. In the file name of a pretrained model, the smoothing parameter is indicated. Read below for more details.
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

>This depends mainly on the **shot noise level** of your dataset. If you want to compute how well the chosen model generalizes to unseen data for a given noise level, check out the Github repository and use [the demo script](https://github.com/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Demo_benchmark_model.py) which computes the performance of a given model.
>
>To get a good idea about the quality of predictions to unseen data, check out **Figure 3** and the associated discussion in the [paper](https://www.nature.com/articles/s41593-021-00895-5).

#### Can I apply CASCADE to data recorded with GCaMP8? 

>Yes. We have tested the global models with the new GCaMP8 datasets (available via the [GCaMP8 paper](https://www.nature.com/articles/s41586-023-05828-9)). The standard pretrained models were in general good; the only caveat is that predictions were shifted in time - due to the fast rise time of GCaMP8, inferred spike rates occur earlier than true spike rates. We are currently (March 2024) in the process of making more in-depth analysis of spike inference with GCaMP8 data with systematic validation and benchmarking.

#### Can I apply CASCADE to data recorded with interneurons? 

> Since April 2025, models based on interneuron ground truth have become available. These models are named, for example, `Interneurons_GC8+_30Hz_smoothing50ms_high_noise`. The ground truth dataset for these interneurons consists of 17 putative fast-spiking interneurons, recorded primarily with various GCaMP8 variants, from [Zhang et al. (2023)](https://www.nature.com/articles/s41586-023-05828-9). The models themselves are described in Figure 6 of our preprint [Rupprecht et al. (2025)](https://www.biorxiv.org/content/10.1101/2025.03.03.641129).
>
> It is important to keep in mind that these models trained for application to interneurons are not yet as reliable and well-tested as the models for excitatory neurons. First, the ground truth dataset is still relatively small (17 neurons). Second, as with all ground truth data, the recordings were obtained under light anesthesia. It is likely that the activity patterns of interneurons differ more between awake vs. anesthetized states than those of excitatory neurons. Therefore, please keep these caveats in mind when interpreting the output of spike inference for interneuron data. And if you want to discuss, don't hesitate to reach out or open an issue on GitHub.
>
> It was also noticed (kudos to [Hal Rockwell](https://github.com/hal-rock)) that the computed noise levels may differ slightly between interneurons and excitatory neurons at otherwise similar recording conditions. This may be because the fluorescence levels, which change with spike rate for fast-spiking interneurons, might result in systematically lower noise levels compared to excitatory neurons, which exhibit sharp fluorescence transients from single spikes or bursts. It's not crucial, but if you are interested in noise level comparisons, keep it in mind.
>
> Finally, if you need any further pretrained interneuron models that are not uploaded yet, just let us know via GitHub issues or [e-Mail](mailto:p.t.r.rupprecht+cascade@gmail.com)!

#### Why is the output of the algorithm a probability, why not discrete spikes?

>Good question! We think that providing spike times instead of spike rates or spiking probabilities is misleading, since it suggests a false precision and certainty of the spiking estimates. In addition, we found (**Fig. S7** in the [paper](https://www.nature.com/articles/s41593-021-00895-5)) that single-spike precision could not achieved with any of the ground truth datasets.
>
>However, for some cases, discrete spikes still might be a good approach. We provide a Python function that converts the spiking probability into the most likely underlying discrete spikes (**[demo](https://github.com/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Demo_discrete_spikes.py)** on Github).


#### Why are the first and last datapoints of the predictions NaNs?

>The deep network uses a window that looks at the calcium trace around the current time point to better understand the context of the current time point. For the first and last points in time, the network is unable to look into the environment and therefore gives back NaNs. If the window size of the network is 64 datapoints (which is the default), the first and last 32 time points will be NaNs.

#### Should I smooth the dF/F traces before I feed them into CASCADE?

>No! Check out [issue #53](https://github.com/HelmchenLabSoftware/Cascade/issues/53).

#### I get a certain noise level for my recordings. What is good or bad?

>For an illustration of different noise levels, check out Extended Data Fig. 3 in the [paper](https://www.nature.com/articles/s41593-021-00895-5). To give an example, the Allen Brain Observatory Visual Coding dataset is of very high imaging quality, with noise levels around **1, which is very good** (unit: $\small \%·s^{-1/2}$ ). A noise level of **3-4 is still decent**, especially for population imaging with many neurons. Noise levels **above 5 indicates rather poor signal** levels. For a definition of the noise level, check out the Methods of the preprint.
>
>However, even for excellent shot noise levels, the recording quality can be bad due to bad imaging resolution, **neuropil contamination** and, most importantly, **movement artifacts**. See Extended Data Fig. 5 in the [paper](https://www.nature.com/articles/s41593-021-00895-5) and the associated text as well as the Discussion for more details.


#### How do I select an appropriate model for my data?

> Each model is trained on a resampled ground truth dataset, as described in the preprint. The training dataset is resampled at the desired frame rate and at multiple noise levels. The model automatically chooses the model with matching noise-levels for each neuron. You only have to select the correct frame rate (which is indicated in the model name).
>
>If you do not have a specific ground truth for your dataset, it is typically best (see Fig. 3 and the associated discussion in the [paper](https://www.nature.com/articles/s41593-021-00895-5)) to use a model that has been trained on all available datasets (called 'Global EXC Model').
>
>There are two additional model specifications that you can choose, "causal" kernels and "smoothing". The choice of these specifications does not make a model better or worse, but better or less well suited for your needs. See the following two questions!

#### What does the "Global EXC" for some of the models mean?

> "Global EXC" indicates that the model has been trained on a diverse set of ground truth datasets from excitatory neurons. It should work very well on unseen data from excitatory neurons without any retraining (as described in Fig. 3 in the [paper](https://www.nature.com/articles/s41593-021-00895-5)).
> 
>  The datasets used to train the "Global EXC model" include diverse indicators (GCaMP6f, GCaMP6s, OGB-1, GCaMP5k, Cal-520, R-CaMP1.07 and jRCaMP) and diverse brain regions (visual cortex, somatosensory cortex, hippocampus, several areas in the zebrafish forebrain and olfactory bulb). The olfactory bulb dataset also includes some inhibitory neurons, which were included in the training dataset because their spike-to-calcium relationship is similar to the excitatory datasets. Interneuron datasets (datasets #22-#26) were not included in the training dataset because their inclusion would compromise the overall performance of the global model for excitatory neurons.

#### What does the **smoothing** parameter for the models mean?

> The ground truth which has been used to train the model has been slightly smoothed with a Gaussian kernel. This is a processing step which helps the deep network to learn quicker and more reliably. However, this also means that the predictions will be smoothed in a similar fashion. How to choose these parameters optimally?
>
> From our experience, at a frame rate of 7.5 Hz, a smoothing kernel with standard deviation of 200 ms is appropriate; for nicely visible transients, also a value of 100 or 50 ms can be tried out, and we have had cases where this was the most satisfying choice of parameters. At 30 Hz, a smoothing kernel of 50 ms works well, but a smoothing kernel of 25 ms could be tried as well if the data quality is good and if one wants to avoid temporally smooth predictions. If the calcium imaging quality is not ideal, it can make sense to increase the smoothing kernel standard deviation. In the end, it is always a trade-off between reliability and optimal learning (more smoothing) and temporal precision (less smoothing of the ground truth). The impact of temporal smoothing on the quality of the inference is discussed in Extended Data Fig. 9 in the [paper](https://www.nature.com/articles/s41593-021-00895-5).
>
> However, if you use our suggested default specifications, you should be good!


#### What does the "causal" mean for some of the models?

> By default, the ground truth is smoothed symmetrically in time. This means, also the predicted spike probabilities are symetrically distributed in time around the true time point. In some cases, this can be a problem because this predicts non-zero neuronal spiking probability before the calcium event had even started. Especially when you want to analyze stimulus-triggered activity patterns, this is an important issue and a common problem for all deconvolution algorithms.
> 
> However, if the ground truth is smoothed not with a symmetric Gaussian but with a smooth causal kernel, this limitation can be circumvented (discussed in detail in Fig. S12 in the [paper](https://www.nature.com/articles/s41593-021-00895-5)), and spiking activity is almost exclusively assigned to time points after the calcium event started. It must be noted that this reliable causal re-assignment of activity works well for high-quality datasets, but in case of higher noise levels, any deconvolution algorithm will assign activity to non-causal time points. Good to keep in mind when you interpret your results!


#### How can I use CASCADE for online spike inference?

> Models for online spike inference can be identified from the model name that contains an "Online" at the beginning. Currently, we have pretrained models for online spike inference with GCaMP6 and GCaMP8 for frame rates of 30 Hz and 60 Hz, at moderate noise levels only (assuming that nobody will attempt online spike inference with low latency for extremely noisy data!). If you need additional models, get in touch by opening a [Github](https://github.com/HelmchenLabSoftware/Cascade) issue or via [e-Mail](mailto:p.t.r.rupprecht+cascade@gmail.com).
> 
> How can you select a good online spike inference model for you? Check out Figure 5 in our [preprint](https://www.biorxiv.org/content/10.1101/2025.03.03.641129v1), and read our related [blog post](https://gcamp6f.com/2025/03/24/online-spike-inference-with-gcamp8/). If you then still have question, get in touch via email or just open an issue on GitHub!


#### None of the models is good for me. What can I do?

> First of all, is this really true? For example, if you have recorded at 30.5 Hz, you can also use a model trained at 30 Hz imaging rates. A deviation by less than 5\% of the imaging rate is totally okay in our experience!
>
> If however you want to use an entirely different model, for example a model trained at a sampling rate of 2 Hz, or a model only trained with a specific ground truth dataset, you have two options. 1) You go to the [Github page](https://github.com/HelmchenLabSoftware/Cascade) and follow the instructions on how to train you own model. This can be done even without GPU-support, but it will take some time (on the other hand, you only have to do this once). 2) You contact us via [e-Mail](mailto:p.t.r.rupprecht+cascade@gmail.com) and tell us what kind of model you would like to have. We will train it for you and upload it to our repository. Not only you, but everybody will then be able to use it further on.



#### I have my own ground truth dataset. How can I use it?

> You have two options.
>
> Either you process the data yourself. You can inspect the ground truth datasets, which consist of Matlab structs saved as a file for each neuron from the [ground truth](https://github.com/HelmchenLabSoftware/Cascade/tree/master/Ground_truth). If you process your ground truth recordings into the same format, you can use it as a training set and train the model yourself. All instructions are indicated at the Github repository.
>
> Or you can contact us, and we help to process your dataset if it meets certain quality standards. We can process raw calcium and ephys recordings, but of course extracted dF/F traces and spike times would be even better. Yes, we will do the work for you. But only under the condition that the processed dataset will then be integrated into the published set of ground truth datasets, where it is openly accessible to everybody. Please get in touch with us to discuss options on how to get credit for the recording of the dataset, which we will discuss case by case.

#### I want to use my own ground truth dataset, but I don't want to share it.

> As mentioned, you can process the ground truth dataset yourself. However, we will only help you with the dataset is made public afterwards.


#### Can I use the algorithm also locally, *e.g.*, within [CaImAn](https://github.com/flatironinstitute/CaImAn), or in my own pipeline?

> Sure! We have done this ourselves with CaImAn and our custom analysis pipelines. Your starting point to do this will not be the Colaboratory Notebook, but rather the [Github repository](https://github.com/HelmchenLabSoftware/Cascade). Check out the demo scripts. They are very easy to understand and will show you which functions you have to use and how. If you have successfully used the Colaboratory Notebook, understanding the demo scripts will be a piece of cake.


#### Can I use Cascade as well for endoscopic 1p calcium imaging data?

> One of the key features of Cascade is that it infers absolute spike rates. To achieve this, it is necessary that dF/F values extracted from neuronal ROIs are approximately correct. For endoscopic 1p calcium imaging data, the background fluorescence is often extremely high, and complex methods for subtraction of global or local background activity are used (e.g., by [CNMF-E](https://elifesciences.org/articles/28728)). As also discussed in the CNMF-E paper, extraced traces therefore cannot be properly transformed into dF/F values in cases of high background. Quantitative deconvolution cannot be applied in such cases (be it with Cascade or another algorithm), but qualitative deconvolution of the timecourse is still possible with Cascade (recommended units are then "arbitrary units" instead of "estimated spike rate [Hz]").

#### I would like to look at the ground truth data.

> We actually recommend this to anybody who is doing calcium imaging at cellular resolution. Looking at the ground truth data of simultaneous calcium and juxtacellular recording is very enlightening. In the [Github repository](https://github.com/HelmchenLabSoftware/Cascade), we have deposited an interactive tool to conveniently visualize all ground truth datasets. It is available as a [Colaboratory Notebook](https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Explore_ground_truth_datasets.ipynb).


#### Which reference should I cite?


> Please cite the [paper](https://www.nature.com/articles/s41593-021-00895-5) as a reference for Cascade:
>
> Rupprecht P, Carta S, Hoffmann A, Echizen M, Blot A, Kwan AC, Dan Y, Hofer SB, Kitamura K, Helmchen F\*, Friedrich RW\*, *A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging*, Nature Neuroscience (2021).
>
> (\* = co-senior authors)
>
> If you use models trained for spinal cord or GCaMP8 data, please also refer to the following papers:
>
> Rupprecht P, Rózsa M, Fang X, Svoboda K, Helmchen F. *[Spike inference from calcium imaging data acquired with GCaMP8 indicators](https://www.biorxiv.org/content/10.1101/2025.03.03.641129)*, bioRxiv (2025).
>
> Rupprecht P, Fan W, Sullivan S, Helmchen F, Sdrulla A. *[Spike rate inference from mouse spinal cord calcium imaging data](https://www.biorxiv.org/content/10.1101/2024.07.17.603957)*, bioRxiv (2024).
>
> If you use the respective ground truth datasets directly, please also refer to the original papers and the associated dataset:
> 
> Rupprecht P, Carta S, Hoffmann A, Echizen M, Blot A, AC Kwan, Dan Y, Hofer SB, Kitamura K, Helmchen F\*, Friedrich RW\*, *A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging*, Nature Neuroscience (2021), for datasets \#3-8, \#19 and \#27.
> 
> Schoenfeld G, Carta S, Rupprecht P, Ayaz A, Helmchen F, *In vivo calcium imaging of CA3 pyramidal neuron populations in adult mouse hippocampus*, eNeuro (2021), for dataset \#18.
> 
> Chen TW, Wardill TJ, Sun Y, Pulver SR, Renninger SL, Baohan A, Schreiter ER, Kerr RA, Orger MB, Jayaraman V, Looger LL. *Ultrasensitive fluorescent proteins for imaging neuronal activity*, Nature (2013), for datasets \#9 and \#14.
>
> Huang L, Ledochowitsch P, Knoblich U, Lecoq J, Murphy GJ, Reid RC, de Vries SE, Koch C, Zeng H., Buice MA, Waters J, Lu Li, *Relationship between simultaneously recorded spiking activity and fluorescence signal in GCaMP6 transgenic mice*, eLife (2021), for datasets \#10, \#11, \#12 and \#13.
>
> Berens P, et al. *Community-based benchmarking improves spike rate inference from two-photon calcium imaging data*, PLoS Comp Biol (2018), for datasets \#1, \#15, \#16.
> 
> Akerboom J, Chen TW, Wardill TJ, Tian L, Marvin JS, Mutlu S, Calderón NC, Esposti F, Borghuis BG, Sun XR, Gordus A. *Optimization of a GCaMP calcium indicator for neural activity imaging*, J Neuroscience (2012), for dataset \#17.
> 
> Bethge P, Carta S, Lorenzo DA, Egolf L, Goniotaki D, Madisen L, Voigt FF, Chen JL, Schneider B, Ohkura M, Nakai J. *An R-CaMP1.07 reporter mouse for cell-type-specific expression of a sensitive red fluorescent calcium indicator*, PloS ONE (2017), for dataset \#19.
>
> Tada M, Takeuchi A, Hashizume M, Kitamura K, Kano M. *A highly sensitive fluorescent indicator dye for calcium imaging of neural activity in vitro and in vivo*, EJN (2014), for dataset \#3.
> 
> Dana H, Mohar B, Sun Y, Narayan S, Gordus A, Hasseman JP, Tsegaye G, Holt GT, Hu A, Walpita D, Patel R. *Sensitive red protein calcium indicators for imaging neural activity*, Elife (2016), for datasets \#20 and \#21.
>
> Khan AG, Poort J, Chadwick A, Blot A, Sahani M, Mrsic-Flogel TD, Hofer SB. *Distinct learning-induced changes in stimulus selectivity and interactions of GABAergic interneuron classes in visual cortex*, Nature Neuroscience (2018), for datasets \#24-26.
>
> Kwan AC, Dan Y. *Dissection of cortical microcircuits by single-neuron stimulation in vivo*, Current Biology (2012), for datasets \#2 and \#22-23.
