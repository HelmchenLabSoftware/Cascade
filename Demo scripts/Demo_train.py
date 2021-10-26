#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Demo script to train a network based on:

  - a range of noise levels
  - a given frame rate
  - a selection of ground truth data sets

The model configuration is defined based on a couple of parameters (sampling rate,
training data sets, noise levels, ground truth smoothing). A folder is generated
on the hard disk with the name 'model_name'.

Finally, the model is trained, "cascade.train_model( model_name )" and the trained
models are saved to disk.

"""


"""

Import python packages

"""

import os

if "Demo scripts" in os.getcwd():
    sys.path.append(os.path.abspath(".."))  # add parent directory to path for imports
    os.chdir("..")  # change to main directory
print("Current directory: {}".format(os.getcwd()))

# perform checks to catch most likly import errors
from cascade2p import checks  # TODO: put all of this in one function

print("\nChecks for packages:")
checks.check_packages()

from cascade2p import cascade


"""

Configure model and its parameters

"""

cfg = dict(
    model_name="Universal_30Hz",  # Model name (and name of the save folder)
    sampling_rate=30,  # Sampling rate in Hz (round to next integer)
    training_datasets=[
        "DS03-Cal520-m-S1",
        "DS04-OGB1-zf-pDp",
        "DS05-Cal520-zf-pDp",
        "DS06-GCaMP6f-zf-aDp",
        "DS07-GCaMP6f-zf-dD",
        "DS08-GCaMP6f-zf-OB",
        "DS09-GCaMP6f-m-V1",
        "DS10-GCaMP6f-m-V1-neuropil-corrected",
        "DS11-GCaMP6f-m-V1-neuropil-corrected",
        "DS12-GCaMP6s-m-V1-neuropil-corrected",
        "DS13-GCaMP6s-m-V1-neuropil-corrected",
        "DS14-GCaMP6s-m-V1",
        "DS15-GCaMP6s-m-V1",
        "DS16-GCaMP6s-m-V1",
        "DS17-GCaMP5k-m-V1",
        "DS18-R-CaMP-m-CA3",
        "DS19-R-CaMP-m-S1",
        "DS20-jRCaMP1a-m-V1",
    ],
    noise_levels=[
        noise for noise in range(2, 9)
    ],  # int values of noise values (do not use numpy here => representer error!)
    smoothing=0.2,  # std of Gaussian smoothing in time (sec)
    causal_kernel=0,  # causal ground truth smoothing kernel
    # Advanced:
    # For additional parameters, you can find their names in the helper_scripts/config.py
    # file in the config_template string
)


"""

Generate folder on hard disk for model

"""

cascade.create_model_folder(cfg)

print('\nTo load this model, use the model name "{}"'.format(cfg["model_name"]))


"""

Train model for all datasets and noise levels

"""

model_name = cfg["model_name"]
cascade.train_model(model_name)
