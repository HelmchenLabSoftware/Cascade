#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Demo script to quantify out-of-dataset generalizaton for an already trained model:

The input of this script is an existing model. The output is the estimated performance
of the model ('X') when applied to unseen data.

To compute this estimate, models with all parameters identicial to model 'X' are trained,
with all datasets except a single dataset, and then applied to the remaining dataset.
For each model, correlation, error and bias will be computed.
The distribution of these values is the estimate for performance on unseen datasets.

Please be aware that this procedure involves training of many models (as many as datasets were
used to train model 'X'), which can take a lot of time, also on computers equipped with good GPUs.


"""



"""

Import python packages

"""

import os
import shutil

if 'Demo scripts' in os.getcwd():
    sys.path.append( os.path.abspath('..') ) # add parent directory to path for imports
    os.chdir('..')  # change to main directory
print('Current directory: {}'.format( os.getcwd() ))

import keras
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter
import numpy as np

# perform checks to catch most likly import errors
from cascade2p import checks    # TODO: put all of this in one function
print('\nChecks for packages:')
checks.check_packages()

from cascade2p import cascade
from cascade2p import config
from cascade2p import utils

"""

Check which model to benchmark, and at which noise level

"""

model_name =  'Global_EXC_30Hz_smoothing100ms'
noise_level = 2


"""

Load model and parameters

"""


model_path = os.path.join('Pretrained_models', model_name)
cfg_file = os.path.join( model_path, 'config.yaml')

# Load config file
cfg = config.read_config( cfg_file )

all_training_datasets = cfg['training_datasets']


# Find the corresponding filename of the training dataset
# This additional step is due to name changes of ground truth folders
# Discussed on Github issue #42 on https://github.com/HelmchenLabSoftware/Cascade/issues/42

all_training_datasets_new = all_training_datasets
if 'DS08-GCaMP6f-m-V1' in all_training_datasets:
    
    for k,this_dataset in enumerate(all_training_datasets):
        
        temporary_string = f'{int(this_dataset[2:4])+1:02}' 
        all_training_datasets_new[k] = this_dataset[0:2]+temporary_string+this_dataset[4:]


"""

Train a model with all datasets and then test with the remaining dataset
For each model, correlation, error and bias will be computed
The distribution of these values is the estimate for performance on unseen datasets

"""


correlation = []
error = []
bias = []

for index in range(len(all_training_datasets_new)):

  training_datasets = deepcopy(all_training_datasets_new)
  test_dataset = training_datasets.pop(index);

  cfg['training_datasets'] = training_datasets
  cfg['model_name'] = 'temporary model'
  cfg['noise_levels'] = [noise_level]


  cascade.create_model_folder( cfg )

  cascade.train_model( cfg['model_name'])



  # extract values from config file into variables
  test_dataset = [os.path.join('Ground_truth', ds) for ds in [test_dataset]]

  # test model with the one remaining test_dataset
  calcium, ground_truth = utils.preprocess_groundtruth_artificial_noise_balanced(
                              ground_truth_folders = test_dataset,
                              before_frac = cfg['before_frac'],
                              windowsize = cfg['windowsize'],
                              after_frac = 1 - cfg['before_frac'],
                              noise_level = noise_level,
                              sampling_rate = cfg['sampling_rate'],
                              smoothing = cfg['smoothing'] * cfg['sampling_rate'],
                              omission_list = [],
                              permute = 0,
                              verbose = cfg['verbose'],
                              replicas = 0)
  calcium = calcium[:,32,]
  ground_truth = ground_truth[:,]

  # perform predictions
  spike_rates = cascade.predict( model_name, calcium.T )
  spike_rates = np.squeeze(spike_rates)

  # take only non-nan values
  nnan_ix = ~np.isnan(spike_rates)
  ground_truth = ground_truth[nnan_ix]
  spike_rates = spike_rates[nnan_ix]

  # compute performance metrics
  ground_truth_smooth = gaussian_filter(ground_truth.astype(float), sigma=cfg['smoothing'] * cfg['sampling_rate'])
  spike_rates_smooth = gaussian_filter(spike_rates.astype(float), sigma=cfg['smoothing'] * cfg['sampling_rate'])

  error_diff = (spike_rates_smooth - ground_truth_smooth.T)
  error_pos = np.sum(error_diff[error_diff>0])
  error_neg = np.sum(error_diff[error_diff<0])
  error_total = np.sum(np.abs(error_diff))
  signal = np.sum(ground_truth_smooth)

  error.append(error_total/signal)
  bias.append((error_pos+error_neg)/signal)
  correlation.append(np.corrcoef(ground_truth,spike_rates,rowvar=False)[0,1])

  # delete temporary model from disk
  model_path = os.path.join('Pretrained_models', cfg['model_name'])
  shutil.rmtree(model_path)
"""

Results

- correlation with ground truth, across different datasets
- error
- bias

"""

m_correlation,std_correlation = np.nanmedian(correlation),np.nanstd(correlation)
m_error,std_error = np.nanmedian(error),np.nanstd(error)
m_bias,stdbias = np.nanmedian(bias),np.nanstd(bias)
