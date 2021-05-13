
"""

This script evaluates spike inference as a function of the number of data points after the time point of interest. For the purpose of online processing, it would be desirable to keep the number of used data points after the time point of interest as low as possible, to reduce possible delays during closed-loop processing.

More context and the results of this script are describe in this blog post: https://gcamp6f.com/2021/05/13/online-spike-rate-inference-with-cascade/

Peter Rupprecht, 2021-05-13


"""



"""

Import dependencies

"""


import os
if 'Demo scripts' in os.getcwd(): os.chdir('..')  # change to main directory
print('Current directory: {}'.format( os.getcwd() ))

# perform checks to catch import errors
from cascade2p import checks    # TODO: put all of this in one function
print('\nChecks for packages:')
checks.check_packages()

from cascade2p import cascade

import keras
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter
import numpy as np

# perform checks to catch most likly import errors
from cascade2p import checks
print('\nChecks for packages:')
checks.check_packages()

from cascade2p import cascade
from cascade2p import config
from cascade2p import utils



"""

Select datasets used for training (always: leave-one-dataset-out strategy)
Evaluate not on all datasets (only GCaMP6s/f in mouse pyramidal cells) (indices 6-14)

"""


training_datasetsX = ['DS03-Cal520-m-S1',
   'DS04-OGB1-zf-pDp',
   'DS05-Cal520-zf-pDp',
   'DS06-GCaMP6f-zf-aDp',
   'DS07-GCaMP6f-zf-dD',
   'DS08-GCaMP6f-zf-OB',
   'DS09-GCaMP6f-m-V1',
   'DS10-GCaMP6f-m-V1-neuropil-corrected',
   'DS11-GCaMP6f-m-V1-neuropil-corrected',
   'DS12-GCaMP6s-m-V1-neuropil-corrected',
   'DS13-GCaMP6s-m-V1-neuropil-corrected',
   'DS14-GCaMP6s-m-V1',
   'DS15-GCaMP6s-m-V1',
   'DS16-GCaMP6s-m-V1',
   'DS17-GCaMP5k-m-V1',
   'DS18-R-CaMP-m-CA3',
   'DS19-R-CaMP-m-S1',
   'DS20-jRCaMP1a-m-V1',
                   ]


"""

Train a model with all datasets and then test with the remaining dataset
For each model, correlation, error and bias will be computed

Initialize results: 3 noise levels, 6 different fraction of the window after time t, 8 datasets

"""


correlation = np.zeros((3,6,8))
error = np.zeros((3,6,8))
bias = np.zeros((3,6,8))


# go through 3 noise levels (2 = good, 4 = average, 8 = rather high noise)
for iii,noise_xx in enumerate([2,4,8]):
    
    # "post_fraction" is the fraction of the input window that falls after the current time point. This results in 1, 2, 4, 8, 16 or 32 data points.
    for kkk,post_fraction in enumerate([0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5]):

	# go through all datasets
        for index in range(len(reduced_evaluation_set)):
            
	  # Used previously to selectively evaluate program (not necessary)
          if 1: 
              
              print(iii,kkk,index)

              ## Define and train model


	      training_datasetsXX = deepcopy(training_datasetsX)
              test_dataset = training_datasetsXX.pop(index+6);

	      # generate model with certain properties; important property here: "before_frac"
              cfg = dict(
                  model_name = 'Test_30Hz_'+str(noise_xx)+'_'+str(post_fraction)+'_'+test_dataset,    # Model name (and name of the save folder)
                  sampling_rate = 30,    # Sampling rate in Hz (round to next integer)

                  training_datasets = training_datasetsXX,

                  noise_levels = [noise for noise in range(noise_xx,noise_xx+1)],  # int values of noise values (do not use numpy here => representer error!)

                  smoothing = 0.05,     # std of Gaussian smoothing in time (sec)
                  causal_kernel = 0,   # causal ground truth smoothing kernel
                  before_frac = 1-post_fraction,
                  windowsize = 64,
                # Advanced:
                # For additional parameters, you can find their names in the helper_scripts/config.py
                # file in the config_template string
                  )

              cfg['training_datasets'] = training_datasetsXX
              #cfg['model_name'] = 'temporary model52'
              cfg['noise_levels'] = [noise_xx]

              cascade.create_model_folder( cfg )

	      # Train model
              cascade.train_model( cfg['model_name'])


	      ## Evaluate model on withheld dataset
		
              # extract values from config file into variables
              test_dataset = [os.path.join('Ground_truth', ds) for ds in [test_dataset]]

              # test model with the one remaining test_dataset
              calcium, ground_truth = utils.preprocess_groundtruth_artificial_noise_balanced(
                                          ground_truth_folders = test_dataset,
                                          before_frac = cfg['before_frac'],
                                          windowsize = cfg['windowsize'],
                                          after_frac = 1 - cfg['before_frac'],
                                          noise_level = noise_xx,
                                          sampling_rate = cfg['sampling_rate'],
                                          smoothing = cfg['smoothing'] * cfg['sampling_rate'],
                                          omission_list = [],
                                          permute = 0,
                                          verbose = 1,
                                          replicas = 0)
              calcium = calcium[:,int(cfg['before_frac']*cfg['windowsize']-1),]
              ground_truth = ground_truth[:,]

              # perform predictions
              spike_rates = cascade.predict( cfg['model_name'], calcium.T )
              spike_rates = np.squeeze(spike_rates)

              # take only non-nan values
              nnan_ix = ~np.isnan(spike_rates)
              ground_truth = ground_truth[nnan_ix]
              spike_rates = spike_rates[nnan_ix]


              ## Compute performance metrics
              ground_truth_smooth = gaussian_filter(ground_truth.astype(float), sigma=cfg['smoothing'] * cfg['sampling_rate'])
              spike_rates_smooth = gaussian_filter(spike_rates.astype(float), sigma=cfg['smoothing'] * cfg['sampling_rate'])

              error_diff = (spike_rates_smooth - ground_truth_smooth.T)
              error_pos = np.sum(error_diff[error_diff>0])
              error_neg = np.sum(error_diff[error_diff<0])
              error_total = np.sum(np.abs(error_diff))
              signal = np.sum(ground_truth_smooth)

              error[iii,kkk,index] = error_total/signal
              bias[iii,kkk,index] =  (error_pos+error_neg)/signal
              correlation[iii,kkk,index] = np.corrcoef(ground_truth,spike_rates,rowvar=False)[0,1]

              # delete temporary model from disk
              #model_path = os.path.join('Pretrained_models', cfg['model_name'])
              #shutil.rmtree(model_path)

	      ## Save results to Matlab-compatible file
              import scipy.io as sio
              sio.savemat('results_delays_online_.mat',{'correlation':correlation,'error':error,'bias': bias})
    
