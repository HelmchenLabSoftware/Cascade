#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:25:01 2020

@author: pierre
"""

import os
if 'Notebooks' in os.getcwd(): os.chdir('..')  # change to main directory
print('Current directory: {}'.format( os.getcwd() ))

# perform checks to catch most likly import errors
from cascade2p import checks    # TODO: put all of this in one function
print('\nChecks for packages:')
checks.check_packages()


from cascade2p import cascade

cfg = dict( 
    model_name = 'Universal_30Hz',    # Model name (and name of the save folder)
    sampling_rate = 30,    # Sampling rate in Hz (round to next integer)
    
    training_datasets = ['DS02-Cal520-m-S1',
       'DS03-OGB1-zf-pDp',
       'DS04-Cal520-zf-pDp',
       'DS05-GCaMP6f-zf-aDp',
       'DS06-GCaMP6f-zf-dD',
       'DS07-GCaMP6f-zf-dD',
       'DS08-GCaMP6f-m-V1',
       'DS09-GCaMP6f-m-V1-neuropil-corrected',
       'DS10-GCaMP6f-m-V1-neuropil-corrected',
       'DS11-GCaMP6s-m-V1-neuropil-corrected',
       'DS12-GCaMP6s-m-V1-neuropil-corrected',
       'DS13-GCaMP6s-m-V1',
       'DS14-GCaMP6s-m-V1',
       'DS15-GCaMP6s-m-V1',
       'DS16-GCaMP5k-m-V1',
       'DS17-R-CaMP-m-CA3',
       'DS18-R-CaMP-m-S1',
       'DS19-jRCaMP1a-m-V1',
                       ],
    
    noise_levels = [noise for noise in range(2,10)],  # int values of noise values (do not use numpy here => representer error!)
    
    smoothing = 0.2,     # std of Gaussian smoothing in time (sec)
    
    # Advanced:
    # For additional parameters, you can find their names in the helper_scripts/config.py
    # file in the config_template string
          )


# save parameter as config.yaml file
cascade.create_model_folder( cfg )

print('\nTo load this model, use the model name "{}"'.format( cfg['model_name'] ))


model_name = cfg['model_name']

# train model for all ensembles and noise levels
cascade.train_model( model_name )



