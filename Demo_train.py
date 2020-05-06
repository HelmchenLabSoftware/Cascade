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
    model_name = 'OGB_pDp_7.5Hz',    # Model name (and name of the save folder)
    sampling_rate = 5,    # Sampling rate in Hz (round to next integer)
    
    training_datasets = [
        'DS03-OGB1-zf-pDp',
                        ],
    
    noise_levels = [noise for noise in range(2,7)],  # int values of noise values (do not use numpy here => representer error!)
    
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



