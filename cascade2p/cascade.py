#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""  High level interface to the CASCADE package 

This file contains functions to train networks for spike prediction ('train_model')
and to use existing networks to predict spiking activity ('predict').


A typical workflow for applying an existing network to calcium imaging data,
shown in the "demo_predict.py" script:
  
  1)  Load calcium imaging data as a dF/F matrix
  2)  Load a predefined model; the model should match the properties of the calcium
      imaging dataset (frame rate, noise levels, ground truth datasets)
  3)  Use the model and the dF/F matrix as inputs for the function 'predict'
  4)  Predictions will be saved. Done!

A typical workflow for training a new network would be the following,
shown in the "demo_train.py" script:
  
  1)  Define a model (frame rate, noise levels, ground truth datasets; additional parameters)
  2)  Use the model as input to the function 'train_model'
  3)  The trained models will be saved together with a configuration file (YAML). Done!


Additional functions in this file are used to navigate different models ('get_model_paths', 'create_model_folder',  'verify_config_dict').


"""

import os
import time
import numpy as np

from cascade2p import config



def train_model( model_name ):
  
    """ Train neural network with parameters specified in the config.yaml file in the model folder

    In this function, a model is configured (defined in the input 'model_name': frame rate, noise levels, ground truth datasets, etc.).
    The ground truth is resampled (function 'preprocess_groundtruth_artificial_noise_balanced', defined in "utils.py").
    The network architecture is defined (function 'define_model', defined in "utils.py").
    The thereby defined model is trained with the resampled ground truth data.
    The trained model with its weight and configuration details is saved to disk.
    
    """
    import keras
    from cascade2p import utils

    # TODO: check here for relative vs absolute path definition
    model_folder = os.path.join('Pretrained_models', model_name)

    # load cfg dictionary from config.yaml file
    cfg = config.read_config( os.path.join(model_folder, 'config.yaml') )
    verbose = cfg['verbose']

    if verbose:
        print('Used configuration for model fitting (file {}):\n'.format( os.path.join(model_folder, 'config.yaml') ))
        for key in cfg:
            print('{}:\t{}'.format(key, cfg[key]))

        print('\n\nModels will be saved into this folder:', model_folder)

    start = time.time()


    # add base folder to selected training datasets
    training_folders = [os.path.join('Ground_truth', ds) for ds in cfg['training_datasets']]

    # Update model fitting status
    cfg['training_finished'] = 'Running'
    config.write_config(cfg, os.path.join( model_folder,'config.yaml' ))

    nr_model_fits = len( cfg['noise_levels'] ) * cfg['ensemble_size']
    print('Fitting a total of {} models:'.format( nr_model_fits))

    curr_model_nr = 0

    print(training_folders[0])
            
    for noise_level in cfg['noise_levels']:
        for ensemble in range( cfg['ensemble_size'] ):
            # train 'ensemble_size' (e.g. 5) models for each noise level

            curr_model_nr += 1
            print('\nFitting model {} with noise level {} (total {} out of {}).'.format(
                    ensemble+1, noise_level, curr_model_nr, nr_model_fits))

            
            # preprocess dataset to get uniform dataset for training
            X,Y = utils.preprocess_groundtruth_artificial_noise_balanced(
                                ground_truth_folders = training_folders,
                                before_frac = cfg['before_frac'],
                                windowsize = cfg['windowsize'],
                                after_frac = 1 - cfg['before_frac'],
                                noise_level = noise_level,
                                sampling_rate = cfg['sampling_rate'],
                                smoothing = cfg['smoothing'] * cfg['sampling_rate'],
                                omission_list = [],
                                permute = 1,
                                verbose = cfg['verbose'],
                                replicas=1,
                                causal_kernel=cfg['causal_kernel'])
            

            model = utils.define_model(
                                filter_sizes = cfg['filter_sizes'],
                                filter_numbers = cfg['filter_numbers'],
                                dense_expansion = cfg['dense_expansion'],
                                windowsize = cfg['windowsize'],
                                loss_function = cfg['loss_function'],
                                optimizer = cfg['optimizer']
                                        )

            model.compile( loss = cfg['loss_function'],
                           optimizer = cfg['optimizer'] )

            model.fit(X,Y,
                      batch_size = cfg['batch_size'],
                      epochs = cfg['nr_of_epochs'],
                      verbose = cfg['verbose'])

            # save model
            file_name = 'Model_NoiseLevel_{}_Ensemble_{}.h5'.format(int(noise_level), ensemble)
            model.save( os.path.join( model_folder,file_name ) )
            print('Saved model:', file_name)

    # Update model fitting status
    cfg['training_finished'] = 'Yes'
    config.write_config(cfg, os.path.join( model_folder,'config.yaml' ))

    print('\n\nDone!')
    print('Runtime: {:.0f} min'.format((time.time() - start)/60))


def predict( model_name, traces, threshold=0, padding=np.nan ):
    
    """ Use a specific trained neural network ('model_name') to predict spiking activity for calcium traces ('traces')

    In this function, a already trained model (generated by 'train_model') is loaded.
    The model (frame rate, noise levels, ground truth datasets) should be chosen
      to match the properties of the calcium recordings in 'traces'.
    An ensemble of 5 models is loaded for each noise level.
    These models are used to predict spiking activitz of neurons from 'traces' with the same noise levels.
    The predictions are made in the line with 'model.predict()'.
    The predictions are returned as a matrix 'Y_predict'.
    
    input:  model configuration identifier 'model_name'
            dF/F recording matrix 'traces'
    output: spiking activity 'Y_predict' (same matrix shape as 'traces')
    
    """
    import keras
    from cascade2p import utils

    model_folder = os.path.join('Pretrained_models', model_name)

    # Load config file
    cfg = config.read_config( os.path.join( model_folder, 'config.yaml'))

    # extract values from config file into variables
    verbose = cfg['verbose']
    batch_size = cfg['batch_size']
    sampling_rate = cfg['sampling_rate']
    before_frac = cfg['before_frac']
    window_size = cfg['windowsize']
    noise_levels_model = cfg['noise_levels']
    smoothing = cfg['smoothing']

    if verbose: print('Loaded model was trained at frame rate {} Hz'.format(sampling_rate))
    if verbose: print('Given argument traces contains {} neurons and {} frames.'.format( traces.shape[0], traces.shape[1]))

    # calculate noise levels for each trace
    trace_noise_levels = utils.calculate_noise_levels(traces, sampling_rate)
    
    print('Noise levels (mean, std; in standard units): '+str(int(np.nanmean(trace_noise_levels*100))/100)+', '+str(int(np.nanstd(trace_noise_levels*100))/100))

    # Get model paths as dictionary (key: noise_level) with lists of model
    # paths for the different ensembles
    model_dict = get_model_paths( model_folder )
    if verbose > 2: print('Loaded models:', str(model_dict))

    # XX has shape: (neurons, timepoints, windowsize)
    XX = utils.preprocess_traces(traces,
                            before_frac = before_frac,
                            window_size = window_size)
    Y_predict = np.zeros( (XX.shape[0], XX.shape[1]) )


    # Use for each noise level the matching model
    for i, model_noise in enumerate(noise_levels_model):

        if verbose: print('\nPredictions for noise level {}:'.format(model_noise))

        # TODO make more general (e.g. argmin(abs(diff)))
        # select neurons which have this noise level:
        if i == 0:   # lowest noise
            neuron_idx = np.where( trace_noise_levels < model_noise + 0.5 )[0]
        elif i == len(noise_levels_model)-1:   # highest noise
            neuron_idx = np.where( trace_noise_levels >= model_noise - 0.5 )[0]
        else:
            neuron_idx = np.where( (trace_noise_levels >= model_noise - 0.5) & (trace_noise_levels < model_noise + 0.5) )[0]

        if len(neuron_idx) == 0:  # no neurons were selected
            if verbose: print('\tNo neurons for this noise level')
            continue   # jump to next noise level

        # load keras models for the given noise level
        models = list()
        for model_path in model_dict[model_noise]:
            models.append( keras.models.load_model( model_path ) )

        # select neurons and merge neurons and timepoints into one dimension
        XX_sel = XX[neuron_idx, :, :]

        XX_sel = np.reshape( XX_sel, (XX_sel.shape[0]*XX_sel.shape[1], XX_sel.shape[2]) )
        XX_sel = np.expand_dims(XX_sel,axis=2)   # add empty third dimension to match training shape

        for j, model in enumerate(models):
            if verbose: print('\t... ensemble', j)

            prediction_flat = model.predict(XX_sel, batch_size, verbose=verbose )
            prediction = np.reshape(prediction_flat, (len(neuron_idx),XX.shape[1]))

            Y_predict[neuron_idx,:] += prediction / len(models)  # average predictions

        # remove models from memory
        keras.backend.clear_session()
    
    # Cut off noise floor (lower than 1/e of a single action potential);
    if threshold:
      
      from scipy.ndimage.filters import gaussian_filter
      from scipy.ndimage.morphology import binary_dilation
      
      # find out empirically  how large a single AP is (depends on frame rate and smoothing)
      single_spike = np.zeros(1001,)
      single_spike[501] = 1
      single_spike_smoothed = gaussian_filter(single_spike.astype(float), sigma=smoothing*sampling_rate)
      threshold_value = np.max(single_spike_smoothed)/np.exp(1)
      
      # Set everything below threshold to zero.
      # Use binary dilation to avoid clipping of true events.
      for neuron in range(Y_predict.shape[0]):
      
        activity_mask = Y_predict[neuron,:] > threshold_value
        activity_mask = binary_dilation(activity_mask,iterations = int(smoothing*sampling_rate))
        
        Y_predict[neuron,~activity_mask] = 0
        
    else:
      
      Y_predict[Y_predict<0] = 0
      
    # NaN or 0 for first and last datapoints, for which no predictions can be made
    Y_predict[:,0:int(before_frac*window_size)] = padding
    Y_predict[:,-int((1-before_frac)*window_size):] = padding

    print('Done')

    return Y_predict





def verify_config_dict( config_dictionary ):
  
    """ Perform some test to catch the most likely errors when creating config files """

    # TODO: Implement
    print('Not implemented yet...')





def create_model_folder( config_dictionary ):
  
    """ Creates a new folder in 'trained_models' and saves config.yaml file there

    Parameters
    ----------
    config_dictionary : dict
        Dictionary with keys like 'model_name' or 'training_datasets'
        Values which are not specified will be set to default values defined in
        the config_template in config.py
    """
    cfg = config_dictionary  # shorter name

    # TODO: call here verify_config_dict

    # TODO: check here the current directory, might not be the main folder...
    model_folder = os.path.join('Pretrained_models', cfg['model_name'])

    if not os.path.exists( model_folder ):
        # create folder
        os.mkdir(model_folder)

        # save config file into the folder
        config.write_config(cfg, os.path.join(model_folder, 'config.yaml') )

    else:
        raise Warning('There is already a folder called {}. '.format(cfg['model_name']) +
              'Please rename your model.')




def get_model_paths( model_folder ):
  
    """ Find all models in the model folder and return as dictionary

    Returns
    -------
    model_dict : dict
        Dictionary with noise_level (int) as keys and entries are lists of model paths

    """
    import glob, re

    all_models = glob.glob( os.path.join(model_folder, '*.h5') )
    all_models = sorted( all_models )  # sort

    # dictionary with key for noise level, entries are lists of models
    model_dict = dict()

    for model_path in all_models:
        try:
            noise_level = int( re.findall('_NoiseLevel_(\d+)', model_path)[0] )
        except:
            print('Error while processing the file with name: ', model_path)
            raise

        # add model path to the model dictionary
        if noise_level not in model_dict:
            model_dict[noise_level] = list()
        model_dict[noise_level].append(model_path)

    return model_dict
