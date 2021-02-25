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
import warnings
from . import config, utils



def train_model( model_name, model_folder='Pretrained_models', ground_truth_folder='Ground_truth' ):

    """ Train neural network with parameters specified in the config.yaml file in the model folder

    In this function, a model is configured (defined in the input 'model_name': frame rate, noise levels, ground truth datasets, etc.).
    The ground truth is resampled (function 'preprocess_groundtruth_artificial_noise_balanced', defined in "utils.py").
    The network architecture is defined (function 'define_model', defined in "utils.py").
    The thereby defined model is trained with the resampled ground truth data.
    The trained model with its weight and configuration details is saved to disk.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. 'Universal_30Hz_smoothing100ms'
        This name has to correspond to the folder with the config.yaml file which defines the model parameters

    model_folder: str
        Absolute or relative path, which defines the location of the specified model_name folder
        Default value 'Pretrained_models' assumes a current working directory in the Cascade folder

    ground_truth_folder : str
        Absolute or relative path, which defines the location of the ground truth datasets
        Default value 'Ground_truth'  assumes a current working directory in the Cascade folder

    Returns
    --------
    None
        All results are saved in the folder model_name as .h5 files containing the trained model

    """
    import tensorflow.keras
    from tensorflow.keras.optimizers import Adagrad

    model_path = os.path.join(model_folder, model_name)
    cfg_file = os.path.join( model_path, 'config.yaml')

    # check if configuration file can be found
    if not os.path.isfile(cfg_file):
        m = 'The configuration file "config.yaml" can not be found at the location "{}".\n'.format( os.path.abspath(cfg_file) ) + \
            'You have provided the model "{}" at the absolute or relative path "{}".\n'.format( model_name, model_folder) + \
            'Please check if there is a folder for model "{}" at the location "{}".'.format( model_name, os.path.abspath(model_folder))
        print(m)
        raise Exception(m)

    # load cfg dictionary from config.yaml file
    cfg = config.read_config( cfg_file )
    verbose = cfg['verbose']

    if verbose:
        print('Used configuration for model fitting (file {}):\n'.format( os.path.abspath(cfg_file) ))
        for key in cfg:
            print('{}:\t{}'.format(key, cfg[key]))

        print('\n\nModels will be saved into this folder:', os.path.abspath(model_path))

    # add base folder to selected training datasets
    training_folders = [os.path.join(ground_truth_folder, ds) for ds in cfg['training_datasets']]

    # check if the training datasets can be found
    missing = False
    for folder in training_folders:
        if not os.path.isdir(folder):
            print('The folder "{}" could not be found at the specified location "{}"'.format(folder, os.path.abspath(folder)))
            missing = True
    if missing:
        m = 'At least one training dataset could not be located.\nThis could mean that the given path "{}" '.format(ground_truth_folder) + \
            'does not specify the correct location or that e.g. a training dataset referenced in the config.yaml file ' + \
            'contained a typo.'
        print(m)
        raise Exception(m)


    start = time.time()
    # Update model fitting status
    cfg['training_finished'] = 'Running'
    config.write_config(cfg, os.path.join( model_path,'config.yaml' ))

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
            
            if cfg['sampling_rate'] > 30:
                
                cfg['windowsize'] = int(np.power(cfg['sampling_rate']/30,0.25)*64)

                print('Window size enlarged to '+str(cfg['windowsize']) +' time points due to the high calcium imaging sampling rate('+str(cfg['sampling_rate'])+').')

            model = utils.define_model(
                                filter_sizes = cfg['filter_sizes'],
                                filter_numbers = cfg['filter_numbers'],
                                dense_expansion = cfg['dense_expansion'],
                                windowsize = cfg['windowsize'],
                                loss_function = cfg['loss_function'],
                                optimizer = cfg['optimizer']
                                        )
            
            optimizer = Adagrad(learning_rate=0.05)
            model.compile( loss = cfg['loss_function'],
                           optimizer = optimizer)

            cfg['nr_of_epochs'] = np.minimum(cfg['nr_of_epochs'], np.int(10 * np.floor(5e6/ len(X))))

            model.fit(X,Y,
                      batch_size = cfg['batch_size'],
                      epochs = cfg['nr_of_epochs'],
                      verbose = cfg['verbose'])

            # save model
            file_name = 'Model_NoiseLevel_{}_Ensemble_{}.h5'.format(int(noise_level), ensemble)
            model.save( os.path.join( model_path, file_name ) )
            print('Saved model:', file_name)

    # Update model fitting status
    # cfg['training_finished'] = 'Yes'
    # config.write_config(cfg, os.path.join( model_path, 'config.yaml' ))

    print('\n\nDone!')
    print('Runtime: {:.0f} min'.format((time.time() - start)/60))


def predict( model_name, traces, model_folder='Pretrained_models', threshold=0, padding=np.nan ):

    """ Use a specific trained neural network ('model_name') to predict spiking activity for calcium traces ('traces')

    In this function, a already trained model (generated by 'train_model' or downloaded) is loaded.
    The model (frame rate, noise levels, ground truth datasets) should be chosen
      to match the properties of the calcium recordings in 'traces'.
    An ensemble of 5 models is loaded for each noise level.
    These models are used to predict spiking activitz of neurons from 'traces' with the same noise levels.
    The predictions are made in the line with 'model.predict()'.
    The predictions are returned as a matrix 'Y_predict'.


    Parameters
    ------------
    model_name : str
        Name of the model, e.g. 'Universal_30Hz_smoothing100ms'
        This name has to correspond to the folder in which the config.yaml and .h5 files are stored which define
        the trained model

    traces : 2d numpy array (neurons x nr_timepoints)
        Df/f traces with recorded fluorescence (as fractions, not in percents) on which the spiking activity will
        be predicted. Required shape: (neurons x nr_timepoints)

    model_folder: str
        Absolute or relative path, which defines the location of the specified model_name folder
        Default value 'Pretrained_models' assumes a current working directory in the Cascade folder

    threshold : int or boolean
        Allowed values: 0, 1 or False
            0: All negative values are set to 0
            1 or True: Threshold signal to set every signal which is smaller than the expected signal size
                       of an action potential to zero (with dialated mask)
            False: No thresholding. The result can contain negative values as well

    padding : 0 or np.nan
        Value which is inserted for datapoints, where no prediction can be made (because of window around timepoint of prediction)
        Default value: np.nan, another recommended value would be 0 which circumvents some problems with following analysis.

    Returns
    --------
    predicted_activity: 2d numpy array (neurons x nr_timepoints)
        Spiking activity as predicted by the model. The shape is the same as 'traces'
        This array can contain NaNs if the value 'padding' was np.nan as input argument

    """
    import tensorflow.keras
    from tensorflow.keras.models import load_model

    model_path = os.path.join(model_folder, model_name)
    cfg_file = os.path.join( model_path, 'config.yaml')

    # check if configuration file can be found
    if not os.path.isfile(cfg_file):
        m = 'The configuration file "config.yaml" can not be found at the location "{}".\n'.format( os.path.abspath(cfg_file) ) + \
            'You have provided the model "{}" at the absolute or relative path "{}".\n'.format( model_name, model_folder) + \
            'Please check if there is a folder for model "{}" at the location "{}".'.format( model_name, os.path.abspath(model_folder))
        print(m)
        raise Exception(m)

    # Load config file
    cfg = config.read_config( cfg_file )

    # extract values from config file into variables
    verbose = cfg['verbose']
    training_data = cfg['training_datasets']
    ensemble_size = cfg['ensemble_size']
    batch_size = cfg['batch_size']
    sampling_rate = cfg['sampling_rate']
    before_frac = cfg['before_frac']
    window_size = cfg['windowsize']
    noise_levels_model = cfg['noise_levels']
    smoothing = cfg['smoothing']
    causal_kernel = cfg['causal_kernel']

    model_description = '\n \nThe selected model was trained on '+str(len(training_data))+' datasets, with '+str(ensemble_size)+' ensembles for each noise level, at a sampling rate of '+str(sampling_rate)+'Hz,'
    if causal_kernel:
      model_description += ' with a resampled ground truth that was smoothed with a causal kernel'
    else:
      model_description += ' with a resampled ground truth that was smoothed with a Gaussian kernel'
    model_description += ' of a standard deviation of '+str(int(1000*smoothing))+' milliseconds. \n \n'
    print(model_description)

    if verbose: print('Loaded model was trained at frame rate {} Hz'.format(sampling_rate))
    if verbose: print('Given argument traces contains {} neurons and {} frames.'.format( traces.shape[0], traces.shape[1]))

    # calculate noise levels for each trace
    trace_noise_levels = utils.calculate_noise_levels(traces, sampling_rate)

    print('Noise levels (mean, std; in standard units): '+str(int(np.nanmean(trace_noise_levels*100))/100)+', '+str(int(np.nanstd(trace_noise_levels*100))/100))

    # Get model paths as dictionary (key: noise_level) with lists of model
    # paths for the different ensembles
    model_dict = get_model_paths( model_path )  # function defined below
    if verbose > 2: print('Loaded models:', str(model_dict))

    # XX has shape: (neurons, timepoints, windowsize)
    XX = utils.preprocess_traces(traces,
                            before_frac = before_frac,
                            window_size = window_size)
    Y_predict = np.zeros( (XX.shape[0], XX.shape[1]) )


    # Use for each noise level the matching model
    for i, model_noise in enumerate(noise_levels_model):

        if verbose: print('\nPredictions for noise level {}:'.format(model_noise))

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
            models.append( load_model( model_path ) )

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
        tensorflow.keras.backend.clear_session()


    if threshold is False:  # only if 'False' is passed as argument
        if verbose: print('Skipping the thresholding. There can be negative values in the result.')

    elif threshold == 1:     # (1 or True)
      # Cut off noise floor (lower than 1/e of a single action potential)

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
        # ignore warning because of nan's in Y_predict in comparison with value
        with np.errstate(invalid='ignore'):
            activity_mask = Y_predict[neuron,:] > threshold_value
        activity_mask = binary_dilation(activity_mask,iterations = int(smoothing*sampling_rate))

        Y_predict[neuron,~activity_mask] = 0

        Y_predict[Y_predict<0] = 0  # set possible negative values in dilated mask to 0

    elif threshold == 0:
      # ignore warning because of nan's in Y_predict in comparison with value
      with np.errstate(invalid='ignore'):
          Y_predict[Y_predict<0] = 0

    else:
        raise Exception('Invalid value of threshold "{}". Only 0, 1 (or True) or False allowed'.format(threshold))

    # NaN or 0 for first and last datapoints, for which no predictions can be made
    Y_predict[:,0:int(before_frac*window_size)] = padding
    Y_predict[:,-int((1-before_frac)*window_size):] = padding

    print('Done')

    return Y_predict





def verify_config_dict( config_dictionary ):

    """ Perform some test to catch the most likely errors when creating config files """

    # TODO: Implement
    print('Not implemented yet...')





def create_model_folder( config_dictionary, model_folder='Pretrained_models' ):

    """ Creates a new folder in model_folder and saves config.yaml file there

    Parameters
    ----------
    config_dictionary : dict
        Dictionary with keys like 'model_name' or 'training_datasets'
        Values which are not specified will be set to default values defined in
        the config_template in config.py

    model_folder : str
        Absolute or relative path, which defines the location at which the new
        folder containing the config file will be created
        Default value 'Pretrained_models' assumes a current working directory
        in the Cascade folder

    """
    cfg = config_dictionary  # shorter name

    # TODO: call here verify_config_dict

    # TODO: check here the current directory, might not be the main folder...
    model_path = os.path.join(model_folder, cfg['model_name'])

    if not os.path.exists( model_path ):
        # create folder
        try:
            os.mkdir(model_path)
            print('Created new directory "{}"'.format( os.path.abspath(model_path) ))
        except:
            print(model_path+' already exists')        

        # save config file into the folder
        config.write_config(cfg, os.path.join(model_path, 'config.yaml') )

    else:
        warnings.warn('There is already a folder called {}. '.format(cfg['model_name']) + \
              'Please rename your model.')




def get_model_paths( model_path ):

    """ Find all models in the model folder and return as dictionary
    ( Helper function called by predict() )

    Returns
    -------
    model_dict : dict
        Dictionary with noise_level (int) as keys and entries are lists of model paths

    """
    import glob, re

    all_models = glob.glob( os.path.join(model_path, '*.h5') )
    all_models = sorted( all_models )  # sort

    # Exception in case no model was found to catch this mistake where it happened
    if len(all_models) == 0:
        m = 'No models (*.h5 files) were found in the specified folder "{}".'.format( os.path.abspath(model_path) )
        raise Exception(m)

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



def download_model( model_name,
                    model_folder='Pretrained_models',
                    info_file_link = 'https://drive.switch.ch/index.php/s/IJWlhwjDT3aw2ro/download',
                    verbose = 1):
    """ Download and unzip pretrained model from the online repository

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. 'Universal_30Hz_smoothing100ms'
        This name has to correspond to a pretrained model that is available for download
        To see available models, run this function with model_name='update_models' and
        check the downloaded file 'available_models.yaml'

    model_folder: str
        Absolute or relative path, which defines the location of the specified model_name folder
        Default value 'Pretrained_models' assumes a current working directory in the Cascade folder

    info_file_link: str
        Direct download link to yaml file which contains download links for new models.
        Default value is official repository of models.

    verbose : int
        If 0, no messages are printed. if larger than 0, the user is informed about status.

    """

    from urllib.request import urlopen
    import zipfile

    # Download the current yaml file with information about available models first
    new_file = os.path.join( model_folder, 'available_models.yaml')
    with urlopen( info_file_link ) as response:
        text = response.read()

    with open(new_file, 'wb') as f:
        f.write(text)

    # check if the specified model_name is present
    download_config = config.read_config( new_file )  # orderedDict with model names as keys

    if model_name not in download_config.keys():
        if model_name == 'update_models':
            print('You can now check the updated available_models.yaml file for valid model names.')
            print('File location:', os.path.abspath(new_file))
            return

        raise Exception( 'The specified model_name "{}" is not in the list of available models. '.format(model_name) + \
                         'Available models for download are: {}'.format( list( download_config.keys()) ))


    if verbose: print('Downloading and extracting new model "{}"...'.format( model_name ) )

    # download and save .zip file of model
    download_link = download_config[model_name]['Link']
    with urlopen( download_link ) as response:
        data = response.read()

    tmp_file = os.path.join( model_folder, 'tmp_zipped_model.zip')
    with open(tmp_file, 'wb') as f:
        f.write(data)

    # unzip the model and save in the corresponding folder
    with zipfile.ZipFile( tmp_file, 'r') as zip_ref:
        zip_ref.extractall( path=os.path.join(model_folder,model_name))

    os.remove(tmp_file)

    if verbose: print('Pretrained model was saved in folder "{}"'.format(
                                           os.path.abspath(os.path.join(model_folder, model_name)) ))

    return
