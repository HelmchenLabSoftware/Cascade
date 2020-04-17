
""" High level interface to the CASCADE package

TODO: lot of documentation here, since this is the entry point for people checking / debugging code

"""

import os
import time

from helper_scripts import config

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
    model_folder = os.path.join('trained_models', cfg['model_name'])

    if not os.path.exists( model_folder ):
        # create folder
        os.mkdir(model_folder)

        # save config file into the folder
        config.write_config(cfg, os.path.join(model_folder, 'config.yaml') )

    else:
        raise Warning('There is already a folder called {}. '.format(cfg['model_name']) +
              'Please rename your model.')



def train_model( model_name ):
    """ Train neural network with parameters specified in the config.yaml file in the model folder

    # TODO: more documentation

    """

    import keras
    from helper_scripts import utils

    # TODO: check here for relative vs absolute path definition
    model_folder = os.path.join('trained_models', model_name)

    # load cfg dictionary from config.yaml file
    cfg = config.read_config( os.path.join(model_folder, 'config.yaml') )

    print('Used configuration for model fitting (file {}):\n'.format( os.path.join(model_folder, 'config.yaml') ))
    for key in cfg:
        print('{}:\t{}'.format(key, cfg[key]))

    print('\n\nModels will be saved into this folder:', model_folder)

    start = time.time()


    # add base folder to selected training datasets
    training_folders = [os.path.join('GT_datasets', ds) for ds in cfg['training_datasets']]

    # Update model fitting status
    cfg['training_finished'] = 'Running'
    config.write_config(cfg, os.path.join( model_folder,'config.yaml' ))

    nr_model_fits = len( cfg['noise_levels'] ) * cfg['ensemble_size']
    print('Fitting a total of {} models:'.format( nr_model_fits))

    curr_model_nr = 0

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
                                verbose = cfg['verbose'])

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
