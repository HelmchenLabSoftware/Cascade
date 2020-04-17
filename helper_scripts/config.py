
# if this results in an error, install the package with: pip install ruamel.yaml
import ruamel.yaml as yaml

config_template = """

###
### Main parameter of this model:
###

model_name :                    # Name of the model
sampling_rate :                 # Sampling rate in Hz

# Dataset of ground truth data (in folder 'GT_datasets')   Example: GT_dataset_GC6s_Chen
training_datasets :
    - placeholder_1
    - placeholder_2


# Noise levels for training (integers, normally 1-9)
noise_levels:
    - 1
    - 2
    - 3
    - 4
    - 5
    - 6
    - 7
    - 8
    - 9

# Standard deviation of Gaussian smoothing in time (sec)
smoothing : 0.2


###
### Additional parameters for model specification:
###


windowsize : 64                   # Windowsize in timepoints
before_frac : 0.5                 # Fraction of timepoints before prediction point (0-1)


filter_sizes :                    # Filter sizes for each convolutional layer
    - 31
    - 19
    - 5

filter_numbers :                  # Filter numbers for each convolutional layer
    - 30
    - 40
    - 50

dense_expansion : 30              # For dense layer


loss_function : mean_squared_error     # gradient-descent loss function
optimizer : Adagrad                    #                  optimizer

nr_of_epochs : 10                 # Number of training epochs per model
ensemble_size : 5                 # Number of models trained for one noise level
batch_size : 8192                 # Batch size

###
### Information about status of fitting
###

training_finished :               # Yes / No / Running


###
### Additional parameter not specified in template
###
"""


def read_config(config_yaml_file):
    """Read given yaml file and return dictionary with entries"""

    # TODO: add handling of file not found error

    yaml_config = yaml.YAML()
    with open(config_yaml_file, 'r') as file:
        config_dict = yaml_config.load(file)

    return config_dict


def write_config(config_dict, save_file):
    """Write config file from dictionary, use yaml_template_file to define file structure"""

    # TODO: include error handling for wrong values, missing files, overwrite warnings of template
    # TODO: add .yml ending if not in file name

    # read in template
    yml_config = yaml.YAML()
    yml_dict = yml_config.load(config_template)

    # update values of config dict (to keep default values)
    for key in config_dict:
        yml_dict[key] = config_dict[key]

    # save updated configs in save_file
    with open(save_file, 'w') as file:
        yml_config.dump(yml_dict, file)


    print('Created file', save_file)
