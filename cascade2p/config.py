
import os
from pip._internal import main as pip

config_template = """\

## Main parameter of this model:

model_name: YOUR_MODEL_NAME                   # Name of the model
sampling_rate: YOUR_SAMPLING_RATE             # Sampling rate in Hz

# Dataset of ground truth data (in folder 'Ground_truth')   Example: DS14-GCaMP6s-m-V1
training_datasets:
- placeholder_1
- placeholder_2

placeholder_1: 0       # protect formatting


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

placeholder_2: 0       # protect formatting


# Standard deviation of Gaussian smoothing in time (sec)
smoothing: 0.2

# Smoothing kernel is symmetric in time (0) or is causal (1)
causal_kernel: 0

## Additional parameters for model specification:


windowsize: 64                   # Windowsize in timepoints
before_frac: 0.5                 # Fraction of timepoints before prediction point (0-1)

# Filter sizes for each convolutional layer
filter_sizes:
- 31
- 19
- 5

# Filter numbers for each convolutional layer
filter_numbers:
- 30
- 40
- 50

dense_expansion: 10              # For dense layer


loss_function: mean_squared_error     # gradient-descent loss function
optimizer: Adagrad                    #                  optimizer

nr_of_epochs: 20                 # Number of training epochs per model
ensemble_size: 5                 # Number of models trained for one noise level
batch_size: 1024                 # Batch size

## Information about status of fitting

training_finished: No            # Yes / No / Running
verbose : 1                      # level of status messages (0: minimal, 1: standard, 2: most, 3: all)


## Additional parameter not specified in template

"""


def read_config(config_yaml_file):
    """Read given yaml file and return dictionary with entries"""
    # if this results in an error, install the package with: pip install ruamel.yaml
    try:
        import ruamel.yaml as yaml   # install the package with: pip install ruamel.yaml
    except ImportError:
        pip.main(['install', '--user', 'ruamel'])
        import ruamel.yaml as yaml   # install the package with: pip install ruamel.yaml

    # TODO: add handling of file not found error

    yaml_config = yaml.YAML()
    with open(config_yaml_file, 'r') as file:
        config_dict = yaml_config.load(file)

    return config_dict


def write_config(config_dict, save_file):
    """Write config file from dictionary, use config_template string to define file structure"""

    # if this results in an error, install the package with: pip install ruamel.yaml
    try:
        import ruamel.yaml as yaml   # install the package with: pip install ruamel.yaml
    except ImportError:
        pip.main(['install', '--user', 'ruamel'])
        import ruamel.yaml as yaml   # install the package with: pip install ruamel.yaml

    # TODO: some error handling in case of missing default values?

    # read in template
    yml_config = yaml.YAML()
    yml_dict = yml_config.load(config_template)

    # update values of config dict (to keep default values)
    for key in config_dict:
        yml_dict[key] = config_dict[key]

    file_existed = os.path.exists( save_file )

    # save updated configs in save_file
    with open(save_file, 'w') as file:
        yml_config.dump(yml_dict, file)

    if not file_existed:
        print('Created file', save_file)
    else:
        pass  # file was updated, no need to notify user
