

"""
General checks to simplify jupyter notebooks and GUI
"""

def check_packages():
    """ Wrapper for check_yaml and check_keras_version """
    check_yaml()
    check_keras_version()


def check_yaml():
    """ Check if ruamel.yaml is installed, otherwise notify user with instructions """

    try:
        import ruamel.yaml
    except ModuleNotFoundError:
        print('\nModuleNotFoundError: The package "ruamel.yaml" does not seem to be installed on this PC.',
              'This package is necessary to load the configuration files of the models.\n',
              'Please install it with "pip install ruamel.yaml"')
        return

    print('\tYAML reader installed (version {}).'.format(ruamel.yaml.__version__))

def check_keras_version():
    """ Import keras and tensorflow and check versions """
    try:
        import keras
    except ModuleNotFoundError:
        print('\nModuleNotFoundError: The package "keras" does not seem to be installed on this PC.',
              'It is not possible to train models or predict neural activity without keras.\n',
              'Please install keras with "pip install keras==2.3.1".')
        return

    try:
        import tensorflow
    except ModuleNotFoundError:
        print('ModuleNotFoundError: The package "tensorflow" does not seem to be installed on this PC.',
              'Please install tensorflow with "pip install tensorflow==2.1.0".')
        return

    print('\tKeras installed (version {}).'.format(keras.__version__) )
    print('\tTensorflow installed (version {}).'.format(tensorflow.__version__) )

    ## TODO: perform check that versions are compatible, notify user
