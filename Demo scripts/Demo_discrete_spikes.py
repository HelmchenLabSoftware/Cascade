#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Demo script to convert spiking probabilities into discrete spikes.

The input is a matrix (output of cascade predictions), the output are estimate
discrete spike times. The main function is infer_discrete_spikes().

The procedure is brute-force and not optimized for fast processing.

Please be aware of the limitations of the prediction of single spikes, as
discussed in more detail in the paper.


"""



"""

Import python packages

"""

import os
if 'Demo scripts' in os.getcwd():
    sys.path.append( os.path.abspath('..') ) # add parent directory to path for imports
    os.chdir('..')  # change to main directory
print('Current directory: {}'.format( os.getcwd() ))

from cascade2p import checks
checks.check_packages()

import numpy as np
import scipy.io as sio

from cascade2p.utils import plot_dFF_traces
from cascade2p.utils_discrete_spikes import infer_discrete_spikes



"""

Define functions to load calcium recordings and spike rate predictions from disk

"""


def load_neurons_x_time(file_path):
    """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""

    traces = sio.loadmat(file_path)['dF_traces']

    return traces/100

def load_predictions(file_path):
    """Custom method to load spike predictions produced by "Demo_predict.py" """

    spike_prob = sio.loadmat(file_path)['spike_prob']

    return spike_prob



"""

Load dF/F traces and the parameters of the model/dataset

"""

example_file = 'Example_datasets/Multiplane-OGB1-zf-pDp-Rupprecht-7.5Hz/Calcium_traces_04.mat'
example_file_predictions = 'Example_datasets/Multiplane-OGB1-zf-pDp-Rupprecht-7.5Hz/full_prediction_Calcium_traces_04.mat'
model_name = 'OGB_zf_pDp_7.5Hz_smoothing200ms'
frame_rate = 7.5

traces = load_neurons_x_time(example_file)
spike_prob = load_predictions(example_file_predictions)




"""

Fill up probabilities (output of the network) with discrete spikes

"""

discrete_approximation, spike_time_estimates = infer_discrete_spikes(spike_prob,model_name)



"""

Plot example predictions together with discrete spikes

"""

neuron_indices = np.random.randint(spike_prob.shape[0], size=10)
plot_dFF_traces(traces,neuron_indices,frame_rate,spiking=spike_prob,discrete_spikes=spike_time_estimates )


"""

Save predictions to disk

"""

folder = os.path.dirname(example_file)
save_path = os.path.join(folder, 'discrete_spikes_'+os.path.basename(example_file))

# save as numpy file
#np.save(save_path, spike_prob)
sio.savemat(save_path, {'spike_prob':spike_prob,'discrete_approximation':discrete_approximation,'spike_time_estimates':spike_time_estimates})
