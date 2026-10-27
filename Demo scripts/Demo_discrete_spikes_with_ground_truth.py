# -*- coding: utf-8 -*-

"""
Created on Tue Oct 27 16:04:33 2026

@author: peter
"""


"""
Demo script to test spike rate inference and discrete spike inference on ground truth data.
This addresses issue #90, so it is possible to check the temporal alignment of inferred discrete and continuous
spikes compared to the ground truth spikes.

1. Ground truth recordings (dF/F traces and simultaneously recorded spikes) are resampled
   to the frame rate of the chosen model and brought to a chosen noise level (as for training).
2. Spike rates are inferred with the selected model (the model is downloaded if it is not
   yet available in the folder 'Pretrained_models').
3. Discrete spikes are inferred from the spike rates.
4. dF/F, ground truth spikes and spike rates, inferred spike rates and inferred discrete
   spikes are plotted together for several neurons.

"""



"""

Import python packages

"""

import os
import sys
if 'Demo scripts' in os.getcwd():
    sys.path.append( os.path.abspath('..') ) # add parent directory to path for imports
    os.chdir('..')  # change to main directory
print('Current directory: {}'.format( os.getcwd() ))

import glob
import numpy as np
import matplotlib.pyplot as plt

from cascade2p import cascade, utils, config
from cascade2p.utils_discrete_spikes import infer_discrete_spikes, causal_smoothing_kernel, smooth_spikes



"""

Selected model names, ground truth folder, and parameters

"""

model_name = 'GC8s_EXC_30Hz_smoothing50ms_causalkernel'           # model used for spike inference
ground_truth_folder = os.path.join('Ground_truth','DS32-GCaMP8s-m-V1')  # ground truth dataset

noise_level = 2          # noise level of the resampled ground truth (standardized noise level, as in the model names)
nb_neurons = 4           # number of (randomly chosen) ground truth neurons that are processed and plotted
plot_duration = 20       # duration (in seconds) of the plotted excerpt for each neuron (None: entire recording)
random_seed = 1

model_folder = 'Pretrained_models'
np.random.seed(random_seed)

"""

Download the model (if necessary) and load its parameters

"""

if not os.path.isfile( os.path.join(model_folder, model_name, 'config.yaml') ):
    cascade.download_model( model_name, model_folder=model_folder, verbose=1 )

cfg = config.read_config( os.path.join(model_folder, model_name, 'config.yaml') )
sampling_rate = cfg['sampling_rate']
smoothing = cfg['smoothing']
causal_kernel = 'causal_kernel' in cfg and cfg['causal_kernel']
window_size = cfg['windowsize']
before_frac = cfg['before_frac']


"""

Generate ground truth: dF/F traces at the chosen noise level and frame rate, with the
simultaneously recorded spikes binned into the same time bins

"""

print('Resampling the ground truth from "{}" at {} Hz and noise level {} ...'.format(ground_truth_folder, sampling_rate, noise_level))

traces_gt, spikes_gt, _, _ = utils.calibrated_ground_truth_artificial_noise( ground_truth_folder, noise_level, sampling_rate, replicas=0, verbose=0 )
neuron_files = sorted(list(set(glob.glob( os.path.join(ground_truth_folder,'*_mini.mat') ))))

# for each neuron, use the recording (trial) with the largest number of spikes
available_neurons = [k for k in range(len(traces_gt)) if traces_gt[k] is not None]
if len(available_neurons) == 0:
    raise Exception('No ground truth found in "{}" for noise level {}. Check the folder name, or use a higher noise level.'.format(ground_truth_folder, noise_level))
selected_neurons = np.random.choice(available_neurons, size=np.minimum(nb_neurons, len(available_neurons)), replace=False)

dFF_traces = []; gt_spikes = []; labels = []
for k in selected_neurons:
    trial = np.argmax( np.nansum(spikes_gt[k], axis=0) )
    valid = ~np.isnan( traces_gt[k][:,trial] )
    dFF_traces.append( traces_gt[k][valid,trial] )
    gt_spikes.append( spikes_gt[k][valid,trial] )
    labels.append( '{} (recording {})'.format( os.path.basename(neuron_files[k]).replace('_mini.mat',''), trial+1 ) )

# array (neurons x time) for spike inference; shorter recordings are padded with NaNs
lengths = [len(trace) for trace in dFF_traces]
traces = np.nan*np.ones( (len(dFF_traces), np.max(lengths)) )
for i,trace in enumerate(dFF_traces):
    traces[i,:len(trace)] = trace



"""

Infer spike rates and discrete spikes

"""

spike_rates = cascade.predict( model_name, traces, model_folder=model_folder, threshold=1, padding=np.nan )

# no predictions for the last frames of each recording (window of the network extends beyond the recording);
# this is needed here for the recordings that were padded with NaNs
frames_after = int((1-before_frac)*window_size)
for i,length in enumerate(lengths):
    spike_rates[i,length-frames_after:] = np.nan

discrete_approximation, discrete_spikes = infer_discrete_spikes( spike_rates, model_name, model_folder=model_folder )



"""

Ground truth spike rates: ground truth spikes smoothed with the same kernel as for training

"""

if causal_kernel:
    kernel = causal_smoothing_kernel( sampling_rate, smoothing )
else:
    kernel = None
gt_rates = [ smooth_spikes( spikes, smoothing*sampling_rate, kernel ) for spikes in gt_spikes ]



"""

Summary statistics (only for frames with predictions)

"""

print('\nKernel used for training and discrete spike inference: {}'.format('causal' if causal_kernel else 'Gaussian'))
for i in range(len(selected_neurons)):
    length = lengths[i]
    valid = ~np.isnan(spike_rates[i,:length])
    nb_gt = np.sum(gt_spikes[i][valid])
    nb_rate = np.sum(spike_rates[i,:length][valid])
    discrete = np.array(discrete_spikes[i], dtype=int)
    nb_discrete = np.sum( valid[ discrete[(discrete >= 0) & (discrete < length)] ] )
    correlation = np.corrcoef( gt_rates[i][valid], spike_rates[i,:length][valid] )[0,1]
    print('{}: {} ground truth spikes, {:.1f} spikes (sum of inferred spike rate), {} inferred discrete spikes; correlation of spike rates: {:.2f}'.format(
          labels[i], int(nb_gt), nb_rate, nb_discrete, correlation))



"""

Plot dF/F, ground truth spikes and spike rates, inferred spike rates and inferred discrete spikes

"""

for i in range(len(selected_neurons)):

    length = lengths[i]
    time = np.arange(length)/sampling_rate
    valid_indices = np.where(~np.isnan(spike_rates[i,:length]))[0]

    # excerpt with the largest number of ground truth spikes
    if plot_duration is None or plot_duration*sampling_rate >= len(valid_indices):
        start, stop = valid_indices[0], valid_indices[-1]+1
    else:
        nb_frames = int(plot_duration*sampling_rate)
        spikes_in_window = np.convolve( gt_spikes[i][valid_indices], np.ones(nb_frames), mode='valid' )
        start = valid_indices[0] + np.argmax(spikes_in_window)
        stop = start + nb_frames
    excerpt = np.arange(start, stop)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 2, 1]})
    fig.suptitle( '{}\nmodel: {}, noise level {}'.format(labels[i], model_name, noise_level), fontsize=10 )

    # dF/F
    axs[0].plot( time[excerpt], dFF_traces[i][excerpt], color='tab:green', linewidth=1 )
    axs[0].set_ylabel('dF/F')

    # spike rates (in Hz)
    axs[1].plot( time[excerpt], gt_rates[i][excerpt]*sampling_rate, color='k', linewidth=1.5, label='ground truth (smoothed spikes)' )
    axs[1].plot( time[excerpt], spike_rates[i,excerpt]*sampling_rate, color='tab:blue', linewidth=1.5, label='inferred spike rate' )
    axs[1].plot( time[excerpt], discrete_approximation[i,excerpt]*sampling_rate, color='tab:orange', linewidth=1, linestyle='--', label='smoothed inferred discrete spikes' )
    axs[1].set_ylabel('Spike rate (Hz)')
    axs[1].legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=8, frameon=False)

    # discrete spikes (ground truth: upper row; inferred: lower row); numbers indicate several spikes in the same frame
    for row, spike_train, color in [(1, gt_spikes[i], 'k'), (0, None, 'tab:orange')]:
        if spike_train is None:
            spike_indices = np.array(discrete_spikes[i], dtype=int)
            spike_train = np.bincount( spike_indices[(spike_indices >= 0) & (spike_indices < length)], minlength=length )
        for index in excerpt[ spike_train[excerpt] > 0 ]:
            axs[2].plot( [time[index]]*2, [row-0.35, row+0.35], color=color, linewidth=1.5 )
            if spike_train[index] > 1:
                axs[2].text( time[index], row+0.4, str(int(spike_train[index])), color=color, fontsize=7, ha='center', va='bottom' )
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(['inferred\ndiscrete spikes', 'ground truth\nspikes'])
    axs[2].set_ylim(-0.6, 1.9)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_xlim( time[excerpt[0]], time[excerpt[-1]]-5 )

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()

plt.show()