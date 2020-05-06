#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Aug 29 22:32:59 2019

@author: pierre
"""


import os
from os.path import normpath, basename
import numpy as np
import glob as glob

import scipy.io as sio
from scipy import optimize
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import resample, convolve
from scipy.interpolate import interp1d
from scipy.stats import invgauss
          
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, MaxPooling1D, Conv1D, Input, LSTM,BatchNormalization, LocallyConnected1D, Activation, concatenate
from keras import backend as K


""""

Define the model using the function API of Keras.

"""


def define_model(filter_sizes,filter_numbers,dense_expansion,windowsize,loss_function,optimizer, conv_filter=Conv1D):

  inputs = Input(shape=(windowsize,1))

  outX = conv_filter(filter_numbers[0], filter_sizes[0], strides=1, activation='relu')(inputs)
  outX = conv_filter(filter_numbers[1], filter_sizes[1], activation='relu')(outX)
  outX = MaxPooling1D(2)(outX)
  outX = conv_filter(filter_numbers[2], filter_sizes[2], activation='relu')(outX)
  outX = MaxPooling1D(2)(outX)

  outX = Dense(dense_expansion, activation='relu')(outX) # 'linear' units work here as well!
  outX = Flatten()(outX)
  predictions = Dense(1,activation='linear')(outX)
  model = Model(inputs=[inputs],outputs=predictions)
  model.compile(loss=loss_function, optimizer=optimizer)

  return model




def calculate_noise_levels(neurons_x_time, frame_rate):
    """
    TODO: documentation

    # TODO Peter: maybe short explanation of the calculation here

    """
    dF_traces = neurons_x_time

    nb_neurons = dF_traces.shape[0]
    noise_levels = np.zeros( nb_neurons )

    for neuron in range(nb_neurons):
        noise_levels[neuron] = np.nanmedian( np.abs(np.diff(dF_traces[neuron,:])))/np.sqrt(frame_rate)

    return noise_levels * 100     # scale noise levels to percent


"""


"""


def preprocess_traces(neurons_x_time, before_frac, window_size):
    """ Transform df/f data to format (neurons, timepoints, window_size)

    Creates a large matrix X that contains for each timepoint of each
    calcium trace a vector of length 'window_size' around the timepoint.
    """

    before = int( before_frac * window_size )
    after = int( window_size - before )

    dF_traces = neurons_x_time

    nb_neurons = dF_traces.shape[0]
    nb_timepoints = dF_traces.shape[1]

    X = np.zeros( (nb_neurons,nb_timepoints,window_size) ) * np.nan

    for neuron in range(nb_neurons):
        for timepoint in range(nb_timepoints-window_size):

            X[neuron,timepoint+before,:] = dF_traces[neuron, timepoint:(timepoint+window_size)]

    return X




"""
Created on Thu Jan 16 22:45:12 2020

@author: pierre
"""


def noiselevels_test_dataset(test_file,before_frac,windowsize,after_frac,framerate):

  dF_traces = sio.loadmat(test_file)['dF_traces']

  nb_neurons = dF_traces.shape[1]

  noise_levels = np.nan*np.zeros((nb_neurons,))
  for neuron in range(nb_neurons):

    noise_levels[neuron] = np.nanmedian(np.abs(np.diff(dF_traces[:,neuron])))/np.sqrt(framerate)

  return noise_levels


"""
Created on Thu Aug 29 22:22:38 2019

@author: pierre
"""


"""
Creates a large matrix X that contains for each timepoint of each calcium trace a vector of
length 'windowsize' around the timepoint.

Also creates a vector Y that contains the corresponding spikes/non-spikes.
Random permutations un-do the original sequence of the timepoints.

"""


def preprocess_groundtruth_artificial_noise(ground_truth_folders,before_frac,windowsize,after_frac,noise_level,sampling_rate,smoothing,omission_list=[],permute=1):


  sub_traces_all = [None]*500
  sub_traces_events_all = [None]*500
  events_all = [None]*500

  neuron_counter = 0
  for dataset_index,training_dataset in enumerate(ground_truth_folders):

    try:

        sub_traces_allX, sub_traces_events_allX, frame_rate, events_allX = calibrated_ground_truth_artificial_noise(ground_truth_folders[dataset_index],noise_level,sampling_rate,omission_list)
        sub_traces_all[neuron_counter:neuron_counter+len(sub_traces_allX)] = sub_traces_allX
        sub_traces_events_all[neuron_counter:neuron_counter+len(sub_traces_allX)] = sub_traces_events_allX
        events_all[neuron_counter:neuron_counter+len(sub_traces_allX)] = events_allX

        neuron_counter += len(sub_traces_allX)

    except:
         sub_traces_allX = None

  sub_traces_events_all = sub_traces_events_all[:neuron_counter]
  sub_traces_events_all = sub_traces_events_all[:neuron_counter]
  events_all = events_all[:neuron_counter]

  print(len(sub_traces_events_all))

  before = int(before_frac*windowsize)
  after = int(after_frac*windowsize)

  X = np.zeros((15000000,windowsize,))
  Y = np.zeros((15000000,))


  sample_counter = 0
  for sub_traces,sub_traces_events in zip(sub_traces_all,sub_traces_events_all):

    if sub_traces is not None:
      sample_counter += sub_traces.shape[0]*sub_traces.shape[1]
  oversampling = np.maximum(1,np.floor(sample_counter/5e6))

  counter = 0
  for sub_traces,sub_traces_events in zip(sub_traces_all,sub_traces_events_all):

    if sub_traces is not None:

      for trace_index in range(sub_traces.shape[1]):

        single_trace = sub_traces[:,trace_index]
        single_spikes = sub_traces_events[:,trace_index]

        single_spikes = gaussian_filter(single_spikes.astype(float), sigma=smoothing)

        recording_length = np.sum(~np.isnan(single_trace))

        datapoints_used = np.minimum(len(single_spikes)-windowsize,recording_length-windowsize)

        if oversampling > 1:

#          print('Discarding some samples (randomized) to reduce dataset.')

          datapoints_used_rand = np.random.permutation(datapoints_used)
          datapoints_used_rand = datapoints_used_rand[0:int(len(datapoints_used_rand)/oversampling)]

        else:

          datapoints_used_rand = np.arange(datapoints_used)

        for time_points in datapoints_used_rand:

          Y[counter,] = single_spikes[time_points+before]
          X[counter,:,] = single_trace[time_points:(time_points+before+after)]

          counter += 1

  Y = np.expand_dims(Y[:counter],axis=1)
  X = np.expand_dims(X[:counter,:],axis=2)

  if permute == 1:

    p = np.random.permutation(len(X))
    X = X[p,:,:]
    Y = Y[p,:]

    # Maximum of 5e6 training samples
    X = X[:5000000,:,:]
    Y = Y[:5000000,:]

  return X,Y




"""
Created on Thu Aug 29 22:22:38 2019

@author: pierre
"""


"""
Creates a large matrix X that contains for each timepoint of each calcium trace a vector of
length 'windowsize' around the timepoint.

"""


def preprocess_test_dataset(test_file,before_frac,windowsize,after_frac):

  before = int(before_frac*windowsize)
  after = int(after_frac*windowsize)

  dF_traces = sio.loadmat(test_file)['dF_traces']

  nb_neurons = dF_traces.shape[1]
  nb_timepoints = dF_traces.shape[0]

  X = np.nan*np.zeros((nb_neurons,nb_timepoints,windowsize,))
  for neuron in range(nb_neurons):

    for timepoint in range(nb_timepoints-windowsize):

      X[neuron,timepoint+before,:] = dF_traces[timepoint:(timepoint+before+after),neuron]


  return X



"""
Created on Sat Aug 24 23:54:00 2019

@author: Peter Rupprecht, p.t.r.rupprecht(at)gmail.com, August 2019


Main function:

sub_traces_all, sub_traces_events_all, frame_rate = calibrated_ground_truth(ground_truth_folder,noise_level,sampling_rate)

Inputs:

  Folder with ground truth pixel traces (dF/F) in *.mat files
  Noise level at which the ground truth should be resampled
  >> The noise level is defined as the median different between subsequent samples (i.e., high-frequency noise)
  Temporal sampling rate at which the ground truth should be resampled

Outputs:

  The extracted subtraces as a matrix ('sub' refers to the subset of ROI pixels used for the respective subtrace)
  The simultaneously recorded spikes, with the same time bins
  The frame rate, usually identical to the input target sampling rate
  >> If the input sampling rate does not differ >5% from the original sampling rate of the ground truth, there will be no resampling

"""



def calibrated_ground_truth_artificial_noise(ground_truth_folder,noise_level,sampling_rate,omission_list=[], verbose=3):

  #  ground_truth_folder = '/media/pierre/Der Hort/Peter/FMI PhD/Project calibrated deconvolution/Extracted_highQuality_datasets/GT_dataset_GC_aDp'
  #
  #  noise_level = 8
  #  sampling_rate = 10

  # Noise level normalized by sampling rate
  noise_level_normalized = noise_level

  # Go to target folder

  # Iterate through all ground truth files of the selected dataset
  # Datasets that were too large (> 2GB) have been split in two parts and have to receive special treatment
  fileList = sorted(list(set(glob.glob( os.path.join(ground_truth_folder,'*_mini.mat')))))

  # Omit neurons from the training data, if indicated in the omission_list
  for index in sorted(omission_list, reverse=True):
      del fileList[index]

  # Initialize lists which will later contain the resampled ground truth
  sub_traces_all = [None] * len(fileList)
  sub_traces_events_all = [None] * len(fileList)
  events_all = [None] * len(fileList)
  framerate_all = [None] * len(fileList)

  # for loop over all mat files / neurons in this dataset
  for file_index,neuron_file in enumerate(fileList):

    if verbose > 2: print('Resampling neuron '+str(file_index+1)+' from a total of '+str(len(fileList))+' neurons.')

    # Load mat file
    dataset_neuron_all = sio.loadmat(neuron_file)['CAttached'][0]

    # Initialize arrays that will contain the ground truth extracted from this neuron
    sub_traces = None
    sub_traces_events = None
    events_all[file_index] = [None] * 100000
    counter = 0

    # for loop over all trials of this neuron
    for i,trial in enumerate(dataset_neuron_all):

      # Find the relevant elements in the data structure
      # (dF/F traces; spike events; time stamps of fluorescence recording)
      keys = trial[0][0].dtype.descr
      keys_unfolded = list(sum(keys, ()))

      traces_index = int(keys_unfolded.index("fluo_mean")/2)
      fluo_time_index = int(keys_unfolded.index("fluo_time")/2)
      events_index = int(keys_unfolded.index("events_AP")/2)


      events = trial[0][0][events_index]
      events = events[~np.isnan(events)]
      ephys_sampling_rate = 1e4

      fluo_times = np.squeeze(trial[0][0][fluo_time_index])
      frame_rate = 1/np.nanmean(np.diff(fluo_times))

      traces_mean = np.squeeze(trial[0][0][traces_index])
      traces_mean = traces_mean[:fluo_times.shape[0]]

      traces_mean = traces_mean[~np.isnan(fluo_times)]
      fluo_times = fluo_times[~np.isnan(fluo_times)]

      base_noise = np.nanmedian(np.abs(np.diff(traces_mean)))*100/np.sqrt(frame_rate)

      test_noise = np.zeros((20,))
      for test_i in np.arange(20):
        noise_trace = np.random.normal(0,test_i/100*np.sqrt(frame_rate), traces_mean.shape)
        test_noise[test_i] = np.nanmedian(np.abs(np.diff(noise_trace+traces_mean)))*100/np.sqrt(frame_rate)

      interpolating_function = interp1d(test_noise,np.arange(20), kind='linear')


      if noise_level_normalized >= test_noise[0]:

        noise_std = interpolating_function(noise_level_normalized)/100*np.sqrt(frame_rate)
        # Get as many artificial noisified traces that natural noise is not dominating
        nb_subROIs = np.minimum(500,np.ceil( 1.2*(noise_level_normalized/base_noise)**2 ))

      else:

        nb_subROIs = 0

      if nb_subROIs >= 1:

        # Considered not necessary if sampling rates of ground truth and target sampling rate similar (<5% relative difference)
        if np.abs(sampling_rate - frame_rate)/frame_rate > 0.05:

          num_samples = int(round(traces_mean.shape[0]*sampling_rate/frame_rate))

          (traces_mean,fluo_times_resampled) = resample(traces_mean,num_samples,np.squeeze(fluo_times),axis=0)

          noise_std = noise_std*np.sqrt(sampling_rate/frame_rate)

        else:

          fluo_times_resampled = fluo_times

        frame_rate = 1/np.nanmean(np.diff(fluo_times_resampled))

        # Bin the ground truth (spike times) into time bins determined by calcium recording
        fluo_times_bin_centers = fluo_times_resampled
        fluo_times_bin_edges = np.append(fluo_times_bin_centers,fluo_times_bin_centers[-1]+1/frame_rate/2) - 1/frame_rate/2

        [events_binned,event_bins] = np.histogram(events/ephys_sampling_rate, bins=fluo_times_bin_edges)


        for iii in range(int(nb_subROIs)):

          fluo_level = np.sqrt(np.abs(traces_mean + 1))
          fluo_level /= np.median(fluo_level)
          
          noise_additional = np.random.normal(0,noise_std*fluo_level, traces_mean.shape)
          sub_traces_single = traces_mean + noise_additional
          
          # If 'sub_traces' exists already, append the subROI-trace; else, generate it
          # The nested if-clause covers cases where different trials for the same neuron have variable number of time points; the rest is filled up with NaNs
          if np.any(sub_traces):

            if sub_traces.shape[0]-len(sub_traces_single) >= 0:

              sub_traces_single = np.append(sub_traces_single, np.zeros(sub_traces.shape[0]-len(sub_traces_single)) + np.nan )
              events_binned = np.append(events_binned, np.zeros(sub_traces_events.shape[0]-len(events_binned)) + np.nan )

            else:
              sub_traces = np.append(sub_traces,np.zeros((len(sub_traces_single)-sub_traces.shape[0],sub_traces.shape[1])) + np.nan, axis=0)
              sub_traces_events = np.append(sub_traces_events,np.zeros((len(events_binned)-sub_traces_events.shape[0],sub_traces_events.shape[1])) + np.nan, axis=0)

            sub_traces = np.append(sub_traces,sub_traces_single.reshape(-1, 1),axis=1)
            sub_traces_events = np.append(sub_traces_events,events_binned.reshape(-1, 1),axis=1)

          else:

            sub_traces = sub_traces_single.reshape(-1, 1)
            sub_traces_events = events_binned.reshape(-1, 1)

          events_all[file_index][counter] = events/ephys_sampling_rate
          counter += 1

        # Write the subROI-traces for each neuron into a list item of 'sub_traces_all' (calcium) and 'sub_traces_events_all' (spikes)
        sub_traces_all[file_index] = sub_traces
        sub_traces_events_all[file_index] = sub_traces_events

    try:
      events_all[file_index] = events_all[file_index][0:sub_traces.shape[1]]
    except:
      pass
    framerate_all[file_index] = frame_rate

  if verbose > 2: print('Resampled ground truth from the neurons in this dataset. Done!')
  # Function output
  return sub_traces_all, sub_traces_events_all, framerate_all, events_all





def preprocess_groundtruth_artificial_noise_balanced(ground_truth_folders,before_frac,windowsize,after_frac,noise_level,sampling_rate,smoothing,omission_list=[],permute=1,maximum_traces=5000000,verbose=3,causal_kernel=0):


  sub_traces_all = [None]*500
  sub_traces_events_all = [None]*500
  events_all = [None]*500

  neuron_counter = 0
  nbx_datapoints = [None]*500
  dataset_sizes = np.zeros(len(ground_truth_folders),)
  dataset_indices = [None]*500

  for dataset_index,training_dataset in enumerate(ground_truth_folders):

    base_folder = os.getcwd()
    dataset_name = basename(normpath(ground_truth_folders[dataset_index]))

    try:
        if verbose > 2: print('Preprocessing dataset number', dataset_index)

        sub_traces_allX, sub_traces_events_allX, frame_rate, events_allX = calibrated_ground_truth_artificial_noise(ground_truth_folders[dataset_index],noise_level,sampling_rate,omission_list, verbose)

        datapoint_counter = 0
        for k in range(len(sub_traces_allX)):
          try:
             datapoint_counter += sub_traces_allX[k].shape[1]*sub_traces_allX[k].shape[0]
          except:
            if verbose > 2: print('No things for k={}'.format(k))

        dataset_sizes[dataset_index] = datapoint_counter

        nbx_datapoints[neuron_counter:neuron_counter+len(sub_traces_allX)] = datapoint_counter*np.ones(len(sub_traces_allX),)
        sub_traces_all[neuron_counter:neuron_counter+len(sub_traces_allX)] = sub_traces_allX
        sub_traces_events_all[neuron_counter:neuron_counter+len(sub_traces_allX)] = sub_traces_events_allX
        events_all[neuron_counter:neuron_counter+len(sub_traces_allX)] = events_allX
        dataset_indices[neuron_counter:neuron_counter+len(sub_traces_allX)] = dataset_index*np.ones(len(sub_traces_allX),)

        neuron_counter += len(sub_traces_allX)

    except:
         sub_traces_allX = None
         dataset_sizes[dataset_index] = np.NaN
    os.chdir(base_folder)

  mininum_traces = 15e6/len(ground_truth_folders)

  reduction_factors = dataset_sizes/mininum_traces

  if np.nanmax(reduction_factors) > 1:
    oversampling = 1
  else:
    oversampling = 0

  if verbose>1: print('Reducing ground truth by a factor of ca. '+str(int(3*np.nanmean(reduction_factors)))+'.')

  nbx_datapoints = nbx_datapoints[:neuron_counter]
  sub_traces_all = sub_traces_all[:neuron_counter]
  sub_traces_events_all = sub_traces_events_all[:neuron_counter]
  events_all = events_all[:neuron_counter]
  dataset_indices = dataset_indices[:neuron_counter]

  if verbose>1: print('Number of neurons in the ground truth: '+str(len(sub_traces_events_all)))

  before = int(before_frac*windowsize)
  after = int(after_frac*windowsize)

  X = np.zeros((15000000,windowsize,))
  Y = np.zeros((15000000,))


  counter = 0
  for neuron_ix,(sub_traces,sub_traces_events) in enumerate(zip(sub_traces_all,sub_traces_events_all)):

    if sub_traces is not None:

      for trace_index in range(sub_traces.shape[1]):

        single_trace = sub_traces[:,trace_index]
        single_spikes = sub_traces_events[:,trace_index]
        
        if causal_kernel:
          
          xx = np.arange(0,199)/sampling_rate
          yy = invgauss.pdf(xx,smoothing/sampling_rate*2,101/sampling_rate,1)
          ix = np.argmax(yy)
          yy = np.roll(yy,int((99-ix)/1.5))
          yy = yy/np.sum(yy)
          single_spikes = convolve(single_spikes,yy,mode='same')
          
        else:
          
          single_spikes = gaussian_filter(single_spikes.astype(float), sigma=smoothing)
          
        recording_length = np.sum(~np.isnan(single_trace))

        datapoints_used = np.minimum(len(single_spikes)-windowsize,recording_length-windowsize)

        if oversampling:

          # Discarding some samples (randomly chosen) to reduce ground truth dataset size

          datapoints_used_rand = np.random.permutation(datapoints_used)

          reduce_samples = reduction_factors[int(dataset_indices[neuron_ix])]

          datapoints_used_rand = datapoints_used_rand[0:int(len(datapoints_used_rand)/( max(reduce_samples,1)  ))]

        else:

          datapoints_used_rand = np.arange(datapoints_used)

        for time_points in datapoints_used_rand:

          Y[counter,] = single_spikes[time_points+before]
          X[counter,:,] = single_trace[time_points:(time_points+before+after)]

          counter += 1

  Y = np.expand_dims(Y[:counter],axis=1)
  X = np.expand_dims(X[:counter,:],axis=2)

  if permute == 1:

    p = np.random.permutation(len(X))
    X = X[p,:,:]
    Y = Y[p,:]

    # Maximum of 5e6 training samples
    X = X[:5000000,:,:]
    Y = Y[:5000000,:]

  os.chdir(base_folder)

  if verbose > 1: print('Shape of training dataset X: {}    Y: {}'.format(X.shape, Y.shape))
  return X,Y
