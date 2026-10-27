#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 2020

Infer discrete spikes from probabilities: define helper functions

"""


from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.stats import invgauss
import scipy.ndimage as ndim
import numpy as np
import os
import scipy.io as sio

from . import config





def infer_discrete_spikes(spike_rates,model_name,model_folder='Pretrained_models',verbosity=1,kernel='auto'):

  """

  Main function. The discrete spikes are smoothed with the same kernel that was used to smooth the ground truth
  during training of the model ('kernel'='auto'; Gaussian or causal, see 'causal_kernel' in the
  config file of the model). With 'kernel'='gaussian' or 'kernel'='causal', the respective kernel
  is used instead.
  """

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
  sampling_rate = cfg['sampling_rate']
  smoothing = cfg['smoothing']
  
  if kernel == 'auto':
    kernel = 'causal' if ('causal_kernel' in cfg and cfg['causal_kernel']) else 'gaussian'
  if kernel == 'causal':
    smoothing_kernel = causal_smoothing_kernel(sampling_rate,smoothing)
  elif kernel == 'gaussian':
    smoothing_kernel = None
    
  if verbosity:
    print('Discrete spikes are inferred using a {} smoothing kernel.'.format(kernel))

  spikes_all = []
  approximations_all = np.nan*np.ones(spike_rates.shape)

  for neuron in range(spike_rates.shape[0]):
    if verbosity:
      print('Infer spikes for neuron '+str(neuron+1)+' out of '+str(spike_rates.shape[0]))

    prob_density = spike_rates[neuron,:]

    spike_locs_all = []

    # find non-nan indices (first and last frames of predictions are NaNs)
    nnan_indices = ~np.isnan(prob_density)
    # offset in time to assign inferred spikes to correct positions in the end
    offset = np.argmax(nnan_indices==True)

    if np.sum(nnan_indices) > 0:

      prob_density = prob_density[nnan_indices]

      vector_of_indices = np.arange(0,len(prob_density))
      # "support_slices", indices of continuous chunks of the array which are non-zero and which might contain spikes
      support_slices = divide_and_conquer(prob_density,smoothing*sampling_rate)

      approximation = np.zeros(prob_density.shape)
      # go through each slice separately
      for k in range(len(support_slices)):

        prob_slice = prob_density[support_slices[k]]

        # smoothed single spikes for all possible spike positions in this slice (computed only once per slice)
        spike_kernels = single_spike_kernels(len(prob_slice),smoothing*sampling_rate,smoothing_kernel)

        spike_locs = []

        nb_spikes = np.sum(prob_slice)

        # Monte Carlo/Metropolis-based sampling, initial guess of spikes
        spike_locs,approximation_slice,counter = fill_up_APs(prob_slice,spike_kernels,nb_spikes,spike_locs)

        # every spike is shifted to any other position (no sub-pixel resolution) and the best position is used
        spike_locs,approximation_slice = systematic_exploration(prob_slice,spike_kernels,nb_spikes,spike_locs,approximation_slice)

        # refine initial guess using random shifts or removal of spikes
        for jj in range(5):
          # remove the worst spikes
          spike_locs,approximation_slice = prune_APs(prob_slice,spike_kernels,nb_spikes,spike_locs,approximation_slice)
          # fill up spikes again
          nb_spikes = np.sum(prob_slice) - np.sum(approximation_slice)
          spike_locs,approximation_slice,counter = fill_up_APs(prob_slice,spike_kernels,nb_spikes,spike_locs)

        # every spike is shifted to any other position (no sub-pixel resolution) and the best position is used
        spike_locs,approximation_slice = systematic_exploration(prob_slice,spike_kernels,nb_spikes,spike_locs,approximation_slice)

        approximation[support_slices[k]] = approximation_slice

        temporal_offset = vector_of_indices[support_slices[k]][0]
        new_spikes = spike_locs+temporal_offset
        spike_locs_all.extend(new_spikes)

      approximations_all[neuron,nnan_indices] = approximation

    spikes_all.append(spike_locs_all+offset)

  return approximations_all, spikes_all


def causal_smoothing_kernel(sampling_rate,smoothing):

  """
  
  This function returns the causal kernel that was used to smooth the ground truth
  for models trained with causal kernel (identical to the kernel defined in
  utils.preprocess_groundtruth_artificial_noise_balanced(); 'smoothing' is given in seconds).
  """
  
  xx = np.arange(0,199)/sampling_rate
  yy = invgauss.pdf(xx,smoothing*2.0,101/sampling_rate,1)
  ix = np.argmax(yy)
  yy = np.roll(yy,int((99-ix)/1.5))
  # values below 1e-9 of the maximum (together <1e-8 of the kernel) are set to zero; this
  # makes the kernel shorter and therefore the computations below faster
  yy[yy < 1e-9*np.max(yy)] = 0
  return yy/np.nansum(yy)



def smooth_spikes(spikes,smoothingX,smoothing_kernel=None):

  """

  This function converts discrete spikes into a smoothed spike rate (along the last axis), with a Gaussian kernel
  (standard deviation 'smoothingX', in frames) if 'smoothing_kernel' is None, otherwise with
  the (causal) kernel 'smoothing_kernel', in the same way as the ground truth was generated
  during training.
  """
  
  if smoothing_kernel is None:
    return gaussian_filter1d(spikes.astype(float), sigma=smoothingX, axis=-1)
  else:
    # same output as scipy.signal.convolve(spikes,smoothing_kernel,mode='same') as used during training
    return convolve1d(spikes.astype(float), smoothing_kernel, axis=-1, mode='constant')


def single_spike_kernels(slice_length,smoothingX,smoothing_kernel=None):

  """

  single_spike_kernels(): computes, for each possible spike position t in a slice of length 'slice_length',
  the smoothed rate generated by a single spike at t (including the effects of the slice boundaries).
  This smoothed spike is non-zero only close to t; therefore only a window of 'width' frames, starting
  at frame t-'before', is stored. All arrays below that are zero-padded by 'before' frames at the beginning
  (and 'width'-'before'-1 frames at the end) contain the window of a spike at t at indices t:t+width.

  Returns a dictionary with the entries:
    'kernels': array (slice_length x width), the smoothed single spikes
    'mass':    vector (slice_length), sum of each smoothed single spike within the slice
    'before':  number of frames before the spike that are covered by the window

  """
  # smoothed single spike far away from the slice boundaries, and its extent
  M = int(4.0*smoothingX + 0.5) + 2 + (0 if smoothing_kernel is None else len(smoothing_kernel))
  delta = np.zeros(2*M+1)
  delta[M] = 1
  single_spike = smooth_spikes(delta,smoothingX,smoothing_kernel)
  nonzero = np.nonzero(single_spike)[0]
  before = max(M - nonzero[0], 0)
  after = max(nonzero[-1] - M, 0)
  width = before + after + 1

  # far away from the slice boundaries, all smoothed single spikes have the same shape
  kernels = np.tile(single_spike[M-before:M+after+1], (slice_length,1))

  # close to the slice boundaries, they are computed explicitly (reflected at the boundary for the Gaussian
  # kernel, truncated for the causal kernel), with the same function as above
  positions = np.arange(slice_length)
  edge_positions = positions[ (positions < before) | (positions > slice_length-1-after) ]
  if len(edge_positions) > 0:
    single_spikes = np.zeros((len(edge_positions),slice_length))
    single_spikes[np.arange(len(edge_positions)),edge_positions] = 1
    edge_kernels = np.zeros((len(edge_positions),slice_length+width-1))
    edge_kernels[:,before:before+slice_length] = smooth_spikes(single_spikes,smoothingX,smoothing_kernel)
    for i,t in enumerate(edge_positions):
      kernels[t] = edge_kernels[i,t:t+width]

  return {'kernels':kernels, 'mass':np.sum(kernels,axis=1), 'before':before}



def approximation_from_spikes(spike_locs,spike_kernels,slice_length):

  """

  approximation_from_spikes(): sum of the smoothed single spikes, as zero-padded array (see single_spike_kernels())

  """
  kernels = spike_kernels['kernels']
  width = kernels.shape[1]
  approximation_padded = np.zeros(slice_length+width-1)
  for spike in spike_locs:
    approximation_padded[spike:spike+width] += kernels[spike]
  return approximation_padded



def fill_up_APs(prob_density,spike_kernels,nb_spikes,spike_locs):

  """

  fill_up_APs(): takes a probability distribution (prob_density) and an initial guess spikes (spike_locs)
  the smoothed spikes generate an approximation of the probability (approximation)
  The difference between the probability distribution and the approximation is then
  compensated with additional spikes. These spikes are sampled according to the distribution of
  the difference over time. This is a variation of a Monte Carlo / Metropolis algorithm.
  Technically, it generates a cumulative distribution and samples randomly along the y-axis of the
  cumulative distribution.

  """
  kernels = spike_kernels['kernels']
  width = kernels.shape[1]
  before = spike_kernels['before']
  L = len(prob_density)

  # produce approximation based on previously inferred spikes (spike_locs), zero-padded
  prob_padded = np.zeros(L+width-1)
  prob_padded[before:before+L] = prob_density
  approximation_padded = approximation_from_spikes(spike_locs,spike_kernels,L)
  approximation = approximation_padded[before:before+L]   # view, without padding

  # sample additional spike guesses using a Monte Carlo/Metropolis sampling scheme;
  # during each loop iteration, a spike is added at a likely location (spike_location)
  # the added spike is accepted or rejected based on whether the
  # error of the approximation decreases or not
  
  # 'nb_spikes' is the number of spikes to be added to the already existing spikes
  initial_sum = np.sum(approximation)
  counter = 0
  while np.sum(approximation) - initial_sum < nb_spikes and counter < nb_spikes*20:

    if np.mod(counter,np.ceil(nb_spikes/10)) == 0:
      # a weighted cumulative probability distribution is computed; it is
      # recomputed every each time 10% of the spikes have been assigned
      norm_cum_distribution = np.cumsum(np.exp(prob_density - approximation) - 1)
      norm_cum_distribution /= np.max(norm_cum_distribution)

    spike_location = np.argmin(np.abs(norm_cum_distribution - np.random.uniform()))

    # change of the error when adding the spike; computed only within the window
    # where the smoothed spike is non-zero (outside of it, the error does not change)
    residual = prob_padded[spike_location:spike_location+width] - approximation_padded[spike_location:spike_location+width]
    error_change = np.sum(np.abs(residual - kernels[spike_location])) - np.sum(np.abs(residual))

    # spikes whose smoothed rate lies mostly outside of the slice are not accepted (relevant for causal kernels)
    if error_change <= 0 and spike_kernels['mass'][spike_location] > 0.5:
      spike_locs.append(spike_location)
      approximation_padded[spike_location:spike_location+width] += kernels[spike_location]

    counter += 1

  return spike_locs,approximation.copy(),counter



def divide_and_conquer(prob_density,smoothingX):

  """

  divide_and_conquer(): splits the probablity density into continous chunks of non-zero values (so-called "support").
  These are returned as "slices", i.e., ranges of indices.

  """
  support = prob_density > 0.03/(smoothingX)

  support = ndim.morphology.binary_dilation(support,np.ones((round(smoothingX*4), )))
  segmentation = ndim.label(support)
  support_slices = ndim.find_objects(segmentation[0])

  return support_slices



def error_changes(residual_padded,kernels,start,stop):

  """

  error_changes(): for all spike positions t in start:stop, the change of the error sum(abs(residual))
  when the smoothed single spike at position t is subtracted from the (zero-padded) residual

  """
  width = kernels.shape[1]
  # windows[i,:] is a read-only view of residual_padded[start+i:start+i+width]
  windows = np.lib.stride_tricks.as_strided(residual_padded[start:], shape=(stop-start,width), strides=(residual_padded.strides[0],)*2, writeable=False)
  return np.sum(np.abs(windows - kernels[start:stop]),axis=1) - np.sum(np.abs(windows),axis=1)



def systematic_exploration(prob_density,spike_kernels,nb_spikes,spike_locs,approximation):

  """

  systematic_exploration(): for each spike, all other possible locations in the probability density are tested.
  If any position is any better than the initial guess, it is accepted, otherwise rejected.

  """
  kernels = spike_kernels['kernels']
  width = kernels.shape[1]
  before = spike_kernels['before']
  L = len(prob_density)

  if len(spike_locs) == 0:
    return spike_locs,approximation

  # positions whose smoothed rate lies mostly outside of the slice are excluded (relevant for causal kernels)
  excluded_positions = spike_kernels['mass'] < 0.5

  # residual (probability minus approximation), zero-padded
  residual_padded = np.zeros(L+width-1)
  residual_padded[before:before+L] = prob_density - approximation

  # change of the error when adding a spike at any of the positions; when a spike is moved, this
  # changes only for positions close to the old and the new position of the spike
  error_change = error_changes(residual_padded,kernels,0,L)

  # subtract the existing spike at "spike", add a spike at any other position; use the best position
  for spike_index,spike in enumerate(spike_locs):

    residual_padded[spike:spike+width] += kernels[spike]
    start = max(spike-width+1,0); stop = min(spike+width,L)
    error_change[start:stop] = error_changes(residual_padded,kernels,start,stop)

    ix = np.argmin(np.where(excluded_positions,np.inf,error_change))

    spike_locs[spike_index] = ix

    residual_padded[ix:ix+width] -= kernels[ix]
    for position in (spike,ix):
      start = max(position-width+1,0); stop = min(position+width,L)
      error_change[start:stop] = error_changes(residual_padded,kernels,start,stop)

  approximation = approximation_from_spikes(spike_locs,spike_kernels,L)[before:before+L]

  return spike_locs,approximation



def prune_APs(prob_density,spike_kernels,nb_spikes,spike_locs,approximation):

  """

  prune_APs(): for each spike, it is tested whether removing the spike improves the fit.
  If the result improves the fit, the spike is removed, otherwise it is kept.

  """
  kernels = spike_kernels['kernels']
  width = kernels.shape[1]
  before = spike_kernels['before']
  L = len(prob_density)

  # residual (probability minus approximation), zero-padded
  residual_padded = np.zeros(L+width-1)
  residual_padded[before:before+L] = prob_density - approximation

  for spike_ix,spike1 in enumerate(spike_locs):

    # change of the error when removing the spike (only within the window of the smoothed spike)
    residual = residual_padded[spike1:spike1+width]
    error_change = np.sum(np.abs(residual + kernels[spike1]))  -  np.sum(np.abs(residual))

    if error_change < 0:
      spike_locs[spike_ix] = -1
      residual_padded[spike1:spike1+width] += kernels[spike1]

  spike_locs = [x for x in spike_locs if x >= 0]

  approximation = prob_density - residual_padded[before:before+L]

  return spike_locs,approximation