#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 2020

Infer discrete spikes from probabilities: define helper functions

"""


from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as ndim
from copy import deepcopy
import numpy as np
import os
import scipy.io as sio

from . import config





def infer_discrete_spikes(spike_rates,model_name,model_folder='Pretrained_models'):

  """

  Main function. Detailed documentation: To be done.

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

  spikes_all = []
  approximations_all = np.nan*np.ones(spike_rates.shape)

  for neuron in range(spike_rates.shape[0]):

    print('Infer spikes for neuron '+str(neuron+1)+' out of '+str(spike_rates.shape[0]))

    prob_density = spike_rates[neuron,:]

    spike_locs_all = []

    # find non-nan indices (first and last frames of predictions are NaNs)
    nnan_indices = ~np.isnan(prob_density)
    # offset in time to assign inferred spikes to correct positions in the end
    offset = np.argmax(nnan_indices==True) - 1

    if np.sum(nnan_indices) > 0:

      prob_density = prob_density[nnan_indices]

      vector_of_indices = np.arange(0,len(prob_density))
      # "support_slices", indices of continuous chunks of the array which are non-zero and which might contain spikes
      support_slices = divide_and_conquer(prob_density,smoothing*sampling_rate)

      approximation = np.zeros(prob_density.shape)
      # go through each slice separately
      for k in range(len(support_slices)):

        spike_locs = []

        nb_spikes = np.sum(prob_density[support_slices[k]])

        # Monte Carlo/Metropolis-based sampling, initial guess of spikes
        spike_locs,approximation[support_slices[k]],counter = fill_up_APs(prob_density[support_slices[k]],smoothing*sampling_rate,nb_spikes,spike_locs)

        # every spike is shifted to any other position (no sub-pixel resolution) and the best position is used
        spike_locs,approximation[support_slices[k]] = systematic_exploration(prob_density[support_slices[k]],smoothing*sampling_rate,nb_spikes,spike_locs,approximation[support_slices[k]])

        # refine initial guess using random shifts or removal of spikes
        for jj in range(5):
          # remove the worst spikes
          spike_locs,approximation[support_slices[k]] = prune_APs(prob_density[support_slices[k]],smoothing*sampling_rate,nb_spikes,spike_locs,approximation[support_slices[k]])
          # fill up spikes again
          nb_spikes = np.sum(prob_density[support_slices[k]]) - np.sum(approximation[support_slices[k]])
          spike_locs,approximation[support_slices[k]],counter = fill_up_APs(prob_density[support_slices[k]],smoothing*sampling_rate,nb_spikes,spike_locs)

        # every spike is shifted to any other position (no sub-pixel resolution) and the best position is used
        spike_locs,approximation[support_slices[k]] = systematic_exploration(prob_density[support_slices[k]],smoothing*sampling_rate,nb_spikes,spike_locs,approximation[support_slices[k]])

        temporal_offset = vector_of_indices[support_slices[k]][0]
        new_spikes = spike_locs+temporal_offset
        spike_locs_all.extend(new_spikes)

      approximations_all[neuron,nnan_indices] = approximation

    spikes_all.append(spike_locs_all+offset)

  return approximations_all, spikes_all






def fill_up_APs(prob_density,smoothingX,nb_spikes,spike_locs):

  """

  fill_up_APs(): takes a probability distribution (prob_density) and an initial guess spikes (spike_locs)
  the smoothed spikes generate an approximation of the probability (approximation)
  The difference between the probability distribution and the approximation is then
  compensated with additional spikes. These spikes are sampled according to the distribution of
  the difference over time. This is a variation of a Monte Carlo / Metropolis algorithm.
  Technically, it generates a cumulative distribution and samples randomly along the y-axis of the
  cumulative distribution.

  """
  # produce approximation based on previously inferred spikes (spike_locs)
  approximation = np.zeros(prob_density.shape)
  for spike in spike_locs:
    approximation[spike] += 1
  approximation =  gaussian_filter(approximation.astype(float), sigma=smoothingX)

  # sample additional spike guesses using a Monte Carlo/Metropolis sampling scheme
  #
  # during each loop iteration, a spike is added at a likely location (spike_location)
  # the added spike is accepted or rejected based on whether the
  # error of the approximation decreases or not
  counter = 0
  while np.sum(approximation) < nb_spikes and counter < nb_spikes*20:

    if np.mod(counter,np.ceil(nb_spikes/10)) == 0:
      # a weighted cumulative probability distribution is computed; it is
      # recomputed every each time 10% of the spikes have been assigned
      norm_cum_distribution = np.cumsum(np.exp(prob_density - approximation) - 1)
      norm_cum_distribution /= np.max(norm_cum_distribution)

    spike_location = np.argmin(np.abs(norm_cum_distribution - np.random.uniform()))

    approximation_temp = deepcopy(approximation)
    this_spike =  np.zeros(prob_density.shape)
    this_spike[spike_location] = 1
    this_spike =  gaussian_filter(this_spike.astype(float), sigma=smoothingX)
    approximation += this_spike

    error_change = np.sum(np.abs(prob_density - approximation))  -  np.sum(np.abs(prob_density - approximation_temp))

    if error_change <= 0:
      spike_locs.append(spike_location)
    else:
      approximation = deepcopy(approximation_temp)

    counter += 1

  return spike_locs,approximation,counter



def divide_and_conquer(prob_density,smoothingX):

  """

  divide_and_conquer(): plits the probablity density in continous chunks of non-zero values (so-called "support").
  These are returned as "slices", i.e., ranges of indices.

  """
  support = prob_density > 0.03/(smoothingX)

  support = ndim.morphology.binary_dilation(support,np.ones((round(smoothingX*4), )))
  segmentation = ndim.label(support)
  support_slices = ndim.find_objects(segmentation[0])

  return support_slices



def systematic_exploration(prob_density,smoothingX,nb_spikes,spike_locs,approximation):

  """

  systematic_exploration(): for each spike, all other possible locations in the probability density are tested.
  If any position is any better than the initial guess, it is accepted, otherwise rejected.

  """
  # smoothed single spikes, initialized now beforehand because the creation takes time
  spike_reservoir = np.zeros((len(approximation),len(approximation)))
  for timepoint in range(len(approximation)):
    spike_reservoir[timepoint,timepoint] = 1
    spike_reservoir[timepoint,:] =  gaussian_filter( (spike_reservoir[timepoint,:]).astype(float), sigma=smoothingX)

  # add a spike at "timepoint", subtract the existing spike at "spike"
  error = np.zeros(approximation.shape)
  for spike_index,spike in enumerate(spike_locs):
    for timepoint in range(len(approximation)):
      approximation_suggestion = approximation + spike_reservoir[timepoint] - spike_reservoir[spike]
      error[timepoint] = np.sum(np.abs(prob_density - approximation_suggestion))

    ix = np.argmin(error)

    spike_locs[spike_index] = ix

    approximation = np.zeros(prob_density.shape)
    for spike in spike_locs:
      approximation[spike] += 1
    approximation =  gaussian_filter(approximation.astype(float), sigma=smoothingX)

  return spike_locs,approximation



def prune_APs(prob_density,smoothing,nb_spikes,spike_locs,approximation):

  """

  prune_APs(): chooses a random pair of two spikes and moves them randomly in small steps.
  If the result improves the fit, it is accepted, otherwise rejected.

  """
  # produce approximation based on previously inferred spikes (spike_locs)
  for spike_ix,spike1 in enumerate(spike_locs):

    spike_neg = np.zeros(prob_density.shape)
    spike_neg[spike1] = 1
    spike_neg =  gaussian_filter(spike_neg.astype(float), sigma=smoothing)

    approximation_temp = approximation - spike_neg

    error_change = np.sum(np.abs(prob_density - approximation_temp))  -  np.sum(np.abs(prob_density - approximation))

    if error_change < 0:
      spike_locs[spike_ix] = -1
      approximation -= spike_neg

  spike_locs = [x for x in spike_locs if x >= 0]

#  # produce approximation based on previously inferred spikes (spike_locs)
#  approximation = np.zeros(prob_density.shape)
#  for spike in spike_locs:
#
#    this_spike =  np.zeros(prob_density.shape)
#    this_spike[spike] = 1
#    this_spike =  gaussian_filter(this_spike.astype(float), sigma=smoothing)
#    approximation += this_spike

  return spike_locs,approximation
