

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:00:57 2019

@author: pierre
"""


## Load packages

#cd '/media/pierre/Der Hort/Peter/FMI PhD/Project calibrated deconvolution/Calibrated Inference of Firing Rates/Package with GUI'


import os
import glob as glob
import scipy.io as sio
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
plt.style.use('seaborn-darkgrid')

import numpy as np
import scipy.io as sio
from scipy import optimize
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import resample
from scipy.interpolate import interp1d

import os
from os.path import normpath, basename
import glob

import matplotlib.pyplot as plt

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, MaxPooling1D, Conv1D, Input, LSTM,BatchNormalization, LocallyConnected1D, Activation, concatenate
from keras import backend as K


from helper_scripts.utils import define_model, noiselevels_test_dataset, preprocess_test_dataset
from helper_scripts.utils import preprocess_groundtruth_artificial_noise, calibrated_ground_truth_artificial_noise
from helper_scripts.utils_discrete_spikes import divide_and_conquer, fill_up_APs, systematic_exploration, random_motion, prune_APs



"""

Set model configuration and parameters.

"""

# Smoothing of ground truth
smoothing = 0.2 # std of Gaussian smoothing in time (sec)

# Determines how the time window used as input is positioned around the actual time point
windowsize = 64 # 64 timepoints
before_frac, after_frac = 0.5, 0.5 # symmetric

# Set model parameters 
conv_filter = Conv1D
filter_sizes = (31, 19, 5) # for each conv layer
filter_numbers = (30,40,50) # for each conv layer
dense_expansion = 30 # for dense layer

# Set parameters of gradient descent learning
loss_function = 'mean_squared_error'
optimizer = 'Adagrad'
nr_of_epochs = 10

# Use ensemble learning
ensemble_size = 1


"""

Use test dataset & set sampling/imaging rate

"""

test_dataset_folder = os.path.join('Example_datasets','Test_dataset_OGB_zf_4i3pdp')
sampling_rate = 7.5





"""

Extract noise level distribution for the test dataset; plot distribution

"""

fileList = glob.glob( os.path.join( test_dataset_folder, '*.mat'))

noise_levels_all = [None] * len(fileList)
for file_index,file in enumerate(fileList):
  noise_levels_all[file_index] = noiselevels_test_dataset(file,before_frac,windowsize,after_frac,sampling_rate)

noise_levels_pooled = np.array(noise_levels_all)
noise_levels_pooled = noise_levels_pooled[~np.isnan(noise_levels_pooled)]

percent99 = np.percentile(noise_levels_pooled,99)
percent999 = np.percentile(noise_levels_pooled,99.9)
percent1 = np.percentile(noise_levels_pooled,1)

plt.figure(1121); plt.hist(noise_levels_pooled,normed=True, bins=300);
plt.plot([percent99, percent99],[0, 1]);
plt.plot([percent1, percent1],[0, 1]);
plt.ylim([0, 1]); plt.xlim([0, percent999])

noise_levels_model = np.arange(2,np.ceil(percent99)+1)

nb_noise_levels = len(noise_levels_model)


""" 

Train network(s)

"""

load_pretrained_models = 0 # load pre-trained models


training_dataset_folders = [None]*2
training_dataset_folders[0] = os.path.join('GT_datasets','GT_dataset_OGB')
training_dataset_folders[1] = os.path.join('GT_datasets','GT_dataset_Cal520')


X = [[None]*ensemble_size for _ in range(nb_noise_levels)] 
Y = [[None]*ensemble_size for _ in range(nb_noise_levels)] 
set_of_models = [[None]*ensemble_size for _ in range(nb_noise_levels)] 

for noise_level_index,noise_level in enumerate(noise_levels_model):
  
  
  for ensemble in range(ensemble_size):
    
    print('Training model '+str(ensemble+1)+' with noise level '+str(noise_level))
    
    if load_pretrained_models:
      
      set_of_models[noise_level_index][ensemble] = load_model( os.path.join( test_dataset_folder,'Models', 'Model_noise_'+str(int(noise_level))+'_'+str(ensemble)+'.h5') )
      
    else:
    
      omission_list = []
      permute = 1
      
      X[noise_level_index][ensemble],Y[noise_level_index][ensemble] = preprocess_groundtruth_artificial_noise(training_dataset_folders,before_frac,windowsize,after_frac,noise_level,sampling_rate,smoothing*sampling_rate,omission_list,permute)
      set_of_models[noise_level_index][ensemble] = define_model(filter_sizes,filter_numbers,dense_expansion,windowsize,conv_filter,loss_function,optimizer)
      set_of_models[noise_level_index][ensemble].compile(loss=loss_function, optimizer=optimizer)
      set_of_models[noise_level_index][ensemble].fit(X[noise_level_index][ensemble], Y[noise_level_index][ensemble], batch_size=1024, epochs=nr_of_epochs,verbose=1)


""" 

Save network weights to file (unless pretrained models have been loaded)

"""

if load_pretrained_models == 0:
  
  foldername = os.path.join(test_dataset_folder,'Models')
  if not os.path.exists(foldername):
    os.mkdir(foldername)
  
  print('Saving models to disk for '+str(len(noise_levels_model))+' noise levels, each an ensemble of '+str(ensemble_size)+' models.')
  
  for noise_level_index,noise_level in enumerate(noise_levels_model):
    
    for ensemble in range(ensemble_size):
      
      set_of_models[noise_level_index][ensemble].save(os.path.join(foldername,'Model_noise_'+str(int(noise_level))+'_'+str(ensemble)+'.h5') )



"""

Preprocess and process test data

"""

# fileList is a list of mat-files, for which predictions should be made
fileList = glob.glob( os.path.join( test_dataset_folder, 'Calcium*.mat'))

for file_index,file in enumerate(fileList):
  
  XX = preprocess_test_dataset(file,before_frac,windowsize,after_frac)
  
  Y_predict = np.zeros((XX.shape[0],XX.shape[1]))
  
  for model_noise_index,model_noise in enumerate(noise_levels_model):
    
    print('Predictions for file '+str(file_index+1)+' out of ',str(len(fileList))+'; noise level '+str(int(model_noise)) )
    
    # Find indices of neurons with a given noise level ('model_noise')
    if model_noise == noise_levels_model[-1]: # Highest noise bin (or even higher)
      neurons_ixs = np.where(noise_levels_all[file_index] >= noise_levels_model[-1])[0]-1
    else: # Lower noise bins
      neurons_ixs = np.where(noise_levels_all[file_index] < model_noise)[0]

    Calcium_this_noise = XX[neurons_ixs,:,:]/100 # division by 100 to have dF/F values NOT in %
    Calcium_this_noise = np.reshape(Calcium_this_noise,(Calcium_this_noise.shape[0]*Calcium_this_noise.shape[1],Calcium_this_noise.shape[2]))

    for ensemble in range(ensemble_size):
    
      prediction = set_of_models[model_noise_index][ensemble].predict( np.expand_dims(Calcium_this_noise,axis=2),batch_size = 4096 )
      
      prediction = np.reshape(prediction,(len(neurons_ixs),XX.shape[1]))
      
      Y_predict[neurons_ixs,:] += prediction/ensemble_size

  # NaN for first and last datapoints, for which no predictions can be made
  Y_predict[:,0:int(before_frac*windowsize)] = np.nan
  Y_predict[:,-int(after_frac*windowsize):] = np.nan
  Y_predict[Y_predict==0] = np.nan
  
  # Enfore non-negative spike prediction values
  Y_predict[Y_predict<0] = 0

  if not os.path.exists('Predictions'):
    os.mkdir('Predictions')
  stripped_path = os.path.basename(os.path.normpath(file))
  sio.savemat(os.path.join(test_dataset_folder,'Predictions','Predictions_'+stripped_path),{'Y_predict':Y_predict})

# Clear Keras models from memory (otherwise, they accumulate and slow down things)
K.clear_session()



"""

Plot sample predictions together with raw calcium data

"""

import seaborn as sns; sns.set()

plt.style.use('seaborn-darkgrid')
index = [127, 7, 777, 941]
time = np.arange(0,XX.shape[1])/sampling_rate

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(time,XX[index[0],:,32]/100); axs[0, 0].plot(time,Y_predict[index[0],:]/2+1); axs[0, 0].set_ylim(-0.2, 3) 
axs[1, 0].plot(time,XX[index[1],:,32]/100); axs[1, 0].plot(time,Y_predict[index[1],:]/2+1); axs[1, 0].set_ylim(-0.2, 3) 
axs[0, 1].plot(time,XX[index[2],:,32]/100); axs[0, 1].plot(time,Y_predict[index[2],:]/2+1); axs[0, 1].set_ylim(-0.2, 3) 
axs[1, 1].plot(time,XX[index[3],:,32]/100); axs[1, 1].plot(time,Y_predict[index[3],:]/2+1); axs[1, 1].set_ylim(-0.2, 3) 




"""

Fill up probabilities (output of the network) with discrete spikes

"""


# fileList is a list of mat-files with predictions
# fileList2 a list of mat-files with the corresponding calcium data
fileList = glob.glob( os.path.join( test_dataset_folder, 'Predictions','Predictions_*.mat'))
fileList2 = glob.glob( os.path.join( test_dataset_folder, 'Calcium*.mat'))

for file,file2 in zip(fileList,fileList2):

  prob_density_all = sio.loadmat(file)['Y_predict']
  calcium_all = sio.loadmat(file2)['dF_traces']
  
  # initialize resulting list of spikes / matrix of approximations
  # "approximations" show how well the inferred spikes match the input probabilities
  # they are generated by convolving each inferred spike with the Gaussian kernel that
  # was used for generating the ground truth
  
  spikes_all = []
  approximations_all = np.nan*np.ones(prob_density_all.shape)
  
  for neuron in range(prob_density_all.shape[0]):
    
    print('Infer spikes for neuron '+str(neuron+1)+' out of '+str(prob_density_all.shape[0])+' for file '+basename(file2))
    
    prob_density = prob_density_all[neuron,:]
    Calcium = calcium_all[:,neuron]/100
    
    spike_locs_all = []
    
    # find non-nan indices (first and last frames of predictions are NaNs)
    nnan_indices = ~np.isnan(prob_density)
    # offset in time to assign inferred spikes to correct positions in the end
    offset = np.argmax(nnan_indices==True) - 1
    
    if np.sum(nnan_indices) > 0:
    
      prob_density = prob_density[nnan_indices]
      Calcium = Calcium[nnan_indices]
      
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
        
      
        temporal_offset = vector_of_indices[support_slices[k]][0]
        new_spikes = spike_locs+temporal_offset
        spike_locs_all.extend(new_spikes)
        
      approximations_all[neuron,nnan_indices] = approximation
      
    spikes_all.append(spike_locs_all+offset)
  
  # save results
  stripped_path = os.path.basename(os.path.normpath(file))
  sio.savemat(os.path.join(test_dataset_folder,'Predictions','Spikes_'+stripped_path),{'approximations_all':approximations_all,'spikes_all':spikes_all})



time = np.arange(0,len(prob_density_all[0,:]))/sampling_rate

index = [1,2,3,4]
index = [59, 7, 8, 57]
index = [11,12,13,14]
fig, axs = plt.subplots(2,2)
axs[0, 0].plot(time,prob_density_all[index[0],:]); axs[0, 0].plot(time,approximations_all[index[0],:]); axs[0, 0].set_ylim(-0.4, 3) 
axs[0, 0].plot(time,calcium_all[:,index[0]]/100+1.5); 
for spike in spikes_all[index[0]]:
  axs[0,0].plot([spike/sampling_rate,spike/sampling_rate],[-0.2, -0.1],'k')
axs[1, 0].plot(time,prob_density_all[index[1],:]); axs[1, 0].plot(time,approximations_all[index[1],:]); axs[1, 0].set_ylim(-0.4, 3) 
axs[1, 0].plot(time,calcium_all[:,index[1]]/100+1.5); 
for spike in spikes_all[index[1]]:
  axs[1,0].plot([spike/sampling_rate,spike/sampling_rate],[-0.2, -0.1],'k')
axs[0, 1].plot(time,prob_density_all[index[2],:]); axs[0, 1].plot(time,approximations_all[index[2],:]); axs[0, 1].set_ylim(-0.4, 3) 
axs[0, 1].plot(time,calcium_all[:,index[2]]/100+1.5); 
for spike in spikes_all[index[2]]:
  axs[0,1].plot([spike/sampling_rate,spike/sampling_rate],[-0.2, -0.1],'k')
axs[1, 1].plot(time,prob_density_all[index[3],:]); axs[1, 1].plot(time,approximations_all[index[3],:]); axs[1, 1].set_ylim(-0.4, 3) 
axs[1, 1].plot(time,calcium_all[:,index[3]]/100+1.5); 
for spike in spikes_all[index[3]]:
  axs[1,1].plot([spike/sampling_rate,spike/sampling_rate],[-0.2, -0.1],'k')







