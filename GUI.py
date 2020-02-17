#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 01:35:23 2020

@author: Peter Rupprecht, Adrian Hoffmann
"""






# =============================================================================
# 0. Option to save and load pretrained models
# 1. Resize and beautify buttons and text elements; relief ("sunken in")
# 2. Catch errors (window size too small; no dataset selected; wrong order of button clicks; ...)
# 3. Options to save the model and to save the predictions to defined folders
# 4. Infer spikes
# =============================================================================











# =============================================================================
# HARDCODED PARAMETER FOR GUI
# =============================================================================



WINDOW_WIDTH = 800
WINDOW_HEIGHT = 700

WINDOW_WIDTH_L = 400

BUTTON_WIDTH = 130
BUTTON_HEIGHT = 40

# session info
S_LEFT = 30+20    # position of the first element in session box
S_TOP = 20+30
ROW = 70
COL = 220
S_HEIGHT = 550   # height of session box

# behavior
B_LEFT = S_LEFT
B_TOP = S_HEIGHT + 70

# imaging
I_LEFT = WINDOW_WIDTH_L-20
I_TOP = B_TOP

L_TOP = B_TOP + 500


before_frac = 0.5
after_frac = 0.5


import wx
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


from helper_scripts.utils import define_model,noiselevels_test_dataset,preprocess_test_dataset
from helper_scripts.utils import preprocess_groundtruth_artificial_noise,calibrated_ground_truth_artificial_noise
from helper_scripts.utils_discrete_spikes import divide_and_conquer, fill_up_APs, systematic_exploration, random_motion, prune_APs


class window(wx.Frame):

    def __init__(self,parent,id):
        
        self.var_ensemble = 5.0
        self.var_windowsize = 64.0
        self.var_smoothing = 0.2
        self.var_epochs = 10.0
        self.var_framerate = 30.0
        
        self.var_show_noise_histogram = False
        self.var_show_example_predictions = False
        
        self.var_pretrained_model = ''
        self.var_testdataset_folder = ''
        
        self.noise_levels = None
        self.noise_levels_model = [2.0,3.0,4.0,5.0,6.0,7.0]
        
        self.X = None
        self.Y = None
        self.set_of_models = None
        
        wx.Frame.__init__(self,parent,id,'Use deep networks to predict spiking',size=(WINDOW_WIDTH,WINDOW_HEIGHT))
        panel=wx.Panel(self)


        self.statusbar = self.CreateStatusBar(1)
        self.statusbar.SetStatusText('Ready to load training/testing datasets...')

        wx.StaticBox(panel, label='Training dataset',
                    pos=(S_LEFT-20, S_TOP-30), size=(WINDOW_WIDTH_L-2*S_LEFT, S_HEIGHT))
        wx.StaticBox(panel, label='Test dataset',
                    pos=(S_LEFT-20+WINDOW_WIDTH_L, S_TOP-30), size=(WINDOW_WIDTH_L-2*S_LEFT, S_HEIGHT))

        
        self.pretrained = wx.CheckBox(panel, label='Use pretrained models', pos=(S_LEFT,S_TOP-0))
        self.pretrained.SetValue(False)
        wx.EVT_CHECKBOX(self, self.pretrained.GetId(), self.load_or_train)
        
        # Load session button
        self.load_pretrained = wx.Button(panel,label="Load models",
                                       pos=(S_LEFT+160, S_TOP+26),
                                       size=(100, 25) )
        self.Bind( wx.EVT_BUTTON, self.event_load_pretrained, self.load_pretrained)
        self.loaded_file = wx.StaticText(panel,label="No model", pos=(S_LEFT,S_TOP+30))
        self.load_pretrained.Disable()
        
        
        # select multiple entries with holding Ctrl
        
        
        fileList = glob.glob( 'GT_*')
        
        wx.StaticText(panel,label="Select training datasets", pos=(S_LEFT,S_TOP+ROW))
        self.training_list = wx.ListBox(panel,
                size=(250, 150),
                pos = (S_LEFT,S_TOP+95),
                style=wx.LB_MULTIPLE,
                choices=fileList)
        self.training_list.Bind(wx.EVT_LISTBOX, self.change_training_set)
        
        #  Ensemble size
        wx.StaticText(panel,label="Ensemble size", pos=(S_LEFT+55,S_TOP+255))
        self.ensemble = wx.TextCtrl(panel,style=wx.TE_CENTRE, pos=(S_LEFT,S_TOP+255), size=(40,-1))
        self.ensemble.SetValue( str(self.var_ensemble) )
        #  Window size
        wx.StaticText(panel,label="Window size (datapoints)", pos=(S_LEFT+55,S_TOP+295))
        self.windowsize = wx.TextCtrl(panel,style=wx.TE_CENTRE, pos=(S_LEFT,S_TOP+295), size=(40,-1))
        self.windowsize.SetValue( str(self.var_windowsize) )
        #  Smoothing kernel (std of Gaussian (sec))
        wx.StaticText(panel,label="Smoothing (std of Gaussian, sec)", pos=(S_LEFT+55,S_TOP+335))
        self.smoothing = wx.TextCtrl(panel,style=wx.TE_CENTRE, pos=(S_LEFT,S_TOP+335), size=(40,-1))
        self.smoothing.SetValue( str(self.var_smoothing) )
        #  Training epochs
        wx.StaticText(panel,label="Training epochs",  pos=(S_LEFT+55,S_TOP+375))
        self.epochs = wx.TextCtrl(panel,style=wx.TE_CENTRE, pos=(S_LEFT,S_TOP+375), size=(40,-1))
        self.epochs.SetValue( str(self.var_epochs) )
        
        self.plothistogram = wx.CheckBox(panel, label='Histogram of noise levels', pos=(S_LEFT,S_TOP+415))
        self.plothistogram.SetValue(self.var_show_noise_histogram)

        self.plotexamples = wx.CheckBox(panel, label='Plot example predictions', pos=(S_LEFT,S_TOP+445))
        self.plotexamples.SetValue(self.var_show_example_predictions)

        
        
        # Load session button
        self.load_testdata = wx.Button(panel,label="Select test data",
                                       pos=(S_LEFT+160+WINDOW_WIDTH_L, S_TOP+26),
                                       size=(100, 25) )
        self.Bind( wx.EVT_BUTTON, self.event_load_testdata, self.load_testdata)
        self.loaded_file_test = wx.StaticText(panel,label="Test dataset", pos=(S_LEFT+WINDOW_WIDTH_L,S_TOP+30))

        #  Framerate
        wx.StaticText(panel,label="Framerate (Hz)", pos=(S_LEFT+55+WINDOW_WIDTH_L,S_TOP+70))
        self.framerate = wx.TextCtrl(panel,style=wx.TE_CENTRE, pos=(S_LEFT+WINDOW_WIDTH_L,S_TOP+70), size=(40,-1))
        self.framerate.SetValue( str(self.var_framerate) )
        
        
        
        
        # Compute noise levels
        self.noise_level_button = wx.Button(panel,label="Noise levels",
                                       pos=(S_LEFT+2*COL, S_TOP+2*ROW+20),
                                       size=(180, 60) )
        self.Bind( wx.EVT_BUTTON, self.compute_noise_levels, self.noise_level_button)
        
        # Train model
        self.train_model_button = wx.Button(panel,label="Train model",
                                       pos=(S_LEFT+2*COL, S_TOP+3*ROW+20),
                                       size=(180, 60) )
        self.Bind( wx.EVT_BUTTON, self.train_model_levels, self.train_model_button)
        
        # Predict spiking probabilities
        self.predict_button = wx.Button(panel,label="Predict!",
                                       pos=(S_LEFT+2*COL, S_TOP+4*ROW+20),
                                       size=(180, 60) )
        self.Bind( wx.EVT_BUTTON, self.predict_levels, self.predict_button)
        
        # Infer descrete, single spikes
        self.infer_spikes_button = wx.Button(panel,label="Infer spikes",
                                       pos=(S_LEFT+2*COL, S_TOP+5*ROW+20),
                                       size=(180, 60) )
        self.Bind( wx.EVT_BUTTON, self.infer_spikes_levels, self.infer_spikes_button)
        


    def update_variables(self, event): 
        """ xxx """
        self.var_ensemble = float(self.ensemble.GetValue())
        self.var_windowsize = float(self.windowsize.GetValue())
        self.var_smoothing = float(self.smoothing.GetValue())
        self.var_epochs = float(self.epochs.GetValue())
        self.var_framerate = float(self.framerate.GetValue())
        
        self.var_show_example_predictions = self.plotexamples.GetValue()
        self.var_show_noise_histogram = self.plothistogram.GetValue()

    def load_or_train(self,event):
        
        if self.pretrained.GetValue():
            
            self.training_list.Disable()
            self.load_pretrained.Enable(True)
            self.loaded_file.SetLabel('N/A')
            
        else:
          
            self.training_list.Enable(True)
            self.load_pretrained.Disable()
            self.loaded_file.SetLabel('No model')
            
    def event_load_testdata(self,event):
        
        dialog = wx.DirDialog(None, "Choose directory containing test data", os.getcwd())
        if dialog.ShowModal() == wx.ID_OK:
            self.var_testdataset_folder = dialog.GetPath()
        dialog.Destroy()
        
        short_path = os.path.basename(os.path.normpath(self.var_testdataset_folder))
        
        self.loaded_file_test.SetLabel(short_path)

        self.statusbar.SetStatusText('Loaded test dataset '+short_path)

    def event_load_pretrained(self, event):
        
        dialog = wx.DirDialog(None, "Choose directory containing pretrained models", os.getcwd())
        if dialog.ShowModal() == wx.ID_OK:
            self.var_pretrained_model = dialog.GetPath()
        dialog.Destroy()
        
        short_path = os.path.basename(os.path.normpath(self.var_pretrained_model))
        
        self.loaded_file.SetLabel(short_path)
        
        print(short_path)
        
        self.set_of_models = [[None]*int(self.var_ensemble) for _ in range(len(self.noise_levels_model))] 
        
        for noise_level_index,noise_level in enumerate(self.noise_levels_model):
          
          self.statusbar.SetStatusText('Load ensemble of '+str(self.var_ensemble)+' models with noise level '+str(int(noise_level)))
          
          for ensemble in range(int(self.var_ensemble)):
            
            self.set_of_models[noise_level_index][ensemble] = load_model( os.path.join( self.var_testdataset_folder,short_path, 'Model_noise_'+str(int(noise_level))+'_'+str(ensemble)+'.h5') )  
            
        
        self.statusbar.SetStatusText('Loaded model '+short_path)


    def change_training_set(self,event):
        
        selection = self.training_list.GetSelections()
        
        all_items = self.training_list.Items
        
        self.statusbar.SetStatusText('Selected '+str(len(selection))+' training set(s).')
        
        
    def compute_noise_levels(self, event):
        
        self.update_variables(-1)
        
        testdata_files = glob.glob(os.path.join(self.var_testdataset_folder,'*.mat'))
        self.noise_levels = [None]*len(testdata_files)
        
        for file_index,file in enumerate(testdata_files):
            self.noise_levels[file_index] = noiselevels_test_dataset(file,before_frac,self.var_windowsize,after_frac,float(self.var_framerate))
        
        noise_levels_pooled = np.array(self.noise_levels)
        noise_levels_pooled = noise_levels_pooled[~np.isnan(noise_levels_pooled)]
        
        percent99 = np.percentile(noise_levels_pooled,99)
        percent999 = np.percentile(noise_levels_pooled,99.9)
        percent1 = np.percentile(noise_levels_pooled,1)
        
        if self.var_show_noise_histogram:
            plt.figure; plt.hist(noise_levels_pooled,normed=True, bins=300);
            plt.plot([percent99, percent99],[0, 1]);
            plt.plot([percent1, percent1],[0, 1]);
            plt.ylim([0, 1]); plt.xlim([0, percent999])
            plt.show()
            
        self.noise_levels_model = np.arange(2,np.maximum(3,np.ceil(percent99)+1))
        
        print(self.noise_levels_model)
        
        self.statusbar.SetStatusText('Noise levels computed, with a median of '+str(np.nanmedian(self.noise_levels)))
        
    def train_model_levels(self, event):
        self.update_variables(-1)
        
        conv_filter = Conv1D
        filter_sizes = (31, 19, 5) # for each conv layer
        filter_numbers = (30,40,50) # for each conv layer
        dense_expansion = 30 # for dense layer

        loss_function = 'mean_squared_error'
        optimizer = 'Adagrad'
        
        selection = self.training_list.GetSelections()
        
        fileList = glob.glob( 'GT_*')
        training_dataset_folders = [None]*len(selection)
        for k in range(len(selection)):
          training_dataset_folders[k] = fileList[selection[k]]
        
        self.X = [[None]*int(self.var_ensemble) for _ in range(len(self.noise_levels_model))] 
        self.Y = [[None]*int(self.var_ensemble) for _ in range(len(self.noise_levels_model))] 
        self.set_of_models = [[None]*int(self.var_ensemble) for _ in range(len(self.noise_levels_model))] 
        
        for noise_level_index,noise_level in enumerate(self.noise_levels_model):
          
          self.statusbar.SetStatusText('Train ensemble of '+str(self.var_ensemble)+' models with noise level '+str(int(noise_level)))
          
          for ensemble in range(int(self.var_ensemble)):
            
            omission_list = []
            permute = 1
            
            self.X[noise_level_index][ensemble],self.Y[noise_level_index][ensemble] = preprocess_groundtruth_artificial_noise(training_dataset_folders,before_frac,int(self.var_windowsize),after_frac,noise_level,self.var_framerate,self.var_smoothing*self.var_framerate,omission_list,permute)
            self.set_of_models[noise_level_index][ensemble] = define_model(filter_sizes,filter_numbers,dense_expansion,int(self.var_windowsize),conv_filter,loss_function,optimizer)
            self.set_of_models[noise_level_index][ensemble].compile(loss=loss_function, optimizer=optimizer)
            self.set_of_models[noise_level_index][ensemble].fit(self.X[noise_level_index][ensemble], self.Y[noise_level_index][ensemble], batch_size=1024, epochs=int(self.var_epochs),verbose=1)
            
            foldername = os.path.join(self.var_testdataset_folder,'Models')
            if not os.path.exists(foldername):
              os.mkdir(foldername)
            
            print('Saving model to disk for '+str(noise_level)+' noise level, ensemble '+str(ensemble)+'.')
                 
            self.set_of_models[noise_level_index][ensemble].save(os.path.join(foldername,'Model_noise_'+str(int(noise_level))+'_'+str(ensemble)+'.h5') )

              
        self.statusbar.SetStatusText('Model trained')
        
    def predict_levels(self, event):
      
        fileList = glob.glob( os.path.join( self.var_testdataset_folder, 'Calcium*.mat'))
        
        for file_index,file in enumerate(fileList):
          
            XX = preprocess_test_dataset(file,before_frac,int(self.var_windowsize),after_frac)
            
            Y_predict = np.zeros((XX.shape[0],XX.shape[1]))
            
            for model_noise_index,model_noise in enumerate(self.noise_levels_model):
              
                print('Predictions for file '+str(file_index+1)+' out of ',str(len(fileList))+'; noise level '+str(int(model_noise)) )
              
                # Find indices of neurons with a given noise level ('model_noise')
                if (model_noise < 2 and  model_noise == self.noise_levels_model[-1]): # Highest noise bin (or even higher)
                    neurons_ixs = np.where(self.noise_levels[file_index] >= self.noise_levels_model[-1])[0]
                      
                else: # Lower noise bins
                    neurons_ixs = np.where(self.noise_levels[file_index] < model_noise)[0]
                      
                    
                
                Calcium_this_noise = XX[neurons_ixs,:,:]/100 # division by 100 to have dF/F values NOT in %
                Calcium_this_noise = np.reshape(Calcium_this_noise,(Calcium_this_noise.shape[0]*Calcium_this_noise.shape[1],Calcium_this_noise.shape[2]))
            
                for ensemble in range(int(self.var_ensemble)):
                
                    prediction = self.set_of_models[model_noise_index][ensemble].predict( np.expand_dims(Calcium_this_noise,axis=2),batch_size = 4096 )
                    
                    prediction = np.reshape(prediction,(len(neurons_ixs),XX.shape[1]))
                    
                    Y_predict[neurons_ixs,:] += prediction/self.var_ensemble
          
            # NaN for first and last datapoints, for which no predictions can be made
            Y_predict[:,0:int(before_frac*int(self.var_windowsize))] = np.nan
            Y_predict[:,-int(after_frac*int(self.var_windowsize)):] = np.nan
            Y_predict[Y_predict==0] = np.nan
            
            # Enfore non-negative spike prediction values
            Y_predict[Y_predict<0] = 0
          
            if not os.path.exists('Predictions'):
                os.mkdir('Predictions')
            stripped_path = os.path.basename(os.path.normpath(file))
            sio.savemat(os.path.join(self.var_testdataset_folder,'Predictions','Predictions_'+stripped_path),{'Y_predict':Y_predict})
            self.statusbar.SetStatusText('Predictions complete for file '+file)
      
        self.statusbar.SetStatusText('Predictions complete')
        
    def infer_spikes_levels(self, event):
        self.update_variables(-1)
        
        smoothing = self.var_smoothing
        sampling_rate = self.var_framerate
        
        # fileList is a list of mat-files with predictions
        # fileList2 a list of mat-files with the corresponding calcium data)
        fileList = glob.glob( os.path.join( self.var_testdataset_folder, 'Predictions','Predictions_*.mat'))
        fileList2 = glob.glob( os.path.join( self.var_testdataset_folder, 'Calcium*.mat'))
        
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
            
            
            self.statusbar.SetStatusText('Infer spikes for neuron '+str(neuron+1)+' out of '+str(prob_density_all.shape[0])+' for file '+basename(file2))
#            print('Infer spikes for neuron '+str(neuron+1)+' out of '+str(prob_density_all.shape[0])+' for file '+basename(file2))
            
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
          sio.savemat(os.path.join(self.var_testdataset_folder,'Predictions','Spikes_'+stripped_path),{'approximations_all':approximations_all,'spikes_all':spikes_all})
          self.statusbar.SetStatusText('All spikes inferred.')

        
        
        
        
        
        
        
        
        
        
        
        
        
        

# run the GUI
if __name__ == '__main__':
    app = wx.App()
    frame = window(parent=None,id=-1)
    frame.Show()
    app.MainLoop()