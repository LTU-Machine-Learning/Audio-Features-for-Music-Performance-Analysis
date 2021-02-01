#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple module for extracting audio features from an audio file and save them into a pandas dataframe


Federico Visi 
Luleå University of Technology

Created on Wed Jan 27 22:14:48 2021

@author: federicovisi
"""
#%%
import numpy as np
import pandas as pd
import librosa
import sklearn as sk

def normalize(x, axis=0):
    return sk.preprocessing.minmax_scale(x, axis=axis)

def audio_features(path,hop_length=512):
    
    # Load the audio
    y, sr = librosa.load(path)
    
    # Default hop length = 512 samples ~= 23ms at 22050 Hz,

    
    # Compute spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, 
                                                          sr=sr)[0]
    
    # Compute spectral rollof
    rolloff = librosa.feature.spectral_rolloff(y=y, 
                                               sr=sr, 
                                               roll_percent=0.99)[0]
    
    # Compute root-mean-square
    rms = librosa.feature.rms(y=y)[0]
    
    # Compute spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, 
                                                 sr=sr)[0]
    
    # Compute spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    
    # Compute zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    # Compute MFCC (20)
    mfcc = librosa.feature.mfcc(y=y, sr=sr).T
    
    #time vector
    t2 = np.linspace(0, len(y)/sr, len(spectral_centroid)) #fixed num of steps assuming all features have the same length
    #TODO  steps = len(y)//hop_length+1 more genereal
    
    # Collect in pandas dataframe
    feature_array = [t2, spectral_centroid, rolloff, rms, spec_bw, flatness, zcr, mfcc]
    
    feature_index = ['time',
                     'spectral_centroid',
                     'rolloff',
                     'rms',
                     'spec_bw',
                     'flatness',
                     'zcr',
                     'mfcc']
    
    df = pd.DataFrame(data=feature_array,
                        index=feature_index).T
    return y, sr, df









