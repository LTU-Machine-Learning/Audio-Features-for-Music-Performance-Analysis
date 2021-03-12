#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage examples of the audio_features module


Federico Visi 
Luleå University of Technology

Created on Mon Feb  1 14:40:38 2021

@author: federicovisi
"""
from audio_features import audio_features
from audio_features import normalize

import librosa
import librosa.display
import matplotlib.pyplot as plt

#%% Function call

# get the path of one of the librosa audio examples
path = librosa.ex('nutcracker')

# call the audio_features function
# y = audio; sr = sample rate; df = pandas dataframe containing a time vectore and 26 audio features
y, sr, df = audio_features(path)

#%% Plot waveform and 4 audio features

librosa.display.waveplot(y, sr=sr, alpha=0.4)
plt.plot(df['time'].values, normalize(df['spectral_centroid']), color='r')
plt.plot(df['time'].values, normalize(df['rolloff']), color='g')
plt.plot(df['time'].values, normalize(df['rms']), color='m')
plt.plot(df['time'].values, normalize(df['contrast']), color='y')

