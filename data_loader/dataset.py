import torch
import os
import glob
import pandas as pd
import soundfile as sf
from PIL import Image
import numpy as np
import cv2
import librosa
import torch
from nnAudio.Spectrogram import CQT1992v2, CQT2010v2
import random

import copy
from multiprocessing import Pool
from functools import partial
import pickle
import ast

from scipy import signal

def filterSig(waves, a=None, b=None):
    '''Apply a 20Hz high pass filter to the three events'''
    return np.array([signal.filtfilt(b, a, wave) for wave in waves]) #lfilter introduces a larger spike around 20hz

def apply_qtransform(waves, transform=CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=64)):
    waves = np.hstack(waves)
    waves = waves / np.max(waves)
    waves = torch.from_numpy(waves).float()
    image = transform(waves)
    return image

def apply_qtransform_2(
    waves, 
    transform=CQT1992v2(
        sr=2048, 
        fmin=20, 
        fmax=1024, 
        hop_length=24,
        bins_per_octave= 12,
        filter_scale=0.5
    )):
    #     'n_hop': 24/25/27/28/36 -> 513/492/456/439/342 time frames
    #     'bins_per_octave' and 'filter_scale':(12, 1)/(24, 0.5)/(30, 0.4)/(40, 0.3)/(48, 0.25) -> 69/137/171/228/273 freq bins

    waves = waves / np.max(waves)
    waves = torch.from_numpy(waves).float()
    image = transform(waves)
    return image

def apply_qtransform_3(
    waves, 
    transform=CQT1992v2(
        sr=2048, 
        fmin=20, 
        fmax=1024, 
        hop_length=24,
        bins_per_octave= 12,
        filter_scale=0.5
    )):
    #     'n_hop': 24/25/27/28/36 -> 513/492/456/439/342 time frames
    #     'bins_per_octave' and 'filter_scale':(12, 1)/(24, 0.5)/(30, 0.4)/(40, 0.3)/(48, 0.25) -> 69/137/171/228/273 freq bins

    waves = torch.from_numpy(waves).float()
    image = transform(waves)
    return image

def apply_qtransform_4(
    waves, 
    transform=CQT1992v2(
        sr=2048, 
        fmin=20, 
        fmax=1024, 
        hop_length=24,
        bins_per_octave= 12,
        filter_scale=0.5
    )):
    #     'n_hop': 24/25/27/28/36 -> 513/492/456/439/342 time frames
    #     'bins_per_octave' and 'filter_scale':(12, 1)/(24, 0.5)/(30, 0.4)/(40, 0.3)/(48, 0.25) -> 69/137/171/228/273 freq bins

    waves = waves.T / np.max(waves, axis = 1)
    waves = waves.T
    waves = torch.from_numpy(waves).float()
    image = transform(waves)
    return image

class TrainDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df : pd.DataFrame, 
        config, 
        audio_transforms=None, 
        image_transform=None
        ):
        super().__init__()
        self.df = df
        self.image_transform = image_transform
        self.config = config

        #* 2, 4,5,6,7,
        self.audio_transforms = audio_transforms
        self.cqt_transform = CQT1992v2(**self.config['dataset']['cqt_config'])

        #* 3 
        # self.audio_transforms =  CQT2010v2(**self.config['dataset']['cqt_config'])

        self.sr = self.config['dataset']['cqt_config']['sr']

    def get_label(self, idx):
        return self.df.iloc[idx]["assessment_result"].astype("float32")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        label = item["target"].astype("float32")
        label_encoded = torch.tensor(label, dtype=torch.long)
        
        file_name = item['file_name']
        waves = np.load(file_name)

        bHP, aHP = signal.butter(8, (20, 500), btype='bandpass', fs= 2048)
        waves = filterSig(waves, a=aHP, b=bHP)

        if self.audio_transforms is not None:
            for i in range(len(waves)):
                waves[i] = self.audio_transforms(samples=waves[i], sample_rate=self.sr)

        data = apply_qtransform_4(waves, self.cqt_transform)

        if self.image_transform is not None:
            data = np.transpose(data.detach().numpy(), (1,2,0))
            data = self.image_transform(image=data)["image"]

        info = {
            'data': data,
            'target': label_encoded,
            'uuid': item['id']
        }
        return info