import argparse
import os
import sys
from pathlib import Path

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
pd.options.display.max_columns = 100

from skimage.transform import rescale, resize, downscale_local_mean
from audiomentations import Compose, AddGaussianSNR, AddGaussianNoise, PitchShift, AddBackgroundNoise, AddShortNoises, Gain
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import timm

from scipy.special import logit, expit
from torch import nn

class Backbone(nn.Module):

    
    def __init__(self, name='resnet18', pretrained=True):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained)
        
        if 'regnet' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'vit' in name:
            self.out_features = self.net.head.in_features
        elif backbone == 'vit_deit_base_distilled_patch16_384':
            self.out_features = 768
        elif 'csp' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'res' in name: #works also for resnest
            self.out_features = self.net.fc.in_features
        elif 'efficientnet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'densenet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'senet' in name:
            self.out_features = self.net.fc.in_features
        elif 'inception' in name:
            self.out_features = self.net.last_linear.in_features

        else:
            self.out_features = self.net.classifier.in_features

    def forward(self, x):
        x = self.net.forward_features(x)

        return x
    
class VitModel(nn.Module):
    def __init__(self, backbone, out_dim, embedding_size=512, 
                 loss=False, pretrained=True):
        super(BirdModel, self).__init__()
        self.backbone_name = backbone
        self.loss = loss
        self.embedding_size = embedding_size
        self.out_dim = out_dim
        
        self.backbone = Backbone(backbone, pretrained=pretrained)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
            
        self.head = nn.Linear(self.embedding_size, out_dim)

    def freeze(self):
        # pass
        # print("freeze feature_extractor")
        for param in self.backbone.parameters():
            param.require_grad = False

        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def unfreeze(self):
        # pass
        for param in self.backbone.parameters():
            param.require_grad = True

        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(True)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(True)
                module.train()
        
    def forward(self, melspec, get_embeddings=False, get_attentions=False):

        # x = input_dict['spect']
        x = x.unsqueeze(1)
        x = x.expand(-1, 3, -1, -1)

        x = self.backbone(x)
        
        if 'vit' not in backbone:
            x = self.global_pool(x)
            x = x[:,:,0,0]
        if 'vit_deit_base_distilled_patch16_384' == backbone:
            x = x[0] + x[1]
        
        x = self.neck(x)

        logits = self.head(x)

        return logits
        
        # output_dict = {'logits':logits,
        #               }
        # if self.loss:
        #     target = input_dict['target']
        #     secondary_mask = input_dict['secondary_mask']
        #     loss = criterion(logits, target, secondary_mask)
            
        #     output_dict['loss'] = loss
            
        # return output_dict