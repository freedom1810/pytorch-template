import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import timm

from torch.cuda import amp
import torchaudio

from .coordconv import CoordConv1d, CoordConv2d, CoordConv3d
from .vit import *
from .cnn14 import Cnn14
from .wave2vec import *

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


class TimmBackbone(BaseModel):
    def __init__(self, model_name, inchannels=3, num_classes=1, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(model_name, in_chans=inchannels, pretrained=pretrained)
        n_features = self.backbone.num_features
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(n_features, 128)
        self.fc1 = nn.Linear(n_features, num_classes)
        # self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def freeze(self):
        for param in self.backbone.parameters():
            param.require_grad = False

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.require_grad = True

    def forward(self, x, fp16=False):
        with amp.autocast(enabled=fp16):
            x = x.float()
            feats = self.backbone.forward_features(x)
            x = self.pool(feats).view(x.size(0), -1)
            x = self.drop(x)
            x = self.fc1(x)

        return x

