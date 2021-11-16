import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import timm

from torch.cuda import amp

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim=4,
        lstm_dim=256,
        dense_dim=256,
        logit_dim=256,
        num_classes=1,
    ):
        super().__init__()

        # self.backbone = nn.Sequential(
        #     nn.Linear(input_dim, dense_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(dense_dim // 2, dense_dim),
        #     nn.ReLU(),
        # )

        self.lstm = nn.LSTM(input_dim, lstm_dim, batch_first=True, bidirectional=True, num_layers = 5)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim*2, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, num_classes),
        )
    
    def freeze(self):
        # for param in self.backbone.parameters():
            # param.require_grad = False
        pass

    def unfreeze(self):
        # for param in self.backbone.parameters():
            # param.require_grad = True
        pass

    def forward(self, x, fp16 = False):
        # features = self.backbone(x)
        # features, _ = self.lstm(features)

        features, _ = self.lstm(x)

        pred = self.logits(features)
        return pred

# class CNN_1D_Model(nn.Module):
#     def __init__(
#         self,
#         input_dim=5,
#         num_classes=1,
#     ):
#         super().__init__()

#         self.backbone = nn.Sequential(
#             nn.Linear(input_dim, dense_dim // 2),
#             nn.ReLU(),
#             nn.Linear(dense_dim // 2, dense_dim),
#             nn.ReLU(),
#         )

#         self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)

#         self.logits = nn.Sequential(
#             nn.Linear(lstm_dim*2, logit_dim),
#             nn.ReLU(),
#             nn.Linear(logit_dim, num_classes),
#         )
    
#     def freeze(self):
#         for param in self.backbone.parameters():
#             param.require_grad = False

#     def unfreeze(self):
#         for param in self.backbone.parameters():
#             param.require_grad = True

#     def forward(self, x, fp16 = False):
#         features = self.backbone(x)
#         features, _ = self.lstm(features)
#         pred = self.logits(features)
#         return pred

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

