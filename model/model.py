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

class StackNet(BaseModel):
    def __init__(self, num_classes=1, num_features = 10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(num_features, 32, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes, bias=True),
        )
    def forward(self, x, fp16 = False):
        return self.head(x)

class StackNet2(BaseModel):
    def __init__(self, num_classes=1, num_features = 10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(num_features, 32, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Dropout(0.2),
            nn.Linear(32, num_classes, bias=True),
        )
    def forward(self, x, fp16 = False):
        return self.head(x)

class StackNet3(BaseModel):
    def __init__(self, num_classes=1, num_features = 10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(num_features, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),

            nn.Linear(64, 32, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, num_classes, bias=True),
        )
    def forward(self, x, fp16 = False):
        return self.head(x)

class StackNet4(BaseModel):
    def __init__(self, num_classes=1, num_features = 10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(num_features, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),

            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),

            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),

            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),

            nn.Linear(64, 32, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, num_classes, bias=True),
        )
    def forward(self, x, fp16 = False):
        return self.head(x)

class PlainCNN(BaseModel):
    def __init__(self, inchannels=3, num_classes=1, use_coord=False, pretrained=True, mfcc_config = None, melspec_config = None):
        super().__init__()

        self.mfcc_config = mfcc_config
        if self.mfcc_config is not None:
            self.torch_audio2mfcc = torchaudio.transforms.MFCC(
                sample_rate= mfcc_config['target_sr'], 
                n_mfcc= mfcc_config['n_mfcc'],
                melkwargs={
                    "n_fft": mfcc_config['n_fft'], 
                    "hop_length": mfcc_config['hop_length'],
                    'norm':"slaney",
                    'mel_scale': 'slaney'
                    }
                ).float()

        # self.melspec_config = melspec_config
        # if self.melspec_config is not None:
        #     self.logmelspec_extractor = nn.Sequential(
        #     torchaudio.transforms.MelSpectrogram(
        #         melspec_config['target_sr'],
        #         n_mels=melspec_config['n_mels'],
        #         f_min=20,
        #         n_fft=melspec_config['n_fft'],
        #         hop_length=melspec_config['hop_length'],
        #         normalized=True,
        #     ),
        #     torchaudio.transforms.AmplitudeToDB(top_db=80.0),
        #     NormalizeMelSpec(),
        # )

        if use_coord:
            self.conv1 = CoordConv2d(inchannels,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    with_r=True)
        else:
            self.conv1 = nn.Conv2d(in_channels=inchannels, 
                              out_channels=64,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1))
        self.act0 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=64, 
                              out_channels=256,
                              kernel_size=(2, 2), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, num_classes)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.3)

    def forward(self, x, fp16=False):
        if self.mfcc_config is not None:
            x =  x.unsqueeze(1)
            x = self.torch_audio2mfcc(x.float())

        # if self.melspec_config is not None:
        #     x =  x.unsqueeze(1)
        #     x = self.logmelspec_extractor(x)
        
        
        with amp.autocast(enabled=fp16):
            x = x.float()
            x = self.conv1(x)
            x = self.act0(x)

            x = self.max_pool1(x)

            x = self.conv2(x)
            x = self.act1(x)

            x = self.bn1(x)
            
            x = self.pool(x).view(x.size(0), -1)
            #x = self.flatten(x)

            x = F.relu(self.fc1(x))

            x = self.drop1(x)

            x = F.relu(self.fc2(x))

            x = self.drop2(x)

            x = self.fc3(x)
        return x

class PlainCNNSmall(BaseModel):
    def __init__(self, inchannels=3, num_classes=1, use_coord=False, pretrained=True, mfcc_config = None):
        super().__init__()

        self.mfcc_config = mfcc_config
        if self.mfcc_config is not None:
            self.torch_audio2mfcc = torchaudio.transforms.MFCC(
                sample_rate= mfcc_config['target_sr'], 
                n_mfcc= mfcc_config['n_mfcc'],
                melkwargs={
                    "n_fft": mfcc_config['n_fft'], 
                    "hop_length": mfcc_config['hop_length'],
                    'norm':"slaney",
                    'mel_scale': 'slaney'
                    }
                ).float()
        if use_coord:
            self.conv1 = CoordConv2d(inchannels,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    with_r=True)
        else:
            self.conv1 = nn.Conv2d(in_channels=inchannels, 
                              out_channels=64,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1))
        self.act0 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=64, 
                              out_channels=128,
                              kernel_size=(2, 2), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.act1 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, num_classes)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.3)

    def forward(self, x, fp16=False):
        if self.mfcc_config is not None:
            x =  x.unsqueeze(1).float()
            x = self.torch_audio2mfcc(x.float())
            
        with amp.autocast(enabled=fp16):
            x = x.float()
            x = self.conv1(x)
            x = self.act0(x)

            x = self.max_pool1(x)

            x = self.conv2(x)
            x = self.act1(x)

            x = self.bn1(x)
            
            x = self.pool(x).view(x.size(0), -1)
            #x = self.flatten(x)

            x = F.relu(self.fc1(x))

            x = self.drop1(x)

            x = F.relu(self.fc2(x))

            x = self.drop2(x)

            x = self.fc3(x)
        return x

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)

class Transfer_Cnn14(BaseModel):
    def __init__(self, sample_rate, window_size=512, hop_size=512, mel_bins=128, fmin=0, 
        fmax=48000, classes_num=1, freeze_base=True):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527

        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        # self.head = torch.nn.Sequential(
        #     nn.Linear(2048, 512, bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, 256, bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, classes_num, bias=True)
        # )

        self.head = torch.nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, classes_num, bias=True),
        )
        if freeze_base:
            print("freeze base layer")
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        # self.init_weights()

    def init_weights(self):
        # init_layer(self.fc_transfer)
        pass

    def unfreeze(self):
        pass

    def freeze(self):
        pass

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None, fp16 = False):
        """Input: (batch_size, data_length)
        """
        embedding = self.base(input, mixup_lambda)

        logit =  self.head(embedding) 
        return logit

class Transfer_Wave2Vec(BaseModel):
    def __init__(
        self, 
        model_name_or_path='facebook/wav2vec2-base-960h', 
        pooling_mode='mean', 
        classes_num=1,
        freeze_base=True
        ):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Wave2Vec, self).__init__()

        model_name_or_path = model_name_or_path
        pooling_mode = pooling_mode
        num_labels = classes_num
        # config
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            finetuning_task="wav2vec2_clf",
        )
        setattr(config, 'pooling_mode', pooling_mode)

        self.base = Wav2Vec2ForSpeechClassification.from_pretrained(
            model_name_or_path,
            config=config,
            )
        self.base.freeze_feature_extractor()

        self.head = torch.nn.Sequential(
            nn.Linear(768, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, classes_num, bias=True)
        )

    def init_weights(self):
        # init_layer(self.fc_transfer)
        pass

    def unfreeze(self):
        pass

    def freeze(self):
        pass

    def forward(self, input, mixup_lambda=None, fp16 = False):
        """Input: (batch_size, data_length)
        """
        with amp.autocast(enabled=fp16):
            embedding = self.base(input)
            logit =  self.head(embedding) 
        return logit

class LargerPlainCNN(BaseModel):
    def __init__(self, inchannels=1, num_classes=1, use_coord=False, pretrained=True):
        super().__init__()
        if use_coord:
            self.conv1 = CoordConv2d(inchannels,
                                    out_channels=128,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    with_r=True)
        else:
            self.conv1 = nn.Conv2d(in_channels=inchannels, 
                              out_channels=128,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1))
        self.bn0 = nn.BatchNorm2d(128)
        self.act0 = nn.LeakyReLU(0.1)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=128, 
                              out_channels=256,
                              kernel_size=(2, 2), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.LeakyReLU(0.1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 1)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.2)
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.act0(x)

        x = self.max_pool1(x)

        x = self.conv2(x)

        x = self.bn1(x)
        x = self.act1(x)

        x = self.pool(x).view(x.size(0), -1)
        #x = self.flatten(x)

        x = F.leaky_relu(self.fc1(x), 0.1)

        x = self.drop1(x)

        x = F.leaky_relu(self.fc2(x), 0.05)

        x = self.drop2(x)

        x = self.fc3(x)
        return x

class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            # print(V_fix.shape, norm_min_fix.shape, norm_max_fix.shape)
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V

class TimmBackbone(BaseModel):
    def __init__(self, model_name, inchannels=3, num_classes=1, pretrained=True, melspec_config = None):
        super().__init__()


        # def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        # fmax, classes_num):

        # super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        sample_rate = melspec_config['target_sr']
        window_size = melspec_config['n_fft']
        hop_size =  melspec_config['n_fft']
        mel_bins = melspec_config['n_mels']
        fmin = melspec_config['fmin']
        fmax = melspec_config['fmax']

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        #7s
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=3, 
        #     freq_drop_width=8, freq_stripes_num=3)

        self.bn0 = nn.BatchNorm2d(mel_bins)


        # self.melspec_config = melspec_config
        # if self.melspec_config is not None:
        #     self.logmelspec_extractor = nn.Sequential(
        #     torchaudio.transforms.MelSpectrogram(
        #         melspec_config['target_sr'],
        #         n_mels=melspec_config['n_mels'],
        #         f_min=20,
        #         n_fft=melspec_config['n_fft'],
        #         hop_length=melspec_config['hop_length'],
        #         normalized=True,
        #     ),
        #     torchaudio.transforms.AmplitudeToDB(top_db=80.0),
        #     NormalizeMelSpec(),
        # )
        self.backbone = timm.create_model(model_name, in_chans=inchannels, pretrained=pretrained)
        n_features = self.backbone.num_features
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(n_features, 128)
        self.fc1 = nn.Linear(n_features, num_classes)
        # self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def freeze(self):
        # pass
        # print("freeze feature_extractor")
        for param in self.backbone.parameters():
            param.require_grad = False

    def unfreeze(self):
        # pass
        for param in self.backbone.parameters():
            param.require_grad = True

    def forward(self, x, fp16=False):
        # if self.melspec_config is not None:
        #     x =  x.unsqueeze(1)
        #     x = self.logmelspec_extractor(x)
        

        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        # print(x.size())
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)


        # print(x.size())
        with amp.autocast(enabled=fp16):
            x = x.float()
            feats = self.backbone.forward_features(x)
            x = self.pool(feats).view(x.size(0), -1)
            x = self.drop(x)
            x = self.fc1(x)
            # x = F.relu(self.fc1(x))
            # x = self.drop(x)
            # x = self.fc2(x)
        return x

class TropicModel(BaseModel):
    def __init__(self, model_name, drop_rate=0.2, inchannels=3, num_classes=1, coord=1, pretrained=True):
        super().__init__()
        self.trunk = timm.create_model(model_name,pretrained=pretrained,num_classes=num_classes,in_chans=inchannels+coord)
        self.do = nn.Dropout2d(drop_rate)
        self.coord = coord
    def forward(self, x):
        bs,_,freq_bins,time_bins = x.size()
        coord = torch.linspace(-1,1,freq_bins,dtype=x.dtype,device=x.device).view(1,1,-1,1).expand(bs,1,-1,time_bins)
        if self.coord: x = torch.cat((x,coord),dim=1)
        x = self.do(x)
        return self.trunk(x)

class VitModel(BaseModel):
    def __init__(self, backbone, out_dim, embedding_size=512, 
                 loss=False, pretrained=True):
        super(VitModel, self).__init__()
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
        
    def forward(self, melspec, get_embeddings=False, get_attentions=False, fp16 = False):
        with amp.autocast(enabled=fp16):
            # x = input_dict['spect']
            x = melspec
            x = x.unsqueeze(1)
            x = x.expand(-1, 3, -1, -1)

            x = self.backbone(x)
            
            # if 'vit' not in backbone:
            #     x = self.global_pool(x)
            #     x = x[:,:,0,0]
            # if 'vit_deit_base_distilled_patch16_384' == backbone:
            x = x[0] + x[1]
            
            x = self.neck(x)

            logits = self.head(x)

        return logits