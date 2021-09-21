import argparse
import collections
import os
import gc
import torch
import numpy as np
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from data_loader import TrainDataset
from data_loader import AudioCompose, WhiteNoise, TimeShift, ChangePitch, ChangeSpeed
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, PolarityInversion, AddGaussianSNR, Normalize
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import torchvision
from audiomentations.core.composition import BaseCompose
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as albu
from tqdm import tqdm
import sklearn
from torch.optim.swa_utils import AveragedModel, SWALR
import random

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

class OneOf(BaseCompose):
    # TODO: Name can change to WaveformCompose
    def __init__(self, transforms, p=1.0, shuffle=False):
        super(OneOf, self).__init__(transforms, p, shuffle)
    def __call__(self, samples, sample_rate):
        transforms = self.transforms.copy()
        if random.random() < self.p:
            random.shuffle(transforms)
            for transform in transforms:
                samples = transform(samples, sample_rate)
                break

        return samples

    def randomize_parameters(self, samples, sample_rate):
        for transform in self.transforms:
            transform.randomize_parameters(samples, sample_rate)

def init_dataset(csv_path, config=None, fold_idx=1):
    print("*"*10, " fold {}".format(fold_idx), "*"*10)
    """StratifiedKFold"""

    # df_full = pd.read_csv(csv_path)
    # df_val = df_full[df_full['fold']== fold_idx]

    df_val = pd.read_csv('/home/hana/sonnh/G2Net/data/g2net-gravitational-wave-detection/sample_submission_with_path.csv')


    image_width = config['dataset']['image_config']['width']
    image_height = config['dataset']['image_config']['height']


    val_audio_transform = None
    val_image_transform = albu.Compose([
            albu.Resize(image_width, image_height, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

    validation_dataset = TrainDataset(
                                # df=df_val[:2000],
                                df=df_val,
                                config=config,
                                audio_transforms=val_audio_transform,
                                image_transform=val_image_transform,
                            )
    
    return validation_dataset

def main(config, fold_idx):
    val_dataset = init_dataset(config["dataset"]["csv_path"],
                                              config,  
                                             fold_idx)

    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["dataset"]['validate_batch_size'], 
        # batch_size=1, 
        num_workers=config["dataset"]['num_workers']
    )

    model = config.init_obj('arch', module_arch)
    checkpoint_path = 'saved/models/g2net_8/0918_213540/model_best_fold{}'.format(fold_idx)
    check_point = torch.load(checkpoint_path)
 
    #* normal
    model.load_state_dict(check_point['state_dict'])

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    targets = []
    outputs = []
    model.eval()
    with torch.no_grad():
        # for batch_idx, (data, target) in enumerate(tqdm(eval_loader)):
        for batch_idx, info in enumerate(tqdm(eval_loader)):
            data = info['data']
            target = info['target']
            uuid  = info['uuid']
            data, target = data.to(device), target.to(device)
            target = target.to(dtype=torch.long)
            output = model(data)
            
            targets.append(target.detach().cpu())
            outputs.append(output.detach().cpu())
    targets = torch.cat(targets)
    outputs = torch.cat(outputs)

    print(targets.size())
    print(outputs.size())
    targets = targets.detach().cpu().numpy()
    outputs = torch.sigmoid(outputs)[:, 0].tolist()


    # print('auc: {}'.format(sklearn.metrics.roc_auc_score(targets, outputs)))

    df_test = pd.read_csv('/home/hana/sonnh/G2Net/data/g2net-gravitational-wave-detection/sample_submission.csv')
    df_test['target'] = outputs
    df_test.to_csv('/home/hana/sonnh/G2Net/result/g2net_7.csv'.format(fold_idx), index = False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    for fold_idx in range(1, 6):
        main(config, fold_idx)
