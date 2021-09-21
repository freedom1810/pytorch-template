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
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, PolarityInversion, AddGaussianSNR, Normalize
from parse_config import ConfigParser
from trainer import Trainer
# from trainer import TrainerSwa as Trainer
from utils import prepare_device
import torchvision
from audiomentations.core.composition import BaseCompose
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as albu

# fix random seeds for reproducibility
# SEED = 123
SEED = 1997
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
import random
random.seed(SEED)
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

    df_full = pd.read_csv(csv_path)
    df_train = df_full[df_full['fold']!= fold_idx]
    df_val = df_full[df_full['fold']== fold_idx]

    #* 7
    # train_audio_transform = Compose([
    #     Normalize(p = 1)
    #     ])

    # val_audio_transform = Compose([
    #     Normalize(p = 1)
    #     ])

    train_audio_transform = None
    val_audio_transform = None

    image_width = config['dataset']['image_config']['width']
    image_height = config['dataset']['image_config']['height']
    train_image_transform = albu.Compose([
            albu.Resize(image_width, image_height, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

    val_image_transform = albu.Compose([
            albu.Resize(image_width, image_height, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

    train_dataset = TrainDataset(
            # df=df_train[:2000],
            df=df_train,
            config=config,
            audio_transforms=train_audio_transform,
            image_transform=train_image_transform,
        )

    validation_dataset = TrainDataset(
                                # df=df_val[:2000],
                                df=df_val,
                                config=config,
                                audio_transforms=val_audio_transform,
                                image_transform=val_image_transform,
                            )
    return train_dataset, validation_dataset



def main(config, fold_idx):
    logger = config.get_logger('train')
    train_dataset, val_dataset = init_dataset(config["dataset"]["csv_path"],
                                                config,  
                                                fold_idx)
    # setup data_loader instances
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle = True,
        batch_size=config["dataset"]['training_batch_size'],
        num_workers=config["dataset"]['num_workers'],
        # sampler=ImbalancedDatasetSampler(train_dataset),
        drop_last = True
    )
    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["dataset"]['validate_batch_size'], 
        num_workers=config["dataset"]['num_workers']
    )


    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    pretrained_path = config['arch'].get('pretrained_path', '')
    if pretrained_path:
        model.load_from_pretrain(pretrained_path)
    logger.info(model)
    logger.info(config)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # train_criterion = getattr(module_loss, config['train_loss'])
    # val_criterion = getattr(module_loss, config['val_loss'])

    train_criterion = torch.nn.BCEWithLogitsLoss()
    val_criterion = torch.nn.BCEWithLogitsLoss()
    
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, train_criterion, val_criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_loader,
                      valid_data_loader=eval_loader,
                      lr_scheduler=lr_scheduler,
                      fold_idx=fold_idx
                      )

    trainer.train()
    model = model.to("cpu")
    del model, optimizer, trainer
    del train_loader, eval_loader, train_dataset, val_dataset

    gc.collect()
    torch.cuda.empty_cache()


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
