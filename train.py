import argparse
import collections
import os
import gc
import torch
import numpy as np
import pandas as pd
import torchvision
import albumentations as albu
import random
from torch import nn as nn


import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from data_loader import TrainDataset
from parse_config import ConfigParser
from trainer import Trainer
# from trainer import TrainerSwa as Trainer
from utils import prepare_device


# fix random seeds for reproducibility
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(1997)

def init_dataset(csv_path, config=None, fold_idx=1):
    print("*"*10, " fold {}".format(fold_idx), "*"*10)
    """StratifiedKFold"""

    df_full = pd.read_csv(csv_path)
    # print(df_full.head())
    df_train = df_full[df_full['fold']!= fold_idx]
    df_val = df_full[df_full['fold']== fold_idx]

    train_transform = None
    val_transform = None

    train_dataset = TrainDataset(
            # df=df_train[:2000],
            df=df_train,
            config=config,
            transform=train_transform,
        )

    validation_dataset = TrainDataset(
                                # df=df_val[:2000],
                                df=df_val,
                                config=config,
                                transform=val_transform,
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
    train_criterion = nn.L1Loss()
    val_criterion = module_loss.VentilatorLoss()

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
