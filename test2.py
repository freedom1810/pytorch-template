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
from data_loader import CovidDataset
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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

def roc_auc(output, target):
    output = torch.sigmoid(output.float())
    output = output.numpy()
    target = target.numpy()
    return sklearn.metrics.roc_auc_score(target, output)

def cut_x3(df_train):

    df = pd.DataFrame({})
    for i in range(len(df_train)):
        item = df_train.iloc[i]
        uuid = item['uuid']
        if '_2' in uuid or '_3' in uuid:
        # if uuid[-2] == '_':
            continue
        df = df.append(item, ignore_index=True)
    return df


def cut_x5(df_train):

    df = pd.DataFrame({})
    for i in range(len(df_train)):
        item = df_train.iloc[i]
        uuid = item['uuid']
        if '_2' in uuid or '_3' in uuid or '_1' in uuid or '_0' in uuid:
            continue
        df = df.append(item, ignore_index=True)
    return df

def init_dataset(csv_path, config=None, fold_idx=1, audio_folder=""):
    print("*"*10, " fold {}".format(fold_idx), "*"*10)
    """StratifiedKFold"""
    # df_final = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_train/remove_duplicate/group_included_rmsingle2_kfold_oversampling_new_format.csv')
    # df_final = cut_x3(df_final)
    # df_coswara = pd.read_csv('/home/hana/sonnh/data/AICovidVN/coswara/info_kfold_new_format_oversampling.csv')
    # df_coswara = cut_x3(df_coswara)
    # df_vifury = pd.read_csv('/home/hana/sonnh/data/AICovidVN/virufy-cdf-coughvid/fillter_rm_slient_kfold_new_format_oversampling.csv')
    # df_vifury = df_vifury[df_vifury['assessment_result']==1]
    # df_vifury = cut_x3(df_vifury)

    # df_external = pd.concat([df_final, df_coswara, df_vifury])
    # print(df_external['assessment_result'].value_counts())
    # eval_df = df_external[df_external["fold"] == fold_idx]

    #* train 29
    # df_final = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_train/remove_duplicate/group_included_rmsingle2_kfold_oversampling_new_format.csv')
    # df_final = cut_x3(df_final)

    # df_external = df_final
    # print(df_external['assessment_result'].value_counts())
    # eval_df = df_external[df_external["fold"] == fold_idx]

    #* train 20
    # df_coswara = pd.read_csv('/home/hana/sonnh/data/AICovidVN/coswara/info_kfold_new_format.csv')
    # df_vifury = pd.read_csv('/home/hana/sonnh/data/AICovidVN/virufy-cdf-coughvid/fillter_rm_slient_kfold_new_format.csv')
    # df_vifury = df_vifury[df_vifury['assessment_result']==1]
    # df_vifury = cut_x3(df_vifury)
    # df_external = pd.concat([df_coswara, df_vifury])
    # eval_df = df_external[df_external["fold"] == fold_idx]
    # print(eval_df['assessment_result'].value_counts())

    #* train 31, 32,33
    # fold_idx = 1
    # df_final = pd.read_csv(csv_path)
    # df_final = cut_x5(df_final)
    # df_external = df_final 
    # print(df_external['assessment_result'].value_counts())
    # train_df = df_external[df_external["fold"] != fold_idx]
    # eval_df = df_external[df_external["fold"] == fold_idx]

    # * public
    # df_test = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_test/public_test_sample_submission.csv')
    # audio_dir = '/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_test/public_test_audio_files/'
    # df_test['audio_paths'] = ['{}/{}.wav'.format(audio_dir, uuid) for uuid in list(df_test['uuid'])]
    # eval_df = df_test

    df_test = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_final_private_test/private_test_sample_submission.csv')
    audio_dir = '/home/hana/sonnh/data/AICovidVN/aicv115m_final_private_test/private_test_audio_files/'
    df_test['audio_paths'] = ['{}/{}.wav'.format(audio_dir, uuid) for uuid in list(df_test['uuid'])]
    eval_df = df_test

    # val_audio_transform = Compose([
    #     Normalize(p = 1)
    # ])
    # val_audio_transform = None

    #* tta
    # train_audio_transform = OneOf([
    #                 # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    #                 AddGaussianSNR(p=0.5), # new
    #                 TimeStretch(min_rate=0.9, max_rate=1.2, p=0.5),
    #                 PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    #                 Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    #                 PolarityInversion(p=0.5),
    #                 Gain(
    #                     min_gain_in_db=-15.0,
    #                     max_gain_in_db=5.0,
    #                     p=0.5,),
    #             ], p=1.0)

    #* 76
    # df_warmup = pd.read_csv('/media/sonnh/AICOVIDVN/aicv115m_public_train/metadata_train_challenge_kfold2_new_format.csv')

    # eval_df = df_warmup[df_warmup["fold"] == fold_idx]
    # train_df = df_warmup[df_warmup["fold"] != fold_idx]

    val_audio_transform = Compose([
        Normalize(p = 1)
    ])

    validation_dataset = CovidDataset(
                                df=eval_df,
                                config=config,
                                audio_folder=audio_folder,
                                test = True,
                                audio_transforms=val_audio_transform,
                                image_transform=None,
                            )

    #* normal
    # validation_dataset = CovidDataset(
    #                             df=eval_df,
    #                             config=config,
    #                             audio_folder=audio_folder,
    #                             test = True,
    #                             audio_transforms=val_audio_transform,
    #                             image_transform=None,
    #                         )

    # train_dataset = CovidDataset(
    #                             df=train_df,
    #                             config=config,
    #                             audio_folder=audio_folder,
    #                             test = False,
    #                             audio_transforms=val_audio_transform,
    #                             image_transform=None,
    #                         )

    return validation_dataset

def main(config, fold_idx):
    val_dataset = init_dataset(config["dataset"]["csv_path"],
                                              config,  
                                             fold_idx,
                                             config["dataset"]["audio_folder"])

    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=config["dataset"]['validate_batch_size'], 
        batch_size=1, 
        num_workers=config["dataset"]['num_workers']
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=config["dataset"]['validate_batch_size'], 
    #     # batch_size=1, 
    #     num_workers=config["dataset"]['num_workers']
    # )

    model = config.init_obj('arch', module_arch)
    checkpoint_path = 'saved/models/76/0828_114139/model_best_fold{}'.format(fold_idx)
    check_point = torch.load(checkpoint_path)

    #* swa
    # model = AveragedModel(model)
    # model.load_state_dict(check_point['state_dict'])
    
    #* normal
    model.load_state_dict(check_point['state_dict'])

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # * swa
    # torch.optim.swa_utils.update_bn(train_loader, model, device=torch.device('cuda')) 


    # targets = []
    

    # tta_time = 1
    # outputs_final = None
    # for i in range(tta_time):
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

    #     if outputs_final is None:
    #         outputs_final = outputs
    #     else:
    #         outputs_final += outputs

    # outputs = outputs_final/3
    print(targets.size())
    print(outputs.size())
    targets = targets.detach().cpu().numpy()
    outputs = torch.sigmoid(outputs)[:, 0].tolist()


    # print('auc: {}'.format(sklearn.metrics.roc_auc_score(targets, outputs)))

    # df_test = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_test/public_test_sample_submission.csv')
    df_test = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_final_private_test/private_test_sample_submission.csv')
    df_test['assessment_result'] = outputs
    df_test.to_csv('/home/hana/sonnh/covidaivn/covid19_res/private_76_9_fold{}_best_.csv'.format(fold_idx), index = False)


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
