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
from data_loader import CovidDataset, TestDataset
from data_loader import AudioCompose, WhiteNoise, TimeShift, ChangePitch, ChangeSpeed
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, PolarityInversion
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import torchvision
from tqdm import tqdm

def post_process_function(output_tensor):
    outputs = []
    for output in output_tensor:
        outputs.append(float(output))
    max_prob = max(outputs)
    min_prob = min(outputs)
    sum_prob = sum(outputs)
    fussion_prob = (1/len(outputs))*((sum_prob - len(outputs)*min_prob)/ (max_prob - min_prob))
    return fussion_prob

class EnsembleModel(torch.nn.Module):
    def __init__(self, models, post_process_func=None, weights=[0.2, 0.2, 0.2, 0.2, 0.2]):
        super().__init__()
        self.models = models
        self.post_process_func = post_process_func
        self.weights = weights
    def forward(self, input, post_process_func=None):
        outputs = []
        for i in range(len(self.models)):
            output = self.models[i](input)
            outputs.append(self.weights[i]*torch.sigmoid(output))
        if self.post_process_func is not None:
            return self.post_process_func(outputs)
        return torch.sum(torch.stack(outputs))

def init_unlabeled_dataset(csv_path, audio_folder=""):
    return  TestDataset(csv_path, audio_folder)

def main(config):
    unlabeled_dataset = init_unlabeled_dataset(config["unlabeled_dataset"]["csv_path"],
                                                config["unlabeled_dataset"]["audio_folder"]
                                                )
    device, device_ids = prepare_device(config['n_gpu'])
    CHECKPOINT = ["saved/models/Covid19-PlainCNN/0730_215235/model_best_fold1",
        "saved/models/Covid19-PlainCNN/0730_215235/model_best_fold2",
        "saved/models/Covid19-PlainCNN/0730_215235/model_best_fold3",
        "saved/models/Covid19-PlainCNN/0730_215235/model_best_fold4",
        "saved/models/Covid19-PlainCNN/0730_215235/model_best_fold5"
        ]
    models = []
    for i in range(len(CHECKPOINT)):
        state = torch.load(CHECKPOINT[i])["state_dict"]
        model = config.init_obj('arch', module_arch)
        model.load_state_dict(state)
        model = model.to(device)
        models.append(model.eval())
    ensemble_model = EnsembleModel(models, None)
    ensemble_model.cuda().eval()
    results = pd.DataFrame(columns=['uuid', 'assessment_result'])
    progress_bar = tqdm(total=len(unlabeled_dataset))
    for i in range(len(unlabeled_dataset)):
        feature, id = unlabeled_dataset[i]
        with torch.no_grad():
            feature = feature[None, ...].cuda()
            score = ensemble_model(feature)
            results.loc[i] = [id, float(score.detach())]
        progress_bar.update(1)
    results.to_csv("results.csv", index=False)
    del models, ensemble_model
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
    main(config)
