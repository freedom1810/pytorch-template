import torch
import os
import glob
import pandas as pd
import soundfile as sf
from PIL import Image
import numpy as np
import cv2
import random
import copy
import pickle
import ast

#* multi processing
from multiprocessing import Pool
from functools import partial

def prepare_data(df):
    res = None
    return res

class LivenessDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df : pd.DataFrame,
        config,
        transform=None
        ):
        super().__init__()
        self.df = df.reset_index()
        self.transform = transform
        self.config = config                                            

    def __len__(self):
        return self.df.shape[0]


    def get_image(self, name):
        image_dir = '{}/{}'.format(self.config['dataset']['image_dir'], name[:-4])
        image_paths = glob.glob('{}/*'.format(image_dir))
        image_path = random.choice(image_paths)

        image = cv2.imread(image_path)
        return image

    def __getitem__(self, idx):

        inputs = self.df.iloc[idx]
        image = self.get_image(inputs['fname'])

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = inputs['liveness_score']

        data = {
            "image": torch.tensor(image, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.float),
            # "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data