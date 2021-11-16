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

class TrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df : pd.DataFrame,
        config,
        transform=None
        ):
        super().__init__()
        self.df = df.groupby('breath_id').agg(list).reset_index()
        self.transform = transform
        self.config = config
        self.prepare_data()

    def __len__(self):
        return self.df.shape[0]

    def prepare_data(self):
        self.pressures = np.array(self.df['pressure'].values.tolist())
        self.u_outs = np.array(self.df['u_out'].values.tolist())

        # rs = np.array(self.df['R'].values.tolist())
        # cs = np.array(self.df['C'].values.tolist())
        # u_ins = np.array(self.df['u_in'].values.tolist())
        # self.inputs = np.concatenate([
        #     rs[:, None],
        #     cs[:, None],
        #     u_ins[:, None],
        #     np.cumsum(u_ins, 1)[:, None],
        #     self.u_outs[:, None]
        # ], 1).transpose(0, 2, 1)

        feature = ['R', 'C', 'u_in', 'u_out',
                    'area', 'cross', 'cross2', 'u_in_cumsum', 'one', 'count',
                    'u_in_cummean', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame',
                    'breath_id_lag2same', 'u_in_lag', 'u_in_lag2', 'u_out_lag2', 'RC']
        list_feature = []
        for key in  feature:
            f = np.array(self.df[key].values.tolist())
            list_feature.append(f[:, None])
        self.inputs = np.concatenate(list_feature, 1).transpose(0, 2, 1)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        inputs = inputs/np.max(inputs[0], axis = 0)
        data = {
            "input": torch.tensor(inputs, dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data