import torch
import os
import glob
import pandas as pd
import soundfile as sf
from PIL import Image
import numpy as np
import cv2
import librosa
import random
from .audio_preprocessing import mfcc_feature, extract_mfcc_feature
from .audio_preprocessing import trim_and_pad, padding_repeat, random_crop, padding_0_center_crop

from transformers import Wav2Vec2Processor, Wav2Vec2Model

import copy
from multiprocessing import Pool
from functools import partial
import pickle
import ast

def load_audio_(i, df=None, target_sr=22050):
    item = df.iloc[i]
    uuid = item['uuid']
    audio_path = item['audio_paths']
    audio, sr = sf.read(audio_path, dtype="float32")
    # audio, _ = librosa.effects.trim(audio, top_db = 50)
    audio = librosa.resample(audio, sr, target_sr)
    return uuid, audio

def load_audio(df, target_sr):
    audios = {}

    pool = Pool()
    tmp = pool.map(
        partial(
            load_audio_, df = df,
            target_sr = target_sr
        ),
        range(len(df))
    )
    for key, audio in tmp:
        audios[key] = audio
    return audios

class CovidDataset(torch.utils.data.Dataset):
    def __init__(self, df, audio_folder, config, test = False, audio_transforms=None, image_transform=None):
        super().__init__()
        self.df = df
        self.audio_folder = audio_folder
        self.image_transform = image_transform
        self.audio_transforms = audio_transforms
        self.test = test

        self.audio_config = config['dataset'].get('audio_config', None)
        if self.audio_config is None:
            self.audio_config = {}
        
        self.target_sr = self.audio_config.get('target_sr', 48000)
        max_duration = self.audio_config.get('max_duration', 15)
        self.max_samples = int(max_duration * self.target_sr)

        cache = config['dataset'].get('cache', False)
        if cache:
            self.audios = load_audio(self.df, self.target_sr)
        else:
            self.audios = None

    def get_label(self, idx):
        return self.df.iloc[idx]["assessment_result"].astype("float32")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        label = item["assessment_result"].astype("float32")
        # one hot
        # label_encoded = np.zeros((label.size, 2))
        # label_encoded[np.arange(label.size),int(label)] = 1

        label_encoded = torch.tensor(label, dtype=torch.long)
        
        if self.audios is not None:
            uuid = item['uuid']
            audio = copy.deepcopy(self.audios[uuid])
            sr = self.target_sr
        else:
            try:
                audio_path = item['audio_paths']
                audio, sr = sf.read(audio_path, dtype="float32")
                # audio, sr = librosa.load(audio_path, sr = self.target_sr)
            except:
                audio_path = os.path.join(self.audio_folder, "%s.wav"%item['uuid'])
                audio, sr = sf.read(audio_path, dtype="float32")
            audio = librosa.resample(audio, sr, self.target_sr)
        
        if self.test:
            # audio = padding_repeat(audio, self.max_samples)
            audio = padding_0_center_crop(audio, self.max_samples)

        else:
            audio = random_crop(audio, self.max_samples)


        if self.audio_transforms is not None:
            audio_path = item['audio_paths']
            uuid = audio_path.split('/')[-1].split('.')[0]
            if uuid[-2] != '_':
                audio = self.audio_transforms(samples=audio, sample_rate=sr)
            else:
                # print('oversampling no need to aug')
                pass

        #mfcc
        # image = extract_mfcc_feature(audio, self.target_sr, self.audio_config)
        # image = extract_mfcc_feature(audio, sr, self.mfcc_config)
        
        # melspec
        # image = audio2melspec(audio, self.target_sr, self.melspec_config)

        # if self.image_transform is not None:
        #     image = self.image_transform(image = image)['image']

        # return image, label_encoded

        #* train 44 - 52
        info = {
            'data': torch.from_numpy(audio).float(), #melspec
            # 'data': torch.from_numpy(image).float(), #mfcc
            'target': label_encoded,
            'uuid': item['uuid']
        }
        return info

def get_cough_interval():
    # df_cough_interval = pd.read_csv('/home/hana/sonnh/data/AICovidVN/public_train_with_cough.csv')
    # df_cough_interval = pd.read_csv('/home/hana/sonnh/data/AICovidVN/aicv115m_public_train_full/aicv115m_final_public_train/kfold_multi_feature.csv')
    # df_cough_interval = pd.read_csv('/home/hana/sonnh/data/AICovidVN/public_train_with_cough_and_prob.csv')
    # df_cough_interval = pd.read_csv('/home/hana/sonnh/data/AICovidVN/public_train_with_cough_and_prob.csv')
    # df_cough_interval = pd.read_csv('/home/hana/sonnh/data/AICovidVN/tuannm_export.csv')
    # df_cough_interval = pd.read_csv('/home/hana/sonnh/data/AICovidVN/tuannm_export_test.csv')
    # df_cough_interval = pd.read_csv('/home/hana/sonnh/data/AICovidVN/tuannm_export_test_from_extra.csv')
    # df_cough_interval = pd.read_csv('/home/hana/sonnh/data/AICovidVN/public_test_with_cough_and_prob.csv')
    df_cough_interval = pd.read_csv('/home/hana/sonnh/data/AICovidVN/sed_warmup_metadata_train_challenge_kfold2_new_format.csv')

    data = {}
    for i in range(len(df_cough_interval)):
        times = []
        item = df_cough_interval.iloc[i]
        uuid = item['uuid']
        # cough_times = ast.literal_eval(item['cough_intervals'])
        cough_times = ast.literal_eval(item['pred_interval'])
        # for cough_time in cough_times:
        #     times.append(cough_time['end'] - cough_time['start'])

        data[uuid] = cough_times
    return data


def fillter_cough_interval(cough_times):
    cough_times_res = []
    for cough_time in cough_times:
        prob = cough_time['prob']
        if prob > 0.6:
            cough_times_res.append(cough_time)

    return cough_times_res

def split_with_cough(audio, cough_times, max_samples, sr):
    audios = []

    #Option1 sử dụng model sed
    if len(cough_times) == 0:
        audios.append(audio)
    else:
        cough_times.sort(key=lambda x: x['start'])
        start = 0 
        max_duration = max_samples/sr
        for i, cough_time in enumerate(cough_times):
            end = cough_time['end']
            if end - start > max_duration:
                start_new  = min(
                    cough_time['start'], 
                    start + max_duration
                    )
                audio_temp = audio[int(start*sr):int(start_new*sr)]
                audios.append(audio_temp)
                start = start_new
            if i == len(cough_times)-1:
                start_new  = start + max_duration
                audio_temp = audio[int(start*sr):int(start_new*sr)]
                audios.append(audio_temp)
        return audios

def split_with_cough2(audio, cough_times, max_samples, sr):
    # print('max_samples {} sr {}'.format(max_samples, sr))
    audios = []

    if len(audio) <=  max_samples:
        audios.append(audio)
    else:
        audio_duration = len(audio)/sr
        #Option2 sử dụng sliding window
        max_duration = max_samples/sr
        stride = max_duration//3
        start = 0
        while start + max_duration < audio_duration:
            end = start + max_duration
            audio_temp = audio[int(start*sr):int(end*sr)]
            start = start + stride
            audios.append(audio_temp)

        audio_temp = audio[int(start*sr):]
        audios.append(audio_temp)

        x = [len(i) for i in audios]
        # print(x)
    return audios

def split_with_cough3(audio, cough_times, max_samples, sr):
    #* cắt đoạn audio có nhiểu time ho nhất

    if len(audio) <=  max_samples or len(cough_times) == 0:
        audio_res = [audio]
    else:
        audio_duration = len(audio)/sr
        #Option2 sử dụng sliding window
        max_duration = max_samples/sr

        max_cough_time = 0
        start_res = 0
        end_res = audio_duration

        for cough_time1 in cough_times:
            for cough_time2 in cough_times:
                start = cough_time1['start']
                end = cough_time2['end']
                max_cough_time_temp = 0
                if end - start <= max_duration:
                    for cough_time in cough_times:
                        start_temp = cough_time['start']
                        end_temp = cough_time['end']
                        if start <= start_temp <= end  and start <= end_temp <= end:
                            max_cough_time_temp += end_temp - start_temp
                    
                    if max_cough_time_temp > max_cough_time:
                        max_cough_time = max_cough_time_temp
                        start_res = start
                        end_res = end

        end_res = start_res + max_duration
        if end_res > audio_duration:
            end = audio_duration
        start = audio_duration - max_duration
        audio_res = [audio[int(start_res*sr):int(end_res*sr)]]

        audio_res_temp = []
        for audio in audio_res:
            if len(audio_res) > 10:
                audio_res_temp.append(audio)
        audio_res = audio_res_temp
    return audio_res

def split_with_cough4(audio, cough_times, max_samples, sr):
    #* cắt tất cả đoạn ho rồi nối với nhau

    
    return audio_res

class CovidDataset2(torch.utils.data.Dataset):
    def __init__(self, df, audio_folder, config, test = False, audio_transforms=None, image_transform=None):
        super().__init__()
        self.df = df
        self.data_cough_interval = get_cough_interval()
        self.audio_folder = audio_folder
        self.image_transform = image_transform
        self.audio_transforms = audio_transforms
        self.test = test

        self.audio_config = config['dataset'].get('audio_config', None)
        if self.audio_config is None:
            self.audio_config = {}
        
        self.target_sr = self.audio_config.get('target_sr', 48000)
        max_duration = self.audio_config.get('max_duration', 15)
        self.max_samples = int(max_duration * self.target_sr)

        cache = config['dataset'].get('cache', False)
        if cache:
            self.audios = load_audio(self.df, self.target_sr)
        else:
            self.audios = None

        # self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        # processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
        # target_sampling_rate = processor.feature_extractor.sampling_rate

    def get_label(self, idx):
        return self.df.iloc[idx]["assessment_result"].astype("float32")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        label = item["assessment_result"].astype("float32")
        # one hot
        # label_encoded = np.zeros((label.size, 2))
        # label_encoded[np.arange(label.size),int(label)] = 1

        label_encoded = torch.tensor(label, dtype=torch.long)
        
        if self.audios is not None:
            uuid = item['uuid']
            audio = copy.deepcopy(self.audios[uuid])
            sr = self.target_sr
        else:
            try:
                audio_path = item['audio_paths']
                audio, sr = sf.read(audio_path, dtype="float32")
                # audio, sr = librosa.load(audio_path, sr = self.target_sr)
            except:
                audio_path = os.path.join(self.audio_folder, "%s.wav"%item['uuid'])
                audio, sr = sf.read(audio_path, dtype="float32")
            audio = librosa.resample(audio, sr, self.target_sr)
        
        if uuid in self.data_cough_interval:
            cough_times = self.data_cough_interval[uuid]
            cough_times = fillter_cough_interval(cough_times)
        else:
            cough_times = []
        # audios = split_with_cough3(audio, cough_times, self.max_samples, sr)
        audios = split_with_cough3(audio, cough_times, self.max_samples, sr)
        audios = [audio] + audios
        # audios = [audio]

        # max_duration = random.choice(range(7, 16))
        # max_samples =  max_duration * self.target_sr
        if self.test:
            for i in range(len(audios)):
                audios[i] = padding_repeat(audios[i], self.max_samples)
            # audio = padding_repeat(audio, self.max_samples)
            # audio = padding_repeat(audio, max_samples)
        else:
            audio = random_crop(audio, self.max_samples)
            # audio = random_crop(audio, max_samples)
            # audio = trim_and_pad(audio, self.max_samples)

        if self.audio_transforms is not None:
            audio_path = item['audio_paths']
            uuid = audio_path.split('/')[-1].split('.')[0]
            if uuid[-2] != '_':
                audio = self.audio_transforms(samples=audio, sample_rate=sr)
            else:
                # print('oversampling no need to aug')
                pass

        #mfcc
        # image = extract_mfcc_feature(audio, self.target_sr, self.mfcc_config)
        # image = extract_mfcc_feature(audio, sr, self.mfcc_config)
        
        # melspec
        # image = audio2melspec(audio, self.target_sr, self.melspec_config)

        # if self.image_transform is not None:
        #     image = self.image_transform(image = image)['image']

        # return image, label_encoded

        #* train 44 - 52
        audios = np.array(audios)
        info = {
            'data': torch.from_numpy(audios).float(),
            'target': label_encoded,
            'uuid': item['uuid']
        }
        return info

        # return torch.from_numpy(audio).float(), label_encoded

        #* wav2vec2
        # audio_ = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values[0]  # Batch size 1
        # return audio_.float(), label_encoded
        

class CovidStackDataset(torch.utils.data.Dataset):
    def __init__(self, df, config, test = False):
        super().__init__()
        self.df = df
        data_path = config['dataset']['data_path']


        with open(data_path, 'rb') as handle:
            self.data = pickle.load(handle)

        self.test = test

    def get_label(self, idx):
        return self.df.iloc[idx]["assessment_result"].astype("float32")

    def __len__(self):
        return self.df.shape[0]
    
    def get_feature(self, uuid):
        feature = []
        exps = [
    57,
    58,
    59,
    63,
    64,
    65,
    'ast_v4_stack',
    'ast_v5_stack',
    'ast_v6_stack',
    'ast_v7_stack',
    'ast_v8_stack',
    'ast_v16_stack',
    'ast_v17_stack',
    66,
    67,
    68,
    69,
    70, 
    71,
    72,
    73,
    74,
    75
]

        for exp in exps:
            # print(self.data[exp][uuid])
            prob = sum(self.data[exp][uuid]['feature']) / len(self.data[exp][uuid]['feature'])
            feature.append(prob)

        return feature

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        uuid = item['uuid']
        label = item["assessment_result"].astype("float32")
        label_encoded = torch.tensor(label, dtype=torch.long)

        # feature = self.data[uuid]['feature']
        # feature = [float(item['feature_{}'.format(i)]) for i in range(9, 19)]
        feature = self.get_feature(uuid)

        
        feature = np.array(feature)

        info = {
            'data': torch.from_numpy(feature).float(),
            'target': label_encoded,
            'uuid': item['uuid']
        }
        return info
        # return torch.from_numpy(feature).float(), label_encoded


class CovidStackTestDataset(torch.utils.data.Dataset):
    def __init__(self, df, config, fold, test = False):
        super().__init__()
        self.df = df
        data_path = config['dataset']['data_path']
        self.fold = fold

        with open(data_path, 'rb') as handle:
            self.data = pickle.load(handle)

        self.test = test

    def get_label(self, idx):
        return self.df.iloc[idx]["assessment_result"].astype("float32")

    def __len__(self):
        return self.df.shape[0]
    
    def get_feature(self, uuid):
        feature = []
        exps = [
    57,
    58,
    59,
    63,
    64,
    65,
    'ast_v4_stack',
    'ast_v5_stack',
    'ast_v6_stack',
    'ast_v7_stack',
    'ast_v8_stack',
    'ast_v16_stack',
    'ast_v17_stack',
    66,
    67,
    68,
    69,
    70, 
    71,
    72,
    73,
    74,
    75
]

        for exp in exps:
            # print(self.data[exp][uuid])
            # print(self.data[exp].keys())
            prob = sum(self.data[exp][self.fold][uuid]['feature']) / len(self.data[exp][self.fold][uuid]['feature'])
            feature.append(prob)

        return feature

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        uuid = item['uuid']
        label = item["assessment_result"].astype("float32")
        label_encoded = torch.tensor(label, dtype=torch.long)

        # feature = self.data[uuid]['feature']
        # feature = [float(item['feature_{}'.format(i)]) for i in range(9, 19)]
        feature = self.get_feature(uuid)

        
        feature = np.array(feature)

        info = {
            'data': torch.from_numpy(feature).float(),
            'target': label_encoded,
            'uuid': item['uuid']
        }
        return info
        # return torch.from_numpy(feature).float(), label_encoded


        