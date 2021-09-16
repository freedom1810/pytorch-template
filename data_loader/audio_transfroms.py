from abc import ABC
import numpy as np
import random
import librosa

class BaseAudioTransform(ABC):
    def __call__(self, audio, fs):
        raise NotImplementedError

class AudioCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    def test(self):
        for adt in self.transforms:
            print(adt)
    def __call__(self, audio, fs):
        for adt in self.transforms:
            audio, fs = adt.__call__(audio, fs)
        return audio, fs

def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

class WhiteNoise(BaseAudioTransform):
    def __init__(self, noise_factor=0.005):
        self.noise_factor = noise_factor

    def __call__(self, audio, fs):
        audio = add_noise(audio, self.noise_factor)
        return audio, fs

def change_speed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)

class ChangeSpeed(BaseAudioTransform):
    def __init__(self, min_speed=0.5, max_speed=1.5):
        self.min_speed = min_speed
        self.max_speed = max_speed
    def __call__(self, audio, fs):
        speed_factor = random.uniform(self.min_speed, self.max_speed)
        audio = change_speed(audio, speed_factor)
        return audio, fs

def time_shift(data, sampling_rate, shift_max, shift_direction):
    try:
        shift = np.random.randint(sampling_rate * shift_max)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(data, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data, sampling_rate
    except Exception as e:
        print(e)
        return data, sampling_rate

class TimeShift(BaseAudioTransform):
    def __init__(self, shift_max=0.2, shift_direction='both'):
        self.shift_max = shift_max
        self.shift_direction = shift_direction
    def __call__(self, audio, fs):
        return time_shift(audio, fs, self.shift_max, self.shift_direction)

def change_pitch(data, sampling_rate, pitch_factor=4):
    try:
        changed_pitch_audio = librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
        return changed_pitch_audio, sampling_rate
    except Exception as e:
        print(e)
        return data, sampling_rate

class ChangePitch(BaseAudioTransform):
    def __init__(self, pitch_factor=4):
        self.pitch_factor = pitch_factor

    def __call__(self, audio, fs):
        return change_pitch(audio, fs, self.pitch_factor)
