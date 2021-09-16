from audiomentations.core.composition import BaseCompose
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, PolarityInversion, AddGaussianSNR

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

train_audio_transform = OneOf([
                    # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    AddGaussianSNR(p=0.5), # new
                    TimeStretch(min_rate=0.9, max_rate=1.2, p=0.5),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
                    PolarityInversion(p=0.5),
                    Gain()
                ], p=1.0)


audio = train_audio_transform(samples=audio, sample_rate=sr)
        