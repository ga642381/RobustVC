import torch
import augment
import numpy as np
import random
import os
import torchaudio
from functools import partial


def noise_gen(x):
    return torch.zeros_like(x).uniform_()


class WavAug():
    def __init__(self, sample_rate=16000, p_add=1, p_reverb=0.5, p_band=0.5):
        self.sample_rate = sample_rate
        self.p_add = p_add
        self.p_reverb = p_reverb
        self.p_band = p_band

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        * additive_noise
            : snr
        * reverberation
            : 
            :room_size
        * band_rejection
        """
        wavaug_chain = augment.EffectChain()

        if (random.random() < self.p_add):
            noise_generator = partial(noise_gen, x=x)
            wavaug_chain = wavaug_chain.additive_noise(
                noise_generator, snr=15)

        if (random.random() < self.p_reverb):
            wavaug_chain = wavaug_chain.reverb(
                50, 50, np.random.randint(0, 101)).channels(1)

        if (random.random() < self.p_band):
            wavaug_chain = wavaug_chain.sinc('-a', '120', '500-100')

        noisy_wav = wavaug_chain.apply(x, src_info={'rate': self.sample_rate})
        return noisy_wav
