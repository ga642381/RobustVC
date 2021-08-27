import torch
import augment
import numpy as np
import random
import os
import torchaudio
from functools import partial


def noise_gen(x):
    return torch.zeros_like(x).uniform_()


class WavAug:
    def __init__(
        self, sample_rate=16000, p_clean=0.5, p_add=1, p_reverb=0.5, p_band=0.5
    ):
        self.sample_rate = sample_rate
        self.p_clean = p_clean
        self.p_add = p_add
        self.p_reverb = p_reverb
        self.p_band = p_band
        self.snr_list = [0, 5, 10, 15]

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        reference : http://sox.sourceforge.net/sox.html
        * additive_noise
            : snr (the smaller the noiser)
        * reverberation
            : reverberance (50%)
            : HF-dampling (50%)
            : room-scale (100%)
        * band_rejection (sinc)
            : -a 120 (attenuation of 120dB)
            : (freqHP - freqLP ; if freqHP > freqLp then bandreject)
        """
        wavaug_chain = augment.EffectChain()
        if random.random() < self.p_clean:
            # early return clean wav
            return x

        if random.random() < self.p_add:
            noise_generator = partial(noise_gen, x=x)
            wavaug_chain = wavaug_chain.additive_noise(
                noise_generator, snr=random.choice(self.snr_list)
            )

        if random.random() < self.p_reverb:
            wavaug_chain = wavaug_chain.reverb(
                50, 50, np.random.randint(0, 101)
            ).channels(1)

        if random.random() < self.p_band:
            band_width = random.randint(50, 150)
            freq_LP = random.randint(100, 500)
            wavaug_chain = wavaug_chain.sinc(
                "-a", "120", f"{freq_LP+band_width}-{freq_LP}"
            )

        noisy_wav = wavaug_chain.apply(x, src_info={"rate": self.sample_rate})
        return noisy_wav
