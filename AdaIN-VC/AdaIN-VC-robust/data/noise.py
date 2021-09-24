import random
from functools import partial

import augment
import numpy as np
import torch


class WavAug:
    def __init__(
        self,
        sample_rate=16000,
        p_clean=0,
        p_add=0,
        p_reverb=0.5,
        p_band=0.5,
        mode="train",
    ):
        self.sample_rate = sample_rate
        self.p_clean = p_clean
        self.p_add = p_add
        self.p_reverb = p_reverb
        self.p_band = p_band
        self.mode = mode
        if mode == "train":
            self.snr_list = [0, 5, 10, 15]
        elif mode == "test":
            self.snr_list = [2.5, 7.5, 12.5, 17.5]

    def _noise_gen(self, x):
        return torch.zeros_like(x).uniform_()

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

        # clean (early return clean wav)
        if random.random() < self.p_clean and self.mode == "train":
            return x

        # additive
        if random.random() < self.p_add:
            noise_generator = partial(self._noise_gen, x=x)
            wavaug_chain = wavaug_chain.additive_noise(
                noise_generator, snr=random.choice(self.snr_list)
            )

        # reverb
        if random.random() < self.p_reverb:
            wavaug_chain = wavaug_chain.reverb(
                50, 50, np.random.randint(0, 101)
            ).channels(1)

        # band rejection
        if random.random() < self.p_band:
            band_width = random.randint(50, 150)
            freq_LP = random.randint(100, 500)
            wavaug_chain = wavaug_chain.sinc(
                "-a", "120", f"{freq_LP+band_width}-{freq_LP}"
            )

        noisy_wav = wavaug_chain.apply(x, src_info={"rate": self.sample_rate})
        return noisy_wav
