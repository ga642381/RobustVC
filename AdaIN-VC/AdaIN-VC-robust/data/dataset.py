import json
import os
import random

import torch
import augment
from torch.utils.data import Dataset
from data.noise import WavAug
from data.wav2mel import Wav2Mel


class SpeakerDataset(Dataset):
    def __init__(
        self,
        split_type: str,
        split_spks: list,
        data_dir,
        sample_rate: int = 16000,
        segment: int = 25600,
        n_uttrs: int = 4,
    ):
        self.split_type = split_type
        self.split_spks = split_spks

        self.data_dir = data_dir
        self.meta_data = json.load(open(os.path.join(data_dir, "metadata.json"), "r"))
        self.idx2spk = split_spks
        self.sample_rate = sample_rate
        self.segment = segment
        self.n_uttrs = n_uttrs
        self.wav2mel = Wav2Mel()
        self.wavaug = WavAug(sample_rate=sample_rate)

    def __len__(self):
        return len(self.split_spks)
        # return len(self.meta_data)  # num_speakers

    def __getitem__(self, index):
        # define range of the split
        spk = self.idx2spk[index]

        # read wavs
        wav_files = random.sample(self.meta_data[spk], k=self.n_uttrs)
        wavs = [torch.load(os.path.join(self.data_dir, file)) for file in wav_files]

        # add noise
        aug_wavs = [self.wavaug.add_noise(w) for w in wavs]

        # crop wavs
        starts = [random.randint(0, w.shape[-1] - self.segment) for w in wavs]
        clean_wavs = torch.stack(
            [
                w[:, start : (start + self.segment - 1)]
                for (w, start) in zip(wavs, starts)
            ]
        ).squeeze(1)
        noisy_wavs = torch.stack(
            [
                w[:, start : (start + self.segment - 1)]
                for (w, start) in zip(aug_wavs, starts)
            ]
        ).squeeze(1)

        return (
            self.wav2mel.mel(clean_wavs),
            self.wav2mel.mel(noisy_wavs),
            clean_wavs,
            noisy_wavs,
        )
