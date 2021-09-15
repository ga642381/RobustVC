import json
import os
import random

import augment
import torch
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
        clean_wav_ratio=0.4,
    ):
        self.split_type = split_type
        self.split_spks = split_spks

        self.data_dir = data_dir
        self.meta_data = json.load(open(os.path.join(data_dir, "metadata.json"), "r"))
        self.idx2spk = split_spks
        self.sample_rate = sample_rate
        self.segment = segment
        self.n_uttrs = n_uttrs
        self.clean_wav_ratio = clean_wav_ratio
        # self.wav2mel = Wav2Mel()
        self.wavaug = WavAug(
            sample_rate=sample_rate,
            p_clean=0,
            p_add=0,
            p_reverb=0.5,
            p_band=0.5,
        )

    def __len__(self):
        return len(self.split_spks)
        # return len(self.meta_data)  # num_speakers

    def __getitem__(self, index):
        # define range of the split
        spk = self.idx2spk[index]

        # read wavs
        wav_files = random.sample(self.meta_data[spk], k=self.n_uttrs)
        clean_wavs = []
        noisy_wavs = []
        for wav_file in wav_files:
            # clean
            clean_wav = torch.load(os.path.join(self.data_dir, wav_file["clean"]))
            clean_wavs.append(clean_wav)

            # noisy : 40% clean, 60% aug
            if random.random() < self.clean_wav_ratio:
                noisy_wavs.append(clean_wav)
            else:
                noisy_wav = torch.load(os.path.join(self.data_dir, wav_file["noisy"]))
                aug_noisy_wav = self.wavaug.add_noise(noisy_wav)
                noisy_wavs.append(aug_noisy_wav)

        assert len(clean_wavs) == len(noisy_wavs)
        # crop wavs
        starts = [random.randint(0, w.shape[-1] - self.segment) for w in clean_wavs]

        clean_wav_segs = torch.stack(
            [
                w[:, start : (start + self.segment - 1)]
                for (w, start) in zip(clean_wavs, starts)
            ]
        ).squeeze(1)

        noisy_wav_segs = torch.stack(
            [
                w[:, start : (start + self.segment - 1)]
                for (w, start) in zip(noisy_wavs, starts)
            ]
        ).squeeze(1)

        # return
        return (
            clean_wav_segs,
            noisy_wav_segs,
        )
