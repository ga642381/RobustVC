"""Precompute Wav2Vec features and spectrograms."""

import os
from copy import deepcopy
from pathlib import Path

import sox
import torch
from librosa.util import find_files

from .utils import load_wav, log_mel_spectrogram


class PreprocessDataset(torch.utils.data.Dataset):
    """Prefetch audio data for preprocessing."""

    def __init__(
        self,
        clean_data_dir,
        noisy_data_dir,
        trim_method,
        sample_rate,
    ):
        # dirs
        data = []
        clean_data_dir = Path(clean_data_dir)
        noisy_data_dir = Path(noisy_data_dir)
        # spkers
        speakers = sorted(os.listdir(clean_data_dir))
        speakers_ = sorted(os.listdir(noisy_data_dir))
        assert speakers == speakers_

        # add data
        for spk in speakers:
            clean_spk_dir = clean_data_dir / spk
            noisy_spk_dir = noisy_data_dir / spk

            wav_files = os.listdir(clean_spk_dir)
            for wav_file in wav_files:
                clean_wav_path = clean_spk_dir / wav_file
                noisy_wav_path = noisy_spk_dir / wav_file
                if clean_wav_path.is_file() and noisy_wav_path.is_file():
                    data.append((spk, clean_wav_path, noisy_wav_path))

        # === init === #
        self.trim_method = trim_method
        self.sample_rate = sample_rate
        self.data = data
        if trim_method == "vad":
            tfm = sox.Transformer()
            tfm.vad(location=1)
            tfm.vad(location=-1)
            self.sox_transform = tfm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        speaker_name, clean_audio_path, noisy_audio_path = self.data[index]

        if self.trim_method == "librosa":
            clean_wav = load_wav(clean_audio_path, self.sample_rate, trim=True)
            noisy_wav = load_wav(noisy_audio_path, self.sample_rate, trim=True)

        elif self.trim_method == "vad":
            clean_wav = load_wav(clean_audio_path, self.sample_rate)
            noisy_wav = load_wav(noisy_audio_path, self.sample_rate)

            trim_clean_wav = self.sox_transform.build_array(
                input_array=clean_wav, sample_rate_in=self.sample_rate
            )
            trim_noisy_wav = self.sox_transform.build_array(
                input_array=noisy_wav, sample_rate_in=self.sample_rate
            )

            clean_wav = deepcopy(
                trim_clean_wav if len(trim_clean_wav) > 10 else clean_wav
            )

            noisy_wav = deepcopy(
                trim_noisy_wav if len(trim_noisy_wav) > 10 else noisy_wav
            )

        else:
            clean_wav = load_wav(clean_audio_path, self.sample_rate, trim=False)
            noisy_wav = load_wav(noisy_audio_path, self.sample_rate, trim=False)

        return (
            speaker_name,
            str(clean_audio_path),
            str(noisy_audio_path),
            torch.FloatTensor(clean_wav),
            torch.FloatTensor(noisy_wav),
        )
