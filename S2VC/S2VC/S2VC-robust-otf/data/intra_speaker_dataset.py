"""Dataset for reconstruction scheme."""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# from torch._C import device
from tqdm import tqdm

from .feature_extract import FeatureExtractor
from .noise import WavAug
from .utils import load_wav, log_mel_spectrogram


class IntraSpeakerDataset(Dataset):
    """Dataset for reconstruction scheme.

    Returns:
        speaker_id: speaker id number.
        feat: Wav2Vec feature tensor.
        mel: log mel spectrogram tensor.
    """

    def __init__(
        self,
        split_type: str,
        split_spks: list,
        data_dir,
        metadata_path,
        src_feat,
        ref_feat,
        training=True,
        clean_wav_ratio=0.4,
        wav2vec_path=None,
    ):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # _process_data
        executor = ThreadPoolExecutor(max_workers=4)
        futures = []
        for speaker_name, utterances in metadata.items():
            if speaker_name in split_spks:  # split speakers here
                for utterance in utterances:
                    futures.append(
                        executor.submit(
                            _process_data,
                            speaker_name,
                            data_dir,
                            utterance,
                            src_feat,
                            ref_feat,
                        )
                    )
        # add data
        self.data = []
        self.speaker_to_indices = {}
        for i, future in enumerate(tqdm(futures, ncols=0)):
            result = future.result()
            speaker_name = result[0]
            self.data.append(result)
            if speaker_name not in self.speaker_to_indices:
                self.speaker_to_indices[speaker_name] = [i]
            else:
                self.speaker_to_indices[speaker_name].append(i)

        # init
        self.split_type = split_type
        self.split_spks = split_spks
        self.data_dir = Path(data_dir)
        self.training = training
        self.src_feat = src_feat
        self.ref_feat = ref_feat
        self.clean_wav_ratio = clean_wav_ratio

        # === add noise === #\\
        self.wavaug = WavAug(
            sample_rate=16000, p_clean=0, p_add=0, p_reverb=0.5, p_band=0.5
        )

    def __len__(self):
        return len(self.data)

    # __getitem__ <-- _get_data <-- _load_data
    def __getitem__(self, index):
        (speaker_name, ground_mel, source_wav, target_wav) = self._get_data(index)
        return (speaker_name, ground_mel, source_wav, target_wav)

    def _get_data(self, index):
        # noisy training try this first
        (
            speaker_name,
            clean_wav,  # vad
            noisy_wav,  # demand
            ground_mel,
        ) = _load_data(*self.data[index])

        # === add noise === #
        # wav -> wav (noisy) -> cpc
        if self.src_feat == self.ref_feat:
            # same noisy wav and ssl feature for training
            # 40% clean
            if random.random() < self.clean_wav_ratio:
                source_wav = clean_wav.clone()  # clean
            # 60% noisy + wavaug
            else:
                source_wav = noisy_wav.clone()  # demand
                source_wav = self.wavaug.add_noise(source_wav)

            source_wav = source_wav.squeeze(0)  # shape : (Length,)
            target_wav = source_wav.clone()
        else:
            raise NotImplementedError(
                "not implemented different features, pls check github history before 20210915, it's easy"
            )

        return (
            speaker_name,
            ground_mel,
            source_wav,  # for training : (clean + demand + aug)
            target_wav,  # for training : (clean + demand + aug)
        )


def _process_data(speaker_name, data_dir, feature, src_feat, ref_feat):
    _, _, src_feature_path, ref_feature_path = (
        feature["clean_audio_path"],
        feature["noisy_audio_path"],
        feature[src_feat],
        feature[ref_feat],
    )
    return speaker_name, data_dir, src_feature_path, ref_feature_path


def _load_data(speaker_name, data_dir, src_feature_path, ref_feature_path):
    src_feature = torch.load(Path(data_dir, src_feature_path))

    clean_wav = src_feature["clean_wav"]
    noisy_wav = src_feature["noisy_wav"]
    ground_mel = src_feature["clean_mel"]
    return (speaker_name, clean_wav, noisy_wav, ground_mel)


def collate_pseudo(batch):
    spks, ground_mels, src_wavs, tgt_wavs = zip(*batch)
    return (spks, ground_mels, src_wavs, tgt_wavs)
