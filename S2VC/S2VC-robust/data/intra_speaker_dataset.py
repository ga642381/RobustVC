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
        n_samples=5,
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
        self.n_samples = n_samples
        self.training = training
        self.src_feat = src_feat
        self.ref_feat = ref_feat
        self.clean_wav_ratio = clean_wav_ratio
        self.src_dim = -1
        self.ref_dim = -1
        self.tgt_dim = -1

        # === add noise === #
        self.src_feat_extractor = FeatureExtractor(src_feat, wav2vec_path, device="cpu")
        self.ref_feat_extractor = FeatureExtractor(ref_feat, wav2vec_path, device="cpu")
        self.wavaug = WavAug(
            sample_rate=16000, p_clean=0, p_add=0, p_reverb=0.5, p_band=0.5
        )

    def __len__(self):
        return len(self.data)

    # __getitem__ <-- _get_data <-- _load_data
    def __getitem__(self, index):
        (
            speaker_name,
            source_emb,
            target_emb,
            ground_mel,
            source_wav,
            target_wav,
        ) = self._get_data(index)
        return (
            speaker_name,
            source_emb,
            target_emb,
            ground_mel,
            source_wav,
            target_wav,
        )

    def _get_data(self, index):
        # noisy training try this first
        (
            speaker_name,
            source_clean_wav,  # vad
            source_noisy_wav,  # demand
            target_clean_wav,  # vad
            target_noisy_wav,  # demand
            ground_mel,
        ) = _load_data(*self.data[index])

        # === add noise === #
        # wav -> wav (noisy) -> cpc
        if self.src_feat == self.ref_feat:
            # same noisy wav and ssl feature for training
            # 40% clean
            if random.random() < self.clean_wav_ratio:
                source_wav = source_clean_wav  # clean

            # 60% noisy + wavaug
            else:
                source_wav = source_noisy_wav  # demand
                source_wav = self.wavaug.add_noise(source_wav)

            source_wav = source_wav.squeeze(0)  # shape : (Length,)
            source_emb = (
                self.src_feat_extractor.get_feature([source_wav])[0].detach().cpu()
            )

            target_wav = source_wav
            target_emb = source_emb
        else:
            raise NotImplementedError(
                "not implemented different features, pls check github history before 20210915, it's easy"
            )

        # set shapes
        self.src_dim = source_emb.shape[1]
        self.ref_dim = target_emb.shape[1]
        self.tgt_dim = ground_mel.shape[1]

        return (
            speaker_name,
            source_emb,
            target_emb,
            ground_mel,
            source_wav,  # for training : (clean + demand + aug)
            target_wav,  # for training : (clean + demand + aug)
        )

    def get_feat_dim(self):
        self._get_data(0)
        return self.src_dim, self.ref_dim, self.tgt_dim


def _process_data(speaker_name, data_dir, feature, src_feat, ref_feat):
    _, _, src_feature_path, ref_feature_path = (
        feature["clean_audio_path"],
        feature["noisy_audio_path"],
        feature[src_feat],
        feature[ref_feat],
    )
    return speaker_name, data_dir, src_feature_path, ref_feature_path


def _load_data(speaker_name, data_dir, src_feature_path, ref_feature_path):
    src_feature = torch.load(Path(data_dir, src_feature_path), "cpu")
    ref_feature = torch.load(Path(data_dir, ref_feature_path), "cpu")

    source_clean_wav = src_feature["clean_wav"].detach().cpu()
    source_noisy_wav = src_feature["noisy_wav"].detach().cpu()
    target_clean_wav = ref_feature["clean_wav"].detach().cpu()
    target_noisy_wav = ref_feature["noisy_wav"].detach().cpu()

    ground_mel = src_feature["clean_mel"].detach().cpu()
    return (
        speaker_name,
        source_clean_wav,
        source_noisy_wav,
        target_clean_wav,
        target_noisy_wav,
        ground_mel,
    )


# ==== collate function === #
def collate_batch(batch):
    """Collate a batch of data."""
    """
    srcs      : (batch, max_src_len, feat_dim)
    src_masks : (batch, max_src_len) //False if value

    tgts      : (batch, feat_dim, max_tgt_len)
    tgt_masks : (batch, max_tgt_len) // False if value

    tgt_mels      : (batch, mel_dim, max_tgt_mel_len)
    overlap_lens  : list, len == batch_size 
    """
    # speaker_name,
    # source_emb,
    # target_emb,
    # ground_mel,
    # source_wav,
    # target_wav,
    spks, srcs, tgts, tgt_mels, src_wavs, tgt_wavs = zip(*batch)

    src_lens = [len(src) for src in srcs]
    tgt_lens = [len(tgt) for tgt in tgts]
    tgt_mel_lens = [len(tgt_mel) for tgt_mel in tgt_mels]

    overlap_lens = [
        min(src_len, tgt_mel_len)
        for src_len, tgt_mel_len in zip(src_lens, tgt_mel_lens)
    ]

    srcs = pad_sequence(srcs, batch_first=True)

    src_masks = [torch.arange(srcs.size(1)) >= src_len for src_len in src_lens]
    src_masks = torch.stack(src_masks)

    tgts = pad_sequence(tgts, batch_first=True, padding_value=-20)
    tgts = tgts.transpose(1, 2)  # (batch, mel_dim, max_tgt_len)

    tgt_masks = [torch.arange(tgts.size(2)) >= tgt_len for tgt_len in tgt_lens]
    tgt_masks = torch.stack(tgt_masks)  # (batch, max_tgt_len)

    tgt_mels = pad_sequence(tgt_mels, batch_first=True, padding_value=-20)
    tgt_mels = tgt_mels.transpose(1, 2)  # (batch, mel_dim, max_tgt_len)

    return (
        srcs,
        src_masks,
        tgts,
        tgt_masks,
        tgt_mels,
        overlap_lens,
        src_wavs,
        tgt_wavs,
        spks,
    )
