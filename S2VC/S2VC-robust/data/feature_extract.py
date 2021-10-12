from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List

import torch
from fairseq.checkpoint_utils import load_model_ensemble
from torch.nn.utils.rnn import pad_sequence

from .utils import log_mel_spectrogram


def load_pretrained_wav2vec(ckpt_path: str):
    """Load pretrained Wav2Vec model."""
    ckpt_path = str(ckpt_path)
    model, cfg = load_model_ensemble([ckpt_path])
    model = model[0]
    model.remove_pretraining_modules()
    model.eval()
    return model


class FeatureExtractor:
    def __init__(self, feature_name, wav2vec2_path=None, device=None):
        self.device = device
        if feature_name in ["apc", "cpc", "timit_posteriorgram", "fbank"]:
            self.extractor = (
                torch.hub.load(
                    "ga642381/s3prl:s2vc",
                    feature_name,
                    refresh=True,
                )
                .eval()
                .to(device)
            )
            self.mode = 1

        elif feature_name == "wav2vec2":
            self.extractor = load_pretrained_wav2vec(wav2vec2_path).eval().to(device)
            self.mode = 2

        elif feature_name == "wav2vec2_mel":
            self.extractor = partial(
                log_mel_spectrogram,
                preemph=0.97,
                sample_rate=16000,
                n_mels=80,
                n_fft=400,
                hop_length=320,
                win_length=400,
                f_min=0,
                center=False,
            )
            self.mode = 3

        elif feature_name == "cpc_mel":
            self.extractor = partial(
                log_mel_spectrogram,
                preemph=0.97,
                sample_rate=16000,
                n_mels=80,
                n_fft=465,
                hop_length=160,
                win_length=465,
                f_min=80,
                center=True,
            )
            self.mode = 3

        else:
            print(feature_name)
            print(
                "Please use timit_posteriorgram, apc, wav2vec2, cpc, wav2vec2_mel, cpc_mel, or fbank"
            )
            exit()

    def get_feature(self, wavs: list) -> list:
        # wavs : list of tensors, no padding
        if self.mode == 1:
            return self.extractor(wavs)

        elif self.mode == 2:
            wav_lens = [len(wav) for wav in wavs]
            wavs = pad_sequence(wavs, batch_first=True)
            padding_mask = [
                torch.arange(wavs.size(1)) >= wav_len for wav_len in wav_lens
            ]
            padding_mask = torch.stack(padding_mask).to(self.device)

            feats = self.extractor.extract_features(wavs, padding_mask)["x"]
            feats = [f for f in feats]

        elif self.mode == 3:
            wavs = [wav.cpu().numpy() for wav in wavs]
            feats = [self.extractor(wav) for wav in wavs]
            feats = [torch.FloatTensor(feat).to(self.device) for feat in feats]
            return feats

        return feats
