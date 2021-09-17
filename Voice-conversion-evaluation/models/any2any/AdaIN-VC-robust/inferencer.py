"""Inferencer of AdaIN-VC"""
from pathlib import Path
from typing import List

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from torch import Tensor

from .wav2mel import Wav2Mel


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        checkpoint_dir = Path(root) / "checkpoints"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        config_path = checkpoint_dir / "config.yaml"
        model_path = checkpoint_dir / "model.ckpt"
        vocoder_path = checkpoint_dir / "vocoder.pt"
        config = yaml.load(config_path.open(), Loader=yaml.FullLoader)

        self.model = torch.jit.load(str(model_path)).to(device)
        self.model.eval()

        self.vocoder = torch.jit.load(str(vocoder_path)).to(device)

        self.device = device
        self.frame_size = config["data_loader"]["frame_size"]
        self.sample_rate = 16000
        self.wav2mel = Wav2Mel()

    def inference_from_pair(self, pair, source_dir: str, target_dir: str) -> Tensor:
        source_utt = Path(source_dir) / pair["src_utt"]
        target_utt = Path(target_dir) / pair["tgt_utts"][0]  # ! temp one tgt

        # below modified from cyhung-tw AdaIN-VC inference script
        src, src_sr = torchaudio.load(source_utt)
        tgt, tgt_sr = torchaudio.load(target_utt)

        src = self.wav2mel.mel(src)
        tgt = self.wav2mel.mel(tgt)
        if tgt is None or src is None:
            return None

        src = src[None, :].to(self.device)
        tgt = tgt[None, :].to(self.device)

        cvt = self.model.inference(src, tgt)

        return cvt.squeeze(0).data.T

    def spectrogram2waveform(self, spectrogram: List[Tensor]) -> List[Tensor]:
        """Convert spectrogram to waveform."""
        waveforms = self.vocoder.generate(spectrogram)

        return waveforms
