"""Inferencer of FragmentVC"""
from typing import List, Optional
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

from .feature_extract import FeatureExtractor
from .utils import load_wav

# from .models import load_pretrained_wav2vec
# from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(root) / "checkpoints"
        # wav2vec_path = checkpoint_dir / "wav2vec_small.pt"
        ckpt_path = checkpoint_dir / "model.ckpt"
        vocoder_path = checkpoint_dir / "vocoder.pt"

        # self.wav2vec = load_pretrained_wav2vec(str(wav2vec_path)).to(device)
        self.model = torch.jit.load(str(ckpt_path)).eval().to(device)
        self.vocoder = torch.jit.load(str(vocoder_path)).eval().to(device)
        self.device = device
        self.sample_rate = 16000
        self.ref_feat_model = FeatureExtractor("cpc", None, device)
        self.src_feat_model = FeatureExtractor("cpc", None, device)

    def inference(self, src_wav: np.ndarray, tgt_wavs: List[np.ndarray]) -> Tensor:
        """Inference one utterance."""
        src_wav = torch.FloatTensor(src_wav).to(self.device)
        tgt_wavs = [torch.FloatTensor(tgt_wav).to(self.device) for tgt_wav in tgt_wavs]

        with torch.no_grad():
            tgt_mels = self.ref_feat_model.get_feature(tgt_wavs)
            src_mel = (
                self.ref_feat_model.get_feature([src_wav])[0]
                .transpose(0, 1)
                .unsqueeze(0)
                .to(self.device)
            )
            tgt_mels = [tgt_mel.cpu() for tgt_mel in tgt_mels]
            tgt_mel = np.concatenate(tgt_mels, axis=0)
            tgt_mel = torch.FloatTensor(tgt_mel.T).unsqueeze(0).to(self.device)
            src_feat = self.src_feat_model.get_feature([src_wav])[0].unsqueeze(0)
            out_mel, attn = self.model(src_feat, tgt_mel)
            out_mel = out_mel.transpose(1, 2).squeeze(0)

        return out_mel.to("cpu")

    def inference_from_path(self, src_path: Path, tgt_paths: List[Path]) -> Tensor:
        """Inference from path."""
        src_wav = load_wav(src_path, sample_rate=self.sample_rate, trim=False)
        tgt_wavs = [
            load_wav(tgt_path, sample_rate=self.sample_rate, trim=False)
            for tgt_path in tgt_paths
        ]
        result = self.inference(src_wav, tgt_wavs)

        return result

    def inference_from_pair(self, pair, source_dir: str, target_dir: str) -> Tensor:
        """Inference from pair of metadata."""
        source_utt = Path(source_dir) / pair["src_utt"]
        target_utts = [Path(target_dir) / tgt_utt for tgt_utt in pair["tgt_utts"]]
        conv_mel = self.inference_from_path(
            source_utt, [target_utts[0]]
        )  # ! temp one tgt

        return conv_mel

    def spectrogram2waveform(self, spectrogram: List[Tensor]) -> List[Tensor]:
        """Convert spectrogram to waveform."""
        waveforms = self.vocoder.generate(spectrogram)

        return waveforms
