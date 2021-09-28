import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from tqdm import trange

from data import Wav2Mel


def emb_attack(
    model: nn.Module,
    wav2mel: Wav2Mel,
    src: Tensor,
    tgt: Tensor,
    eps: float,
    n_steps: int,
):
    ptb = torch.randn_like(src).requires_grad_(True)
    # ptb = torch.empty_like(src).randn_(0, 1).requires_grad_(True)
    with torch.no_grad():
        src_mel = wav2mel.log_melspectrogram(src)
        tgt_mel = wav2mel.log_melspectrogram(tgt)
        src_emb = model.speaker_encoder(src_mel[None, :])
        tgt_emb = model.speaker_encoder(tgt_mel[None, :])
    optimizer = torch.optim.Adam([ptb], lr=1e-4)
    criterion = nn.MSELoss()
    pbar = trange(n_steps)
    for _ in pbar:
        adv = src + eps * ptb.tanh()
        adv_mel = wav2mel.log_melspectrogram(adv)
        adv_emb = model.speaker_encoder(adv_mel[None, :])
        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, src_emb)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})
    return (src + eps * ptb.tanh()).detach().cpu()


def main(
    src_path: Path,
    tgt_path: Path,
    out_path: Path,
    model_path: Path,
    eps: float,
    n_steps: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path)
    wav2mel = Wav2Mel()
    src_wav = wav2mel.sox_effects(*torchaudio.load(src_path))
    tgt_wav = wav2mel.sox_effects(*torchaudio.load(tgt_path))
    assert src_wav.ndim == tgt_wav.ndim == 2
    adv_wav = emb_attack(
        model.to(device),
        wav2mel.to(device),
        src_wav.to(device),
        tgt_wav.to(device),
        eps,
        n_steps,
    )
    assert adv_wav.ndim == 2
    torchaudio.save(out_path, adv_wav, 16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_path", type=Path)
    parser.add_argument("tgt_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--eps", type=float, default=0.005)
    parser.add_argument("--n_steps", type=int, default=1000)
    main(**vars(parser.parse_args()))
