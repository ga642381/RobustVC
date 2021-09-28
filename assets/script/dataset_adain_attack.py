import argparse
import copy
import json
import random
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.cuda import device
from tqdm import tqdm, trange

from utils.VCTK_split import train_valid_test
from utils.wav2mel import Wav2Mel


def emb_attack(
    model: nn.Module,
    wav2mel: Wav2Mel,
    src: Tensor,
    tgt: Tensor,
    eps: float,
    n_steps: int,
):
    ptb = torch.randn_like(src).requires_grad_(True)
    with torch.no_grad():
        src_mel = wav2mel.log_melspectrogram(src)
        tgt_mel = wav2mel.log_melspectrogram(tgt)
        src_emb = model.speaker_encoder(src_mel[None, :])
        tgt_emb = model.speaker_encoder(tgt_mel[None, :])
    optimizer = torch.optim.Adam([ptb], lr=1e-4)
    criterion = nn.MSELoss()
    for _ in range(n_steps):
        adv = src + eps * ptb.tanh()
        adv_mel = wav2mel.log_melspectrogram(adv)
        adv_emb = model.speaker_encoder(adv_mel[None, :])
        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, src_emb)
        loss.backward()
        optimizer.step()
    return (src + eps * ptb.tanh()).detach().cpu()


# attack_and_save function modified from attack.py
def attack_and_save(
    src_path: Path,
    tgt_path: Path,
    out_path: Path,
    model: torch.ScriptModule,
    wav2mel: Wav2Mel,
    device,
    eps: float,
    n_steps: int,
):
    src_wav = wav2mel.sox_effects(*torchaudio.load(src_path)).to(device)
    tgt_wav = wav2mel.sox_effects(*torchaudio.load(tgt_path)).to(device)
    assert src_wav.ndim == tgt_wav.ndim == 2
    adv_wav = emb_attack(
        model,
        wav2mel,
        src_wav,
        tgt_wav,
        eps,
        n_steps,
    )
    assert adv_wav.ndim == 2
    torchaudio.save(out_path, adv_wav, 16000)


def main(
    data_dir: Path,
    save_dir: Path,
    model_path: Path,
    metadata_path: Path,
    eps: float,
    n_steps: int,
):
    # === path === #
    data_dir = Path(data_dir).resolve()
    save_dir = Path(save_dir).resolve()
    model_path = Path(model_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Task : Adding Attack Noise (AdaIN-roubst)")
    print(f"[INFO] model path : {model_path}")
    print(f"[INFO] metadata_path : {metadata_path}")
    print(f"[INFO] data_dir : {data_dir}")
    print(f"[INFO] save_dir : {save_dir}")

    # === init === #
    metadata = json.load(open(metadata_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path).to(device)
    # all vctk 16k and vad
    wav2mel = Wav2Mel(resample=False, norm_vad=False).to(device)
    attack_fn = partial(
        attack_and_save,
        model=model,
        wav2mel=wav2mel,
        device=device,
        eps=eps,
        n_steps=n_steps,
    )
    # === main loop === #
    pairs = metadata["pairs"]
    n_samples = len(pairs)
    pbar = tqdm(total=n_samples)
    for pair in pairs:
        tgt_utt = pair["tgt_utts"][0]
        atk_tgt_utt = pair["atk_tgt_utts"][0]

        atk_src_path = data_dir / tgt_utt
        atk_tgt_path = data_dir / atk_tgt_utt
        atk_out_path = Path(str(atk_src_path).replace(str(data_dir), str(save_dir)))
        atk_out_path.parent.mkdir(parents=True, exist_ok=True)
        attack_fn(src_path=atk_src_path, tgt_path=atk_tgt_path, out_path=atk_out_path)
        pbar.update()
    pbar.close()


if __name__ == "__main__":
    torchaudio.set_audio_backend("sox_io")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="../vctk_test_vad")
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--metadata_path", type=Path)
    parser.add_argument("--eps", type=float, default=0.005)
    parser.add_argument("--n_steps", type=int, default=1000)
    main(**vars(parser.parse_args()))
