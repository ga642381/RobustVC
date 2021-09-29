import argparse
import json
import random
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

from utils.feature_extract import FeatureExtractor
from utils.utils import load_wav


def extract_tgt_feats(model: nn.Module, tgts: Tensor):
    tgt1 = model.unet.conv1(tgts)
    tgt2 = model.unet.conv2(F.relu(tgt1))
    tgt3 = model.unet.conv3(F.relu(tgt2))
    spk_emb = tgt3.mean(dim=-1)
    return spk_emb


def emb_attack(
    model: nn.Module,
    feat_extractor: FeatureExtractor,
    src_wavs: list,
    tgt_wavs: list,
    eps: float,
    n_steps: int,
) -> Tensor:
    feat_extractor.extractor.train()
    ptbs = [torch.randn_like(src_wav).requires_grad_(True) for src_wav in src_wavs]
    atk_src_wavs = [src_wav + eps * ptb.tanh() for src_wav, ptb in zip(src_wavs, ptbs)]

    ptb_feats = feat_extractor.get_feature(atk_src_wavs)
    src_feats = feat_extractor.get_feature(src_wavs)
    tgt_feats = feat_extractor.get_feature(tgt_wavs)

    src_feats = pad_sequence(src_feats, batch_first=True).transpose(1, 2)
    tgt_feats = pad_sequence(tgt_feats, batch_first=True).transpose(1, 2)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ptbs, lr=1e-4)
    with torch.no_grad():
        src_emb = extract_tgt_feats(model, src_feats)
        tgt_emb = extract_tgt_feats(model, tgt_feats)

    for _ in range(n_steps):
        atk_src_wavs = [
            src_wav + eps * ptb.tanh() for src_wav, ptb in zip(src_wavs, ptbs)
        ]
        ptb_feats = feat_extractor.get_feature(atk_src_wavs)
        ptb_feats = pad_sequence(ptb_feats, batch_first=True).transpose(1, 2)
        adv_emb = extract_tgt_feats(model, ptb_feats)

        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, src_emb)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        adv_wavs = [src_wav + eps * ptb.tanh() for src_wav, ptb in zip(src_wavs, ptbs)]
        return adv_wavs


# attack_and_save function modified from attack.py
def attack_and_save(
    src_path: Path,
    tgt_path: Path,
    out_path: Path,
    model: torch.ScriptModule,
    feat_extractor: FeatureExtractor,
    device,
    eps: float,
    n_steps: int,
):
    src_wav = load_wav(src_path, 16000, trim=False)
    tgt_wav = load_wav(tgt_path, 16000, trim=False)
    src_wav = torch.from_numpy(src_wav).to(device)
    tgt_wav = torch.from_numpy(tgt_wav).to(device)
    assert src_wav.ndim == tgt_wav.ndim == 1
    adv_wav = emb_attack(
        model,
        feat_extractor,
        [src_wav],
        [tgt_wav],
        eps,
        n_steps,
    )
    adv_wav = adv_wav[0].unsqueeze(0).cpu()
    assert adv_wav.ndim == 2
    torchaudio.save(out_path, adv_wav, 16000)


def main(
    data_dir: Path,
    save_dir: Path,
    model_path: Path,
    feat_type: str,
    wav2vec_path: Path,
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
    print(f"[INFO] Task : Adding Attack Noise (S2VC)")
    print(f"[INFO] model path : {model_path}")
    print(f"[INFO] metadata_path : {metadata_path}")
    print(f"[INFO] data_dir : {data_dir}")
    print(f"[INFO] save_dir : {save_dir}")

    # === init === #
    metadata = json.load(open(metadata_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path).to(device)
    # all vctk 16k and vad

    feat_extractor = FeatureExtractor(feat_type, wav2vec_path, device)
    attack_fn = partial(
        attack_and_save,
        model=model,
        feat_extractor=feat_extractor,
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
    parser.add_argument("--feat_type", type=str)
    parser.add_argument("--wav2vec_path", type=Path)
    parser.add_argument("--metadata_path", type=Path)
    parser.add_argument("--eps", type=float, default=0.005)
    parser.add_argument("--n_steps", type=int, default=1000)
    main(**vars(parser.parse_args()))
