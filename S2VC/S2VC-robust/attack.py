import argparse
from pathlib import Path

from yaml import load

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from data import load_wav
from data.feature_extract import FeatureExtractor


def extract_ref_feats(model: nn.Module, refs: Tensor):
    """
    refs shape : [6, 256, 557]
    masks shape: [6, 557]
    ---
    ref3 shape : [6, 512, 557]
    lens shape : [6] (tensor([235, 129,  89, 189, 471, 557], device='cuda:0'))
    """
    ref1 = model.unet.conv1(refs)
    ref2 = model.unet.conv2(F.relu(ref1))
    ref3 = model.unet.conv3(F.relu(ref2))
    spk_emb = ref3.mean(dim=-1)
    # cum_sum = ref3.cumsum(dim=-1)
    # lens = (~masks).sum(dim=-1)
    # assert lens.ndim == 1
    # tmp = torch.stack([cum_sum[i, :, lens[i] - 1] for i in range(len(cum_sum))])
    # spk_emb = tmp / lens[:, None]
    return spk_emb


def emb_attack(
    model: nn.Module,
    feat_extractor: FeatureExtractor,
    src_wavs: Tensor,
    tgt_wavs: Tensor,
    eps: float,
    n_steps: int,
    ) -> Tensor:
    feat_extractor.extractor.train()
    src_wavs = src_wavs[None, :]
    tgt_wavs = tgt_wavs[None, :]
    ptbs = [
        torch.empty_like(src_wav).randn_(0, 1).requires_grad_(True)
        for src_wav in src_wavs
    ]
    atk_src_wavs = [src_wav + eps * ptb.tanh() for src_wav, ptb in zip(src_wavs, ptbs)]
    ptb_feats = feat_extractor.get_feature(atk_src_wavs)

    src_feats = feat_extractor.get_feature(src_wavs)
    tgt_feats = feat_extractor.get_feature(tgt_wavs)


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ptbs, lr=1e-4)

    with torch.no_grad():
        src_emb = extract_ref_feats(model, src_feats)
        tgt_emb = extract_ref_feats(model, tgt_feats)
    pbar = trange(n_steps)
    for _ in pbar:
        ptb_feats = pad_sequence(ptb_feats, batch_first=True).transpose(1, 2)
        adv_emb = extract_ref_feats(model, ptb_feats)
        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, src_emb)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        adv_wavs = [
            src_wav + eps * ptb.tanh()
            for src_wav, ptb in zip(src_wavs, ptbs)
        ]
        return adv_wavs


def main(
    src_path: Path,
    tgt_path: Path,
    out_path: Path,
    model_path: Path,
    feat_type: str,
    wav2vec_path: Path,
    eps: float,
    n_steps: int,
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path)
    feat_extractor = FeatureExtractor(feat_type, wav2vec_path, device)
    src_wav = load_wav(src_path, 16000, trim=True)
    tgt_wav = load_wav(tgt_path, 16000, trim=True)
    src_wav = torch.from_numpy(src_wav)
    tgt_wav = torch.from_numpy(tgt_wav)
    assert src_wav.ndim == tgt_wav.ndim == 1
    adv_wav = emb_attack(
                model.to(device),
                feat_extractor,
                src_wav.to(device),
                tgt_wav.to(device),
                eps,
                n_steps,
                )
    assert adv_wav.ndim == 2
    torchaudio.save(out_path, adv_wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_path", type=Path)
    parser.add_argument("tgt_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("feat_type", type=str)
    parser.add_argument("wav2vec_path", type=Path)
    parser.add_argument("--eps", type=float, default=0.005)
    parser.add_argument("--n_steps", type=int, default=1000)
    main(**vars(parser.parse_args()))
