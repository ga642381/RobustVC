#!/usr/bin/env python3
"""Train S2VC model."""

import argparse
import datetime
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import IntraSpeakerDataset, collate_pseudo, plot_attn, train_valid_test
from data.feature_extract import FeatureExtractor
from models import S2VC, get_cosine_schedule_with_warmup

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--clean_ratio", type=float, default=0.4)  #
    parser.add_argument("--adv_ratio", type=float, default=0.5)  #
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--valid_steps", type=int, default=5000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--accu_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("-s", "--src_feat", type=str, default="cpc")
    parser.add_argument("-r", "--tgt_feat", type=str, default="cpc")
    parser.add_argument("--lr_reduction", action="store_true")
    parser.add_argument("--comment", type=str)
    parser.add_argument("--wav2vec_path", type=str, default=None)

    return vars(parser.parse_args())


def extract_tgt_feats(model: nn.Module, tgts: Tensor, masks: Tensor):
    """
    tgts shape : [6, 256, 557]
    masks shape: [6, 557]
    ---
    tgt3 shape : [6, 512, 557]
    lens shape : [6] (tensor([235, 129,  89, 189, 471, 557], device='cuda:0'))
    """
    tgt1 = model.unet.conv1(tgts)
    tgt2 = model.unet.conv2(F.relu(tgt1))
    tgt3 = model.unet.conv3(F.relu(tgt2))
    cum_sum = tgt3.cumsum(dim=-1)
    lens = (~masks).sum(dim=-1)
    assert lens.ndim == 1
    tmp = torch.stack([cum_sum[i, :, lens[i] - 1] for i in range(len(cum_sum))])
    spk_emb = tmp / lens[:, None]
    return spk_emb


def permute_spks(spk_lst):
    spk_idx = defaultdict(list)
    for idx, spk in enumerate(spk_lst):
        spk_idx[spk].append(idx)

    if len(spk_idx) == 1:
        tmp = list(range(len(spk_lst)))
        return tmp[1:] + tmp[0]

    final_lst = list(range(len(spk_lst)))
    for spk, idx in spk_idx.items():
        other_spks = [s for s in spk_idx.keys() if s != spk]
        for i in idx:
            tgt_spk = random.choice(other_spks)
            tgt_idx = random.choice(spk_idx[tgt_spk])
            final_lst[i] = tgt_idx
    return final_lst


def emb_attack(
    model,
    feat_extractor,
    tgts,
    tgt_wavs,
    masks,
    spks,
    eps=0.05,
    alpha=0.001,
):
    feat_extractor.extractor.train()
    # ptb = torch.empty_like(tgt_wavs).uniform_(-eps, eps).requires_grad_(True)
    ptbs = [
        torch.empty_like(tgt_wav).uniform_(-eps, eps).requires_grad_(True)
        for tgt_wav in tgt_wavs
    ]
    atk_tgt_wavs = [tgt_wav + ptb for tgt_wav, ptb in zip(tgt_wavs, ptbs)]
    ptb_feats = feat_extractor.get_feature(atk_tgt_wavs)

    ptb_feats = pad_sequence(ptb_feats, batch_first=True).transpose(1, 2)

    with torch.no_grad():
        org_emb = extract_tgt_feats(model, tgts, masks)
        # perform some roll
        ptb_index = permute_spks(spks)
        tgt_emb = org_emb[ptb_index]
    adv_emb = extract_tgt_feats(model, ptb_feats, masks)
    loss = F.mse_loss(adv_emb, tgt_emb) - 0.1 * F.mse_loss(adv_emb, org_emb)
    loss.backward()

    ptbs = [ptb.data - alpha * ptb.grad.data.sign() for ptb in ptbs]
    with torch.no_grad():
        adv_wavs = [
            tgt_wav + torch.clamp(ptb, min=-eps, max=eps)
            for tgt_wav, ptb in zip(tgt_wavs, ptbs)
        ]
        adv_feats = feat_extractor.get_feature(adv_wavs)
        adv_feats = pad_sequence(adv_feats, batch_first=True).transpose(1, 2)
    feat_extractor.extractor.eval()
    return adv_feats


def model_fn(
    batch,
    model,
    adv_ratio,
    criterion,
    device,
    src_feat_extractor=None,
    tgt_feat_extractor=None,
):
    """Forward a batch through model."""
    (spks, ground_mels, src_wavs, tgt_wavs) = batch

    # === on the fly feature extraction === #
    # ! we are using the same feature for both src and tgt
    src_wavs = [src_wav.to(device) for src_wav in src_wavs]
    tgt_wavs = [tgt_wav.to(device) for tgt_wav in tgt_wavs]

    srcs = src_feat_extractor.get_feature(src_wavs)
    # tgts = tgt_feat_extractor.get_feature(tgt_wavs)
    tgts = srcs

    # === original collate function === #
    src_lens = [len(src) for src in srcs]
    tgt_lens = [len(tgt) for tgt in tgts]
    ground_mel_lens = [len(ground_mel) for ground_mel in ground_mels]

    overlap_lens = [
        min(src_len, ground_mel_len)
        for src_len, ground_mel_len in zip(src_lens, ground_mel_lens)
    ]

    srcs = pad_sequence(srcs, batch_first=True)  # (batch, max_src_len, mel_dim)

    src_masks = [torch.arange(srcs.size(1)) >= src_len for src_len in src_lens]
    src_masks = torch.stack(src_masks)

    tgts = pad_sequence(tgts, batch_first=True, padding_value=-20)
    tgts = tgts.transpose(1, 2)  # (batch, mel_dim, max_tgt_len)

    tgt_masks = [torch.arange(tgts.size(2)) >= tgt_len for tgt_len in tgt_lens]
    tgt_masks = torch.stack(tgt_masks)  # (batch, max_tgt_len)

    ground_mels = pad_sequence(ground_mels, batch_first=True, padding_value=-20)
    ground_mels = ground_mels.transpose(1, 2)  # (batch, mel_dim, max_tgt_len)

    # srcs, src_masks, tgts, tgt_masks, ground_mels, overlap_lens
    # srcs : (batch, max_src_len, mel_dim)
    # tgts : (batch, mel_dim, max_tgt_len)
    # src_masks : (batch, max_src_len) //False if value
    # tgt_masks : (batch, max_tgt_len) //False if value
    # ground_mels : (batch, mel_dim, max_mel_len)
    # overlap_lens : list ; len(overlaps_lens) == batch_size
    srcs = srcs.detach()
    tgts = tgts.detach()
    ground_mels.detach()
    src_masks = src_masks.to(device)
    tgt_masks = tgt_masks.to(device)
    ground_mels = ground_mels.to(device)
    # === model forward === #
    # adversarial attack
    if random.random() < adv_ratio:
        advs = emb_attack(model, tgt_feat_extractor, tgts, tgt_wavs, tgt_masks, spks)
        outs, attns = model(srcs, advs, src_masks=src_masks, ref_masks=tgt_masks)
    else:
        outs, attns = model(srcs, tgts, src_masks=src_masks, ref_masks=tgt_masks)

    # === losses === #
    losses = []
    for out, ground_mel, attn, overlap_len in zip(
        outs.unbind(), ground_mels.unbind(), attns[-1], overlap_lens
    ):
        loss = criterion(out[:, :overlap_len], ground_mel[:, :overlap_len])
        losses.append(loss)

    # === attention === #
    try:
        attns_plot = []
        for i in range(len(attns)):
            attns_plot.append(attns[i][0][: overlap_lens[0], : overlap_lens[0]])
    except:
        pass

    return sum(losses) / len(losses), attns_plot


def valid(dataloader, model, criterion, device, feature_extractor):
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, attns = model_fn(
                batch, model, 0, criterion, device, feature_extractor, feature_extractor
            )
            running_loss += loss.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(loss=f"{running_loss / (i+1):.2f}")

    pbar.close()
    model.train()

    return running_loss / len(dataloader), attns


def main(
    data_dir,
    save_dir,
    clean_ratio,
    adv_ratio,
    total_steps,
    warmup_steps,
    valid_steps,
    log_steps,
    save_steps,
    accu_steps,
    batch_size,
    n_workers,
    src_feat,
    tgt_feat,
    lr_reduction,
    comment,
    wav2vec_path,
):
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_path = Path(data_dir) / "metadata.json"

    # !we assume src_feat == tgt_feat fow now
    assert src_feat == tgt_feat
    feature_extractor = FeatureExtractor(tgt_feat, wav2vec_path, device=device)

    # === dataset === #
    trainset = IntraSpeakerDataset(
        "train",
        train_valid_test["train"],
        data_dir,
        metadata_path,
        src_feat,
        tgt_feat,
        clean_wav_ratio=clean_ratio,
        wav2vec_path=wav2vec_path,
    )

    validset = IntraSpeakerDataset(
        "valid",
        train_valid_test["valid"],
        data_dir,
        metadata_path,
        src_feat,
        tgt_feat,
        clean_wav_ratio=clean_ratio,
        wav2vec_path=wav2vec_path,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_pseudo,
    )

    valid_loader = DataLoader(
        validset,
        batch_size=batch_size * accu_steps,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        # shuffle to make the plot on tensorboard differenct
        shuffle=True,
        collate_fn=collate_pseudo,
    )
    train_iterator = iter(train_loader)

    # === model === #
    if src_feat == "cpc" and tgt_feat == "cpc":
        src_dim = 256
        tgt_dim = 256
    elif src_feat == "wav2vec2" and tgt_feat == "wav2vec2":
        src_dim = 768
        tgt_dim = 768
    ground_mel_dim = 80

    print(
        f"[INFO] Source dim: {src_dim}, Target dim: {tgt_dim}, Ground Mel dim: {ground_mel_dim}"
    )
    model = S2VC(src_dim, tgt_dim).to(device)
    model = torch.jit.script(model)

    # === log === #
    if comment is not None:
        log_dir = "logs/"
        log_dir += datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_dir += "_" + comment
        writer = SummaryWriter(log_dir)

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # === hparams === #
    learning_rate = 5e-5
    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_loss = float("inf")
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        batch_loss = 0.0

        for _ in range(accu_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            loss, attns = model_fn(
                batch,
                model,
                adv_ratio,
                criterion,
                device,
                feature_extractor,
                feature_extractor,
            )
            loss = loss / accu_steps
            batch_loss += loss.item()
            loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        pbar.update()
        pbar.set_postfix(loss=f"{batch_loss:.2f}", step=step + 1)

        if step % log_steps == 0 and comment is not None:
            writer.add_scalar("Loss/train", batch_loss, step)
            try:
                attn = [attns[i].unsqueeze(0) for i in range(len(attns))]
                figure = plot_attn(attn, save=False)
                writer.add_figure(f"Image/Train-Attentions.png", figure, step + 1)
            except:
                pass

        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_loss, attns = valid(
                valid_loader, model, criterion, device, feature_extractor
            )

            if comment is not None:
                writer.add_scalar("Loss/valid", valid_loss, step + 1)
                try:
                    attn = [attns[i].unsqueeze(0) for i in range(len(attns))]
                    figure = plot_attn(attn, save=False)
                    writer.add_figure(f"Image/Valid-Attentions.png", figure, step + 1)
                except:
                    pass

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            loss_str = f"{best_loss:.4f}".replace(".", "dot")
            best_ckpt_name = f"retriever-best-loss{loss_str}.pt"

            loss_str = f"{valid_loss:.4f}".replace(".", "dot")
            curr_ckpt_name = f"retriever-step{step+1}-loss{loss_str}.pt"

            current_state_dict = model.state_dict()
            model.cpu()

            model.load_state_dict(best_state_dict)
            model.save(str(save_dir_path / best_ckpt_name))

            model.load_state_dict(current_state_dict)
            model.save(str(save_dir_path / curr_ckpt_name))

            model.to(device)
            pbar.write(f"Step {step + 1}, best model saved. (loss={best_loss:.4f})")

    pbar.close()


if __name__ == "__main__":
    main(**parse_args())
