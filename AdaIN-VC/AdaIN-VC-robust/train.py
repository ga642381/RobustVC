import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from data import InfiniteDataLoader, SpeakerDataset, infinite_iterator
from data.VCTK_split import train_valid_test
from data.wav2mel import Wav2Mel
from model import AdaINVC


def emb_attack(
    model: nn.Module,
    x: Tensor,
    wav2mel: Wav2Mel,
    eps: float = 0.005,
    alpha: float = 0.001,
    n_uttrs: int = 4,
):
    ptb = torch.empty_like(x).uniform_(-eps, eps).requires_grad_(True)
    with torch.no_grad():
        org_emb = model.speaker_encoder(wav2mel.mel(x))
        tgt_emb = org_emb.roll(n_uttrs, 0)
    adv_emb = model.speaker_encoder(wav2mel.mel(x + ptb))
    loss = F.mse_loss(adv_emb, tgt_emb) - 0.1 * F.mse_loss(adv_emb, org_emb)
    loss.backward()
    grad = ptb.grad.data
    ptb = ptb.data - alpha * grad.sign()
    adv_x = x + torch.clamp(ptb, min=-eps, max=eps)
    # adv_x = x + torch.max(torch.min(ptb, eps), -eps)
    return wav2mel.mel(adv_x.detach())


def main(
    config_file: str,
    data_dir: str,
    save_dir: str,
    n_steps: int,
    save_steps: int,
    log_steps: int,
    n_spks: int,
    n_uttrs: int,
    clean_ratio: float,
    adv_ratio: float,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # Load config
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    # Prepare data
    train_set = SpeakerDataset(
        "train",
        train_valid_test["train"],
        data_dir,
        sample_rate=16000,
        clean_wav_ratio=clean_ratio,
    )
    valid_set = SpeakerDataset(
        "valid",
        train_valid_test["valid"],
        data_dir,
        sample_rate=16000,
        clean_wav_ratio=clean_ratio,
    )

    print(f"[INFO] train_set spks : {len(train_set)}")
    print(f"[INFO] valid_set spks : {len(valid_set)}")

    # construct loader
    train_loader = InfiniteDataLoader(
        train_set, batch_size=n_spks, shuffle=True, num_workers=10
    )
    valid_loader = InfiniteDataLoader(
        valid_set, batch_size=n_spks, shuffle=True, num_workers=10
    )

    # construct iterator
    train_iter = infinite_iterator(train_loader)
    valid_iter = infinite_iterator(valid_loader)

    # Build model
    wav2mel = Wav2Mel(sox_effect=False).to(device)
    model = AdaINVC(config["Model"]).to(device)
    model = torch.jit.script(model)

    # Optimizer
    opt = torch.optim.Adam(
        model.parameters(),
        lr=config["Optimizer"]["lr"],
        betas=(config["Optimizer"]["beta1"], config["Optimizer"]["beta2"]),
        amsgrad=config["Optimizer"]["amsgrad"],
        weight_decay=config["Optimizer"]["weight_decay"],
    )

    # Tensorboard logger
    writer = SummaryWriter(save_dir)
    criterion = nn.L1Loss()
    pbar = trange(n_steps, ncols=0)
    valid_steps = 32

    for step in pbar:
        # get features
        clean_wavs, noisy_wavs = next(train_iter)
        clean_wavs = clean_wavs.flatten(0, 1).to(device)
        noisy_wavs = noisy_wavs.flatten(0, 1).to(device)

        clean_mels = wav2mel.mel(clean_wavs)

        if random.random() < adv_ratio:
            noisy_mels = emb_attack(model, noisy_wavs, wav2mel, n_uttrs=n_uttrs)
        else:
            noisy_mels = wav2mel.mel(noisy_wavs)

        # reconstruction
        mu, log_sigma, emb, rec_mels = model(noisy_mels)

        # compute loss
        rec_loss = criterion(rec_mels, clean_mels)
        kl_loss = 0.5 * (log_sigma.exp() + mu ** 2 - 1 - log_sigma).mean()
        rec_lambda = config["Lambda"]["rec"]
        kl_lambda = min(
            config["Lambda"]["kl"] * step / config["Lambda"]["kl_annealing"],
            config["Lambda"]["kl"],
        )
        loss = rec_lambda * rec_loss + kl_lambda * kl_loss

        # update parameters
        opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        opt.step()

        # save model and optimizer
        if (step + 1) % save_steps == 0:
            model_path = os.path.join(save_dir, f"model-{step + 1}.ckpt")
            model.cpu()
            model.save(model_path)
            model.to(device)
            opt_path = os.path.join(save_dir, f"opt-{step + 1}.ckpt")
            torch.save(opt.state_dict(), opt_path)

        # validation
        if (step + 1) % log_steps == 0:
            model.eval()
            valid_loss = 0
            for _ in range(valid_steps):
                clean_wavs, noisy_wavs = next(valid_iter)
                clean_wavs = clean_wavs.flatten(0, 1).to(device)
                noisy_wavs = noisy_wavs.flatten(0, 1).to(device)

                clean_mels = wav2mel.mel(clean_wavs)
                noisy_mels = wav2mel.mel(noisy_wavs)

                mu, log_sigma, emb, rec_mels = model(noisy_mels)

                loss = criterion(rec_mels, clean_mels)
                valid_loss += loss.item()

            valid_loss /= valid_steps
            model.train()

            # record information
            writer.add_scalar("training/rec_loss", rec_loss, step + 1)
            writer.add_scalar("training/kl_loss", kl_loss, step + 1)
            writer.add_scalar("training/grad_norm", grad_norm, step + 1)
            writer.add_scalar("lambda/kl", kl_lambda, step + 1)
            writer.add_scalar("validation/rec_loss", valid_loss, step + 1)

        # update tqdm bar
        pbar.set_postfix({"rec_loss": rec_loss.item(), "kl_loss": kl_loss.item()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--n_steps", type=int, default=int(1e6))
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--log_steps", type=int, default=250)
    parser.add_argument("--n_spks", type=int, default=32)
    parser.add_argument("--n_uttrs", type=int, default=4)
    parser.add_argument("--clean_ratio", type=float, default=0.4)
    parser.add_argument("--adv_ratio", type=float, default=0.5)
    main(**vars(parser.parse_args()))
