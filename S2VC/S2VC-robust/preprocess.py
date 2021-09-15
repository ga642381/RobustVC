#!/usr/bin/env python3
"""Precompute Wav2Vec features."""

import json
import os
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import mkstemp

import torch
import tqdm
from torch.utils.data import DataLoader

from data import PreprocessDataset
from data.feature_extract import FeatureExtractor
from models import load_pretrained_wav2vec


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("clean_data_dir", type=str)
    parser.add_argument("noisy_data_dir", type=str)
    parser.add_argument("feature_name", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--wav2vec_path", type=str, default=None)
    parser.add_argument(
        "--trim_method", choices=["librosa", "vad", "None"], default="None"
    )
    parser.add_argument("--n_workers", type=int, default=cpu_count())
    parser.add_argument("--sample_rate", type=int, default=16000)

    return vars(parser.parse_args())


def main(
    clean_data_dir,
    noisy_data_dir,
    feature_name,
    wav2vec_path,
    save_dir,
    trim_method,
    n_workers,
    sample_rate,
    **kwargs,
):
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === output path === #
    # dirs
    clean_data_dir = Path(clean_data_dir)
    noisy_data_dir = Path(noisy_data_dir)
    save_dir = Path(save_dir)
    assert clean_data_dir.exists(), f"{clean_data_dir} does not exist!"
    assert noisy_data_dir.exists(), f"{noisy_data_dir} does not exist!"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Task : Preprocessing for S2VC training")
    print(f"[INFO] clean_data_dir : {clean_data_dir}")
    print(f"[INFO] noisy_data_dir : {noisy_data_dir}")
    print(f"[INFO] save_dir : {save_dir}")

    # === dataset === #
    dataset = PreprocessDataset(
        clean_data_dir, noisy_data_dir, trim_method, sample_rate
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=n_workers
    )

    # === speaker === #
    speaker_infos = {}
    speaker_infos["feature_name"] = feature_name

    # === feature === #
    mapping = {
        "apc": "fbank",
        "timit_posteriorgram": "fbank",
        "cpc": "cpc_mel",
        "wav2vec2": "wav2vec2_mel",
    }
    feat_extractor = FeatureExtractor(feature_name, wav2vec_path, device)
    mel_extractor = FeatureExtractor(mapping[feature_name], wav2vec_path, device)

    # === loop dataloader === #
    pbar = tqdm.tqdm(total=len(dataset), ncols=0)
    for (
        speaker_name,
        clean_audio_path,
        noisy_audio_path,
        clean_wav,
        noisy_wav,
    ) in dataloader:

        if clean_wav.size(-1) < 10:
            continue

        if noisy_wav.size(-1) < 10:
            continue

        speaker_name = speaker_name[0]
        clean_audio_path = clean_audio_path[0]
        noisy_audio_path = noisy_audio_path[0]
        clean_wav = clean_wav.to(device)
        noisy_wav = noisy_wav.to(device)

        with torch.no_grad():
            clean_mel = mel_extractor.get_feature(clean_wav)[0]

        fd, temp_file = mkstemp(suffix=".tar", prefix="utterance-", dir=save_dir)

        # === save wav tensor === #
        torch.save(
            {
                "clean_wav": clean_wav.detach().cpu(),
                "noisy_wav": noisy_wav.detach().cpu(),
                "clean_mel": clean_mel.detach().cpu(),
            },
            temp_file,
        )
        os.close(fd)

        if speaker_name not in speaker_infos.keys():
            speaker_infos[speaker_name] = []

        speaker_infos[speaker_name].append(
            {
                "feature_path": Path(temp_file).name,
                "clean_audio_path": clean_audio_path,
                "noisy_audio_path": noisy_audio_path,
                "mel_len": len(clean_mel),
            }
        )

        pbar.update(dataloader.batch_size)

    # === metadata === #
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(speaker_infos, f, indent=2)


if __name__ == "__main__":
    main(**parse_args())
