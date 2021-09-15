"""
This file aims to do the preprocessing for "noisy training", the wav augmentation will be performed on the fly (otf)
This preprocessing script do the following job:
(We expected that the wavs have been sampled to 16K and added DEMAND nosie)
1. save the clean and noisy wavs in tensors  
2. create metadata.json
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn as nn
import torchaudio
from tqdm.auto import tqdm


def main(clean_data_dir: str, noisy_data_dir: str, save_dir: str, segment: int):
    # dirs
    clean_data_dir = Path(clean_data_dir)
    noisy_data_dir = Path(noisy_data_dir)
    save_dir = Path(save_dir)
    assert clean_data_dir.exists(), f"{clean_data_dir} does not exist!"
    assert noisy_data_dir.exists(), f"{noisy_data_dir} does not exist!"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Task : Preprocessing for AdaIN-VC training")
    print(f"[INFO] clean_data_dir : {clean_data_dir}")
    print(f"[INFO] noise_data_dir : {noisy_data_dir}")
    print(f"[INFO] save_dir : {save_dir}")

    # spkers
    speakers = sorted(os.listdir(clean_data_dir))
    speakers_ = sorted(os.listdir(noisy_data_dir))
    assert speakers == speakers_

    meta_data = {}
    skip_wavs = []
    for spk in tqdm(speakers):
        clean_spk_dir = clean_data_dir / spk
        noisy_spk_dir = noisy_data_dir / spk
        clean_noisy_wavs = []  # store clean/noisy wav tensors

        # find clean and noisy wav
        wav_files = os.listdir(clean_spk_dir)
        for wav_file in wav_files:
            clean_wav_path = clean_spk_dir / wav_file
            noisy_wav_path = noisy_spk_dir / wav_file
            if not noisy_wav_path.is_file():
                skip_wavs.append(wav_file)
                continue

            # wav to tensor
            clean_wav, _ = torchaudio.load(clean_wav_path)
            noisy_wav, _ = torchaudio.load(noisy_wav_path)
            if clean_wav.shape[-1] > segment and noisy_wav.shape[-1] > segment:
                clean_noisy_wavs.append({"clean": clean_wav, "noisy": noisy_wav})

        # save clean and noisy wav and create meta data
        sub_meta_data = []
        clean_rnd_paths = [f"{uuid4().hex}.pt" for _ in range(len(clean_noisy_wavs))]
        noisy_rnd_paths = [f"{uuid4().hex}.pt" for _ in range(len(clean_noisy_wavs))]
        for clean_noisy_wav, clean_rnd_path, noisy_rnd_path in zip(
            clean_noisy_wavs, clean_rnd_paths, noisy_rnd_paths
        ):
            torch.save(clean_noisy_wav["clean"], os.path.join(save_dir, clean_rnd_path))
            torch.save(clean_noisy_wav["noisy"], os.path.join(save_dir, noisy_rnd_path))
            sub_meta_data.append({"clean": clean_rnd_path, "noisy": noisy_rnd_path})

        meta_data[spk] = sub_meta_data

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta_data, f, indent=4)

    print(f"[INFO] skipped wav files number : {len(skip_wavs)}")


if __name__ == "__main__":
    torchaudio.set_audio_backend("sox_io")
    parser = argparse.ArgumentParser()
    parser.add_argument("clean_data_dir", type=str)
    parser.add_argument("noisy_data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--segment", type=int, default=25600)
    main(**vars(parser.parse_args()))
