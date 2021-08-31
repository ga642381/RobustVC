"""
This file aims to do the preprocessing for on the fly(otf) "noisy training"
This preprocessing script do the following job:
* resample the dataset to 16k Hz 
* trim the beginning and ending silence of the utterence with sox vad
* create metadata.json
"""

import argparse
import json
import os
from functools import partial
from uuid import uuid4

import librosa
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchaudio
from torch import Tensor
from tqdm.auto import tqdm

from data.wav2mel import Wav2Mel
from data.wav2mel import SoxEffects


def process_files(audio_file: str, process_nn: nn.Module) -> Tensor:
    speech_tensor, sample_rate = torchaudio.load(audio_file)
    processed_tensor = process_nn(speech_tensor, sample_rate)

    return processed_tensor


def main(data_dir: str, save_dir: str, segment: int):
    mp.set_sharing_strategy("file_system")
    os.makedirs(save_dir, exist_ok=True)
    sox_effects = SoxEffects(sample_rate=16000, norm_db=-3)
    file2wav = partial(process_files, process_nn=sox_effects)

    meta_data = {}
    speakers = sorted(os.listdir(data_dir))

    for spk in tqdm(speakers):
        spk_dir = os.path.join(data_dir, spk)
        wav_files = librosa.util.find_files(spk_dir)
        wavs = [file2wav(wav_file) for wav_file in wav_files]
        wavs = list(filter(lambda x: x is not None and x.shape[-1] > segment, wavs))
        rnd_paths = [f"{uuid4().hex}.pt" for _ in range(len(wavs))]
        dummy = [
            torch.save(wav, os.path.join(save_dir, path))
            for (wav, path) in zip(wavs, rnd_paths)
        ]
        meta_data[spk] = rnd_paths

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--segment", type=int, default=25600)
    main(**vars(parser.parse_args()))
