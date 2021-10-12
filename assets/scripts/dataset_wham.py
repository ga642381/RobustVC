import argparse
import csv
import os
import random
from functools import partial
from math import ceil, sqrt
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

import librosa
import torch
import torchaudio
from librosa.util import find_files
from torch import Tensor
from torchaudio.sox_effects import apply_effects_tensor
from tqdm import tqdm

SNR = [2.5, 7.5, 12.5, 17.5]
CHANNELS = [1, 2]


def mix_noise(speech_tensor: Tensor, noise_tensor: Tensor, snr_val: float):
    speech_tensor = speech_tensor.squeeze(0)
    noise_tensor = noise_tensor.squeeze(0)
    if len(noise_tensor) <= len(speech_tensor):
        dup_num = int(ceil(len(speech_tensor) / len(noise_tensor))) + 1
        noise_tensor = noise_tensor.repeat(dup_num)

    start = random.randint(0, len(noise_tensor) - len(speech_tensor) - 1)
    noise_tensor = noise_tensor[start : (start + len(speech_tensor))]

    snr_exp = 10.0 ** (snr_val / 10.0)
    speech_var = speech_tensor.dot(speech_tensor)
    noise_var = noise_tensor.dot(noise_tensor)
    scalar = sqrt(speech_var / (snr_exp * noise_var))

    return (speech_tensor + scalar * noise_tensor).unsqueeze(0)


def process_save_wav(wav_file, processor, data_dir, noise_files: list, save_dir):
    # wav
    wav_tensor, sr1 = torchaudio.load(wav_file)

    # noise
    # !TODO resampling can be conducted off-line
    snr = random.choice(SNR)
    channel = random.choice(CHANNELS)
    noise_filename = random.choice(noise_files)
    noise_tensor, sr2 = torchaudio.load(noise_filename)
    assert sr2 == 48000
    assert sr1 == 16000
    effects = [["rate", str(sr1)], ["remix", str(channel)]]
    noise_tensor, sr2_ = apply_effects_tensor(
        noise_tensor, sr2, effects, channels_first=True
    )
    assert sr1 == sr2_, "input wav and DEMAND should have the same sampling rate"

    # add noise
    noisy_wav = processor(wav_tensor, noise_tensor, snr)

    # save
    output_path = wav_file.replace(str(data_dir), str(save_dir))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_path, noisy_wav, sample_rate=sr1)


def main(data_dir, noise_dir, save_dir):
    # dir
    data_dir = Path(data_dir).resolve()
    noise_dir = Path(noise_dir).resolve()
    save_dir = Path(save_dir).resolve()
    assert data_dir != save_dir, f"data_dir and save_dir should not be the same!"
    assert data_dir.exists(), f"{data_dir} does not exist!"
    assert noise_dir.exists(), f"{noise_dir} does not exist!"
    print(f"[INFO] Task : Adding Noise from Noise Dataset WHAM!")
    print(f"[INFO] data_dir : {data_dir}")
    print(f"[INFO] noise_dir : {noise_dir}")
    print(f"[INFO] save_dir : {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # find wav files
    wav_files = librosa.util.find_files(data_dir)
    print(f"[INFO] {len(wav_files)} wav files found in {data_dir}")

    # find noise wav files
    noise_files = []
    metadata = noise_dir / "high_res_metadata.csv"
    with open(metadata) as f:
        rows = csv.reader(f)
        for row in rows:
            if row[-3] == "Test":
                noise_files.append(noise_dir / "audio" / row[0])

    print(f"[INFO] {len(wav_files)} Noisy wav files (Test) found in {noise_dir}")

    # add noise with multiporcessing
    file_to_noisy_file = partial(
        process_save_wav,
        processor=mix_noise,
        data_dir=data_dir,
        noise_files=noise_files,
        save_dir=save_dir,
    )
    N_processes = cpu_count()
    print(f"[INFO] Start multiprocessing with {N_processes} processes")

    with Pool(processes=N_processes) as pool:
        for _ in tqdm(pool.imap(file_to_noisy_file, wav_files), total=len(wav_files)):
            pass

    # copy text files
    txt_in_dir = data_dir / "txt"
    txt_out_dir = save_dir / "txt"
    cmd = f"cp -r {txt_in_dir} {txt_out_dir}"
    print(f'[INFO] Copying text files with command : "{cmd}"')
    os.system(cmd)


if __name__ == "__main__":
    torchaudio.set_audio_backend("sox_io")
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("noise_dir", type=Path)
    parser.add_argument("save_dir", type=Path)
    main(**vars(parser.parse_args()))
