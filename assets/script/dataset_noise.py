import argparse
from random import sample
import librosa
import torch
import torchaudio
from tqdm import tqdm
from utils.noise import WavAug
from pathlib import Path
import os
from functools import partial
from multiprocessing import Pool, cpu_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--out_sample_rate", type=int, default=16000)
    return vars(parser.parse_args())


def create_noisy_wav(wav_file, wavaug, data_dir, save_dir, sr):
    output_path = wav_file.replace(str(data_dir), str(save_dir))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    wav, _ = librosa.load(wav_file, sr=sr)
    noisy_wav = wavaug.add_noise(torch.tensor(wav))
    torchaudio.save(output_path, noisy_wav, sample_rate=sr)


def main(data_dir, save_dir, out_sample_rate):
    # dir
    data_dir = Path(data_dir).resolve()
    save_dir = Path(save_dir).resolve()
    assert data_dir != save_dir, f"data_dir and save_dir should not be the same!"
    assert data_dir.exists(), f"{data_dir} does not exist!"
    print(f"[INFO] data_dir : {data_dir}")
    print(f"[INFO] save_dir : {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # find wav files
    wav_files = librosa.util.find_files(data_dir)
    print(f"[INFO] {len(wav_files)} wav files found in {data_dir}")

    # add noise with multiporcessing
    wavaug = WavAug(sample_rate=out_sample_rate, mode="test")
    file_to_noisy_file = partial(
        create_noisy_wav,
        wavaug=wavaug,
        data_dir=data_dir,
        save_dir=save_dir,
        sr=out_sample_rate,
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
    main(**parse_args())
