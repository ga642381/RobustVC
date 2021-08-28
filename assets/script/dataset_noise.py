import argparse
from random import sample
import librosa
import torch
import torchaudio
from tqdm import tqdm
from utils.noise import WavAug
from pathlib import Path
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--out_sample_rate", type=int, default=16000)
    return vars(parser.parse_args())


def main(data_dir, save_dir, out_sample_rate):
    # dir
    data_dir = Path(data_dir).resolve()
    save_dir = Path(save_dir).resolve()
    assert data_dir != save_dir, f"data_dir and save_dir should not be the same!"
    assert data_dir.exists(), f"{data_dir} does not exist!"
    save_dir.mkdir(parents=True, exist_ok=True)

    # find wav files
    wav_files = librosa.util.find_files(data_dir)
    print(f"[INFO] {len(wav_files)} wav files found in {data_dir}")

    # add noise
    wavaug = WavAug(sample_rate=out_sample_rate, mode="test")
    for wav_file in tqdm(wav_files):
        output_path = wav_file.replace(str(data_dir), str(save_dir))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        wav, sr = librosa.load(wav_file, sr=out_sample_rate)
        noisy_wav = wavaug.add_noise(torch.tensor(wav))
        torchaudio.save(output_path, noisy_wav, sample_rate=out_sample_rate)

    # copy text files
    txt_in_dir = data_dir / "txt"
    txt_out_dir = save_dir / "txt"
    cmd = f"cp -r {txt_in_dir} {txt_out_dir}"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main(**parse_args())
