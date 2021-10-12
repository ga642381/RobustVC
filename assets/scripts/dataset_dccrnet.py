import argparse
import os
import shutil
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from random import sample

import librosa
import soundfile as sf
import torch
import torchaudio
from asteroid.dsp.normalization import normalize_estimates
from asteroid.models import DCCRNet
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    return vars(parser.parse_args())


def process_save_wav(wav_file, processor, data_dir, save_dir, device):
    """
    "normalization" and soundfile write "subtype" are important!
    """
    output_path = wav_file.replace(str(data_dir), str(save_dir))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    noisy_wav, sr = sf.read(wav_file, dtype="float32")
    noisy_wav = torch.tensor(noisy_wav).to(device)
    # noisy_wav, sr = torchaudio.load(wav_file)
    enhanced_wav = processor(noisy_wav.unsqueeze(0)).squeeze(0)
    enhanced_wav_normalized = normalize_estimates(
        enhanced_wav.cpu().data.numpy(), noisy_wav.cpu().data.numpy()
    )
    output_wav = torch.tensor(enhanced_wav_normalized[0]).unsqueeze(0)
    torchaudio.save(output_path, output_wav, sample_rate=16000)
    # sf.write(str(output_path), enhanced_wav_normalized[0], sr, subtype="FLOAT")


def main(data_dir, save_dir):
    # dir
    data_dir = Path(data_dir).resolve()
    save_dir = Path(save_dir).resolve()
    assert data_dir != save_dir, f"data_dir and save_dir should not be the same!"
    assert data_dir.exists(), f"{data_dir} does not exist!"
    print(f"[INFO] Task : Denoise with DCCRNet")
    print(f"[INFO] data_dir : {data_dir}")
    print(f"[INFO] save_dir : {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # find wav files
    wav_files = librosa.util.find_files(data_dir)
    print(f"[INFO] {len(wav_files)} wav files found in {data_dir}")

    # denoise
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enhance_model = DCCRNet.from_pretrained(
        "JorisCos/DCCRNet_Libri1Mix_enhsingle_16k"
    ).to(device)
    print(f"[INFO] DCCRNet loaded from Asteroid!")

    file_to_clean_file = partial(
        process_save_wav,
        processor=enhance_model,
        data_dir=data_dir,
        save_dir=save_dir,
        device=device,
    )
    for wav_path in tqdm(wav_files):
        file_to_clean_file(wav_path)

    # copy text files
    txt_in_dir = data_dir / "txt"
    txt_out_dir = save_dir / "txt"
    cmd = f"cp -r {txt_in_dir} {txt_out_dir}"
    print(f'[INFO] Copying text files with command : "{cmd}"')
    os.system(cmd)


if __name__ == "__main__":
    torchaudio.set_audio_backend("sox_io")
    main(**parse_args())
