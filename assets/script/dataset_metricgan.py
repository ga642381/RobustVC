import argparse
import os
import shutil
from functools import partial
from pathlib import Path
from random import sample

import librosa
import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    return vars(parser.parse_args())


def process_save_wav(wav_file, processor, data_dir, save_dir, tmp_dir="."):
    output_path = wav_file.replace(str(data_dir), str(save_dir))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    noisy_wav = processor.load_audio(wav_file, savedir=tmp_dir).unsqueeze(0)
    enhanced_wav = processor.enhance_batch(noisy_wav, lengths=torch.tensor([1.0]))

    torchaudio.save(output_path, enhanced_wav.cpu(), sample_rate=16000)


def main(data_dir, save_dir):
    # dir
    data_dir = Path(data_dir).resolve()
    save_dir = Path(save_dir).resolve()
    assert data_dir != save_dir, f"data_dir and save_dir should not be the same!"
    assert data_dir.exists(), f"{data_dir} does not exist!"
    print(f"[INFO] Task : Denoise with Facebook Demucs")
    print(f"[INFO] data_dir : {data_dir}")
    print(f"[INFO] save_dir : {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # find wav files
    wav_files = librosa.util.find_files(data_dir)
    print(f"[INFO] {len(wav_files)} wav files found in {data_dir}")

    # denoise
    metricgan_plus_dir = Path("pretrained_models/metricgan-plus-voicebank")
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir=metricgan_plus_dir,
        run_opts={"device": "cuda"},
    )
    print(f"[INFO] MetricGAN+ loaded from speechbrain!")

    file_to_noisy_file = partial(
        process_save_wav,
        processor=enhance_model,
        data_dir=data_dir,
        save_dir=save_dir,
        tmp_dir=metricgan_plus_dir / "wavs",
    )
    for wav_path in tqdm(wav_files):
        file_to_noisy_file(wav_path)

    # copy text files
    txt_in_dir = data_dir / "txt"
    txt_out_dir = save_dir / "txt"
    cmd = f"cp -r {txt_in_dir} {txt_out_dir}"
    print(f'[INFO] Copying text files with command : "{cmd}"')
    os.system(cmd)


if __name__ == "__main__":
    torchaudio.set_audio_backend("sox_io")
    main(**parse_args())
