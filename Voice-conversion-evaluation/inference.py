"""Inference one utterance."""
import importlib
import json
import logging
import warnings
from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)-s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--metadata_path", type=str, help="inference metadata path"
    )
    parser.add_argument("-s", "--source_dir", type=str, help="source dir path")
    parser.add_argument("-t", "--target_dir", type=str, help="target dir path")
    parser.add_argument("-o", "--output_dir", type=str, help="output wav path")
    parser.add_argument("-r", "--root", type=str, help="the model dir")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--reload_dir", type=str, help="reload dir path")
    parser.add_argument("--model_name", type=str, default="model.ckpt")
    parser.add_argument("--feat_type", type=str, default=None)
    parser.add_argument("--vocoder_name", type=str, default="vocoder.pt")

    return vars(parser.parse_args())


def conversion(inferencer, device, root, metadata, source_dir, target_dir, output_dir):
    """Do conversion and save the output of voice conversion model."""
    metadata["vc_model"] = root
    mel_output_dir = output_dir / "mel_files"
    mel_output_dir.mkdir(parents=True, exist_ok=True)

    conv_mels = []
    for pair in tqdm(metadata["pairs"]):
        # Sometimes the wav is too noisy, the sox vad will trim the whole utterence.
        # Then we ignore that sample
        conv_mel = inferencer.inference_from_pair(pair, source_dir, target_dir)
        if conv_mel is None:
            pair["converted"] = "failed"

        else:
            # Only for one target utterence, because AdaIN-VC only support one source and one target now
            prefix = Path(pair["src_utt"]).stem
            postfix = Path(pair["tgt_utts"][0]).stem
            file_path = mel_output_dir / f"{prefix}_to_{postfix}.pt"
            pair["mel_path"] = f"mel_files/{prefix}_to_{postfix}.pt"
            pair["converted"] = "placeholder"

            conv_mel = conv_mel.detach()
            torch.save(conv_mel, file_path)
            conv_mels.append(conv_mel.to(device))

    metadata["pairs"] = metadata.pop("pairs")
    # metadata_output_path = output_dir / "metadata.json"
    # json.dump(metadata, metadata_output_path.open("w"), indent=2)

    return metadata, conv_mels


def reload_from_numpy(device, metadata, reload_dir):
    """Reload the output of voice conversion model."""
    conv_mels = []
    for pair in tqdm(metadata["pairs"]):
        file_path = Path(reload_dir) / pair["mel_path"]
        conv_mel = torch.load(file_path)
        conv_mels.append(conv_mel.to(device))
    return metadata, conv_mels


def main(
    metadata_path,
    source_dir,
    target_dir,
    output_dir,
    root,
    batch_size,
    reload,
    reload_dir,
    model_name,
    vocoder_name,
    **kwargs,
):
    """Main function"""

    # setting up
    inferencer_path = str(Path(root) / "inferencer").replace("/", ".")
    Inferencer = getattr(importlib.import_module(inferencer_path), "Inferencer")
    if kwargs["feat_type"] is not None:
        inferencer = Inferencer(root, model_name, vocoder_name, **kwargs)
    else:
        inferencer = Inferencer(root, model_name, vocoder_name)
    device = inferencer.device
    sample_rate = inferencer.sample_rate
    logger.info("Inferencer is loaded from %s.", root)

    metadata = json.load(open(metadata_path))
    logger.info("Metadata list is loaded from %s.", metadata_path)

    output_dir = (
        Path(output_dir)
        / Path(root).stem
        / model_name
        / f"{metadata['source_corpus']}2{metadata['target_corpus']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Voice Conversion (Mel-Spectrogram)
    if reload:
        metadata, conv_mels = reload_from_numpy(device, metadata, reload_dir)
    else:
        metadata, conv_mels = conversion(
            inferencer, device, root, metadata, source_dir, target_dir, output_dir
        )

    # Mel-Spectrogram -> Wavform
    n_samples = len(conv_mels)
    waveforms = []
    max_memory_use = conv_mels[0].size(0) * batch_size
    with torch.no_grad():
        pbar = tqdm(total=n_samples)
        left = 0
        while left < n_samples:
            batch_size = max_memory_use // conv_mels[left].size(0) - 1
            right = left + min(batch_size, n_samples - left)
            waveforms.extend(inferencer.spectrogram2waveform(conv_mels[left:right]))
            pbar.update(batch_size)
            left += batch_size

        pbar.close()

    # Save Wavforms
    success_metadata = list(
        filter(lambda pair: pair["converted"] != "failed", metadata["pairs"])
    )
    assert len(success_metadata) == len(waveforms)

    for pair, waveform in tqdm(zip(success_metadata, waveforms), total=len(waveforms)):
        waveform = waveform.detach().cpu().numpy()

        prefix = Path(pair["src_utt"]).stem
        postfix = Path(pair["tgt_utts"][0]).stem  # only for one target
        file_path = output_dir / f"{prefix}_to_{postfix}.wav"
        pair["converted"] = f"{prefix}_to_{postfix}.wav"

        sf.write(file_path, waveform, sample_rate, subtype="FLOAT")

    # Save metadata
    metadata_output_path = output_dir / "metadata.json"
    json.dump(metadata, metadata_output_path.open("w"), indent=2)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
