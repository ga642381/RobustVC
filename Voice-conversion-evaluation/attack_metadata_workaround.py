import json
from pathlib import Path

from parsers.VCTK_parser import Parser

metadata = json.load(open("metadata/VCTK_to_VCTK_.json"))

target_dir = "/fortress/vc2021/robust-vc/assets/vctk_test_vad"
atk_target_parser = Parser(target_dir)

n_target_samples = metadata["n_target_samples"]
for i, pair in enumerate(metadata["pairs"]):
    target_speaker_id = pair["target_speaker"]
    atk_target_wavs, atk_target_speaker_id = atk_target_parser.sample_targets(
        n_target_samples, target_speaker_id
    )
    metadata["pairs"][i]["atk_target_speaker"] = atk_target_speaker_id
    metadata["pairs"][i]["atk_tgt_utts"] = atk_target_wavs

json.dump(metadata, Path("metadata/VCTK_to_VCTK.json").open("w"), indent=2)
