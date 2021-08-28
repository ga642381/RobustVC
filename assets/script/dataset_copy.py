from utils.VCTK_split import train_valid_test
from pathlib import Path
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    return vars(parser.parse_args())


def main(split, data_dir, save_dir):
    # dir
    data_dir = Path(data_dir).resolve()
    save_dir = Path(save_dir).resolve()
    assert data_dir != save_dir, f"data_dir and save_dir should not be the same!"
    assert data_dir.exists(), f"{data_dir} does not exist!"

    # copy
    test_spks = train_valid_test[split]
    for data_type in ["txt", "wav48"]:
        Path(save_dir / data_type).mkdir(parents=True, exist_ok=True)
        for spk in test_spks:
            in_dir = data_dir / data_type / spk
            out_dir = save_dir / data_type / spk
            cmd = f"cp -r {in_dir} {out_dir}"
            print(cmd)
            os.system(cmd)


if __name__ == "__main__":
    main(**parse_args())
