import argparse
import os

model_types = ["S2VC-robust", "AdaIN-VC-robust", "S2VC", "AdaIN-VC"]
dataset = "demand"


def make_metadata(args):
    n_samples = args.n
    # === #
    cmd = f"python make_metadata.py VCTK /fortress/vc2021/robust-vc/assets/vctk_test_vad VCTK /fortress/vc2021/robust-vc/assets/vctk_test_vad -n {n_samples} -nt 1 -o ./metadata"
    print("[Command]", cmd)
    os.system(cmd)


def inference(args):
    datasets = args.dataset
    models = args.model
    model_name = args.model_name
    vocoder_name = args.vocoder_name
    # === #
    for dataset in datasets:
        for model in models:
            cmd = f"python inference.py -m metadata/VCTK_to_VCTK.json -s /fortress/vc2021/robust-vc/assets/vctk_test_{dataset} -t /fortress/vc2021/robust-vc/assets/vctk_test_{dataset} -o ./result/{dataset}_data -r models/any2any/{model} --model_name {model_name} --vocoder_anem {vocoder_name}"
            print("[Command]", cmd)
            os.system(cmd)


def metric(args):
    metrics = args.metric
    datasets = args.dataset
    models = args.model
    model_name = args.model_name
    vocoder_name = args.vocoder_name
    mapping = {
        "MOS": "mean_opinion_score",
        "CER": "character_error_rate",
        "SVAR": "speaker_verification",
    }
    # === #
    if "MOS" in metrics:
        for dataset in datasets:
            for model in models:
                cmd = f"python calculate_objective_metric.py -d ./result/{dataset}_data/{model}/VCTK2VCTK/ -r metrics/mean_opinion_score -o ./result/{dataset}_data/{model} --model_name {model_name} --vocoder_anem {vocoder_name}"
                print("[Command]", cmd)
                os.system(cmd)

    if "CER" in metrics:
        for dataset in datasets:
            for model in models:
                cmd = f"python calculate_objective_metric.py -d ./result/{dataset}_data/{model}/VCTK2VCTK/ -r metrics/character_error_rate -o ./result/{dataset}_data/{model} --model_name {model_name} --vocoder_anem {vocoder_name}"
                print("[Command]", cmd)
                os.system(cmd)

    if "SVAR" in metrics:
        for dataset in datasets:
            for model in models:
                cmd = f"python calculate_objective_metric.py -d ./result/{dataset}_data/{model}/VCTK2VCTK/ -r metrics/speaker_verification -o ./result/{dataset}_data/{model} -t /fortress/vc2021/robust-vc/assets/vctk_test_vad --th metrics/speaker_verification/equal_error_rate/VCTK_eer.yaml --model_name {model_name} --vocoder_anem {vocoder_name}"
                print("[Command]", cmd)
                os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # make metadata
    subparsers = parser.add_subparsers()
    parser_metadata = subparsers.add_parser("make_metadata")
    parser_metadata.add_argument("-n", type=int, default=250)
    parser_metadata.set_defaults(func=make_metadata)

    # inference
    parser_inference = subparsers.add_parser("inference")
    parser_inference.add_argument("--dataset", nargs="+")
    parser_inference.add_argument("--model", nargs="+")
    parser_inference.add_argument("--model_name", type=str, default="model.ckpt")
    parser_inference.add_argument("--vocoder_name", type=str, default="vocoder.pt")
    parser_inference.set_defaults(func=inference)

    # metrics
    parser_metric = subparsers.add_parser("metric")
    parser_metric.add_argument("--metric", nargs="+")
    parser_metric.add_argument("--dataset", nargs="+")
    parser_metric.add_argument("--model", nargs="+")
    parser_metric.add_argument("--model_name", type=str, default="model.ckpt")
    parser_metric.add_argument("--vocoder_name", type=str, default="vocoder.pt")
    parser_metric.set_defaults(func=metric)

    # args
    args = parser.parse_args()
    args.func(args)
    # print(args)
