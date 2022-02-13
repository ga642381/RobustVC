import argparse
import os

model_types = ["S2VC-robust", "AdaIN-VC-robust", "S2VC", "AdaIN-VC"]
ROOT_DIR


def make_metadata(args):
    n_samples = args.n
    # === #
    cmd = f"python make_metadata.py VCTK ../assets/vctk_test_vad VCTK ../assets/vctk_test_vad -n {n_samples} -nt 1 -o ./metadata --s_seed 261 --t_seed 444"
    print("[Command]", cmd)
    os.system(cmd)


def inference(args):
    datasets = args.dataset
    model = args.model
    model_name = args.model_name
    feat_type = args.feat_type
    vocoder_name = args.vocoder_name
    # === #
    for dataset in datasets:
        if feat_type is not None:
            cmd = f"python inference.py -m metadata/VCTK_to_VCTK.json -s ../assets/vctk_test_{dataset} -t ../assets/vctk_test_{dataset} -o ./result/{dataset}_data -r models/any2any/{model} --model_name {model_name} --feat_type {feat_type} --vocoder_name {vocoder_name}"
        else:
            cmd = f"python inference.py -m metadata/VCTK_to_VCTK.json -s ../assets/vctk_test_{dataset} -t ../assets/vctk_test_{dataset} -o ./result/{dataset}_data -r models/any2any/{model} --model_name {model_name} --vocoder_name {vocoder_name}"
        print("[Command]", cmd)
        os.system(cmd)


def inference_attack(args):
    model = args.model
    datasets = args.dataset
    model_name = args.model_name
    vocoder_name = args.vocoder_name
    # === #
    for dataset in datasets:
        cmd = f"python inference.py -m metadata/VCTK_to_VCTK.json -s ../assets/vctk_test_vad -t ../assets/vctk_test_{dataset} -o ./result/{dataset}_data -r models/any2any/{model} --model_name {model_name} --vocoder_name {vocoder_name}"
        print("[Command]", cmd)
        os.system(cmd)


def metric(args):
    metrics = args.metric
    datasets = args.dataset
    model = args.model
    model_name = args.model_name
    mapping = {
        "MOS": "mean_opinion_score",
        "CER": "character_error_rate",
        "SVAR": "speaker_verification",
    }
    # === #
    if "MOS" in metrics:
        for dataset in datasets:
            cmd = f"python calculate_objective_metric.py -d ./result/{dataset}_data/{model}/{model_name}/VCTK2VCTK/ -r metrics/mean_opinion_score -o ./result/{dataset}_data/{model}/{model_name}"
            print("[Command]", cmd)
            os.system(cmd)

    if "CER" in metrics:
        for dataset in datasets:
            cmd = f"python calculate_objective_metric.py -d ./result/{dataset}_data/{model}/{model_name}/VCTK2VCTK/ -r metrics/character_error_rate -o ./result/{dataset}_data/{model}/{model_name}"
            print("[Command]", cmd)
            os.system(cmd)

    if "SVAR" in metrics:
        for dataset in datasets:
            cmd = f"python calculate_objective_metric.py -d ./result/{dataset}_data/{model}/{model_name}/VCTK2VCTK/ -r metrics/speaker_verification -o ./result/{dataset}_data/{model}/{model_name} -t ../assets/vctk_test_vad --th metrics/speaker_verification/equal_error_rate/VCTK_eer.yaml"
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
    parser_inference.add_argument("--model", type=str)
    parser_inference.add_argument("--model_name", type=str, default="model.ckpt")
    parser_inference.add_argument(
        "--feat_type", type=str, default=None
    )  # for S2VC only
    parser_inference.add_argument("--vocoder_name", type=str, default="vocoder.pt")
    parser_inference.set_defaults(func=inference)

    # inference_attack
    parser_inference_atk = subparsers.add_parser("inference_attack")
    parser_inference_atk.add_argument("--dataset", nargs="+")
    parser_inference_atk.add_argument("--model", type=str)
    parser_inference_atk.add_argument("--model_name", type=str, default="model.ckpt")
    parser_inference_atk.add_argument("--vocoder_name", type=str, default="vocoder.pt")
    parser_inference_atk.set_defaults(func=inference_attack)

    # metrics
    parser_metric = subparsers.add_parser("metric")
    parser_metric.add_argument("--metric", nargs="+")
    parser_metric.add_argument("--dataset", nargs="+")
    parser_metric.add_argument("--model", type=str)
    parser_metric.add_argument("--model_name", type=str, default="model.ckpt")
    parser_metric.set_defaults(func=metric)

    # args
    args = parser.parse_args()
    args.func(args)
    # print(args)
