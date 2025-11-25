import argparse
import os
import subprocess
import sys

from pathlib import Path

from src.utils import write_config, get_cfg_from_args


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("bcr-risk-prediction", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command. For Yacs configs, use space-separated \"PATH.KEY VALUE\" pairs. For python-based LazyConfig, use \"path.key=value\".",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="output directory to save logs and checkpoints",
    )
    return parser


def classification(root_dir, config_file):
    print(f"Running {root_dir}/train/classification.py...")
    cmd = [
        sys.executable,
        "src/train/classification.py",
        "--config-file",
        os.path.abspath(config_file),
    ]
    result = subprocess.run(cmd, cwd=root_dir)
    if result.returncode != 0:
        print("Classification failed. Exiting.")
        sys.exit(result.returncode)


def classification_multi(root_dir, config_file):
    print(f"Running {root_dir}/train/classification-multi.py...")
    cmd = [
        sys.executable,
        "src/train/classification-multi.py",
        "--config-file",
        os.path.abspath(config_file),
    ]
    result = subprocess.run(cmd, cwd=root_dir)
    if result.returncode != 0:
        print("Multi-fold classification training failed. Exiting.")
        sys.exit(result.returncode)


def regression(root_dir, config_file):
    print(f"Running {root_dir}/train/regression.py...")
    cmd = [
        sys.executable,
        "src/train/regression.py",
        "--config-file",
        os.path.abspath(config_file),
    ]
    result = subprocess.run(cmd, cwd=root_dir)
    if result.returncode != 0:
        print("Regression failed. Exiting.")
        sys.exit(result.returncode)


def regression_multi(root_dir, config_file):
    print(f"Running {root_dir}/train/regression-multi.py...")
    cmd = [
        sys.executable,
        "src/train/regression-multi.py",
        "--config-file",
        os.path.abspath(config_file),
    ]
    result = subprocess.run(cmd, cwd=root_dir)
    if result.returncode != 0:
        print("Multi-fold regression training failed. Exiting.")
        sys.exit(result.returncode)


def survival(root_dir, config_file):
    print(f"Running {root_dir}/train/survival.py...")
    cmd = [
        sys.executable,
        "src/train/survival.py",
        "--config-file",
        os.path.abspath(config_file),
    ]
    result = subprocess.run(cmd, cwd=root_dir)
    if result.returncode != 0:
        print("Survival training failed. Exiting.")
        sys.exit(result.returncode)


def survival_multi(root_dir, config_file):
    print(f"Running {root_dir}/train/survival-multi.py...")
    cmd = [
        sys.executable,
        "src/train/survival-multi.py",
        "--config-file",
        os.path.abspath(config_file),
    ]
    result = subprocess.run(cmd, cwd=root_dir)
    if result.returncode != 0:
        print("Multi-fold survival training failed. Exiting.")
        sys.exit(result.returncode)


def main(args):

    cfg = get_cfg_from_args(args)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    config_file = write_config(cfg, output_dir)

    multi_fold = False
    if cfg.data.fold_dir is not None:
        multi_fold = True

    root_dir = "hipt"

    if cfg.task == "classification":
        if multi_fold:
            classification_multi(root_dir, config_file)
        else:
            classification(root_dir, config_file)
    elif cfg.task == "regression":
        if multi_fold:
            regression_multi(root_dir, config_file)
        else:
            regression(root_dir, config_file)
    elif cfg.task == "survival":
        if multi_fold:
            survival_multi(root_dir, config_file)
        else:
            survival(root_dir, config_file)
    else:
        print(f"Unsupported task: {cfg.task}. Exiting.")
        sys.exit(1)


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()
    main(args)
