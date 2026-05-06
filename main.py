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


def survival(root_dir, config_file, output_dir):
    root_dir = Path(root_dir).resolve()
    print(f"Running {root_dir}/train/survival.py...")
    cmd = [
        sys.executable,
        "src/train/survival.py",
        "--config-file",
        os.path.abspath(config_file),
        "--output-dir",
        os.path.abspath(output_dir),
        "task=survival",
    ]
    result = subprocess.run(cmd, cwd=root_dir, env=build_training_env(root_dir))
    if result.returncode != 0:
        print("Survival training failed. Exiting.")
        sys.exit(result.returncode)


def survival_multi(root_dir, config_file, output_dir):
    root_dir = Path(root_dir).resolve()
    print(f"Running {root_dir}/train/survival-multi.py...")
    cmd = [
        sys.executable,
        "src/train/survival-multi.py",
        "--config-file",
        os.path.abspath(config_file),
        "--output-dir",
        os.path.abspath(output_dir),
        "task=survival",
    ]
    result = subprocess.run(cmd, cwd=root_dir, env=build_training_env(root_dir))
    if result.returncode != 0:
        print("Multi-fold survival training failed. Exiting.")
        sys.exit(result.returncode)


def survival_dctm(root_dir, config_file, output_dir):
    """Run DCTM survival training (single-fold)."""
    print(f"Running {root_dir}/src/train/survival_dctm.py...")
    cmd = [
        sys.executable,
        "src/train/survival_dctm.py",
        "--config-file",
        os.path.abspath(config_file),
        "--output-dir",
        os.path.abspath(output_dir),
    ]
    result = subprocess.run(cmd, cwd=root_dir)
    if result.returncode != 0:
        print("DCTM survival training failed. Exiting.")
        sys.exit(result.returncode)


def survival_dctm_multi(root_dir, config_file, output_dir):
    """Run DCTM survival training (multi-fold cross-validation)."""
    print(f"Running {root_dir}/src/train/survival_dctm_multi.py...")
    cmd = [
        sys.executable,
        "src/train/survival_dctm_multi.py",
        "--config-file",
        os.path.abspath(config_file),
        "--output-dir",
        os.path.abspath(output_dir),
    ]
    result = subprocess.run(cmd, cwd=root_dir)
    if result.returncode != 0:
        print("Multi-fold DCTM survival training failed. Exiting.")
        sys.exit(result.returncode)


def build_training_env(root_dir: Path):
    env = os.environ.copy()
    root_dir = str(root_dir.resolve())
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{root_dir}{os.pathsep}{pythonpath}" if pythonpath else root_dir
    )
    return env


def main(args):

    cfg = get_cfg_from_args(args)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    config_file = write_config(cfg, output_dir)

    multi_fold = False
    if cfg.data.fold_dir is not None:
        multi_fold = True

    # Check if DCTM is enabled
    use_dctm = cfg.get("dctm", {}).get("enable", False)

    if use_dctm:
        # Run DCTM training from main repo
        root_dir = Path(__file__).resolve().parent
        if multi_fold:
            survival_dctm_multi(root_dir, config_file, output_dir)
        else:
            survival_dctm(root_dir, config_file, output_dir)
    else:
        # Run original training from hipt submodule
        root_dir = Path(__file__).resolve().parent / "hipt"
        if multi_fold:
            survival_multi(root_dir, config_file, output_dir)
        else:
            survival(root_dir, config_file, output_dir)


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()
    main(args)
