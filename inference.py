"""CV-ensemble inference launcher.

Routes to discrete-bin or DCTM inference based on the trained run's config,
mirroring train.py's dispatch on `cfg.dctm.enable`.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("bcr-inference", add_help=add_help)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training output dir (contains checkpoints/fold-*/best.pt and config.yaml)",
    )
    parser.add_argument(
        "--test-csv",
        required=True,
        help="Shared held-out test CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <run-dir>/inference/<test-csv stem>/",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint stem under each fold dir (best | latest | epoch_<n>)",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Override cfg.features_dir for the test set's pre-extracted features",
    )
    parser.add_argument(
        "--label-name",
        default=None,
        help="Override cfg.label_name if the test CSV uses a different event-time column name",
    )
    return parser


def build_env(root_dir: Path):
    env = os.environ.copy()
    project_dir = str(root_dir.resolve())
    hipt_dir = str((root_dir / "hipt").resolve())
    dctm_dir = str((root_dir / "DCTM").resolve())
    pythonpath = env.get("PYTHONPATH", "")
    parts = [project_dir, hipt_dir, dctm_dir] + [
        p for p in pythonpath.split(os.pathsep) if p
    ]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(parts))
    return env


def resolve_output_dir(args) -> str:
    if args.output_dir:
        return os.path.abspath(args.output_dir)
    test_stem = Path(args.test_csv).stem
    return str(Path(args.run_dir).resolve() / "inference" / test_stem)


def run(script: str, args, root_dir: Path):
    cmd = [
        sys.executable,
        script,
        "--run-dir", os.path.abspath(args.run_dir),
        "--test-csv", os.path.abspath(args.test_csv),
        "--output-dir", resolve_output_dir(args),
        "--checkpoint", args.checkpoint,
    ]
    if args.features_dir:
        cmd += ["--features-dir", os.path.abspath(args.features_dir)]
    if args.label_name:
        cmd += ["--label-name", args.label_name]
    print(f"Running {script}...")
    result = subprocess.run(cmd, cwd=root_dir, env=build_env(root_dir))
    sys.exit(result.returncode)


def main():
    args = get_args_parser().parse_args()
    root_dir = Path(__file__).resolve().parent

    cfg = OmegaConf.load(Path(args.run_dir).resolve() / "config.yaml")
    use_dctm = bool(cfg.get("dctm", {}).get("enable", False))

    if use_dctm:
        run("bcr/eval/inference_dctm.py", args, root_dir)
    else:
        run("bcr/eval/inference.py", args, root_dir)


if __name__ == "__main__":
    main()
