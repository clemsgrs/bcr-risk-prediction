"""CV-ensemble inference for discrete-bin survival HIPT runs.

Loads each fold's best checkpoint (bare state_dict), runs hipt's
`inference` (risk = -sum(survival across bins)) on a shared test set,
reports per-fold + mean ± std + ensemble c-index.
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from bcr.eval.aggregate import aggregate_and_save, find_fold_dirs, resolve_id_column
from bcr.eval.dataset import InferenceSurvivalDataset
from hipt.src.data.dataset import DatasetOptions
from src.models import ModelFactory
from src.utils.survival_utils import inference as inference_survival


def get_args():
    p = argparse.ArgumentParser("discrete-survival-inference")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument(
        "--features-dir",
        default=None,
        help="Override cfg.features_dir (pre-extracted .pt features for the test set)",
    )
    p.add_argument(
        "--label-name",
        default=None,
        help="Override cfg.label_name if the test CSV uses a different event-time column name",
    )
    return p.parse_args()


def main():
    args = get_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(run_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    features_dir = Path(args.features_dir) if args.features_dir else Path(cfg.features_dir)
    label_name = args.label_name or cfg.label_name
    print(f"Loading features from: {features_dir}")
    print(f"Event-time column: {label_name}")
    test_df = pd.read_csv(args.test_csv)
    id_col = resolve_id_column(test_df)
    if "case_id" not in test_df.columns:
        test_df["case_id"] = test_df[id_col]  # dataset keys on case_id
    if "censored" not in test_df.columns:
        if "event" not in test_df.columns:
            raise SystemExit("test CSV must contain either 'censored' or 'event'")
        test_df["censored"] = 1 - test_df["event"].astype(int)

    fold_dirs = find_fold_dirs(run_dir / "checkpoints")
    if not fold_dirs:
        raise SystemExit(f"No fold checkpoints under {run_dir/'checkpoints'}")
    print(f"Found {len(fold_dirs)} fold(s): {[f.name for f in fold_dirs]}")

    per_fold_c: dict[str, float] = {}
    per_fold_risks: list[pd.Series] = []

    for fold_dir in fold_dirs:
        fold_tag = fold_dir.name
        ckpt_path = fold_dir / f"{args.checkpoint}.pt"
        print(f"\n=== {fold_tag}: {ckpt_path} ===")
        state_dict = torch.load(ckpt_path, map_location=device)

        ds_opts = DatasetOptions(
            df=test_df.copy(),
            features_dir=features_dir,
            label_name=label_name,
            label_mapping=cfg.label_mapping,
        )
        dataset = InferenceSurvivalDataset(ds_opts)

        model = ModelFactory(
            level=cfg.model.level,
            num_classes=cfg.num_classes,
            options=cfg.model,
        ).get_model().to(device)
        model.load_state_dict(state_dict)

        results = inference_survival(
            model,
            dataset,
            metric_names=cfg.metrics,
            batch_size=1,
            num_workers=num_workers,
            device=device,
        )

        fold_out = output_dir / fold_tag
        fold_out.mkdir(parents=True, exist_ok=True)
        dataset.df.to_csv(fold_out / "test.csv", index=False)

        c = results.get("c-index", float("nan"))
        per_fold_c[fold_tag] = c
        per_fold_risks.append(
            dataset.df.set_index(id_col)["risk"].rename(fold_tag)
        )
        print(f"  c-index: {c:.5f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_and_save(
        {"risk": (per_fold_c, per_fold_risks)},
        test_df,
        label_name,
        output_dir,
        id_col=id_col,
    )


if __name__ == "__main__":
    main()
