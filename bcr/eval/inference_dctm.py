"""Run each CV fold's best checkpoint on a shared held-out test set.

Reports per-fold c-index, mean ± std across folds, and an ensemble c-index
computed from per-slide risks averaged across folds.
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from bcr.eval.aggregate import (
    aggregate_and_save,
    censored_c_index,
    find_fold_dirs,
    resolve_id_column,
)
from bcr.train.dctm_eval import deserialize_horizons, horizon_label
from bcr.train.survival_dctm import (
    create_model,
    get_dctm_evaluation_options,
    inference_dctm,
)
from bcr.eval.dataset import InferenceSurvivalDataset
from hipt.src.data.dataset import DatasetOptions


def get_args():
    p = argparse.ArgumentParser("dctm-inference")
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
    eval_options = get_dctm_evaluation_options(cfg)
    risk_alias_label = horizon_label(eval_options["risk_alias_quantile"])

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

    # horizon label -> {fold_tag: c_index}, horizon label -> [risk Series per fold]
    per_horizon_c: dict[str, dict[str, float]] = {}
    per_horizon_risks: dict[str, list[pd.Series]] = {}

    for fold_dir in fold_dirs:
        fold_tag = fold_dir.name
        ckpt_path = fold_dir / f"{args.checkpoint}.pt"
        print(f"\n=== {fold_tag}: {ckpt_path} ===")
        ckpt = torch.load(ckpt_path, map_location=device)

        max_time = float(ckpt["max_time"])
        horizons = deserialize_horizons(ckpt["horizons"])

        ds_opts = DatasetOptions(
            df=test_df.copy(),
            features_dir=features_dir,
            label_name=label_name,
            label_mapping=cfg.label_mapping,
        )
        dataset = InferenceSurvivalDataset(ds_opts)

        model = create_model(cfg, device)
        model.load_state_dict(ckpt["model_state_dict"])

        results = inference_dctm(
            model,
            dataset,
            max_time,
            metric_names=cfg.metrics,
            horizons=horizons,
            risk_alias_label=risk_alias_label,
            train_event_times=None,
            train_events=None,
            compute_ibs_metric=False,  # IBS skipped — only c-index requested
            ibs_times=None,
            batch_size=1,
            num_workers=num_workers,
            device=device,
        )

        fold_out = output_dir / fold_tag
        fold_out.mkdir(parents=True, exist_ok=True)
        dataset.df.to_csv(fold_out / "test.csv", index=False)

        # Compute overall c-index for each horizon's risk score
        time_arr = dataset.df[label_name].to_numpy(dtype=float)
        event_arr = 1 - dataset.df["censored"].to_numpy(dtype=float)
        for h in horizons:
            risk_col = f"risk_{h.label}"
            risk_arr = dataset.df[risk_col].to_numpy(dtype=float)
            c = censored_c_index(risk_arr, time_arr, event_arr)
            per_horizon_c.setdefault(h.label, {})[fold_tag] = c
            per_horizon_risks.setdefault(h.label, []).append(
                dataset.df.set_index(id_col)[risk_col].rename(fold_tag)
            )
            print(f"  c-index {h.label}: {c:.5f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_horizon = {
        label: (per_horizon_c[label], per_horizon_risks[label])
        for label in per_horizon_c
    }
    aggregate_and_save(per_horizon, test_df, label_name, output_dir, id_col=id_col)


if __name__ == "__main__":
    main()
