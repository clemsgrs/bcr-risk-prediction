"""Shared CV-ensemble aggregation for inference scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored


def resolve_id_column(df: pd.DataFrame) -> str:
    has_slide = "slide_id" in df.columns
    has_case = "case_id" in df.columns
    if has_slide and has_case:
        raise ValueError("test CSV must contain only one of slide_id / case_id, not both")
    if not (has_slide or has_case):
        raise ValueError("test CSV must contain either slide_id or case_id")
    return "slide_id" if has_slide else "case_id"


def find_fold_dirs(checkpoint_root: Path) -> list[Path]:
    folds = sorted(p for p in checkpoint_root.glob("fold-*") if p.is_dir())
    if not folds:
        folds = sorted(p for p in checkpoint_root.glob("fold_*") if p.is_dir())
    return folds


def censored_c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            event.astype(bool),
            time.astype(float),
            risk.astype(float),
            tied_tol=1e-8,
        )[0]
    )


def aggregate_and_save(
    per_horizon: dict[str, tuple[dict[str, float], list[pd.Series]]],
    test_df: pd.DataFrame,
    label_name: str,
    output_dir: Path,
    id_col: str = "slide_id",
) -> None:
    """Write summary.csv + ensemble.csv across one or more risk scores.

    `per_horizon` maps a label (e.g. 'q100' for DCTM, 'risk' for discrete) to
    a tuple of (per_fold_c_index, per_fold_risk_series).
    """
    summary_rows: list[dict] = []
    ensemble_columns: dict[str, pd.Series] = {}

    for horizon, (per_fold_c, per_fold_risks) in per_horizon.items():
        c_values = np.array(list(per_fold_c.values()), dtype=float)
        mean_c = float(np.nanmean(c_values))
        std_c = float(np.nanstd(c_values, ddof=1)) if len(c_values) > 1 else 0.0

        risk_df = pd.concat(per_fold_risks, axis=1)
        ensemble_risk = risk_df.mean(axis=1)
        aligned = test_df.set_index(id_col).loc[ensemble_risk.index]
        ens_time = aligned[label_name].to_numpy(dtype=float)
        ens_event = 1 - aligned["censored"].to_numpy(dtype=float)
        ensemble_c = censored_c_index(ensemble_risk.to_numpy(), ens_time, ens_event)

        for fold, c in per_fold_c.items():
            summary_rows.append({"horizon": horizon, "fold": fold, "c_index": c})
        summary_rows.append({"horizon": horizon, "fold": "mean", "c_index": mean_c})
        summary_rows.append({"horizon": horizon, "fold": "std", "c_index": std_c})
        summary_rows.append({"horizon": horizon, "fold": "ensemble", "c_index": ensemble_c})

        for col in risk_df.columns:
            ensemble_columns[f"{horizon}_{col}"] = risk_df[col]
        ensemble_columns[f"{horizon}_ensemble"] = ensemble_risk

    ensemble_df = pd.DataFrame(ensemble_columns).join(
        test_df.set_index(id_col)[[label_name, "censored"]]
    )
    ensemble_df.to_csv(output_dir / "ensemble.csv")

    summary_df = pd.DataFrame(summary_rows)
    summary_wide = summary_df.pivot(index="fold", columns="horizon", values="c_index")
    fold_rows = sorted(
        [f for f in summary_wide.index if f not in ("mean", "std", "ensemble")]
    )
    summary_wide = summary_wide.loc[fold_rows + ["mean", "std", "ensemble"]]
    summary_wide.to_csv(output_dir / "summary.csv")

    print()
    print(summary_wide.round(5).to_string())
    print(f"\nWrote {output_dir}/summary.csv and ensemble.csv")
