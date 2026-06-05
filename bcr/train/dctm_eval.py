from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.utils.metrics import get_metrics


TIME_EPS = 1e-7


@dataclass(frozen=True)
class DCTMHorizon:
    label: str
    quantile: float
    time: float
    normalized_time: float


def serialize_horizons(horizons: list[DCTMHorizon]) -> list[dict[str, float | str]]:
    return [
        {
            "label": h.label,
            "quantile": h.quantile,
            "time": h.time,
            "normalized_time": h.normalized_time,
        }
        for h in horizons
    ]


def deserialize_horizons(
    horizons: list[DCTMHorizon] | list[dict[str, float | str]],
) -> list[DCTMHorizon]:
    if not horizons:
        return []
    if isinstance(horizons[0], DCTMHorizon):
        return horizons
    return [
        DCTMHorizon(
            label=str(h["label"]),
            quantile=float(h["quantile"]),
            time=float(h["time"]),
            normalized_time=float(h["normalized_time"]),
        )
        for h in horizons
    ]


def normalize_time(event_time: np.ndarray, max_time: float) -> np.ndarray:
    """Normalize event times to the DCTM Bernstein basis interval."""
    if max_time <= 0:
        raise ValueError("max_time must be positive")
    return np.clip(np.asarray(event_time, dtype=float) / max_time, 0.0, 1.0 - TIME_EPS)


def horizon_label(quantile: float) -> str:
    return f"q{int(round(float(quantile) * 100))}"


def compute_train_event_horizons(
    train_df: pd.DataFrame,
    time_col: str,
    quantiles: Iterable[float],
    max_time: float,
    censored_col: str = "censored",
) -> list[DCTMHorizon]:
    event_mask = train_df[censored_col].to_numpy(dtype=float) == 0
    event_times = train_df.loc[event_mask, time_col].to_numpy(dtype=float)
    if event_times.size == 0:
        raise ValueError("Cannot compute DCTM horizons without uncensored events")

    horizons = []
    for quantile in quantiles:
        q = float(quantile)
        time = float(np.quantile(event_times, q))
        norm = float(normalize_time(np.array([time]), max_time)[0])
        horizons.append(
            DCTMHorizon(
                label=horizon_label(q),
                quantile=q,
                time=time,
                normalized_time=norm,
            )
        )
    return horizons


def dctm_survival_from_transform(
    transform: torch.Tensor | np.ndarray,
    family: str,
) -> torch.Tensor | np.ndarray:
    if family == "logistic":
        if isinstance(transform, torch.Tensor):
            return 1 - torch.sigmoid(transform)
        return 1 - (1 / (1 + np.exp(-transform)))
    if family == "gompertz":
        if isinstance(transform, torch.Tensor):
            return torch.exp(-torch.exp(transform))
        return np.exp(-np.exp(transform))
    raise ValueError(f"Unsupported DCTM family: {family}")


def concordance_index_at_horizon(
    risk: np.ndarray,
    event_times: np.ndarray,
    events: np.ndarray,
    horizon: float,
) -> float:
    risk = np.asarray(risk, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    events = np.asarray(events, dtype=bool) & (event_times <= float(horizon))

    concordant = 0.0
    comparable = 0.0
    n = len(event_times)
    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if event_times[i] >= event_times[j]:
                continue
            comparable += 1.0
            if risk[i] > risk[j]:
                concordant += 1.0
            elif risk[i] == risk[j]:
                concordant += 0.5

    if comparable == 0:
        return float("nan")
    return concordant / comparable


def compute_horizon_c_indices(
    risks_by_label: dict[str, np.ndarray],
    event_times: np.ndarray,
    events: np.ndarray,
    horizons: dict[str, float],
) -> dict[str, float]:
    metrics = {}
    for label, horizon in horizons.items():
        metrics[f"c-index/{label}"] = concordance_index_at_horizon(
            risks_by_label[label],
            event_times=event_times,
            events=events,
            horizon=horizon,
        )
    return metrics


def _censoring_km_values(
    train_time: np.ndarray,
    train_event: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    train_time = np.asarray(train_time, dtype=float)
    censoring_event = 1 - np.asarray(train_event, dtype=int)
    query_times = np.asarray(query_times, dtype=float)

    unique_times = np.sort(np.unique(train_time[censoring_event == 1]))
    surv = 1.0
    step_times = []
    step_surv = []
    for time in unique_times:
        at_risk = np.sum(train_time >= time)
        censored_at_time = np.sum((train_time == time) & (censoring_event == 1))
        if at_risk > 0:
            surv *= 1 - (censored_at_time / at_risk)
        step_times.append(time)
        step_surv.append(surv)

    values = np.ones_like(query_times, dtype=float)
    if not step_times:
        return values

    step_times = np.asarray(step_times)
    step_surv = np.asarray(step_surv)
    for i, query_time in enumerate(query_times):
        idx = np.searchsorted(step_times, query_time, side="right") - 1
        if idx >= 0:
            values[i] = step_surv[idx]
    return np.clip(values, 1e-6, None)


def compute_ibs(
    train_time: np.ndarray,
    train_event: np.ndarray,
    eval_time: np.ndarray,
    eval_event: np.ndarray,
    survival: np.ndarray,
    eval_grid: np.ndarray,
) -> float:
    eval_time = np.asarray(eval_time, dtype=float)
    eval_event = np.asarray(eval_event, dtype=bool)
    eval_grid = np.asarray(eval_grid, dtype=float)
    survival = np.asarray(survival, dtype=float)

    if eval_grid.size == 0 or survival.size == 0:
        return float("nan")
    if eval_grid.size == 1:
        scale = 1.0
    else:
        scale = eval_grid[-1] - eval_grid[0]
        if scale <= 0:
            scale = 1.0

    brier_scores = []
    for col, time in enumerate(eval_grid):
        pred_survival = survival[:, col]
        g_time = _censoring_km_values(train_time, train_event, np.array([time]))[0]
        observed_before = (eval_time <= time) & eval_event
        still_at_risk = eval_time > time

        score = np.zeros_like(eval_time, dtype=float)
        if observed_before.any():
            g_observed = _censoring_km_values(
                train_time, train_event, eval_time[observed_before]
            )
            score[observed_before] = pred_survival[observed_before] ** 2 / g_observed
        if still_at_risk.any():
            score[still_at_risk] = (1 - pred_survival[still_at_risk]) ** 2 / g_time
        brier_scores.append(float(np.mean(score)))

    if eval_grid.size == 1:
        return brier_scores[0]
    return float(np.trapz(brier_scores, eval_grid) / scale)


def make_ibs_grid(max_time: float, num_times: int) -> np.ndarray:
    if num_times <= 0:
        raise ValueError("num_times must be positive")
    upper = max(float(max_time), TIME_EPS)
    return np.linspace(TIME_EPS, upper, num_times)


def add_horizon_risks_to_frame(
    df: pd.DataFrame,
    idxs: list[int],
    risks_by_label: dict[str, Iterable[float]],
    risk_alias_label: str,
) -> None:
    for label, risks in risks_by_label.items():
        df.loc[idxs, f"risk_{label}"] = list(risks)
    if risk_alias_label in risks_by_label:
        df.loc[idxs, "risk"] = list(risks_by_label[risk_alias_label])


def compute_dctm_metrics(
    metric_names: list[str],
    risks_by_label: dict[str, np.ndarray],
    event_times: np.ndarray,
    events: np.ndarray,
    horizons: list[DCTMHorizon],
    risk_alias_label: str | None = None,
    survival: np.ndarray | None = None,
    survival_times: np.ndarray | None = None,
    train_event_times: np.ndarray | None = None,
    train_events: np.ndarray | None = None,
    compute_ibs_metric: bool = False,
) -> dict[str, float]:
    results = {}
    should_compute_c_index = "c-index" in metric_names or not set(metric_names) & {
        "c-index"
    }
    if should_compute_c_index:
        if risk_alias_label is not None and risk_alias_label in risks_by_label:
            results.update(
                get_metrics(
                    ["c-index"],
                    preds=risks_by_label[risk_alias_label],
                    labels=event_times,
                    event_indicator=[bool(e) for e in events],
                )
            )
        results.update(
            compute_horizon_c_indices(
                risks_by_label=risks_by_label,
                event_times=event_times,
                events=events,
                horizons={h.label: h.time for h in horizons},
            )
        )

    if compute_ibs_metric:
        if (
            survival is None
            or survival_times is None
            or train_event_times is None
            or train_events is None
        ):
            raise ValueError("IBS requires survival curves, grid times, and train labels")
        results["ibs"] = compute_ibs(
            train_time=train_event_times,
            train_event=train_events,
            eval_time=event_times,
            eval_event=events,
            survival=survival,
            eval_grid=survival_times,
        )
    return results
