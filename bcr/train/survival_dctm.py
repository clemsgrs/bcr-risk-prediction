"""
Single-fold survival training with DCTM head.

This script trains a HIPT model with DCTM (Deep Conditional Transformation Model)
survival head for continuous-time survival analysis.
"""

import argparse
import logging
import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
import wandb

from bcr.models.hipt_dctm import LocalHIPTWithDCTM
from bcr.train.dctm_eval import (
    DCTMHorizon,
    add_horizon_risks_to_frame,
    compute_dctm_metrics,
    compute_train_event_horizons,
    deserialize_horizons,
    horizon_label,
    make_ibs_grid,
    normalize_time,
    serialize_horizons,
)
from hipt.src.data.dataset import DatasetOptions, ExtractedFeaturesSurvivalDataset
from src.models.parameter_counts import log_trainable_parameter_breakdown
from hipt.src.utils import (
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
    compute_time,
    setup,
    update_log_dict,
)
from hipt.src.utils.train_utils import collate_features_survival

logger = logging.getLogger("bcr-risk-prediction")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("survival-dctm", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "opts",
        help='Modify config options at the end of the command. Use "path.key=value".',
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


def get_dctm_evaluation_options(cfg):
    """Read DCTM evaluation options with backward-compatible defaults."""
    dctm_cfg = cfg.get("dctm", {})
    eval_cfg = dctm_cfg.get("evaluation", {}) if dctm_cfg else {}
    return {
        "horizon_quantiles": list(
            eval_cfg.get("horizon_quantiles", [0.25, 0.5, 0.75, 1.0])
        ),
        "horizon_source": eval_cfg.get("horizon_source", "train_events"),
        "compute_ibs": bool(eval_cfg.get("compute_ibs", True)),
        "ibs_num_times": int(eval_cfg.get("ibs_num_times", 100)),
        "risk_alias_quantile": float(eval_cfg.get("risk_alias_quantile", 1.0)),
    }


def get_survival_arrays(df: pd.DataFrame, time_col: str):
    event_times = df[time_col].to_numpy(dtype=float)
    events = 1 - df["censored"].to_numpy(dtype=float)
    return event_times, events


def build_dctm_evaluation_context(cfg, train_df: pd.DataFrame, max_time: float):
    eval_options = get_dctm_evaluation_options(cfg)
    if eval_options["horizon_source"] != "train_events":
        raise ValueError("Only dctm.evaluation.horizon_source='train_events' is supported")

    horizons = compute_train_event_horizons(
        train_df,
        time_col=cfg.label_name,
        quantiles=eval_options["horizon_quantiles"],
        max_time=max_time,
    )
    risk_alias_label = horizon_label(eval_options["risk_alias_quantile"])
    if risk_alias_label not in {h.label for h in horizons}:
        raise ValueError(
            f"risk_alias_quantile={eval_options['risk_alias_quantile']} does not "
            "match any configured horizon quantile"
        )

    train_event_times, train_events = get_survival_arrays(train_df, cfg.label_name)
    max_ibs_time = max(h.time for h in horizons)
    ibs_times = make_ibs_grid(max_ibs_time, eval_options["ibs_num_times"])
    return {
        "horizons": horizons,
        "risk_alias_label": risk_alias_label,
        "compute_ibs_metric": eval_options["compute_ibs"],
        "ibs_times": ibs_times,
        "train_event_times": train_event_times,
        "train_events": train_events,
    }


def predict_dctm_evaluation_outputs(
    model: torch.nn.Module,
    x: torch.Tensor,
    horizons: list[DCTMHorizon],
    ibs_times: np.ndarray,
    max_time: float,
    compute_ibs_metric: bool,
):
    horizon_times = np.asarray([h.normalized_time for h in horizons], dtype=float)
    risks = model.predict_transform(x, horizon_times).detach().cpu().numpy()

    survival = None
    if compute_ibs_metric:
        survival_times_norm = normalize_time(ibs_times, max_time)
        survival = model.predict_survival(x, survival_times_norm).detach().cpu().numpy()

    return risks, survival


def train_dctm(
    epoch: int,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    max_time: float,
    metric_names: list[str],
    horizons: list[DCTMHorizon],
    risk_alias_label: str,
    train_event_times: np.ndarray,
    train_events: np.ndarray,
    compute_ibs_metric: bool = True,
    ibs_times: np.ndarray | None = None,
    batch_size: int = 1,
    collate_fn=partial(collate_features_survival, label_type="int"),
    gradient_accumulation: int | None = None,
    num_workers: int = 0,
    device: torch.device | None = None,
):
    """Training loop for DCTM survival model."""
    model.train()
    epoch_loss = 0
    censoring, event_times = [], []
    risks_by_label = {h.label: [] for h in horizons}
    survival_curves = []
    idxs = []
    if ibs_times is None:
        ibs_times = np.asarray([h.time for h in horizons], dtype=float)

    sampler = torch.utils.data.RandomSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Train"),
        unit=" case",
        unit_scale=batch_size,
        leave=False,
    ) as t:
        for i, batch in enumerate(t):
            idx, x, label, event_time, censored = batch
            x = x.to(device, non_blocking=True)

            # Normalize time for DCTM [0, 1]
            time_norm = normalize_time(event_time.numpy(), max_time)

            # Convert censoring to event indicator
            # DCTM: event=1 means occurred, event=0 means censored
            # Current data: censored=1 means no event, censored=0 means event
            event = (1 - censored).to(device, non_blocking=True)

            # Forward pass and compute loss
            loss = model.compute_loss(x, time_norm, event)

            loss_value = loss.item()
            epoch_loss += loss_value

            if gradient_accumulation:
                loss = loss / gradient_accumulation

            loss.backward()

            if not gradient_accumulation:
                optimizer.step()
                optimizer.zero_grad()
            elif (i + 1) % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                risks, survival = predict_dctm_evaluation_outputs(
                    model,
                    x,
                    horizons=horizons,
                    ibs_times=ibs_times,
                    max_time=max_time,
                    compute_ibs_metric=compute_ibs_metric,
                )
                for horizon, risk in zip(horizons, risks):
                    risks_by_label[horizon.label].append(float(risk))
                if survival is not None:
                    survival_curves.append(survival)

            censoring.append(censored.item())
            event_times.append(event_time.item())
            idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    add_horizon_risks_to_frame(
        dataset.df,
        idxs,
        risks_by_label,
        risk_alias_label=risk_alias_label,
    )

    event_times = np.asarray(event_times, dtype=float)
    events = 1 - np.asarray(censoring, dtype=float)
    survival_matrix = np.asarray(survival_curves) if survival_curves else None
    metrics = compute_dctm_metrics(
        metric_names,
        risks_by_label={k: np.asarray(v, dtype=float) for k, v in risks_by_label.items()},
        event_times=event_times,
        events=events,
        horizons=horizons,
        risk_alias_label=risk_alias_label,
        survival=survival_matrix,
        survival_times=ibs_times,
        train_event_times=train_event_times,
        train_events=train_events,
        compute_ibs_metric=compute_ibs_metric,
    )
    results.update(metrics)
    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results


def tune_dctm(
    epoch: int,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    max_time: float,
    metric_names: list[str],
    horizons: list[DCTMHorizon],
    risk_alias_label: str,
    train_event_times: np.ndarray,
    train_events: np.ndarray,
    compute_ibs_metric: bool = True,
    ibs_times: np.ndarray | None = None,
    batch_size: int = 1,
    collate_fn=partial(collate_features_survival, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):
    """Validation loop for DCTM survival model."""
    model.eval()
    epoch_loss = 0
    censoring, event_times = [], []
    risks_by_label = {h.label: [] for h in horizons}
    survival_curves = []
    idxs = []
    if ibs_times is None:
        ibs_times = np.asarray([h.time for h in horizons], dtype=float)

    sampler = torch.utils.data.SequentialSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Tune"),
        unit=" case",
        unit_scale=batch_size,
        leave=False,
    ) as t:
        with torch.no_grad():
            for batch in t:
                idx, x, label, event_time, censored = batch
                x = x.to(device, non_blocking=True)

                # Normalize time for DCTM [0, 1]
                time_norm = normalize_time(event_time.numpy(), max_time)

                # Convert censoring to event indicator
                event = (1 - censored).to(device, non_blocking=True)

                # Compute loss
                loss = model.compute_loss(x, time_norm, event)
                epoch_loss += loss.item()

                risks, survival = predict_dctm_evaluation_outputs(
                    model,
                    x,
                    horizons=horizons,
                    ibs_times=ibs_times,
                    max_time=max_time,
                    compute_ibs_metric=compute_ibs_metric,
                )
                for horizon, risk in zip(horizons, risks):
                    risks_by_label[horizon.label].append(float(risk))
                if survival is not None:
                    survival_curves.append(survival)
                censoring.append(censored.item())
                event_times.append(event_time.item())
                idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    add_horizon_risks_to_frame(
        dataset.df,
        idxs,
        risks_by_label,
        risk_alias_label=risk_alias_label,
    )

    event_times = np.asarray(event_times, dtype=float)
    events = 1 - np.asarray(censoring, dtype=float)
    survival_matrix = np.asarray(survival_curves) if survival_curves else None
    metrics = compute_dctm_metrics(
        metric_names,
        risks_by_label={k: np.asarray(v, dtype=float) for k, v in risks_by_label.items()},
        event_times=event_times,
        events=events,
        horizons=horizons,
        risk_alias_label=risk_alias_label,
        survival=survival_matrix,
        survival_times=ibs_times,
        train_event_times=train_event_times,
        train_events=train_events,
        compute_ibs_metric=compute_ibs_metric,
    )
    results.update(metrics)
    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def inference_dctm(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    max_time: float,
    metric_names: list[str],
    horizons: list[DCTMHorizon],
    risk_alias_label: str,
    train_event_times: np.ndarray,
    train_events: np.ndarray,
    compute_ibs_metric: bool = True,
    ibs_times: np.ndarray | None = None,
    batch_size: int = 1,
    collate_fn=partial(collate_features_survival, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):
    """Inference for DCTM survival model."""
    model.eval()
    epoch_loss = 0
    censoring, event_times = [], []
    risks_by_label = {h.label: [] for h in horizons}
    survival_curves = []
    idxs = []
    if ibs_times is None:
        ibs_times = np.asarray([h.time for h in horizons], dtype=float)

    sampler = torch.utils.data.SequentialSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc="Inference",
        unit=" case",
        unit_scale=batch_size,
        leave=True,
    ) as t:
        with torch.no_grad():
            for batch in t:
                idx, x, _, event_time, censored = batch
                x = x.to(device, non_blocking=True)

                # Normalize time for DCTM [0, 1]
                time_norm = normalize_time(event_time.numpy(), max_time)

                # Convert censoring to event indicator
                event = (1 - censored).to(device, non_blocking=True)

                # Compute loss
                loss = model.compute_loss(x, time_norm, event)
                epoch_loss += loss.item()

                risks, survival = predict_dctm_evaluation_outputs(
                    model,
                    x,
                    horizons=horizons,
                    ibs_times=ibs_times,
                    max_time=max_time,
                    compute_ibs_metric=compute_ibs_metric,
                )
                for horizon, risk in zip(horizons, risks):
                    risks_by_label[horizon.label].append(float(risk))
                if survival is not None:
                    survival_curves.append(survival)
                censoring.append(censored.item())
                event_times.append(event_time.item())
                idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    add_horizon_risks_to_frame(
        dataset.df,
        idxs,
        risks_by_label,
        risk_alias_label=risk_alias_label,
    )

    event_times = np.asarray(event_times, dtype=float)
    events = 1 - np.asarray(censoring, dtype=float)
    survival_matrix = np.asarray(survival_curves) if survival_curves else None
    metrics = compute_dctm_metrics(
        metric_names,
        risks_by_label={k: np.asarray(v, dtype=float) for k, v in risks_by_label.items()},
        event_times=event_times,
        events=events,
        horizons=horizons,
        risk_alias_label=risk_alias_label,
        survival=survival_matrix,
        survival_times=ibs_times,
        train_event_times=train_event_times,
        train_events=train_events,
        compute_ibs_metric=compute_ibs_metric,
    )
    results.update(metrics)
    results["loss"] = epoch_loss / len(loader)
    return results


def create_model(cfg, device):
    """Create HIPT model with DCTM head based on config."""
    dctm_cfg = cfg.dctm
    model = LocalHIPTWithDCTM(
        region_size=cfg.model.region_size,
        patch_size=cfg.model.patch_size,
        embed_dim_patch=cfg.model.embed_dim_patch,
        embed_dim_region=cfg.model.embed_dim_region,
        embed_dim_slide=cfg.model.embed_dim_slide,
        dropout=cfg.model.dropout,
        mask_attn=cfg.model.mask_attn,
        num_register_tokens=cfg.model.num_register_tokens,
        num_heads=cfg.model.num_heads,
        pretrained_weights=cfg.model.pretrained_weights,
        img_size_pretrained=cfg.model.img_size_pretrained,
        dctm_variant=dctm_cfg.variant,
        basis_features=dctm_cfg.basis_features,
        family=dctm_cfg.family,
    )
    return model.to(device)


class EarlyStoppingDCTM(EarlyStopping):
    """Extended EarlyStopping that saves max_time with checkpoint."""

    def __init__(
        self,
        *args,
        max_time: float = 1.0,
        horizons: list[DCTMHorizon] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_time = max_time
        self.horizons = horizons or []

    def _checkpoint(self, model):
        return {
            "model_state_dict": model.state_dict(),
            "max_time": self.max_time,
            "horizons": serialize_horizons(self.horizons),
        }

    def __call__(self, epoch, model, results):
        score = results[self.tracking]
        if self.min_max == "min":
            score = -1 * score

        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            fname = "best.pt"
            torch.save(self._checkpoint(model), Path(self.checkpoint_dir, fname))
            self.counter = 0

        elif score < self.best_score:
            self.counter += 1
            if epoch <= self.min_epoch + 1 and self.verbose:
                print(
                    f"EarlyStopping counter: {min(self.counter,self.patience)}/{self.patience}"
                )
            elif self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience and epoch > self.min_epoch:
                self.early_stop = True

        if self.save_all:
            fname = f"epoch_{epoch+1}.pt"
            torch.save(self._checkpoint(model), Path(self.checkpoint_dir, fname))

        # override latest
        torch.save(self._checkpoint(model), Path(self.checkpoint_dir, "latest.pt"))


def main(args):
    cfg = setup(args)

    output_dir = Path(cfg.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(cfg.features_dir)

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    print("Loading data")
    train_df = pd.read_csv(cfg.data.train_csv)
    tune_df = pd.read_csv(cfg.data.tune_csv)
    if cfg.data.test_csv:
        test_df = pd.read_csv(cfg.data.test_csv)

    # Compute max_time from training set only (avoid data leakage)
    max_time = train_df[cfg.label_name].max()
    print(f"Max time from training set: {max_time}")
    eval_context = build_dctm_evaluation_context(cfg, train_df, max_time)

    train_dataset_options = DatasetOptions(
        df=train_df,
        features_dir=features_dir,
        label_name=cfg.label_name,
        label_mapping=cfg.label_mapping,
    )
    tune_dataset_options = DatasetOptions(
        df=tune_df,
        features_dir=features_dir,
        label_name=cfg.label_name,
        label_mapping=cfg.label_mapping,
    )
    if cfg.data.test_csv:
        test_dataset_options = DatasetOptions(
            df=test_df,
            features_dir=features_dir,
            label_name=cfg.label_name,
            label_mapping=cfg.label_mapping,
        )

    print("Initializing datasets")
    train_dataset = ExtractedFeaturesSurvivalDataset(train_dataset_options)
    tune_dataset = ExtractedFeaturesSurvivalDataset(tune_dataset_options)
    if cfg.data.test_csv:
        test_dataset = ExtractedFeaturesSurvivalDataset(test_dataset_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing model")
    model = create_model(cfg, device)
    print(model)
    log_trainable_parameter_breakdown(model, logger)

    print("Configuring optimizer & scheduler")
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = OptimizerFactory(
        cfg.optim.name, model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd
    ).get_optimizer()
    scheduler = SchedulerFactory(optimizer, cfg.optim.lr_scheduler).get_scheduler()

    early_stopping = EarlyStoppingDCTM(
        cfg.early_stopping.tracking,
        cfg.early_stopping.min_max,
        cfg.early_stopping.patience,
        cfg.early_stopping.min_epoch,
        checkpoint_dir=checkpoint_dir,
        save_all=cfg.early_stopping.save_all,
        max_time=max_time,
        horizons=eval_context["horizons"],
    )

    stop = False
    start_time = time.time()

    print()
    with tqdm.tqdm(
        range(cfg.training.nepochs),
        desc="Training",
        unit=" epoch",
        leave=True,
    ) as t:
        for epoch in t:
            epoch_start_time = time.time()
            train_dataset.seed = epoch

            if cfg.wandb.enable:
                log_dict = {"epoch": epoch + 1}

            train_results = train_dctm(
                epoch + 1,
                model,
                train_dataset,
                optimizer,
                max_time,
                metric_names=cfg.metrics,
                horizons=eval_context["horizons"],
                risk_alias_label=eval_context["risk_alias_label"],
                train_event_times=eval_context["train_event_times"],
                train_events=eval_context["train_events"],
                compute_ibs_metric=eval_context["compute_ibs_metric"],
                ibs_times=eval_context["ibs_times"],
                batch_size=cfg.training.batch_size,
                gradient_accumulation=cfg.training.gradient_accumulation,
                num_workers=num_workers,
                device=device,
            )

            if cfg.wandb.enable:
                update_log_dict("train", train_results, log_dict)

            train_dataset.df.to_csv(
                Path(result_dir, f"train-{epoch+1}.csv"), index=False
            )

            if epoch % cfg.tuning.tune_every == 0:
                tune_results = tune_dctm(
                    epoch + 1,
                    model,
                    tune_dataset,
                    max_time,
                    metric_names=cfg.metrics,
                    horizons=eval_context["horizons"],
                    risk_alias_label=eval_context["risk_alias_label"],
                    train_event_times=eval_context["train_event_times"],
                    train_events=eval_context["train_events"],
                    compute_ibs_metric=eval_context["compute_ibs_metric"],
                    ibs_times=eval_context["ibs_times"],
                    batch_size=cfg.tuning.batch_size,
                    num_workers=num_workers,
                    device=device,
                )

                if cfg.wandb.enable:
                    update_log_dict("tune", tune_results, log_dict)

                tune_dataset.df.to_csv(
                    Path(result_dir, f"tune-{epoch+1}.csv"), index=False
                )

                early_stopping(epoch, model, tune_results)
                if early_stopping.early_stop and cfg.early_stopping.enable:
                    stop = True

            lr = cfg.optim.lr
            if scheduler:
                lr = scheduler.get_last_lr()[0]
                scheduler.step()
            if cfg.wandb.enable:
                log_dict.update({"train/lr": lr})
                wandb.log(log_dict)

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
            tqdm.tqdm.write(
                f"End of epoch {epoch+1} / {cfg.training.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s"
            )

            if stop:
                tqdm.tqdm.write(
                    f"Stopping early because best {cfg.early_stopping.tracking} was reached {cfg.early_stopping.patience} epochs ago"
                )
                break

    # Load best model
    best_model_fp = Path(checkpoint_dir, f"{cfg.testing.retrieve_checkpoint}.pt")
    if cfg.wandb.enable:
        wandb.save(str(best_model_fp))
    checkpoint = torch.load(best_model_fp)
    model.load_state_dict(checkpoint["model_state_dict"])
    saved_max_time = checkpoint["max_time"]
    saved_horizons = deserialize_horizons(
        checkpoint.get("horizons", eval_context["horizons"])
    )

    # Tune set inference
    best_tune_results = inference_dctm(
        model,
        tune_dataset,
        saved_max_time,
        metric_names=cfg.metrics,
        horizons=saved_horizons,
        risk_alias_label=eval_context["risk_alias_label"],
        train_event_times=eval_context["train_event_times"],
        train_events=eval_context["train_events"],
        compute_ibs_metric=eval_context["compute_ibs_metric"],
        ibs_times=eval_context["ibs_times"],
        batch_size=1,
        num_workers=num_workers,
        device=device,
    )
    tune_dataset.df.to_csv(
        Path(result_dir, f"tune-{cfg.testing.retrieve_checkpoint}.csv"), index=False
    )

    for r, v in best_tune_results.items():
        if isinstance(v, float):
            v = round(v, 5)
        if cfg.wandb.enable:
            wandb.log({f"tune/{r}-{cfg.testing.retrieve_checkpoint}": v})
        else:
            print(f"tune {r}-{cfg.testing.retrieve_checkpoint}: {v}")

    if cfg.data.test_csv:
        # Test set inference
        test_results = inference_dctm(
            model,
            test_dataset,
            saved_max_time,
            metric_names=cfg.metrics,
            horizons=saved_horizons,
            risk_alias_label=eval_context["risk_alias_label"],
            train_event_times=eval_context["train_event_times"],
            train_events=eval_context["train_events"],
            compute_ibs_metric=eval_context["compute_ibs_metric"],
            ibs_times=eval_context["ibs_times"],
            batch_size=1,
            num_workers=num_workers,
            device=device,
        )
        test_dataset.df.to_csv(Path(result_dir, "test.csv"), index=False)

        for r, v in test_results.items():
            if isinstance(v, float):
                v = round(v, 5)
            if cfg.wandb.enable:
                wandb.log({f"test/{r}": v})
            else:
                print(f"test {r}: {v}")

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
