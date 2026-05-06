"""
Single-fold survival training with DCTM head.

This script trains a HIPT model with DCTM (Deep Conditional Transformation Model)
survival head for continuous-time survival analysis.
"""

import argparse
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

from src.models.hipt_dctm import LocalHIPTWithDCTM
from hipt.src.data.dataset import DatasetOptions, ExtractedFeaturesSurvivalDataset
from hipt.src.utils import (
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
    compute_time,
    setup,
    update_log_dict,
)
from hipt.src.utils.train_utils import collate_features_survival
from hipt.src.utils.metrics import get_metrics


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


def normalize_time(event_time: np.ndarray, max_time: float) -> np.ndarray:
    """Normalize event times to [0, 1] range."""
    return np.clip(event_time / max_time, 0.0, 1.0 - 1e-7)


def train_dctm(
    epoch: int,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    max_time: float,
    metric_names: list[str],
    batch_size: int = 1,
    collate_fn=partial(collate_features_survival, label_type="int"),
    gradient_accumulation: int | None = None,
    num_workers: int = 0,
    device: torch.device | None = None,
):
    """Training loop for DCTM survival model."""
    model.train()
    epoch_loss = 0
    censoring, event_times, risk_scores = [], [], []
    idxs = []

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

            # Compute risk score: log cumulative hazard at t=1 (max time)
            with torch.no_grad():
                risk = model(x, np.array([1.0])).detach()
                risk_scores.append(risk.cpu().item())

            censoring.append(censored.item())
            event_times.append(event_time.item())
            idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    dataset.df.loc[idxs, "risk"] = risk_scores

    event_indicator = [bool(1 - c) for c in censoring]
    metrics = get_metrics(
        metric_names,
        risk_scores,
        event_times,
        event_indicator=event_indicator,
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
    batch_size: int = 1,
    collate_fn=partial(collate_features_survival, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):
    """Validation loop for DCTM survival model."""
    model.eval()
    epoch_loss = 0
    censoring, event_times, risk_scores = [], [], []
    idxs = []

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

                # Compute risk score
                risk = model(x, np.array([1.0])).detach()
                risk_scores.append(risk.cpu().item())
                censoring.append(censored.item())
                event_times.append(event_time.item())
                idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    dataset.df.loc[idxs, "risk"] = risk_scores

    event_indicator = [bool(1 - c) for c in censoring]
    metrics = get_metrics(
        metric_names,
        risk_scores,
        event_times,
        event_indicator=event_indicator,
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
    batch_size: int = 1,
    collate_fn=partial(collate_features_survival, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):
    """Inference for DCTM survival model."""
    model.eval()
    epoch_loss = 0
    censoring, event_times, risk_scores = [], [], []
    idxs = []

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

                # Compute risk score
                risk = model(x, np.array([1.0])).detach()
                risk_scores.append(risk.cpu().item())
                censoring.append(censored.item())
                event_times.append(event_time.item())
                idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    dataset.df.loc[idxs, "risk"] = risk_scores

    event_indicator = [bool(1 - c) for c in censoring]
    metrics = get_metrics(
        metric_names,
        risk_scores,
        event_times,
        event_indicator=event_indicator,
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

    def __init__(self, *args, max_time: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_time = max_time

    def __call__(self, epoch, model, results):
        score = results[self.tracking]
        if self.min_max == "min":
            score = -1 * score

        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            fname = "best.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "max_time": self.max_time},
                Path(self.checkpoint_dir, fname),
            )
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
            torch.save(
                {"model_state_dict": model.state_dict(), "max_time": self.max_time},
                Path(self.checkpoint_dir, fname),
            )

        # override latest
        torch.save(
            {"model_state_dict": model.state_dict(), "max_time": self.max_time},
            Path(self.checkpoint_dir, "latest.pt"),
        )


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

    # Tune set inference
    best_tune_results = inference_dctm(
        model,
        tune_dataset,
        saved_max_time,
        metric_names=cfg.metrics,
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
