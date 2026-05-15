import argparse
import os
import yaml

import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
from tabulate import tabulate, SEPARATING_LINE
from sksurv.metrics import concordance_index_censored

from utils import (
    preprocess_labels,
    preprocess_capra_s,
    read_model_predictions,
)


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(add_help=add_help)
    default_outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_config = os.path.join(os.path.dirname(__file__), "config.yml")
    parser.add_argument(
        "--output-dir", type=str, default=default_outdir, help="directory to save the summary csv",
    )
    parser.add_argument(
        "--config-file", type=str, default=default_config, help="path to the config yaml file",
    )
    parser.add_argument(
        "--skip-ensemble", action="store_true", help="whether to show ensemble predictions when combining with CAPRA-S",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="whether to print verbose output",
    )
    return parser



def main(config, training_set: str, num_folds: int = 5, verbose: bool = False) -> pd.DataFrame:

    if training_set not in config["model_info"]:
        raise ValueError(f"Unknown training_set: {training_set}, must be one of {list(config['model_info'].keys())}")

    model_info = config["model_info"][training_set]

    labels = preprocess_labels(config)
    capra_s = preprocess_capra_s(config)

    model_root_dir = Path(config["model_root_dir"])
    summary_dfs = []
    for test_set in ["rumc", "plco", "imp", "uhc"]:
        if verbose:
            print(f"Processing test set: {test_set.upper()}")
        label_df = labels[test_set]
        capra_df = capra_s[test_set]
        summary_rows = []
        for model in model_info:
            model_id = model["id"]
            encoder = model["encoder"]

            rows = {
                "test_set": test_set.upper(),
                "training_set": training_set,
                "encoder": encoder,
            }

            # load model predictions
            model_dir = model_root_dir / model_id
            csv_name = config["cohort_to_csv_name"][test_set]
            df = read_model_predictions(config, model_dir, csv_name, test_set, label_df=label_df, capra_df=capra_df)
            expected_n = config["num_test_cases"][test_set]
            assert len(df) == expected_n, f"Expected {expected_n} cases, found {len(df)}"


            # scale capra-s and model predictions to mean 0, std 1
            scaler = StandardScaler()
            cols = ["capra_s_score"] + [f"fold_{i}" for i in range(num_folds)] + ["ensemble"]
            df[cols] = scaler.fit_transform(df[cols])

            # get c-index from CAPRA-S
            cindex_capra = concordance_index_censored(
                df.event.values.astype(bool),
                df.event_time.values,
                df.capra_s_score.values,
            )[0]

            rows.update({"capra_s_c_index": f"{cindex_capra:.3f}"})

            # get c-index from model

            ## average over folds
            concordance_model = []
            for i in range(num_folds):
                c_index = concordance_index_censored(
                    df.event.values.astype(bool),
                    df.event_time.values,
                    df[f"fold_{i}"].values,
                )[0]
                concordance_model.append(c_index)
            cindex_model_mean = np.mean(concordance_model)
            cindex_model_std = np.std(concordance_model)

            rows.update({"model_c_index": f"{cindex_model_mean:.3f} ± {cindex_model_std:.3f}"})

            ## ensemble
            cindex_ensemble = concordance_index_censored(
                df.event.values.astype(bool),
                df.event_time.values,
                df.ensemble.values,
            )[0]

            rows.update({"ens_c_index": f"{cindex_ensemble:.3f}"})

            # get c-index when combining CAPRA-S + and model

            # 1. average over folds
            concordance_combined = []
            for i in range(num_folds):
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(df[["event_time", "event", "capra_s_score", f"fold_{i}"]], duration_col="event_time", event_col="event")
                concordance_combined.append(cph.concordance_index_)
            cindex_combined_mean = np.mean(concordance_combined)
            cindex_combined_std = np.std(concordance_combined)

            rows.update({"combined_c_index_avg": f"{cindex_combined_mean:.3f} ± {cindex_combined_std:.3f}"})

            # 2. use model ensemble
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(df[["event_time", "event", "capra_s_score", "ensemble"]], duration_col="event_time", event_col="event")
            cindex_combined_ens = cph.concordance_index_

            # extract HR, 95% CI, and p-value for ensemble
            summary = cph.summary
            hr_ens = summary.loc["ensemble", "exp(coef)"]
            ci_lower_ens = summary.loc["ensemble", "exp(coef) lower 95%"]
            ci_upper_ens = summary.loc["ensemble", "exp(coef) upper 95%"]
            p_val_ens = summary.loc["ensemble", "p"]

            rows.update({
                "combined_c_index_ens": f"{cindex_combined_ens:.3f}",
                "ens_hr_95_ci": f"{hr_ens:.2f} ({ci_lower_ens:.2f}–{ci_upper_ens:.2f})",
                "ens_p_value": f"{p_val_ens:.4f}",
            })

            summary_rows.append(rows)

        summary_df = pd.DataFrame(summary_rows)
        summary_dfs.append(summary_df)

    summary_df = pd.concat(summary_dfs, ignore_index=True)
    return summary_df


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()
    verbose = args.verbose
    skip_ensemble = args.skip_ensemble

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    dfs = []
    for training_set in ["rumc", "rumc+tcga"]:
        if verbose:
            print(f"Processing training set: {training_set}")
        df_ = main(config, training_set, verbose=verbose)
        dfs.append(df_)

    df = pd.concat(dfs, ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "summary.csv"

    df.to_csv(out_csv, index=False)
    print(f"Summary saved to {out_csv}")
    print()

    # prepare display table
    display_cols = ["test_set", "encoder", "capra_s_c_index", "model_c_index", "ens_c_index"]
    if skip_ensemble:
        display_cols.append("combined_c_index_avg")
    else:
        display_cols.append("combined_c_index_ens")
        display_cols.extend(["ens_hr_95_ci", "ens_p_value"])

    for tset in df.training_set.unique():
        print(f"--- Training Set: {tset} ---")
        subset_df = df[df.training_set == tset][display_cols]

        table = []
        prev_test_set = None
        for _, row in subset_df.iterrows():
            if prev_test_set is not None and row["test_set"] != prev_test_set:
                table.append(SEPARATING_LINE)
            table.append(row.values.tolist())
            prev_test_set = row["test_set"]

        print(tabulate(table, headers=subset_df.columns, tablefmt="psql"))
        print()
