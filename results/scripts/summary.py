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


def preprocess_labels(config):
    pp_labels = {}
    # RUMC
    rumc_labels = pd.read_csv(config["labels"]["rumc"])
    rumc_labels["cohort"] = rumc_labels.case_id.apply(lambda x: "radboud" if "radboud" in x else "plco")
    rumc_labels = rumc_labels[rumc_labels.cohort == "radboud"]
    pp_labels["rumc"] = rumc_labels
    # PLCO
    plco_labels = pd.read_csv(config["labels"]["plco"])
    pp_labels["plco"] = plco_labels
    # IMP
    imp_labels = pd.read_csv(config["labels"]["imp"])
    pp_labels["imp"] = imp_labels
    # UHC
    uhc_labels = pd.read_csv(config["labels"]["uhc"])
    pp_labels["uhc"] = uhc_labels
    return pp_labels


def preprocess_capra_s(config):
    pp_capra = {}
    # RUMC
    rumc_capra = pd.read_csv(config["capra_s"]["rumc"])
    rumc_capra = rumc_capra[rumc_capra.partition == "testing"]
    pp_capra["rumc"] = rumc_capra
    # PLCO
    plco_capra = pd.read_csv(config["capra_s"]["plco"])
    pp_capra["plco"] = plco_capra
    # IMP
    imp_capra = pd.read_csv(config["capra_s"]["imp"])
    pp_capra["imp"] = imp_capra
    # UHC
    uhc_capra = pd.read_csv(config["capra_s"]["uhc"])
    pp_capra["uhc"] = uhc_capra
    return pp_capra


def read_model_predictions(
    config,
    model_dir: Path,
    csv_name: str,
    test_set: str,
    label_df: pd.DataFrame,
    capra_df: pd.DataFrame,
    num_folds: int = 5,
) -> pd.DataFrame:
    csv_paths = sorted([x for x in model_dir.glob("*.csv") if csv_name in str(x)])
    assert len(csv_paths) == num_folds, f"Expected {num_folds} folds, found {len(csv_paths)}"
    dfs = []
    cols = ["case_id", "overall_survival_years"]
    for fp in csv_paths:
        fold_num = fp.stem.split("-")[-1]
        preds_df = pd.read_csv(fp)[cols]
        preds_df["risk"] = preds_df.overall_survival_years.apply(lambda x: -np.abs(x))
        preds_df = preds_df.rename(columns={"risk": f"fold_{fold_num}"})
        preds_df = preds_df[["case_id", f"fold_{fold_num}"]]
        if test_set == "rumc":
            test_preds_df = preds_df[preds_df.case_id.str.contains("radboud")]
        elif test_set == "plco":
            test_preds_df = preds_df[~preds_df.case_id.str.contains("radboud")]
        else:
            test_preds_df = preds_df
        dfs.append(test_preds_df)
    # merge folds
    preds_df = dfs[0]
    for df_ in dfs[1:]:
        preds_df = pd.merge(preds_df, df_, on="case_id", how="inner")
    # ensemble fold predictions
    preds_df["ensemble"] = preds_df[[f"fold_{i}" for i in range(num_folds)]].apply(lambda row: np.mean(row), axis=1)
    # merge with CAPRA-S and labels
    df = pd.merge(preds_df, capra_df, on="case_id", how="inner")
    follow_up_col = config["follow_up_cols"][test_set]
    df = pd.merge(df, label_df[["case_id", follow_up_col, "event"]], on="case_id", how="inner")
    df = df.rename(columns={follow_up_col: "event_time"})
    return df


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
