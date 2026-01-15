import argparse
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sksurv.metrics import concordance_index_censored


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(add_help=add_help)
    default_outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_config = os.path.join(os.path.dirname(__file__), "config.yml")
    parser.add_argument(
        "-o", "--output-dir", type=str, default=default_outdir, help="output directory for saving results",
    )
    parser.add_argument(
        "--config-file", type=str, default=default_config, help="path to the config yaml file",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="whether to print verbose output",
    )
    parser.add_argument(
        "--min-events", type=int, default=0, help="minimum number of events for a subgroup to be included",
    )
    parser.add_argument(
        "--min-samples", type=int, default=5, help="minimum number of samples for a subgroup to be included",
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
    if "partition" in rumc_capra.columns:
        rumc_capra = rumc_capra[rumc_capra.partition.str.contains("test")]
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


def preprocess_clinical(config):
    pp_clinical = {}
    # RUMC
    pp_clinical["rumc"] = pd.read_csv(config["clinical"]["rumc"])[["case_id", "isup"]]
    # PLCO
    plco_clinical = pd.read_csv(config["clinical"]["plco"])[["plco_id", "isup"]]
    plco_mapping = pd.read_csv(config["clinical"]["plco_mapping"])[["plco_id", "fake_id"]]
    plco_clinical = pd.merge(plco_clinical, plco_mapping, on="plco_id", how="inner").drop(columns=["plco_id"])
    plco_clinical = plco_clinical.rename(columns={"fake_id": "case_id"})
    pp_clinical["plco"] = plco_clinical
    # IMP
    pp_clinical["imp"] = pd.read_csv(config["clinical"]["imp"])[["case_id", "isup"]]
    # UHC
    pp_clinical["uhc"] = pd.read_csv(config["clinical"]["uhc"])[["case_id", "isup"]]
    return pp_clinical


def read_model_predictions(
    config,
    model_dir: Path,
    csv_name: str,
    test_set: str,
    label_df: pd.DataFrame,
    capra_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    num_folds: int = 5,
) -> pd.DataFrame:
    csv_paths = sorted([x for x in model_dir.glob("*.csv") if csv_name in str(x)])
    if len(csv_paths) != num_folds:
        print(f"Warning: Expected {num_folds} folds, found {len(csv_paths)} in {model_dir}")
        if len(csv_paths) == 0:
            return None
    
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
    actual_folds = [c for c in preds_df.columns if c.startswith("fold_")]
    preds_df["ensemble"] = preds_df[actual_folds].apply(lambda row: np.mean(row), axis=1)
    
    # merge with CAPRA-S, labels, and clinical
    df = pd.merge(preds_df, capra_df, on="case_id", how="inner")
    follow_up_col = config["follow_up_cols"][test_set]
    df = pd.merge(df, label_df[["case_id", follow_up_col, "event"]], on="case_id", how="inner")
    df = df.rename(columns={follow_up_col: "event_time"})
    df = pd.merge(df, clinical_df, on="case_id", how="inner")
    
    return df


def main(
    *,
    output_dir: Path,
    config_path: Path,
    min_events: int = 0,
    min_samples: int = 5,
    verbose: bool = False,
):
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    labels = preprocess_labels(config)
    capra_s = preprocess_capra_s(config)
    clinicals = preprocess_clinical(config)

    model_root_dir = Path(config["model_root_dir"])
    
    all_results = []

    for training_set_key in ["rumc", "rumc+tcga"]:
        model_info = config["model_info"][training_set_key]
        training_label = "RUMC-only" if training_set_key == "rumc" else "RUMC+TCGA"
        
        for test_set in ["rumc", "plco", "imp", "uhc"]:
            if verbose:
                print(f"Processing training={training_label}, test={test_set}")
            
            label_df = labels[test_set]
            capra_df = capra_s[test_set]
            clinical_df = clinicals[test_set]
            csv_name = config["cohort_to_csv_name"][test_set]

            for model in model_info:
                model_id = model["id"]
                encoder = model["encoder"]
                
                model_dir = model_root_dir / model_id
                df = read_model_predictions(
                    config, model_dir, csv_name, test_set, label_df, capra_df, clinical_df
                )
                
                if df is None or df.empty:
                    continue

                for isup in sorted(df.isup.unique()):
                    sub_df = df[df.isup == isup]
                    n_samples = len(sub_df)
                    num_events = int(sub_df.event.sum())

                    if n_samples < min_samples or num_events < min_events or num_events == 0:
                        if n_samples < min_samples:
                            reason = f"only {n_samples} samples (min_samples={min_samples})"
                        elif num_events < min_events:
                            reason = f"only {num_events} events (min_events={min_events})"
                        else:
                            reason = "no events"
                        
                        print(
                            f"Warning: Skipping ISUP {isup} for {encoder} on {test_set} (train={training_set_key}) "
                            f"because it has {reason}"
                        )
                        continue
                    
                    try:
                        c_index = concordance_index_censored(
                            sub_df.event.astype(bool).values,
                            sub_df.event_time.values,
                            sub_df.ensemble.values,
                            tied_tol=1e-08,
                        )[0]
                        
                        all_results.append({
                            "encoder": encoder,
                            "training_set": training_label,
                            "test_set": test_set.upper(),
                            "ISUP": str(int(isup)),
                            "c_index": c_index,
                            "n": len(sub_df),
                            "n_events": int(num_events)
                        })
                    except Exception as e:
                        if verbose:
                            print(f"Error calculating c-index for {encoder} on {test_set} ISUP {isup}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "cindex-isup-subgroup.csv", index=False)
    print(f"Results saved to {output_dir / 'cindex-isup-subgroup.csv'}")
    return results_df


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main(
        output_dir=output_dir,
        config_path=Path(args.config_file),
        min_events=args.min_events,
        min_samples=args.min_samples,
        verbose=args.verbose,
    )
