import argparse
import os
import yaml
import pandas as pd
from pathlib import Path
from sksurv.metrics import concordance_index_censored

from utils import (
    preprocess_labels,
    preprocess_capra_s,
    preprocess_clinical,
    read_model_predictions,
)


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
