import pandas as pd
import numpy as np
import sys
import argparse
import os
import yaml
from pathlib import Path
from itertools import combinations
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(add_help=add_help)
    default_outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_config = os.path.join(os.path.dirname(__file__), "config.yml")
    parser.add_argument(
        "--output-dir", type=str, default=default_outdir, help="directory to save the output csvs",
    )
    parser.add_argument(
        "--config-file", type=str, default=default_config, help="path to the config yaml file",
    )
    return parser


def _cindex(time, event, risk):
    return concordance_index_censored(event.astype(bool), time, risk)[0]


def _bootstrap_indices(n, event, rng):
    idx = np.arange(n)
    ev = idx[event == 1]
    ce = idx[event == 0]
    ev_bs = rng.choice(ev, size=len(ev), replace=True) if len(ev) else np.array([], int)
    ce_bs = rng.choice(ce, size=len(ce), replace=True) if len(ce) else np.array([], int)
    return rng.permutation(np.concatenate([ev_bs, ce_bs]))


def bootstrap_cindex(time, event, risk, n_boot=4000, seed=42):
    time_ = np.asarray(time, dtype=float)
    event_ = np.asarray(event, dtype=int)
    risk_ = np.asarray(risk, dtype=float)
    c_point = _cindex(time_, event_, risk_)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = len(time_)
    for b in range(n_boot):
        idx = _bootstrap_indices(n, event_, rng)
        boots[b] = _cindex(time_[idx], event_[idx], risk_[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return c_point, (lo, hi)


def paired_bootstrap_diff(time, event, ra, rb, n_boot=4000, seed=42):
    delta = _cindex(time, event, ra) - _cindex(time, event, rb)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = _bootstrap_indices(len(time), event, rng)
        deltas.append(_cindex(time[idx], event[idx], ra[idx]) -
                      _cindex(time[idx], event[idx], rb[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(deltas) <= 0),
                np.mean(np.array(deltas) >= 0))
    return dict(delta=delta, ci_low=lo, ci_high=hi, p=min(1.0, p))


def bh_fdr(pvals):
    m = len(pvals)
    ranked = pvals.sort_values().reset_index()
    ranked["rank"] = np.arange(1, m+1)
    ranked["q"] = ranked["p"] * m / ranked["rank"]
    ranked["q"] = np.minimum.accumulate(ranked["q"][::-1])[::-1]
    q = pd.Series(index=pvals.index, dtype=float)
    q.loc[ranked["index"]] = ranked["q"].values
    return q.clip(upper=1.0)


def summarize_cindices(df, n_boot, seed):
    rows = []
    for (cohort, setting, encoder), g in tqdm(df.groupby(["cohort","train_setting","encoder"]), desc="Bootstrap C-indices"):
        c, (lo,hi) = bootstrap_cindex(g["time_to_event"], g["event"].astype(int),
                                      g["ensemble"], n_boot=n_boot, seed=seed)
        rows.append(dict(cohort=cohort, train_setting=setting,
                         encoder=encoder, c_index=c, ci_low=lo, ci_high=hi, n=len(g)))
    return pd.DataFrame(rows)


def pairwise_tests(df, n_boot, seed, out_dir):
    winners = []
    pairwise_out_dir = out_dir / "pairwise-tests"
    pairwise_out_dir.mkdir(parents=True, exist_ok=True)

    for (cohort, setting), g in tqdm(df.groupby(["cohort", "train_setting"]), desc="Pairwise Tests"):
        encoders = sorted(g["encoder"].unique())
        if len(encoders) < 2:
            continue

        # Build encoder-specific frames indexed by case_id
        enc_dfs = {}
        common_ids = None
        for enc in encoders:
            gi = (g[g["encoder"] == enc]
                  .set_index("case_id")[["ensemble", "time_to_event", "event"]]
                  .rename(columns={"ensemble": f"risk_{enc}"}))
            enc_dfs[enc] = gi
            common_ids = gi.index if common_ids is None else common_ids.intersection(gi.index)

        # Base = take time/event once (from first encoder), then add each risk column
        first = encoders[0]
        base = enc_dfs[first][["time_to_event", "event"]].copy()
        for enc in encoders:
            base[f"risk_{enc}"] = enc_dfs[enc].loc[common_ids, f"risk_{enc}"]

        aligned = base.loc[common_ids].reset_index(drop=True)

        # 1-D numpy arrays for sksurv
        time  = aligned["time_to_event"].to_numpy(dtype=float)
        event = aligned["event"].to_numpy(dtype=int)

        results = []
        for a, b in combinations(encoders, 2):
            ra = aligned[f"risk_{a}"].to_numpy(dtype=float)
            rb = aligned[f"risk_{b}"].to_numpy(dtype=float)
            stats = paired_bootstrap_diff(time, event, ra, rb, n_boot=n_boot, seed=seed)
            results.append(dict(encoder_a=a, encoder_b=b, **stats))

        out = pd.DataFrame(results)
        if out.empty:
            continue

        out["q"] = bh_fdr(out["p"])
        out["cohort"] = cohort
        out["train_setting"] = setting
        out.sort_values(["q", "encoder_a", "encoder_b"]).to_csv(
            pairwise_out_dir / f"pairwise-{cohort}-{setting}.csv", index=False
        )

        # winners = encoders not significantly beaten by another (q<0.05)
        beaten = {e: set() for e in encoders}
        for _, row in out.iterrows():
            if row["q"] < 0.05:
                if row["delta"] > 0:
                    beaten[row["encoder_b"]].add(row["encoder_a"])
                elif row["delta"] < 0:
                    beaten[row["encoder_a"]].add(row["encoder_b"])

        winner_set = sorted([e for e in encoders if not beaten[e]])
        winners.append(dict(cohort=cohort, train_setting=setting, winners=";".join(winner_set)))

    return pd.DataFrame(winners)


def dataset_effect(df, n_boot, seed):
    rows = []
    for (cohort, encoder), g in tqdm(df.groupby(["cohort","encoder"]), desc="Dataset Effect"):
        if set(g["train_setting"].unique()) >= {"RUMC","RUMC+TCGA"}:
            gA = g[g["train_setting"]=="RUMC"].set_index("case_id")
            gB = g[g["train_setting"]=="RUMC+TCGA"].set_index("case_id")
            common = gA.index.intersection(gB.index)
            if len(common) == 0: continue
            aligned = pd.concat([gA.loc[common], gB.loc[common]], axis=1,
                                keys=["A","B"])
            time, event = aligned["A","time_to_event"], aligned["A","event"].astype(int)
            ra, rb = aligned["A","ensemble"], aligned["B","ensemble"]
            stats = paired_bootstrap_diff(time.values, event.values, rb.values, ra.values,
                                          n_boot=n_boot, seed=seed)
            rows.append(dict(cohort=cohort, encoder=encoder, n=len(common), **stats))
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["p"])
    return out


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


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess labels
    labels = preprocess_labels(config)

    ensemble_dfs = []

    model_root_dir = Path(config["model_root_dir"])
    
    # Iterate over training sets (RUMC vs RUMC+TCGA)
    for training_set_key, models_list in config["model_info"].items():
        # Map config key to canonical training setting name for script logic
        if training_set_key.lower() == "rumc":
            train_setting_name = "RUMC"
        elif training_set_key.lower() == "rumc+tcga":
            train_setting_name = "RUMC+TCGA"
        else:
            train_setting_name = training_set_key.upper()

        print(f"Processing training set: {train_setting_name}")

        for model in tqdm(models_list, desc=f"Loading models for {train_setting_name}"):
            model_id = model["id"]
            encoder = model["encoder"]

            # Load predictions for all cohorts
            for test_set in ["rumc", "plco", "imp", "uhc"]:
                label_df = labels[test_set]
                csv_name = config["cohort_to_csv_name"][test_set]
                
                # Load model predictions logic (similar to read_model_predictions)
                model_dir = model_root_dir / model_id
                csv_paths = sorted([x for x in model_dir.glob("*.csv") if csv_name in str(x)])
                
                # Assuming 5 folds as per usual, but let's handle what we find
                if not csv_paths:
                    tqdm.write(f"Warning: No CSVs found for {encoder} ({model_id}) on {test_set}")
                    continue

                dfs = []
                cols = ["case_id", "overall_survival_years"]
                
                # Filter for correct test set if RUMC/PLCO share csv file names or content
                for fp in csv_paths:
                    fold_num = fp.stem.split("-")[-1]
                    preds_df = pd.read_csv(fp)[cols]
                    
                    # Convert survival years to risk score (negative of survival)
                    # This ensures correct C-index directionality (Higher Risk = Worse Survival)
                    pass_val = preds_df["overall_survival_years"].apply(lambda x: -np.abs(x))
                    
                    preds_df["risk_" + fold_num] = pass_val
                    
                    if test_set == "rumc":
                        preds_df = preds_df[preds_df.case_id.str.contains("radboud")]
                    elif test_set == "plco":
                        preds_df = preds_df[~preds_df.case_id.str.contains("radboud")]
                    
                    dfs.append(preds_df[["case_id", "risk_" + fold_num]])

                if not dfs: continue
                
                # Merge folds
                preds_df = dfs[0]
                for df_ in dfs[1:]:
                    preds_df = pd.merge(preds_df, df_, on="case_id", how="inner")
                
                # Ensemble
                risk_cols = [c for c in preds_df.columns if "risk_" in c]
                preds_df["ensemble"] = preds_df[risk_cols].mean(axis=1)

                # Merge with labels (time_to_event, event)
                # Config has follow_up_cols
                follow_up_col = config["follow_up_cols"][test_set]
                
                # Merge
                df = pd.merge(preds_df, label_df[["case_id", follow_up_col, "event"]], on="case_id", how="inner")
                df = df.rename(columns={follow_up_col: "time_to_event"})
                
                # Check length
                expected_n = config["num_test_cases"][test_set]
                if len(df) != expected_n:
                    tqdm.write(f"Warning: {test_set} expected {expected_n} but got {len(df)} for {encoder}")
                
                # Add metadata
                df["encoder"] = encoder
                df["train_setting"] = train_setting_name
                df["cohort"] = test_set.upper()

                ensemble_dfs.append(df[["case_id", "ensemble", "time_to_event", "event", "encoder", "train_setting", "cohort"]])

    if not ensemble_dfs:
        print("No data found!")
        sys.exit(1)

    df = pd.concat(ensemble_dfs, axis=0, ignore_index=True)

    n_boot = 4000
    seed = 42

    csum = summarize_cindices(df, n_boot, seed)
    csum_path = out_dir / "cindex-summary.csv"
    csum.to_csv(csum_path, index=False)
    print(f"Saved cindex summary to {csum_path}")

    winners = pairwise_tests(df, n_boot, seed, out_dir)
    winners_path = out_dir / "winners.csv"
    winners.to_csv(winners_path, index=False)
    print(f"Saved winners to {winners_path}")

    dset_effect = dataset_effect(df, n_boot, seed)
    dset_effect_path = out_dir / "dataset-effect.csv"
    dset_effect.to_csv(dset_effect_path, index=False)
    print(f"Saved dataset effect to {dset_effect_path}")