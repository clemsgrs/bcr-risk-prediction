import numpy as np
import pandas as pd
from pathlib import Path
from sksurv.metrics import concordance_index_censored

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
    capra_df: pd.DataFrame = None,
    clinical_df: pd.DataFrame = None,
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
    
    df = preds_df
    # merge with CAPRA-S if provided
    if capra_df is not None:
        df = pd.merge(df, capra_df, on="case_id", how="inner")
    
    # merge with labels
    follow_up_col = config["follow_up_cols"][test_set]
    df = pd.merge(df, label_df[["case_id", follow_up_col, "event"]], on="case_id", how="inner")
    df = df.rename(columns={follow_up_col: "event_time"})
    
    # merge with clinical if provided
    if clinical_df is not None:
        df = pd.merge(df, clinical_df, on="case_id", how="inner")
    
    return df

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
