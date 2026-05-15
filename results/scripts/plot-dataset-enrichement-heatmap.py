import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(add_help=add_help)
    default_outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    default_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset-effect.csv"))
    default_config = os.path.join(os.path.dirname(__file__), "config.yml")
    default_pairwise_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pairwise-tests"))
    parser.add_argument(
        "-o", "--output-dir", type=str, default=default_outdir, help="output directory for saving figures",
    )
    parser.add_argument(
        "--csv", type=str, default=default_csv, help="path to the dataset effect csv file",
    )
    parser.add_argument(
        "--pairwise-tests-dir", type=str, default=default_pairwise_dir,
        help="directory containing pairwise test CSVs (used for unified color scale)",
    )
    parser.add_argument(
        "--config-file", type=str, default=default_config, help="path to the config yaml file",
    )
    return parser


def compute_global_vlim(
    dataset_effect_csv: Path,
    pairwise_tests_dir: Path,
    settings: list,
    cohorts: list,
) -> float:
    """Compute a unified color limit from dataset-effect.csv and all pairwise CSVs."""
    all_abs = []
    if dataset_effect_csv.exists():
        df = pd.read_csv(dataset_effect_csv)
        if "delta" in df.columns:
            all_abs.extend(df["delta"].abs().dropna().tolist())
    for setting in settings:
        for cohort in cohorts:
            fpath = pairwise_tests_dir / f"pairwise-{cohort}-{setting}.csv"
            if fpath.exists():
                df = pd.read_csv(fpath)
                if "delta" in df.columns:
                    all_abs.extend(df["delta"].abs().dropna().tolist())
    if not all_abs:
        return 0.001
    vmax = float(max(all_abs))
    return float(np.ceil(vmax * 1000) / 1000.0)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    low = {c.lower(): c for c in df.columns}
    need = ["cohort", "encoder", "delta", "q"]
    missing = [c for c in need if c not in low]
    if missing:
        raise ValueError(f"Missing required columns {missing}; found {list(df.columns)}")
    return df.rename(columns={
        low["cohort"]: "cohort",
        low["encoder"]: "encoder",
        low["delta"]:  "delta",
        low["q"]:      "q",
    })


def build_mats(df: pd.DataFrame, cohorts, encoders):
    """
    Build Δ matrix M (rows=cohorts, cols=encoders) and q matrix Q
    in the specified order. Missing entries become NaN.
    """
    r = len(encoders)
    c = len(cohorts)
    M = np.full((r, c), np.nan, dtype=float)
    Q = np.full((r, c), np.nan, dtype=float)
    # Map for quick lookups
    idx_row = {k: i for i, k in enumerate(encoders)}
    idx_col = {k: j for j, k in enumerate(cohorts)}
    for _, row in df.iterrows():
        enc, coh = row["encoder"], row["cohort"]
        if enc in idx_row and coh in idx_col:
            i, j = idx_row[enc], idx_col[coh]
            M[i, j] = float(row["delta"])
            Q[i, j] = float(row["q"])
    return M, Q


def plot_heatmap(M, Q, cohorts, encoders, title, output_dir, outname="heatmap-dataset-effect", sig_alpha=0.05, vlim=None):
    if vlim is None:
        vmax = np.nanmax(np.abs(M))
        vmax = 0.001 if not np.isfinite(vmax) or vmax == 0 else vmax
        vlim = float(np.ceil(vmax * 1000) / 1000.0)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.set_facecolor("#f7f7f7")
    cmap = sns.color_palette("RdYlGn", as_cmap=True)

    sns.heatmap(
        M, ax=ax, cmap=cmap,
        vmin=-vlim, vmax=+vlim,
        square=False, cbar=True,
        linewidths=0.5, linecolor="black",
        xticklabels=cohorts, yticklabels=encoders,
        annot=False  # we’ll annotate manually for style control
    )

    # Style
    for spine in ax.spines.values():
        spine.set_visible(False)
    sns.despine(left=False, bottom=False)
    ax.set_xticklabels(ax.get_xticklabels())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    if title is not None:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)

    # Annotate cells with value (and significance)
    n_rows, n_cols = M.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = M[i, j]
            if np.isnan(val):
                continue
            qv = Q[i, j]
            txt = f"{val:+.3f}"
            is_sig = (not np.isnan(qv)) and (qv < sig_alpha)
            # Bold + asterisk for significant entries, normal otherwise
            if is_sig:
                ax.text(j + 0.5, i + 0.5, txt,
                        ha="center", va="center",
                        fontsize=11, color="black", fontweight="bold")
            else:
                ax.text(j + 0.5, i + 0.5, txt,
                        ha="center", va="center",
                        fontsize=11, color="black")

    # Colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Δ c-index", fontsize=12)

    # Save
    outbase = output_dir / outname
    fig.savefig(f"{outbase}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{outbase}.pdf", bbox_inches="tight")
    plt.close(fig)


def main(
    *,
    output_dir: Path,
    csv_path: Path,
    pairwise_tests_dir: Path = None,
):
    sns.set_theme(style="white", context="paper", font_scale=1.4)

    # Fix row/col order for consistent figures
    cohorts = ["RUMC", "PLCO", "IMP", "UHC"]
    encoders = ["Prost40M", "UNI", "Virchow2", "H-optimus-0"]
    settings = ["RUMC", "RUMC+TCGA"]

    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Please run statistical-testing.py first.")
        return

    df = pd.read_csv(csv_path)
    df = _standardize_columns(df)

    M, Q = build_mats(df, cohorts, encoders)

    # Unified color scale across all figures (pairwise + dataset-effect)
    vlim = compute_global_vlim(csv_path, pairwise_tests_dir or Path("."), settings, cohorts)

    plot_heatmap(
        M, Q, cohorts, encoders,
        title=None,
        output_dir=output_dir,
        sig_alpha=0.05,
        vlim=vlim,
    )


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main(
        output_dir=output_dir,
        csv_path=Path(args.csv),
        pairwise_tests_dir=Path(args.pairwise_tests_dir),
    )
