import argparse
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(add_help=add_help)
    # Default directories relative to this script
    script_dir = Path(__file__).parent
    default_outdir = (script_dir / ".." / "figures").resolve()
    default_pairwise_test_dir = (script_dir / ".." / "pairwise-tests").resolve()
    default_config = (script_dir / "config.yml").resolve()
    default_dataset_effect_csv = (script_dir / ".." / "dataset-effect.csv").resolve()

    parser.add_argument(
        "-o", "--output-dir", type=str, default=str(default_outdir),
        help="Directory to save the output figures",
    )
    parser.add_argument(
        "--pairwise-tests-dir", type=str, default=str(default_pairwise_test_dir),
        help="Directory containing pairwise test CSVs",
    )
    parser.add_argument(
        "--dataset-effect-csv", type=str, default=str(default_dataset_effect_csv),
        help="Path to dataset-effect.csv (used for unified color scale)",
    )
    parser.add_argument(
        "--config-file", type=str, default=str(default_config),
        help="Path to the config YAML file",
    )
    return parser


def compute_global_vlim(
    pairwise_tests_dir: Path,
    settings: list,
    cohorts: list,
    dataset_effect_csv: Path = None,
) -> float:
    """Compute a unified color limit from all pairwise CSVs and optionally dataset-effect.csv."""
    all_abs = []
    for setting in settings:
        for cohort in cohorts:
            fpath = pairwise_tests_dir / f"pairwise-{cohort}-{setting}.csv"
            if fpath.exists():
                df = pd.read_csv(fpath)
                if "delta" in df.columns:
                    all_abs.extend(df["delta"].abs().dropna().tolist())
    if dataset_effect_csv is not None and dataset_effect_csv.exists():
        df = pd.read_csv(dataset_effect_csv)
        if "delta" in df.columns:
            all_abs.extend(df["delta"].abs().dropna().tolist())
    if not all_abs:
        return 0.001
    vmax = float(max(all_abs))
    return float(np.ceil(vmax * 1000) / 1000.0)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns are present and correctly named (case-insensitive)."""
    low = {c.lower(): c for c in df.columns}
    need = ["encoder_a", "encoder_b", "delta", "q"]
    missing = [c for c in need if c not in low]
    if missing:
        raise ValueError(f"Missing required columns {missing}; found {list(df.columns)}")
    return df.rename(columns={low["encoder_a"]: "encoder_a",
                              low["encoder_b"]: "encoder_b",
                              low["delta"]: "delta",
                              low["q"]: "q"})


def build_mats(df: pd.DataFrame, encoders: list):
    """
    Return Δ matrix M and q matrix Q in the given encoder order.
    M[i, j] = encoder[i] - encoder[j]
    """
    n = len(encoders)
    M = np.full((n, n), np.nan)
    Q = np.full((n, n), np.nan)
    for _, r in df.iterrows():
        a, b, d, q = r["encoder_a"], r["encoder_b"], float(r["delta"]), float(r["q"])
        if a not in encoders or b not in encoders:
            continue
        i, j = encoders.index(a), encoders.index(b)
        M[i, j] = d
        Q[i, j] = q
        # Fill antisymmetric counterpart if not present
        if np.isnan(M[j, i]):
            M[j, i] = -d
            Q[j, i] = q
    np.fill_diagonal(M, 0.0)
    np.fill_diagonal(Q, np.nan)
    return M, Q


def plot_hm(M, Q, encoders, title, sig_alpha=0.05, vlim=None, ax=None, show_cbar=True):
    """Plot a single pairwise comparison heatmap."""
    n = len(encoders)
    mask = np.zeros_like(M, dtype=bool)
    np.fill_diagonal(mask, True)

    # Symmetric color limits around 0
    if vlim is None:
        vmax = np.nanmax(np.abs(M))
        vmax = 0.001 if not np.isfinite(vmax) or vmax == 0 else vmax
        vlim = float(np.ceil(vmax * 1000) / 1000.0)

    cmap = sns.color_palette("RdYlGn", as_cmap=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
        ax.set_facecolor("#f7f7f7")

    sns.heatmap(
        M, ax=ax, cmap=cmap, vmin=-vlim, vmax=vlim,
        mask=mask, square=True, cbar=show_cbar,
        linewidths=0.5, linecolor="black",
        xticklabels=encoders, yticklabels=encoders
    )

    # Style
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)

    # Annotate only visible (upper-tri) cells
    for i in range(n):
        for j in range(n):
            if mask[i, j]:  # diagonal or empty
                continue
            val = M[i, j]
            if np.isnan(val):
                continue
            txt = f"{val:+.3f}"
            is_significant = not np.isnan(Q[i, j]) and Q[i, j] < sig_alpha
            weight = "bold" if is_significant else "normal"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", 
                    fontsize=11, color="black", fontweight=weight)

    # Colorbar label
    if show_cbar:
        cbar = ax.collections[0].colorbar
        cbar.set_label("Δ c-index (A-B)", fontsize=12)



def plot_all_cohorts_grid(
    *,
    setting: str,
    output_dir: Path,
    pairwise_tests_dir: Path,
    encoders: list,
    cohorts: list,
    n_samples_dict: dict = None,
    display_title: bool = False,
    vlim: float = None,
):
    """Plot all cohort heatmaps in a grid with shared colorbar."""
    mats = []
    qs = []
    titles = []

    for cohort in cohorts:
        fpath = pairwise_tests_dir / f"pairwise-{cohort}-{setting}.csv"
        if not fpath.exists():
            print(f"[warn] missing: {fpath}")
            mats.append(None)
            qs.append(None)
            titles.append(cohort)
            continue

        df = pd.read_csv(fpath)
        df = _standardize_columns(df)
        M, Q = build_mats(df, encoders)
        mats.append(M)
        qs.append(Q)
        
        title = cohort
        if n_samples_dict and cohort.lower() in n_samples_dict:
            title += f" (n={n_samples_dict[cohort.lower()]})"
        titles.append(title)

    valid_mats = [m for m in mats if m is not None]
    if not valid_mats:
        print(f"[warn] No data found for setting {setting}. Skipping grid plot.")
        return

    if vlim is None:
        all_vals = np.concatenate([np.abs(m) for m in valid_mats])
        vmax = np.nanmax(all_vals)
        vmax = 0.001 if not np.isfinite(vmax) or vmax == 0 else vmax
        vlim = float(np.ceil(vmax * 1000) / 1000.0)

    # Plot setup
    n_plots = len(cohorts)
    ncols = 2
    nrows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows), dpi=120)
    axes = axes.flatten()

    for i, (M, Q, title) in enumerate(zip(mats, qs, titles)):
        if M is not None:
            plot_hm(M, Q, encoders, title, vlim=vlim, ax=axes[i], show_cbar=False)
        else:
            axes[i].set_visible(False)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.subplots_adjust(wspace=0.3, hspace=0.6)

    # Shared colorbar
    norm = plt.Normalize(vmin=-vlim, vmax=vlim)
    sm = plt.cm.ScalarMappable(cmap=sns.color_palette("RdYlGn", as_cmap=True), norm=norm)
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="vertical", fraction=0.02, pad=0.05)
    cbar.set_label("Δ c-index", fontsize=14, fontweight="bold")

    if display_title:
        fig.suptitle(f"Pairwise Model Comparisons - {setting}", fontsize=20, fontweight="bold", y=0.98)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pairwise-heatmaps-{setting}"
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(
    *,
    output_dir: Path,
    pairwise_tests_dir: Path,
    config_path: Path,
    dataset_effect_csv: Path = None,
):
    # Setup aesthetics
    sns.set_theme(style="white", context="paper", font_scale=1.4)

    # Load config if available
    n_samples_dict = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        n_samples_dict = config.get("num_test_cases", {})
    else:
        print(f"[warn] Config file not found at {config_path}")

    # Parameters
    encoders = ["Prost40M", "UNI", "Virchow2", "H-optimus-0"]
    cohorts = ["RUMC", "PLCO", "IMP", "UHC"]
    settings = ["RUMC", "RUMC+TCGA"]

    # Unified color scale across all figures (pairwise + dataset-effect)
    vlim = compute_global_vlim(pairwise_tests_dir, settings, cohorts, dataset_effect_csv)

    for setting in settings:
        plot_all_cohorts_grid(
            setting=setting,
            output_dir=output_dir,
            pairwise_tests_dir=pairwise_tests_dir,
            encoders=encoders,
            cohorts=cohorts,
            n_samples_dict=n_samples_dict,
            vlim=vlim,
        )


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    
    main(
        output_dir=Path(args.output_dir),
        pairwise_tests_dir=Path(args.pairwise_tests_dir),
        config_path=Path(args.config_file),
        dataset_effect_csv=Path(args.dataset_effect_csv),
    )

