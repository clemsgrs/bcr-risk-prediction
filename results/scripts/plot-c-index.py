import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import argparse
import yaml

from pathlib import Path


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(add_help=add_help)
    default_outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    default_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "summary.csv"))
    default_config = os.path.join(os.path.dirname(__file__), "config.yml")
    parser.add_argument(
        "-o", "--output-dir", type=str, default=default_outdir, help="output directory for saving figures",
    )
    parser.add_argument(
        "--csv", type=str, default=default_csv, help="path to the summary CSV file",
    )
    parser.add_argument(
        "--config-file", type=str, default=default_config, help="path to the config yaml file",
    )
    return parser


def plot_on_ax(ax, df_test, test_set, n_samples, model_order, colors, is_first=False):
    offsets = {"RUMC": -0.15, "RUMC+TCGA": 0.15}
    jitter = 0.04
    x = range(len(model_order))

    for i, model in enumerate(model_order):
        for train_set in ["RUMC", "RUMC+TCGA"]:
            rows = df_test[(df_test["encoder"] == model) & (df_test["TrainingSet"] == train_set)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            xpos = i + offsets[train_set]

            # Plot DL score (before fusion)
            ax.errorbar(
                xpos - jitter,
                row["CIndex"],
                yerr=row["CIndexStd"],
                fmt="o",
                color="black",
                markerfacecolor=colors[train_set],
                markeredgecolor="black",
                capsize=4,
                markersize=10,
            )

            # Plot combined score (after fusion)
            ax.errorbar(
                xpos + jitter,
                row["CombinedCIndex"],
                yerr=row["CombinedCIndexStd"],
                fmt="s",
                color="black",
                markerfacecolor=colors[train_set],
                markeredgecolor="black",
                capsize=4,
                markersize=10,
            )

            # Line connecting the two
            ax.plot(
                [xpos - jitter, xpos + jitter],
                [row["CIndex"], row["CombinedCIndex"]],
                linestyle="--",
                color="gray",
                alpha=0.7,
            )

    # Aesthetics
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=12, rotation=45, ha='right')
    ax.set_xlabel("Tile Encoder", fontsize=14, fontweight="bold", labelpad=12)
    if is_first:
        ax.set_ylabel("c-index", fontsize=14, fontweight="bold", labelpad=12)
    ax.set_ylim(0.4, 0.95)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_facecolor("#f7f7f7")
    ax.set_title(f"{test_set}\n(n={n_samples})", fontsize=16, fontweight="bold")
    sns.despine(ax=ax)

    # add CAPRA-S baseline
    capra_s_cindex = df_test["CapraSCIndex"].iloc[0]
    capra_s_color = "#A569BD"  # A purple shade, visually distinct from blue/yellow used above
    ax.axhline(y=capra_s_cindex, color=capra_s_color, linestyle='--', linewidth=1)


def plot_combined_test_sets(df, num_test_cases, model_order, colors, output_dir):
    # Desired 2x2 layout: RUMC, PLCO on top; IMP, UHC on bottom
    cohort_layout = [
        ["RUMC", "PLCO"],
        ["IMP", "UHC"]
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), sharey=True, dpi=300)

    for r in range(2):
        for c in range(2):
            tset = cohort_layout[r][c]
            ax = axes[r, c]

            df_test = df[df["test_set"] == tset].copy()
            if df_test.empty:
                print(f"No data for {tset}, skipping subplot...")
                continue

            df_test["encoder"] = pd.Categorical(df_test["encoder"], categories=model_order, ordered=True)
            df_test.sort_values("encoder", inplace=True)
            n_samples = num_test_cases.get(tset.lower(), "N/A")

            # Label y-axis only for the left column
            plot_on_ax(ax, df_test, tset, n_samples, model_order, colors, is_first=(c == 0))

    # Legend
    capra_s_color = "#A569BD"
    legend_elements_top = [
        Patch(facecolor=colors["RUMC"], edgecolor="black", label="RUMC"),
        Patch(facecolor=colors["RUMC+TCGA"], edgecolor="black", label="RUMC+TCGA"),
    ]
    legend_elements_bottom = [
        Line2D([0], [0], label="CAPRA-S", color=capra_s_color, linestyle='--'),
        Line2D([0], [0], marker="o", label="DLRS", color="black", markerfacecolor="white", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="s", label="DLRS + CAPRA-S", color="black", markerfacecolor="white", markeredgecolor="black", markersize=10),
    ]

    # First legend (patches)
    legend1 = fig.legend(
        handles=legend_elements_top,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(legend_elements_top),
        frameon=True,
        fancybox=True,
        shadow=True,
        columnspacing=1.5,
        handletextpad=1.2,
        borderaxespad=0.5,
        fontsize=14
    )

    # Second legend (lines and markers), placed below the first
    legend2 = fig.legend(
        handles=legend_elements_bottom,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(legend_elements_bottom),
        frameon=True,
        fancybox=True,
        shadow=True,
        columnspacing=1.5,
        handletextpad=1.2,
        borderaxespad=0.5,
        fontsize=14
    )

    fig.add_artist(legend1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.7)  # More space for legends and between rows
    out_path = output_dir / "c-index-all-cohorts.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_test_set(df, test_set, n_samples, model_order, colors, output_dir):
    # Filter for the test set
    df_test = df[df["test_set"] == test_set].copy()
    if df_test.empty:
        print(f"No data for {test_set}, skipping...")
        return

    df_test["encoder"] = pd.Categorical(df_test["encoder"], categories=model_order, ordered=True)
    df_test.sort_values("encoder", inplace=True)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    offsets = {"RUMC": -0.15, "RUMC+TCGA": 0.15}
    jitter = 0.04
    x = range(len(model_order))

    for i, model in enumerate(model_order):
        for train_set in ["RUMC", "RUMC+TCGA"]:
            rows = df_test[(df_test["encoder"] == model) & (df_test["TrainingSet"] == train_set)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            xpos = i + offsets[train_set]

            # Plot DL score (before fusion)
            ax.errorbar(
                xpos - jitter,
                row["CIndex"],
                yerr=row["CIndexStd"],
                fmt="o",
                color="black",
                markerfacecolor=colors[train_set],
                markeredgecolor="black",
                capsize=4,
                markersize=10,
            )

            # Plot combined score (after fusion)
            ax.errorbar(
                xpos + jitter,
                row["CombinedCIndex"],
                yerr=row["CombinedCIndexStd"],
                fmt="s",
                color="black",
                markerfacecolor=colors[train_set],
                markeredgecolor="black",
                capsize=4,
                markersize=10,
            )

            # Line connecting the two
            ax.plot(
                [xpos - jitter, xpos + jitter],
                [row["CIndex"], row["CombinedCIndex"]],
                linestyle="--",
                color="gray",
                alpha=0.7,
            )

    # Aesthetics
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=12)
    ax.set_xlabel("Tile Encoder", fontsize=14, fontweight="bold", labelpad=12)
    ax.set_ylabel("c-index", fontsize=14, fontweight="bold", labelpad=12)
    ax.set_ylim(0.4, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_facecolor("#f7f7f7")
    sns.despine()

    # add CAPRA-S baseline
    capra_s_cindex = df_test["CapraSCIndex"].iloc[0]
    capra_s_color = "#A569BD"  # A purple shade, visually distinct from blue/yellow used above
    ax.axhline(y=capra_s_cindex, color=capra_s_color, linestyle='--', linewidth=1)

    # Legend
    legend_elements_top = [
        Patch(facecolor=colors["RUMC"], edgecolor="black", label="RUMC"),
        Patch(facecolor=colors["RUMC+TCGA"], edgecolor="black", label="RUMC+TCGA"),
    ]
    legend_elements_bottom = [
        Line2D([0], [0], label="CAPRA-S", color=capra_s_color, linestyle='--'),
        Line2D([0], [0], marker="o", label="DLRS", color="black", markerfacecolor="white", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="s", label="DLRS + CAPRA-S", color="black", markerfacecolor="white", markeredgecolor="black", markersize=10),
    ]

    # First legend (patches)
    legend1 = fig.legend(
        handles=legend_elements_top,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(legend_elements_top),
        frameon=True,
        fancybox=True,
        shadow=True,
        columnspacing=1.5,
        handletextpad=1.2,
        borderaxespad=0.5,
    )

    # Second legend (lines and markers), placed below the first
    legend2 = fig.legend(
        handles=legend_elements_bottom,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(legend_elements_bottom),
        frameon=True,
        fancybox=True,
        shadow=True,
        columnspacing=1.5,
        handletextpad=1.2,
        borderaxespad=0.5,
    )

    fig.add_artist(legend1)

    fig.suptitle(f"Combining DLRS with CAPRA-S\n{test_set} (n={n_samples})", fontsize=16, fontweight="bold", y=1.12)
    plt.tight_layout()
    out_path = output_dir / f"c-index-{test_set}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(
    *,
    output_dir: Path,
    csv_path: Path,
    config_path: Path,
):

    sns.set_theme(style="white", context="paper", font_scale=1.4)
    colors = {"RUMC": "#7FB3D5", "RUMC+TCGA": "#F7DC6F"}
    model_order = ["Prost40M", "UNI", "Virchow2", "H-optimus-0"]

    # Retrieve data
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Please run summary.py first.")
        return

    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    num_test_cases = config.get("num_test_cases", {})

    summary_df = pd.read_csv(csv_path)
    summary_df["TrainingSet"] = summary_df.training_set.apply(lambda x: "RUMC" if x == "rumc" else "RUMC+TCGA")

    # Parse columns
    summary_df["CIndex"] = summary_df.ens_c_index.astype(float)
    summary_df["CIndexStd"] = summary_df.model_c_index.apply(lambda x: float(x.split(" ± ")[1]))

    summary_df["CombinedCIndex"] = summary_df.combined_c_index_ens.astype(float)
    summary_df["CombinedCIndexStd"] = summary_df.combined_c_index_avg.apply(lambda x: float(x.split(" ± ")[1]))

    summary_df["CapraSCIndex"] = summary_df.capra_s_c_index.astype(float)

    # Use cohorts present in the summary data, ordering them by name
    test_sets = sorted(summary_df.test_set.unique())

    for tset in test_sets:
        n_samples = num_test_cases.get(tset.lower(), "N/A")
        plot_test_set(
            summary_df,
            tset,
            n_samples,
            model_order,
            colors,
            output_dir
        )

    plot_combined_test_sets(
        summary_df,
        num_test_cases,
        model_order,
        colors,
        output_dir
    )


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main(
        output_dir=output_dir,
        csv_path=Path(args.csv),
        config_path=Path(args.config_file)
    )