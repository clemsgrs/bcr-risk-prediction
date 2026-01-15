import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(add_help=add_help)
    default_outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    default_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cindex-isup-subgroup.csv"))
    parser.add_argument(
        "-o", "--output-dir", type=str, default=default_outdir, help="output directory for saving figures",
    )
    parser.add_argument(
        "--csv", type=str, default=default_csv, help="path to the c-index per ISUP csv file",
    )
    return parser


def main(
    *,
    csv_path: Path,
    output_dir: Path,
    display_title: bool = False,
):
    
    results_df = pd.read_csv(csv_path)
    # Plotting
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    palette = {"RUMC-only": "#4EBBDF", "RUMC+TCGA": "#B631CE"}
    encoder_order = ["Prost40M", "UNI", "Virchow2", "H-optimus-0"]

    for tset in results_df.test_set.unique():
        subset_tset = results_df[results_df.test_set == tset]
        
        # Filter ISUPs that have data for all models if possible, or just common ones
        isup_order = sorted(subset_tset.ISUP.unique(), key=lambda x: int(x))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100, sharey=True)
        axes = axes.flatten()

        handles, labels = None, None
        for i, encoder in enumerate(encoder_order):
            ax = axes[i]
            subset = subset_tset[subset_tset["encoder"] == encoder]
            if subset.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_title(encoder, fontweight="bold")
                continue

            sns.pointplot(
                data=subset,
                x="ISUP", y="c_index",
                hue="training_set",
                order=isup_order,
                palette=palette,
                markers=["o", "s"],
                linestyles=["--", "-"],
                ax=ax,
                errorbar=None
            )

            ax.set_title(encoder, fontweight="bold", fontsize=16, pad=10)
            ax.set_xlabel("ISUP Grade", fontsize=12, fontweight="bold", labelpad=10)
            ax.set_ylabel("c-index", fontsize=12, fontweight="bold", labelpad=10)
            ax.set_ylim(0.4, 0.8)
            ax.set_facecolor("#f7f7f7")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

        if handles:
            fig.legend(
                handles, labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.98),
                ncol=2,
                frameon=True,
                fancybox=True,
                shadow=True,
                columnspacing=1.5,
                handletextpad=1.0
            )

        sns.despine(fig=fig)
        if display_title:
            plt.suptitle(f"Performance Breakdown by ISUP Grade ({tset})", fontsize=18, fontweight="bold", y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        plt.savefig(output_dir / f"isup-subgroup-cindex-{tset.lower()}.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / f"isup-subgroup-cindex-{tset.lower()}.pdf", bbox_inches="tight")
        plt.close()

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":

    args = get_args_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main(
        csv_path=Path(args.csv),
        output_dir=output_dir,
    )
