import argparse
import pandas as pd
import numpy as np
import wholeslidedata as wsd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def get_thumb(fp, spacing=16):
    try:
        wsi = wsd.WholeSlideImage(fp, backend="asap")
        thumb = wsi.get_slide(spacing=spacing)
        return thumb
    except Exception as e:
        print(f"Error loading {fp}: {e}")
        return None

def pad_image(img, target_shape, padding_color=247):
    """Pad image to target_shape with specified background color."""
    h, w = img.shape[:2]
    th, tw = target_shape
    
    # Create background
    padded = np.ones((th, tw, 3), dtype=np.uint8) * padding_color
    
    # Calculate offsets to center the image
    y_offset = (th - h) // 2
    x_offset = (tw - w) // 2
    
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img
    return padded

def main():
    parser = argparse.ArgumentParser(description="Plot WSI thumbnails from different cohorts.")
    parser.add_argument("--rumc-idx", type=int, default=-26, help="Index for RUMC cohort")
    parser.add_argument("--plco-idx", type=int, default=-5, help="Index for PLCO cohort")
    parser.add_argument("--uhc-idx", type=int, default=-3, help="Index for UHC cohort")
    parser.add_argument("--imp-idx", type=int, default=-11, help="Index for IMP cohort")
    parser.add_argument("--spacing", type=int, default=16, help="Spacing for thumbnail extraction")
    parser.add_argument("-o", "--output-dir", type=str, default="results/figures", help="Output directory")
    
    args = parser.parse_args()

    rumc_index = args.rumc_idx
    plco_index = args.plco_idx
    uhc_index = args.uhc_idx
    imp_index = args.imp_idx
    spacing = args.spacing
    output_dir = Path(args.output_dir)

    test_csv = "/data/pathology/projects/clement/leopard/csvs/test-slide2vec.csv"
    uhc_csv = "/data/pathology/projects/clement/leopard/csvs/cologne-slide2vec.csv"
    imp_csv = "/data/pathology/projects/clement/leopard/csvs/brazil-slide2vec-august-2025-revision.csv"

    print("Loading dataframes...")
    test_df = pd.read_csv(test_csv)
    uhc_df = pd.read_csv(uhc_csv)
    imp_df = pd.read_csv(imp_csv)

    rumc_test_df = test_df[test_df.wsi_path.str.contains("radboud")]
    plco_test_df = test_df[~test_df.wsi_path.str.contains("radboud")]

    print(f"RUMC cohort size: {len(rumc_test_df)}")
    print(f"PLCO cohort size: {len(plco_test_df)}")
    print(f"UHC cohort size:  {len(uhc_df)}")
    print(f"IMP cohort size:  {len(imp_df)}")

    cohorts = [
        ("RUMC", rumc_test_df, rumc_index),
        ("PLCO", plco_test_df, plco_index),
        ("IMP", imp_df, imp_index),
        ("UHC", uhc_df, uhc_index)
    ]

    thumbs = []
    names = []
    for name, df, idx in cohorts:
        if idx >= len(df) or idx < -len(df):
            print(f"Index {idx} out of range for {name} (size {len(df)}). Using default -1.")
            idx = -1
        
        wsi_fp = df.wsi_path.values[idx]
        print(f"Processing {name} (index {idx}): {wsi_fp}")
        
        thumb = get_thumb(wsi_fp, spacing=spacing)
        thumbs.append(thumb)
        names.append(name)

    # Filter out None values and find max dimensions
    valid_thumbs = [t for t in thumbs if t is not None]
    if not valid_thumbs:
        print("No thumbnails loaded successfully.")
        return

    max_h = max(t.shape[0] for t in valid_thumbs)
    max_w = max(t.shape[1] for t in valid_thumbs)
    target_shape = (max_h, max_w)

    # Set theme to match other figures
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), constrained_layout=True, dpi=300)
    fig.set_facecolor("#f7f7f7")
    axes = axes.flatten()

    for i, (name, thumb) in enumerate(zip(names, thumbs)):
        axes[i].set_facecolor("#f7f7f7")
        if thumb is not None:
            # Pad to ensure alignment, matching the background color
            padded_thumb = pad_image(thumb, target_shape, padding_color=247)
            axes[i].imshow(padded_thumb)
            axes[i].set_title(name, fontsize=28, fontweight='bold', pad=25)
        else:
            axes[i].text(0.5, 0.5, f"Failed to load {name}", ha='center', va='center', 
                         transform=axes[i].transAxes, fontsize=20, fontweight='bold')
        
        axes[i].axis("off")
        axes[i].set_anchor('C')

    output_path = Path(output_dir) / "cohort-thumbnails.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved plot to {output_path} and {output_path.with_suffix('.pdf')}")

if __name__ == "__main__":
    main()
