import argparse
import os
import random
from typing import List

import numpy as np


def discover_npz_files(folder: str) -> List[str]:
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".npz") and os.path.isfile(os.path.join(folder, f))
    ]


def load_flux_from_npz(path: str) -> np.ndarray:
    data = np.load(path)
    # Project README specifies: arr_0 = flux, arr_1 = period weights
    flux = data["arr_0"] if "arr_0" in data else data[list(data.files)[0]]
    return np.asarray(flux, dtype=float)


def plot_lightcurves(files: List[str], cols: int, title: str | None, save_path: str | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to plot lightcurves. Install it with: pip install matplotlib"
        ) from exc

    n = len(files)
    cols = max(1, cols)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
    if title:
        fig.suptitle(title)

    for idx, f in enumerate(files):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        try:
            flux = load_flux_from_npz(f)
            x = np.arange(len(flux))
            ax.plot(x, flux, lw=0.8)
            ax.set_title(os.path.basename(f), fontsize=8)
            ax.set_xlabel("time index")
            ax.set_ylabel("flux (norm)")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", fontsize=8)
            ax.set_title(os.path.basename(f), fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide any unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot a random set of lightcurves from .npz files in a folder"
    )
    parser.add_argument("folder", help="Folder containing .npz files")
    parser.add_argument(
        "-n", "--num", type=int, default=12, help="Number of lightcurves to plot (default: 12)"
    )
    parser.add_argument(
        "-c", "--cols", type=int, default=4, help="Number of columns in the plot grid (default: 4)"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Optional path to save the figure instead of showing"
    )
    parser.add_argument(
        "-t", "--title", default=None, help="Optional overall figure title"
    )

    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder does not exist or is not a directory: {folder}")

    files = discover_npz_files(folder)
    if not files:
        raise SystemExit(f"No .npz files found in: {folder}")

    if args.seed is not None:
        random.seed(args.seed)
    count = max(1, min(args.num, len(files)))
    sampled = random.sample(files, count)

    plot_lightcurves(sampled, cols=args.cols, title=args.title, save_path=args.output)


if __name__ == "__main__":
    main()



