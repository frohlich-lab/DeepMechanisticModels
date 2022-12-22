import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path


def plot_and_save_fig(filename: str, figdir: Optional[Path] = None):
    if figdir is None:
        figdir = figdir
    plt.tight_layout()
    figdir.mkdir(exist_ok=True, parents=True)
    if filename is not None:
        plt.savefig(figdir / filename)


MODEL_FEATURE_PREFIX = "INPUT_"
