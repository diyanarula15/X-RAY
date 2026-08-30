"""Deck-matching visual style. Dark, flat, no matplotlib default blue anywhere."""
from __future__ import annotations

import matplotlib as mpl

BG = "#0B0B0F"
CARD = "#16161C"
RED = "#E10600"
AMBER = "#FFC300"
WHITE = "#FFFFFF"
GRAY = "#9AA0A6"
DIM = "#5A5A66"
GREEN = "#2ECC71"
GRIDLINE = "#26262E"
FONT = "DejaVu Sans"

OURS = AMBER
RIVAL = RED


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": FONT,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": DIM,
        "axes.labelcolor": GRAY,
        "axes.titlecolor": WHITE,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "text.color": WHITE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 120,
        "axes.prop_cycle": mpl.cycler(color=[AMBER, RED, GREEN, GRAY, WHITE]),
    })


def strip(ax, keep_x: bool = False) -> None:
    ax.set_facecolor(BG)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if keep_x:
        ax.get_xaxis().set_visible(True)
