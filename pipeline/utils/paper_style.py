import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

COLUMN_WIDTH = 252 / 72.27  # inches
TEXT_WIDTH = 516 / 72.27

# One color per arm family; the KB is told apart by the figure panel.
FAMILY_COLORS = {
    "B_only": "#0072B2", "Placebo": "#666666", "SY_raw": "#009E73", "SY": "#D55E00",
    "KB_neighbor": "#E69F00", "KB_hop": "#CC79A7", "Transformer": "#56B4E9", "Transformer_raw": "#1F6F8B",
}

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,

    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "axes.edgecolor": "0.2", "xtick.color": "0.3", "ytick.color": "0.3",
    "text.color": "black", "axes.labelcolor": "black",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,  # see paper_ax

    "legend.frameon": False, "legend.handlelength": 1.1,
    "legend.columnspacing": 0.9, "legend.handletextpad": 0.5,

    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": 0.02, "figure.constrained_layout.h_pad": 0.02,

    # TrueType fonts in the PDF.
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
}


def figsize(cols=1, ratio=0.62):
    """Figure size in inches: one column (`cols=1`) or the full text width."""
    w = COLUMN_WIDTH if cols == 1 else TEXT_WIDTH
    return (w, w * ratio)


def solid(color, alpha=0.8):
    """`color` blended with white as if drawn with `alpha`, but opaque."""
    return tuple(1 - alpha * (1 - v) for v in mcolors.to_rgb(color))


def trim_zero(ax):
    """Y tick labels without the leading zero (0.20 becomes .20). Only for ticks in [-1, 1]."""
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.2f}".replace("0.", ".", 1)))
    return ax


def paper_ax(ax, axis="y"):
    """Grid behind the data, no top/right spines."""
    ax.grid(axis=axis, color="0.85", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax
