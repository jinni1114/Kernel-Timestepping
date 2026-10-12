"""Plot styling shared by the example notebooks (matplotlib only; not part of the package)."""
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE, INK, SECONDARY, MUTED, GRIDLINE, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # categorical series, always in this order
NEUTRAL = "#c3c2b7"                                             # de-emphasised reference series

# signed fields: blue (negative) - neutral gray (zero) - red (positive)
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8f1d1c"])


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 110,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "regular", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.labelcolor": SECONDARY, "axes.titlecolor": INK, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "grid.color": GRIDLINE, "grid.linewidth": 0.8,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 8,
    })


def cell_edges(centers):
    """Edges of the cells around increasing points (for pcolormesh with one cell per point)."""
    centers = np.asarray(centers, dtype=float)
    mid = (centers[1:] + centers[:-1]) / 2
    return np.concatenate([[2 * centers[0] - mid[0]], mid, [2 * centers[-1] - mid[-1]]])


def field_map(ax, t, x, values, limit, title):
    """Signed field values (len(t), len(x)) on the (t, x) plane, colour range [-limit, limit] centred at zero."""
    mesh = ax.pcolormesh(cell_edges(t), cell_edges(x), np.asarray(values).T, cmap=DIVERGING,
                         norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit), shading="flat", rasterized=True)
    ax.set(title=title, xlabel="t", ylabel="x", xlim=(t[0], t[-1]), ylim=(x[0], x[-1]))
    return mesh


def marker(ax, x, y, color, **kwargs):
    """A data marker with a surface-coloured ring, so it stays readable on top of lines."""
    return ax.plot(x, y, "o", color=color, mec=SURFACE, mew=1.5, **kwargs)
