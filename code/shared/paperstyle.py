# -*- coding: utf-8 -*-
"""Common style for manuscript figures — the earlier (clean white background) style.

Keeps the same API but drops all decoration: no panel background, no gradients,
no halos. Lines instead of bands, white-edged dots instead of glossy dots.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

C = {
    "stim": "#1D6B8C", "stim_lt": "#4FA3B0",
    "nost": "#B4651A", "nost_lt": "#E0A067",
    "up": "#1D6B8C", "down": "#9A3B3B",
    "ns": "#C4C9CC", "ink": "#1A1D26", "ink2": "#6B7378",
    "star": "#E8B51F",
    "paper": "white", "panel": "white", "grid": "#E6E3DD",
}
ROI = {"Frontal": "#1D6B8C", "Temporal": "#B4651A",
       "Central": "#4C7A34", "Parietal": "#7A5195"}
ROIS = ["Frontal", "Temporal", "Central", "Parietal"]


def use(base=7.5):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": base,
        "axes.titlesize": base + .9, "axes.labelsize": base,
        "axes.titleweight": "normal",
        "axes.linewidth": .7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "black",
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "xtick.color": "black", "ytick.color": "black",
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.major.width": .7, "ytick.major.width": .7,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.frameon": False,
        "savefig.dpi": 400, "figure.dpi": 140,
    })


def ramp(color, light=.92, n=256):
    r, g, b = to_rgb(color)
    top = tuple(min(1.0, 1 - (1 - v) * (1 - light) * 3.2) for v in (r, g, b))
    return LinearSegmentedColormap.from_list("r", [top, (r, g, b)], N=n)


def panel(ax, **kw):
    """No background — earlier style."""
    return None


def grad_barh(ax, lo, hi, y, color, h=.26, zorder=3, flip=False):
    return ax.plot([lo, hi], [y, y], color=color, lw=2.2,
                   solid_capstyle="round", zorder=zorder)[0]


def grad_ci(ax, lo, hi, y, color, center=None, h=.26, zorder=3):
    return ax.plot([lo, hi], [y, y], color=color, lw=1.9,
                   solid_capstyle="round", zorder=zorder)[0]


def grad_bar(ax, x, h, color, w=.52, zorder=3, radius=None):
    return ax.bar([x], [h], width=w, color=color, lw=0, zorder=zorder)


def grad_area(ax, x, y, color, base=None, alpha=.55, zorder=1):
    return None


def dot(ax, x, y, color, s=40, zorder=5, halo=False):
    return ax.scatter(x, y, s=s, color=color, zorder=zorder, lw=.8,
                      edgecolor="white")


def line(ax, x, y, color, lw=1.6, **kw):
    kw.pop("solid_capstyle", None)
    return ax.plot(x, y, color=color, lw=lw, **kw)[0]


def halfviolin(ax, v, pos, color, width=.42, side=1, alpha=.26):
    from scipy import stats as _st
    v = np.asarray(v, float)
    if v.size < 3 or np.ptp(v) == 0:
        return
    k = _st.gaussian_kde(v)
    yy = np.linspace(v.min(), v.max(), 200)
    d = k(yy); d = d / d.max() * width
    ax.fill_betweenx(yy, pos, pos + side * d, facecolor=color, alpha=alpha,
                     lw=0, zorder=2)
    ax.plot(pos + side * d, yy, color=color, lw=.8, alpha=.7, zorder=3)


def title(ax, main, sub=None, pad=8):
    ax.set_title(main, pad=pad)


def tag(f, ax, s, dx=.052, dy=.030, size=11):
    p = ax.get_position()
    f.text(max(p.x0 - dx, .003), p.y1 + dy, s, fontsize=size,
           fontweight="bold", va="bottom", ha="left")


def _round_rect(ax, lo, hi, y, h):
    from matplotlib.patches import Rectangle
    r = Rectangle((lo, y - h / 2), hi - lo, h, transform=ax.transData,
                  facecolor="none", edgecolor="none")
    ax.add_patch(r)
    return r
