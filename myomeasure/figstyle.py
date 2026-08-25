"""Shared figure typography for the manuscript figures (Figures 2-5).

Why this module exists
----------------------
matplotlib ``fontsize=`` is in points relative to the FIGURE canvas, not the
printed page. Two figures with identical font settings but different figure
WIDTHS render text at different sizes once the journal scales each figure to a
fixed column width:

    rendered_pt = fontsize_pt * (column_width_in / figure_width_in)

The manuscript figures had drifted to widths of 7.10-7.56 in, so nominally
identical tick labels printed at 10.0-10.8 pt at 174 mm. This module fixes the
width (``FIG_W``) and centralizes the type scale (``RC``) so every figure script
uses one width and one set of font sizes. Each figure keeps its own HEIGHT and
panel layout -- only the width and the rcParams are unified here.

Type scale note
---------------
The sizes below preserve the manuscript's established ~10.5 pt tick-label look
(the scale Figures 3-5 already used); they are deliberately NOT shrunk. Because
every figure is rendered at ``FIG_W`` = the Cell Press full column width, the
rendered size equals the nominal size, so tick labels land at a common size
across all figures.

A figure may need a width other than FIG_W for its layout to breathe (e.g. a row
of aspect-equal square panels). Use ``rc(width)`` instead of ``RC`` in that case:
it scales the font sizes by ``width / FIG_W`` so that, once the journal shrinks the
figure from ``width`` to the 174 mm column, the type still renders at the same size
as every other figure. Layout stays under the figure's control; only the type scale
is width-compensated.

Usage
-----
    from myomeasure import figstyle
    plt.rcParams.update(figstyle.rc(width))              # width-compensated type
    fig = plt.figure(figsize=(width, height))            # width chosen for layout

    # Shorthand when the figure is happy at the reference column width:
    plt.rcParams.update(figstyle.RC)
    fig = plt.figure(figsize=(figstyle.FIG_W, height))
"""
from __future__ import annotations

# Cell Press full column width: 174 mm = 6.85 in. The reference width against
# which the type scale is defined; figures may render wider/narrower and have
# their fonts compensated via rc(width).
FIG_W = 6.85

# Font-size keys that scale with figure width (so rendered-at-174mm type is
# constant across figures of different render widths).
_SIZE_KEYS = (
    "font.size", "axes.titlesize", "axes.labelsize",
    "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
)

# Shared type scale + line/spine defaults, defined AT the reference width FIG_W.
# Preserves the ~11 pt tick look Figures 3-5 already carried.
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,   # embed TrueType (editable text), do not outline
    "ps.fonttype": 42,
}


def rc(width: float = FIG_W) -> dict:
    """Return RC with the font sizes scaled by ``width / FIG_W``.

    A figure rendered at ``width`` inches and later scaled to the 174 mm column
    will show type at the same physical size as a figure rendered at FIG_W with
    the base RC. Use for figures whose layout needs a non-reference width.
    """
    scale = width / FIG_W
    out = dict(RC)
    for k in _SIZE_KEYS:
        out[k] = RC[k] * scale
    return out
