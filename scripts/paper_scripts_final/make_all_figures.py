"""Single entry point that regenerates EVERY manuscript figure, in the
corrected colourblind-safe palette, into one folder: reports/figures_final/.

Replaces the old two-tier setup (per-figure make_figure*.py written in an
ad-hoc palette + visual_skill/render_all.py re-rendering into a separate
figures-visual-skill/ folder). Now there is one command and one folder.

Palette (Okabe-Ito, all-pairs distinguishable under the common forms of
colour-vision deficiency):
  * Figures 1-2 already bake the palette in (micrograph reds + the
    sky-blue/vermillion dexa dose ramp), so they are just run.
  * Figures 3-6 + supp1 keep their original plotting code; we override the
    handful of colour constants here so the green/red and other non-safe
    pairings become sky-blue / vermillion / Okabe rater colours.

Run in the cellpose conda env:
    python scripts/paper_scripts_final/make_all_figures.py
"""
from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "scripts" / "paper_scripts_final"
DEXA_ANALYSIS = ROOT / "results" / "IGF_dexa_combined" / "make_figures.py"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT))

# Okabe-Ito colourblind-safe palette.
OKABE_ITO = {
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
    "black":          "#000000",
}


def regen_dexa_panels() -> None:
    """Rebuild the dexa bar/histogram PNGs that figure 2 embeds, so they
    carry the corrected palette before figure 2 is assembled."""
    print(f"[dexa] regenerating embedded panels via {DEXA_ANALYSIS}")
    runpy.run_path(str(DEXA_ANALYSIS), run_name="__main__")


def render_fig1() -> None:
    importlib.import_module("make_figure1").main()


def render_fig2() -> None:
    # Palette baked into make_figure2 (sky-blue control + vermillion ramp).
    importlib.import_module("make_figure2").main()


def render_fig3() -> None:
    mod = importlib.import_module("make_figure3")
    mod.C_CON = OKABE_ITO["sky_blue"]
    mod.C_C26 = OKABE_ITO["vermillion"]   # also drives panel A/B frames
    mod.main()


def render_fig4() -> None:
    mod = importlib.import_module("make_figure4")
    mod.C_CTRL = OKABE_ITO["sky_blue"]
    mod.C_C26 = OKABE_ITO["vermillion"]
    mod.C_R1 = OKABE_ITO["blue"]
    mod.C_R2 = OKABE_ITO["reddish_purple"]
    mod.C_R3 = OKABE_ITO["bluish_green"]
    mod.C_PIPE = OKABE_ITO["orange"]
    mod.main()


def render_fig5() -> None:
    mod = importlib.import_module("make_figure5")
    mod.CTRL_COLOR = OKABE_ITO["blue"]
    mod.C26_COLOR = OKABE_ITO["vermillion"]
    mod.DUP_COLOR = "#9CA3AF"
    mod.R1_COLOR = OKABE_ITO["blue"]
    mod.R2_COLOR = OKABE_ITO["reddish_purple"]
    mod.R3_COLOR = OKABE_ITO["bluish_green"]
    mod.PIPE_COLOR = OKABE_ITO["orange"]
    mod.main()


def render_fig6() -> None:
    mod = importlib.import_module("make_figure6")
    mod.C_R1 = OKABE_ITO["blue"]
    mod.C_R2 = OKABE_ITO["reddish_purple"]
    mod.C_R3 = OKABE_ITO["bluish_green"]
    mod.C_PIPE = OKABE_ITO["orange"]
    mod.main()


def render_supp1() -> None:
    mod = importlib.import_module("make_figure_supp1")
    mod.C_CON = OKABE_ITO["sky_blue"]
    mod.C_C26 = OKABE_ITO["vermillion"]
    mod.ARM_COLORS = [mod.C_CON, mod.C_C26]
    mod.main()


def main() -> None:
    out = ROOT / "reports" / "figures_final"
    print(f"Rendering all manuscript figures into {out}")
    regen_dexa_panels()
    render_fig1()
    render_fig2()
    render_fig3()
    render_fig4()
    render_fig5()
    render_fig6()
    render_supp1()
    print("Done.")


if __name__ == "__main__":
    main()
