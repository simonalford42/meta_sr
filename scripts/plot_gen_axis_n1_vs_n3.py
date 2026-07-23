"""Regenerate the n1 vs n3 (vs n10) per-generation figure.

Uses the 7/16/26 continuation runs (nn_n1o20 / nn_n3o20 continued to 30 more
generations, seeds 0-4) plus the older n10o20 runs. Writes
plots/gen_axis_n1_vs_n3/gen_axis_n1_vs_n3.png.
"""
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_axis_plots as gp

# 7/16/26 continuation runs (continued nn_n1o20 / nn_n3o20 to gen 30), seeds 0-4.
N1_NONE = [883427, 883429, 883431, 883433, 883435]   # nn_n1o20 seeds 0-4
N3_NONE = [883428, 883430, 883432, 883434, 883436]   # nn_n3o20 seeds 0-4
N10_NONE = [568245, 568246]                          # nn_n10o20 seeds 0-1 (older)

api = gp.get_api()
widx = gp.build_wandb_index()

n1 = gp.load_method(api, widx, N1_NONE, "n1")
n3 = gp.load_method(api, widx, N3_NONE, "n3")
n10 = gp.load_method(api, widx, N10_NONE, "n10")

gp.render(
    [("n1 (n_runs 1)",   n1,  gp.COLOR(0), "o"),
     ("n3 (n_runs 3)",   n3,  gp.COLOR(3), "s"),
     ("n10 (n_runs 10)", n10, gp.COLOR(2), "^")],
    gp.OUTDIR / "gen_axis_n1_vs_n3.png",
    "n1 vs n3 vs n10 (all reeval=none, offspring=20) — per generation\n"
    "n1/n3 = 7/16 continuation runs (to gen 30); n3/n10 pay 3x/10x eval cost (factored out here)",
    panels=gp.PANELS_COMPARISON,
)
print("done")
