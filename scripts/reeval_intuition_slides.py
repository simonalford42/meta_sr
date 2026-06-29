"""Intuition slides: why reevaluation helps an evolutionary SR run.

A conceptual slideshow (not tied to a specific run) that motivates the smart
reevaluation logic in `smart_reeval.py` / the MC analysis in `monte_carlo.py`.

Story (parents are assumed drawn *uniformly at random* from the population, so
the quantity that matters is the population's average TRUE fitness):

  1. Each member has a hidden true fitness and a noisy estimate (few seeds).
  2. The population is the top-K by NOISY estimate, so a lucky overestimate
     sneaks in and an unlucky underestimate gets cut -> the population's true
     average sits below the optimum (the optimizer's / winner's curse).
  3. Reevaluating members at/near the selection boundary sharpens the estimate,
     dropping the impostor out and pulling the gem in -> the population's true
     average rises (= higher true fitness of the parents drawn next gen).
  4. Sampling offspring is the complementary lever (it raises the ceiling).
     The algorithm budgets evaluations between the two.

Outputs PNG slides + a combined PDF under plots/reeval_intuition/.

Usage: python scripts/reeval_intuition_slides.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle

# Real method functions (project root) — drives the reeval EI curve and the
# offspring EI number on the method slides.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monte_carlo import (
    simulate_reeval_expected_improvement,
    topk_tourney_batch_selection_fn,
)
from offspring_mc import offspring_expected_improvement

# ---------------------------------------------------------------------------
# Visual theme
# ---------------------------------------------------------------------------
C_TRUE = "#2e8b57"     # true fitness (hidden) — sea green
C_EST = "#e8853b"      # noisy estimate (observed) — orange
C_IMP = "#d1495b"      # impostor (lucky overestimate) — red
C_GEM = "#2c7fb8"      # gem (unlucky underestimate) — blue
C_REEVAL = "#6a4c93"   # reevaluation accent — purple
C_INBAND = "#e9f6ee"   # population band fill
C_BG = "white"
FIG_W, FIG_H = 12.8, 7.2  # 16:9

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "figure.facecolor": C_BG,
    "axes.facecolor": C_BG,
    "savefig.facecolor": C_BG,
})

# ---------------------------------------------------------------------------
# Toy population: hardcoded so the story is legible.
#   true = hidden true fitness;  est = noisy 1-seed estimate.
# Designed so that ranking by `est` lets the impostor (K) in and cuts the gem (D).
# ---------------------------------------------------------------------------
POP_K = 6          # population size (survivors = top-K by estimate)
SIGMA1 = 0.085     # 1-seed estimate noise (std)
N_REEVAL = 9       # seeds after a reevaluation -> sigma/sqrt(N_REEVAL)
SIGMA_RE = SIGMA1 / np.sqrt(N_REEVAL)

CANDS = [
    # name   true   est(1-seed)   The action is at the SELECTION BOUNDARY: noise
    ("A", 0.91, 0.90),   #          swaps the rank-6 and rank-7 members. Arms far
    ("B", 0.85, 0.86),   #          from the cut are estimated about right; the
    ("C", 0.84, 0.83),   #          impostor (K) and gem (D) sit right on the cut.
    ("E", 0.79, 0.80),
    ("F", 0.75, 0.77),
    ("K", 0.62, 0.74),   # IMPOSTOR: estimate sneaks into the top-6, truly weak
    ("D", 0.83, 0.71),   # GEM: estimate just misses the cut, truly strong
    ("H", 0.66, 0.66),
    ("I", 0.61, 0.62),
    ("J", 0.57, 0.58),
    ("G", 0.54, 0.55),
    ("L", 0.49, 0.50),
]
NAME = {c[0]: {"true": c[1], "est": c[2]} for c in CANDS}
IMPOSTOR, GEM = "K", "D"

XLIM = (0.40, 1.00)


def _by(key):
    """Candidates sorted descending by 'true' or 'est'."""
    return sorted(CANDS, key=lambda c: (c[1] if key == "true" else c[2]), reverse=True)


def pop_true_avg(members):
    return float(np.mean([NAME[m]["true"] for m in members]))


def topk_by_est():
    return [c[0] for c in _by("est")[:POP_K]]


def topk_by_true():
    return [c[0] for c in _by("true")[:POP_K]]


# ---------------------------------------------------------------------------
# Shared row drawer: candidates as horizontal "lollipops".
# y goes top (best) -> bottom; each row shows the noise gap est<->true.
# ---------------------------------------------------------------------------
def draw_rows(ax, order, *, show_true=True, show_est=True, est_err=SIGMA1,
              cut_after=None, highlight=None, dim=None, boxes=True):
    """order: list of (name, true, est) top->bottom. cut_after: draw population
    band over the first `cut_after` rows. highlight: {name: color}."""
    highlight = highlight or {}
    dim = dim or set()
    n = len(order)
    ys = list(range(n - 1, -1, -1))  # so first item is at top

    if cut_after is not None:
        ax.axhspan(ys[cut_after - 1] - 0.5, ys[0] + 0.5,
                   xmin=0, xmax=1, color=C_INBAND, zorder=0)
        ax.axhline(ys[cut_after - 1] - 0.5, color="0.45", lw=1.6, ls="--", zorder=1)

    for (name, tval, eval_), y in zip(order, ys):
        a = 0.18 if name in dim else 1.0
        # connecting gap line
        if show_true and show_est:
            ax.plot([eval_, tval], [y, y], color="0.7", lw=2.0,
                    alpha=0.6 * a, zorder=2)
        if show_est:
            ax.errorbar(eval_, y, xerr=est_err, fmt="o", ms=11, color=C_EST,
                        ecolor=C_EST, elinewidth=2, capsize=4, alpha=a, zorder=4,
                        markeredgecolor="white", markeredgewidth=1.0)
        if show_true:
            ax.scatter(tval, y, s=150, color=C_TRUE, alpha=a, zorder=5,
                       edgecolors="white", linewidths=1.2, marker="D")
        # row label + optional highlight box
        lbl_color = highlight.get(name, "0.2")
        weight = "bold" if name in highlight else "normal"
        ax.text(XLIM[0] - 0.012, y, name, ha="right", va="center",
                fontsize=14, color=lbl_color, fontweight=weight)
        if name in highlight and boxes:
            ax.add_patch(Rectangle((XLIM[0], y - 0.46), XLIM[1] - XLIM[0], 0.92,
                                   fill=False, ec=highlight[name], lw=2.2,
                                   zorder=6))

    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("fitness")
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def legend_handles(include_band=False):
    h = [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=C_TRUE,
                   markersize=12, label="true fitness (hidden)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EST,
                   markersize=12, label="noisy estimate (what we measure)"),
    ]
    if include_band:
        h.append(plt.Line2D([0], [0], marker="s", color="w",
                            markerfacecolor=C_INBAND, markeredgecolor="0.45",
                            markersize=14, label=f"population (top-{POP_K} by estimate)"))
    return h


def footnote(fig, text):
    fig.text(0.5, 0.025, text, ha="center", va="bottom", fontsize=13,
             color="0.35", style="italic")


def bell(ax, center, std, y0, height, color, n=140):
    """Draw a horizontal Gaussian 'posterior' bump sitting on baseline y0."""
    xs = np.linspace(center - 3.3 * std, center + 3.3 * std, n)
    pdf = np.exp(-0.5 * ((xs - center) / std) ** 2)
    ax.fill_between(xs, y0, y0 + height * pdf, color=color, alpha=0.22, zorder=2)
    ax.plot(xs, y0 + height * pdf, color=color, alpha=0.7, lw=1.3, zorder=2)


# --- Real computations for the method slides, on the toy pool ----------------
# Per-seed noise (not the ±1σ estimate bar of the earlier slides); each arm has
# N_INIT seeds so its posterior width is SIGMA_SEED/√N_INIT.
SIGMA_SEED = 0.15
N_INIT = 3
METHOD: dict = {}


def compute_method_data():
    if METHOD:
        return METHOD
    sel = topk_tourney_batch_selection_fn(topk=POP_K, n=2)
    order = _by("est")
    mu = np.array([c[2] for c in order], dtype=float)
    N = np.full(mu.size, float(N_INIT))
    curve = simulate_reeval_expected_improvement(
        mu, SIGMA_SEED, N, sel, M=4000, B_max=48, rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)
    emp = np.clip(rng.normal(0.62, 0.09, 36), 0.40, 0.99)
    res = offspring_expected_improvement(
        pop_mu=mu, pop_N=N, offspring_empirical=emp, sigma=SIGMA_SEED,
        n_initial_evals=N_INIT, batch_selection_fn=sel, rng=rng)
    METHOD.update(sel=sel, order=order, mu=mu, N=N, curve=curve, emp=emp, res=res)
    return METHOD


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------
def slide_title(fig):
    fig.clf()
    fig.text(0.5, 0.70, "Why reevaluate?", ha="center", fontsize=40,
             fontweight="bold", color="0.15")
    fig.text(0.5, 0.585, "Noise, selection, and the value of a second look",
             ha="center", fontsize=22, color="0.4")
    # mini lollipop motif
    ax = fig.add_axes([0.30, 0.20, 0.40, 0.28])
    demo = [("", NAME[n]["true"], NAME[n]["est"]) for n in ["A", "K", "D", "L"]]
    draw_rows(ax, demo, est_err=SIGMA1)
    ax.set_xlabel("")
    ax.legend(handles=legend_handles(), loc="lower center",
              bbox_to_anchor=(0.5, -0.55), ncol=2, frameon=False, fontsize=13)


def slide_setup(fig):
    """Slide 1: we only ever see the noisy estimate."""
    fig.clf()
    ax = fig.add_axes([0.10, 0.16, 0.84, 0.66])
    draw_rows(ax, _by("true"), est_err=SIGMA1)
    ax.set_title("Each candidate has a hidden true fitness — we see only a noisy estimate",
                 pad=16, wrap=True)
    ax.legend(handles=legend_handles(), loc="upper left", frameon=True,
              fontsize=13, framealpha=0.95)
    footnote(fig, "Every member is scored on a few random seeds, so the measured "
                  "value = true fitness + evaluation noise (error bars = ±1σ).")


def slide_problem(fig):
    """Slide 2: ranking by estimate misranks; impostor in, gem out."""
    fig.clf()
    ax = fig.add_axes([0.10, 0.16, 0.84, 0.66])
    order = _by("est")
    draw_rows(ax, order, est_err=SIGMA1, cut_after=POP_K,
              highlight={IMPOSTOR: C_IMP, GEM: C_GEM})
    ax.set_title(f"Selecting the top-{POP_K} by estimate lets impostors in and cuts gems out",
                 pad=16, wrap=True)

    # annotations
    ys = list(range(len(order) - 1, -1, -1))
    pos = {c[0]: y for c, y in zip(order, ys)}
    ax.annotate("lucky overestimate:\nsneaks into the pool, truly weak",
                xy=(NAME[IMPOSTOR]["est"], pos[IMPOSTOR]),
                xytext=(0.42, pos[IMPOSTOR] + 2.8),
                fontsize=13, color=C_IMP, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=C_IMP, lw=2))
    ax.annotate("unlucky underestimate:\ntruly strong, cut from the pool",
                xy=(NAME[GEM]["est"], pos[GEM]),
                xytext=(0.42, pos[GEM] - 2.6),
                fontsize=13, color=C_GEM, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=C_GEM, lw=2))

    ax.legend(handles=legend_handles(include_band=True), loc="lower right",
              frameon=True, fontsize=12, framealpha=0.95)
    opt = pop_true_avg(topk_by_true())
    got = pop_true_avg(topk_by_est())
    footnote(fig, f"True avg fitness of the chosen pool = {got:.3f} vs {opt:.3f} "
                  f"for the truly-best {POP_K} — noise costs {opt - got:.3f}.")


def _contested(order, cut_val):
    """Names whose estimate ±1σ straddles the selection cut (uncertain
    membership). These are the arms TTTS reevaluates."""
    return [c[0] for c in order if abs(c[2] - cut_val) < SIGMA1]


def slide_which_to_reeval(fig):
    """Q1: which candidates to reevaluate? -> the ones near the cut whose
    selection is uncertain (top-two Thompson sampling)."""
    fig.clf()
    ax = fig.add_axes([0.10, 0.16, 0.84, 0.66])
    order = _by("est")
    cut_val = 0.5 * (order[POP_K - 1][2] + order[POP_K][2])
    contested = _contested(order, cut_val)
    draw_rows(ax, order, est_err=SIGMA1, cut_after=POP_K,
              highlight={n: C_REEVAL for n in contested}, boxes=False)
    ax.set_title("Q1 — Which candidates to reevaluate?", pad=16, wrap=True)

    ys = list(range(len(order) - 1, -1, -1))
    pos = {c[0]: y for c, y in zip(order, ys)}

    # single bracket around the contested block (uncertain selection)
    y_lo = min(pos[n] for n in contested)
    y_hi = max(pos[n] for n in contested)
    ax.add_patch(Rectangle((XLIM[0], y_lo - 0.5), XLIM[1] - XLIM[0],
                           (y_hi - y_lo) + 1.0, fill=False, ec=C_REEVAL,
                           lw=2.6, zorder=6))
    # callout pointing to the contested block
    ax.annotate("near the cut the ranking is uncertain\n"
                "→ reevaluate these (top-two Thompson sampling)",
                xy=(0.62, y_hi + 0.5),
                xytext=(0.41, 9.5),
                fontsize=14, color=C_REEVAL, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=C_REEVAL, lw=2.2))
    # skip labels for the clearly-decided extremes
    ax.text(0.41, pos["A"], "clearly in → skip (already certain)",
            ha="left", va="center", fontsize=12.5, color="0.45")
    ax.text(0.60, pos["L"], "clearly out → skip (already certain)",
            ha="left", va="center", fontsize=12.5, color="0.45")

    footnote(fig, "A clear winner or loser won't flip — reevaluate only where "
                  "the ranking is still in doubt.")


def slide_how_many(fig):
    """Q2: how big is the reeval budget B? Reevaluate while the next reeval beats
    a fresh offspring; the crossing is B*."""
    fig.clf()
    ax = fig.add_axes([0.10, 0.17, 0.84, 0.63])

    # Synthetic but faithful shapes: the marginal value of the next reeval decays
    # (diminishing returns, like the smoothed MEI in offspring_improvement.py);
    # a new offspring is worth a fixed amount for the same seed cost.
    m0, tau, off = 0.0130, 12.0, 0.0042
    B = np.linspace(0, 48, 400)
    mei = m0 * np.exp(-B / tau)
    b_star = tau * np.log(m0 / off)

    ax.axvspan(0, b_star, color=C_REEVAL, alpha=0.07)
    ax.axvspan(b_star, B[-1], color=C_GEM, alpha=0.07)
    ax.plot(B, mei, color=C_REEVAL, lw=3, zorder=4,
            label="(1) gain from the next reevaluation")
    ax.axhline(off, color=C_GEM, lw=2.6, ls="--", zorder=3,
               label="(2) gain from one new offspring (same cost)")
    ax.axvline(b_star, color="0.4", lw=1.6, ls=":")
    ax.scatter([b_star], [off], s=110, color="k", zorder=6)

    ax.annotate(f"(3) reevaluate until they're equal\n→  B*  ≈ {b_star:.0f} reevaluations",
                xy=(b_star, off), xytext=(b_star + 3, off + 0.0034),
                fontsize=14, fontweight="bold", color="0.15", ha="left",
                arrowprops=dict(arrowstyle="->", color="0.3", lw=2))
    ax.text(b_star / 2, m0 * 0.93, "spend on\nreevaluation", ha="center",
            va="top", fontsize=14, color=C_REEVAL, fontweight="bold")
    ax.text((b_star + B[-1]) / 2, off + 0.0009, "spend on offspring",
            ha="center", va="bottom", fontsize=14, color=C_GEM, fontweight="bold")

    ax.set_xlim(0, B[-1])
    ax.set_ylim(0, m0 * 1.12)
    ax.set_xlabel("reevaluations already spent this generation,  B")
    ax.set_ylabel("expected gain in\nnext-gen parent fitness")
    ax.set_title("Q2 — How many reevaluations? (the budget B)", pad=16, wrap=True)
    ax.legend(loc="upper right", fontsize=13, framealpha=0.95)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    footnote(fig, "Goal: maximize next-gen parent fitness. Each reeval buys less "
                  "than the last — keep going only while it beats one offspring.")


def slide_two_levers(fig):
    """Slide 5: reeval vs offspring — both lift next-gen parent fitness."""
    fig.clf()
    fig.suptitle("Two levers to raise the next generation's parent fitness",
                 fontsize=20, fontweight="bold", y=0.95, wrap=True)
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.96, top=0.80, bottom=0.26,
                          wspace=0.18)

    # Left panel: REEVALUATE (sharpen estimates / fix the pool)
    axL = fig.add_subplot(gs[0])
    axL.set_title("Reevaluate", fontsize=20, color=C_EST, pad=10)
    sub = [(n, NAME[n]["true"], NAME[n]["est"]) for n in ["A", IMPOSTOR, GEM, "L"]]
    draw_rows(axL, sub, est_err=SIGMA1, highlight={IMPOSTOR: C_IMP, GEM: C_GEM})
    axL.set_xlabel("fitness")
    axL.text(0.5, -0.24,
             "Sharpen estimates of existing members\n"
             "→ better accuracy, fix who is in the pool",
             transform=axL.transAxes, ha="center", va="top", fontsize=14,
             color="0.25")

    # Right panel: OFFSPRING (raise the ceiling)
    axR = fig.add_subplot(gs[1])
    axR.set_title("Sample offspring", fontsize=20, color=C_GEM, pad=10)
    cur = sorted([c[1] for c in _by("true")[:POP_K]], reverse=True)
    ys = list(range(len(cur)))
    axR.scatter(cur, ys, s=150, color=C_TRUE, marker="D", zorder=4,
                edgecolors="white", linewidths=1.2)
    # new offspring above the current best
    new_y = len(cur)
    axR.scatter([0.96], [new_y], s=240, color=C_GEM, marker="*", zorder=5,
                edgecolors="white", linewidths=1.2)
    axR.add_patch(FancyArrowPatch((0.93, new_y - 0.55), (0.955, new_y - 0.12),
                  arrowstyle="-|>", mutation_scale=22, color=C_GEM, lw=2.5))
    axR.axhline(max(ys) + 0.5, color="0.6", ls="--", lw=1.4)
    axR.text(XLIM[0] + 0.01, new_y, "new candidate\ncan beat the best",
             ha="left", va="center", fontsize=13, color=C_GEM, fontweight="bold")
    axR.set_xlim(*XLIM)
    axR.set_ylim(-0.7, new_y + 0.7)
    axR.set_yticks([])
    axR.set_xlabel("fitness")
    axR.grid(axis="x", alpha=0.25)
    for s in ("top", "right", "left"):
        axR.spines[s].set_visible(False)
    axR.text(0.5, -0.24,
             "Introduce new members near good parents\n"
             "→ raises the ceiling of achievable fitness",
             transform=axR.transAxes, ha="center", va="top", fontsize=14,
             color="0.25")

    footnote(fig, "Offspring and reevaluations cost the same seeds; smart reeval "
                  "spends them where the marginal gain is larger (the B* point).")


def slide_math(fig):
    """Formal statement: the objective and how each EI is computed."""
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Expected parent-fitness improvement — the math",
                 fontsize=20, fontweight="bold", y=0.965)

    def panel(x0, y0, x1, y1, header, color, body, fc, ls=1.95, fs=14.0,
              hdr_gap=0.075):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True,
                               facecolor=fc, edgecolor=color, lw=2.0, zorder=1))
        ax.text(x0 + 0.012, y1 - 0.018, header, fontsize=15, fontweight="bold",
                color=color, va="top", ha="left", zorder=3)
        ax.text(x0 + 0.018, y1 - hdr_gap, body, fontsize=fs, color="0.12",
                va="top", ha="left", zorder=3, linespacing=ls)

    # Setup / objective
    setup = (
        r"Pool of $k$ arms. Arm $i$:  posterior  $\theta_i \sim \mathcal{N}(\mu_i,\ \sigma^2/N_i)$  from $N_i$ noisy seeds." "\n"
        r"Parent-selection rule  $\pi(\mu)$ :  probability of picking each arm (top-$k$ + tournament)." "\n"
        r"Objective = true fitness of the selected parent:    $\mathrm{PF}(\mu;\theta)=\sum_i \pi_i(\mu)\,\theta_i=\pi(\mu)\cdot\theta$" "\n"
        r"One evaluation, two uses — reevaluate an existing arm, or spend $\Delta=n_{init}$ seeds on a new offspring."
    )
    panel(0.04, 0.700, 0.96, 0.930, "Setup", "0.35", setup, "0.96",
          ls=1.55, fs=13.5, hdr_gap=0.058)

    reeval = (
        r"$\mathrm{EI}_{re}(B)=\mathrm{E}_\theta[\,\pi(\mu^{(B)})\!\cdot\!\theta\,]-\mathrm{E}_\theta[\,\pi(\mu^{(0)})\!\cdot\!\theta\,]$" "\n"
        r"per world  $\theta\sim\mathcal{N}(\mu,\sigma^2/N)$,  then for $b=1\dots B$:" "\n"
        r"    pick  $a\sim\psi_{TTTS}$,    seed  $y\sim\mathcal{N}(\theta_a,\sigma)$" "\n"
        r"    $\mu_a\leftarrow\frac{N_a\mu_a+y}{N_a+1}$,    $N_a\leftarrow N_a+1$" "\n"
        r"marginal:  $\mathrm{MEI}(B;\Delta)=\mathrm{EI}_{re}(B{+}\Delta)-\mathrm{EI}_{re}(B)$"
    )
    panel(0.04, 0.275, 0.49, 0.675, "Reevaluate  (Monte-Carlo)", C_REEVAL,
          reeval, "#f4f1f8")

    offspring = (
        r"draw  $v\sim$ empirical offspring means $\{v_1,\dots,v_E\}$" "\n"
        r"extend pool with arm  $(\mu{=}v,\ N{=}n_{init})$" "\n"
        r"$\mathrm{EI}_{off}=\frac{1}{E}\sum_{e}\pi(\mu^{+v_e})\!\cdot\!\mu^{+v_e}\ -\ \pi(\mu)\!\cdot\!\mu$" "\n\n"
        r"pool fixed per candidate $\Rightarrow$" "\n"
        r"$\mathrm{E}[\pi\!\cdot\!\theta]=\pi\!\cdot\!\mu$   (no truth sampling)"
    )
    panel(0.51, 0.275, 0.96, 0.675, "Add one offspring  (analytic)", C_GEM,
          offspring, "#eef4f8")

    decision = (
        r"Reevaluate while one reeval still beats one offspring:        "
        r"$B^{*}=\min\{\,B:\ \mathrm{MEI}(B;\Delta)\,\leq\,\mathrm{EI}_{off}\,\}$"
    )
    panel(0.04, 0.075, 0.96, 0.245, "Decision", C_TRUE, decision, "#eef6f0")


def slide_reeval_ei_setup(fig):
    """Reeval EI, step 1: sample plausible 'true values' from the posteriors."""
    fig.clf()
    ax = fig.add_axes([0.10, 0.16, 0.56, 0.66])
    std = SIGMA_SEED / np.sqrt(N_INIT)
    names = ["A", "F", "K", "D", "I"]
    truth_off = {"A": -0.6, "F": 0.9, "K": -1.5, "D": 1.6, "I": 0.4}  # in σ units
    ys = list(range(len(names) - 1, -1, -1))
    for nm, y in zip(names, ys):
        mu = NAME[nm]["est"]
        bell(ax, mu, std, y, 0.78, C_EST)
        ax.scatter([mu], [y], s=70, color=C_EST, zorder=4,
                   edgecolors="white", linewidths=1.0)
        truth = mu + truth_off[nm] * std
        ax.scatter([truth], [y], s=150, color=C_TRUE, marker="D", zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.text(XLIM[0] - 0.012, y, nm, ha="right", va="center", fontsize=14,
                color="0.2")
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.6, len(names) - 0.1 + 0.8)
    ax.set_yticks([])
    ax.set_xlabel("fitness")
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EST,
                   markersize=11, label="estimate μ  &  posterior N(μ, σ²/N)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=C_TRUE,
                   markersize=12, label="one sampled 'true value'"),
    ], loc="upper right", fontsize=11.5, framealpha=0.95)
    ax.set_title("Reevaluation EI — step 1: imagine the true values",
                 pad=16, wrap=True)

    ax2 = fig.add_axes([0.69, 0.16, 0.29, 0.66]); ax2.axis("off")
    ax2.text(0, 1.0,
             "Each arm's belief is a\nposterior  θ ~ N(μ, σ²/N).\n\n"
             "① Draw one plausible true\n   value per arm (◆)\n   = one simulated world.\n\n"
             "② Baseline = true fitness of\n   the parent today's\n   estimates would pick.\n\n"
             "Repeat for M ≈ thousands\nof worlds.",
             transform=ax2.transAxes, va="top", ha="left", fontsize=14,
             color="0.2", linespacing=1.35)
    footnote(fig, "The sampled truth is the hidden fitness in that imagined world; "
                  "reevaluations (next slide) reveal noisy glimpses of it.")


def slide_reeval_ei_loop(fig):
    """Reeval EI, step 2: simulate the extra seeds -> the EI(B) curve."""
    fig.clf()
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15],
                          left=0.07, right=0.96, top=0.80, bottom=0.16,
                          wspace=0.24)
    # Left: one arm's running estimate sharpening toward its fixed truth.
    axL = fig.add_subplot(gs[0])
    truth = 0.62
    stages = [(2.0, 0.74, N_INIT, "0 reevals"),
              (1.0, 0.68, N_INIT + 2, "+2 seeds"),
              (0.0, 0.645, N_INIT + 5, "+5 seeds")]
    axL.axvline(truth, color=C_TRUE, ls=":", lw=1.6, zorder=1)
    for y, mu, n, lbl in stages:
        axL.errorbar(mu, y, xerr=SIGMA_SEED / np.sqrt(n), fmt="o", ms=12,
                     color=C_EST, ecolor=C_EST, elinewidth=2.5, capsize=5,
                     zorder=4, markeredgecolor="white", markeredgewidth=1.1)
        axL.text(XLIM[1] - 0.01, y, lbl, ha="right", va="center", fontsize=12,
                 color="0.4")
    axL.scatter([truth], [2.0], s=160, color=C_TRUE, marker="D", zorder=5,
                edgecolors="white", linewidths=1.2)
    axL.annotate("", xy=(0.645, 0.32), xytext=(0.74, 1.72),
                 arrowprops=dict(arrowstyle="-|>", color=C_REEVAL, lw=2.4))
    axL.text(truth - 0.005, 2.45, "fixed truth", color=C_TRUE, ha="center",
             fontsize=12.5, fontweight="bold")
    axL.set_xlim(0.45, 1.0)
    axL.set_ylim(-0.5, 2.9)
    axL.set_yticks([])
    axL.set_xlabel("fitness")
    axL.grid(axis="x", alpha=0.25)
    for s in ("top", "right", "left"):
        axL.spines[s].set_visible(False)
    axL.set_title("each simulated reeval\npulls μ toward the truth", fontsize=15,
                  pad=10)

    # Right: the real cumulative EI(B) curve from simulate_reeval_expected_improvement.
    m = compute_method_data()
    axR = fig.add_subplot(gs[1])
    B = np.arange(len(m["curve"]))
    axR.plot(B, m["curve"], color=C_REEVAL, lw=3, zorder=4)
    axR.fill_between(B, 0, m["curve"], color=C_REEVAL, alpha=0.10)
    axR.set_xlim(0, B[-1])
    axR.set_ylim(0, m["curve"].max() * 1.15)
    axR.set_xlabel("reevaluations,  B")
    axR.set_ylabel("E[gain in parent fitness]\n(cumulative, vs baseline)")
    axR.grid(alpha=0.25)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    axR.set_title("EI(B): averaged over M worlds", fontsize=15, pad=10)
    axR.annotate("slope = the marginal\ngain plotted in Q2",
                 xy=(6, m["curve"][6]), xytext=(16, m["curve"].max() * 0.45),
                 fontsize=12.5, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=1.8))

    fig.suptitle("Reevaluation EI — step 2: simulate the extra seeds",
                 fontsize=18, fontweight="bold", y=0.95, wrap=True)
    footnote(fig, "Per world: TTTS picks an arm → draw a seed ~ N(truth, σ) → "
                  "update μ → re-pick the parent → score its truth vs the baseline.")


def slide_offspring_ei_dist(fig):
    """Offspring EI, step 1: the empirical distribution of recent offspring."""
    fig.clf()
    m = compute_method_data()
    ax = fig.add_axes([0.10, 0.20, 0.84, 0.60])
    emp = m["emp"]
    cut = 0.5 * (m["order"][POP_K - 1][2] + m["order"][POP_K][2])
    best = m["order"][0][2]

    ax.hist(emp, bins=14, range=(0.40, 1.0), color=C_GEM, alpha=0.30,
            edgecolor=C_GEM, zorder=2)
    # rug of individual offspring values
    for v in emp:
        ax.plot([v, v], [-0.6, -0.1], color=C_GEM, lw=1.4, alpha=0.6, zorder=3)
    ax.axvline(cut, color="0.4", ls="--", lw=1.8, zorder=4)
    ax.text(cut, ax.get_ylim()[1] * 0.92, "selection cut", rotation=90,
            va="top", ha="right", fontsize=12, color="0.35")
    ax.axvline(best, color=C_TRUE, ls=":", lw=1.8, zorder=4)
    ax.text(best, ax.get_ylim()[1] * 0.92, "current best", rotation=90,
            va="top", ha="right", fontsize=12, color=C_TRUE)

    ax.annotate("most land below the cut\n(won't be selected)",
                xy=(0.58, 1.0), xytext=(0.44, 5.2), fontsize=13, color=C_GEM,
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=C_GEM, lw=1.8))
    ax.annotate("a few are competitive",
                xy=(0.80, 0.4), xytext=(0.80, 4.0), fontsize=13, color=C_GEM,
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=C_GEM, lw=1.8))

    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.7, ax.get_ylim()[1])
    ax.set_xlabel("offspring posterior mean  v")
    ax.set_ylabel("count")
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Offspring EI — step 1: how good is a new offspring, typically?",
                 pad=16, wrap=True)
    footnote(fig, f"We can't know a new offspring's value, so we sample from the "
                  f"empirical spread of recent offspring (last K=3 gens, E={m['res']['E']} values).")


def slide_offspring_ei_calc(fig):
    """Offspring EI, step 2: insert each candidate, re-select, average the gain."""
    fig.clf()
    m = compute_method_data()
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0],
                          left=0.08, right=0.94, top=0.80, bottom=0.18,
                          wspace=0.28)
    # Left: pool on a number line + two example insertions.
    axL = fig.add_subplot(gs[0])
    mu = m["mu"]
    cut = 0.5 * (m["order"][POP_K - 1][2] + m["order"][POP_K][2])
    axL.axvspan(cut, XLIM[1], color=C_INBAND, zorder=0)
    axL.axvline(cut, color="0.4", ls="--", lw=1.8, zorder=2)
    axL.scatter(mu, np.zeros_like(mu), s=90, color=C_EST, zorder=4,
                edgecolors="white", linewidths=1.0)
    axL.text(cut + 0.005, 0.62, f"selected pool (top-{POP_K})", fontsize=12.5,
             color=C_TRUE, ha="left")
    # high candidate -> enters; low candidate -> no change
    axL.scatter([0.83], [0.0], s=320, color=C_GEM, marker="*", zorder=6,
                edgecolors="white", linewidths=1.0)
    axL.annotate("competitive v\n→ enters, raises parent fitness",
                 xy=(0.83, 0.06), xytext=(0.66, 0.78), fontsize=12.5,
                 color=C_GEM, fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_GEM, lw=1.8))
    axL.scatter([0.57], [0.0], s=320, color="0.6", marker="*", zorder=6,
                edgecolors="white", linewidths=1.0)
    axL.annotate("weak v\n→ no change",
                 xy=(0.57, 0.06), xytext=(0.46, -0.85), fontsize=12.5,
                 color="0.4", fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=1.8))
    axL.set_xlim(*XLIM)
    axL.set_ylim(-1.1, 1.1)
    axL.set_yticks([])
    axL.set_xlabel("fitness")
    axL.grid(axis="x", alpha=0.25)
    for s in ("top", "right", "left"):
        axL.spines[s].set_visible(False)
    axL.set_title("insert one offspring, re-run selection", fontsize=15, pad=10)

    # Right: baseline vs averaged new fitness; the gap is the offspring EI.
    res = m["res"]
    axR = fig.add_subplot(gs[1])
    base, new = res["baseline"], res["new_fitness"]
    bars = axR.bar(["pool", "pool +\noffspring"], [base, new],
                   color=["0.6", C_GEM], width=0.6, edgecolor="white")
    axR.set_ylim(base - 0.006, new + 0.006)
    axR.set_ylabel("E[parent fitness]  (parent_dist · μ)")
    for b, v in zip(bars, [base, new]):
        axR.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center",
                 va="bottom" if v == new else "top", fontsize=13,
                 fontweight="bold")
    axR.text(0.5, 0.96, f"offspring EI = {res['improvement']:+.5f}",
             transform=axR.transAxes, ha="center", va="top", color=C_GEM,
             fontweight="bold", fontsize=14)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    axR.grid(axis="y", alpha=0.25)
    axR.set_title("average over the E candidates", fontsize=15, pad=10)

    fig.suptitle("Offspring EI — step 2: insert each candidate, re-select",
                 fontsize=18, fontweight="bold", y=0.95, wrap=True)
    footnote(fig, "Analytic: parent_dist · μ averaged over the E candidates "
                  "(no truth-sampling — the pool is fixed). This is Q2's offspring line.")


def main():
    out_dir = Path("plots") / "reeval_intuition"
    out_dir.mkdir(parents=True, exist_ok=True)

    slides = [
        ("0_title", slide_title),
        ("1_setup", slide_setup),
        ("2_problem", slide_problem),
        ("3_two_levers", slide_two_levers),
        ("4_which_reeval", slide_which_to_reeval),
        ("5_how_many", slide_how_many),
        ("6_reeval_ei_setup", slide_reeval_ei_setup),
        ("7_reeval_ei_loop", slide_reeval_ei_loop),
        ("8_offspring_ei_dist", slide_offspring_ei_dist),
        ("9_offspring_ei_calc", slide_offspring_ei_calc),
        ("10_math", slide_math),
    ]

    pdf_path = out_dir / "reeval_intuition.pdf"
    with PdfPages(pdf_path) as pdf:
        for name, fn in slides:
            fig = plt.figure(figsize=(FIG_W, FIG_H))
            fn(fig)
            png = out_dir / f"slide_{name}.png"
            fig.savefig(png, dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Wrote {png}")
    print(f"Wrote {pdf_path}")

    print(f"\nToy numbers:")
    print(f"  top-{POP_K} by estimate (true avg): {pop_true_avg(topk_by_est()):.3f}  "
          f"members={topk_by_est()}")
    print(f"  top-{POP_K} by truth    (true avg): {pop_true_avg(topk_by_true()):.3f}  "
          f"members={topk_by_true()}")


if __name__ == "__main__":
    main()
