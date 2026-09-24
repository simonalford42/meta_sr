"""Round-3 Figure 1 mockups: refinements of F (racetrack) with a bigger Evaluate block,
budgeted reevaluation, and a compact population."""
import math
import sys
from svgkit import Svg, C, OPS, OPNAME, text_w
from mockups2 import (chevron, meta_mutation, offspring, fitness_chips, pareto, task_stack, TRACK)

W, H = 1100, 470
INIT = '#7a8494'   # initial-evaluation seeds
REEV = C['warn']   # reevaluation seeds

# racetrack geometry shared by all round-3 mockups
XL, XR, CY, R = 184, 884, 252, 126


# =====================================================================
# small glyphs for the inner loop
# =====================================================================
def tiny_tree(s, x, y, sc=0.5, color=C['ink2'], fill=C['white'], hi=None, hi_stroke=None):
    nodes = [(0, 0), (-14, 16), (14, 16), (-22, 32), (-6, 32), (20, 32)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    P = [(x + a * sc, y + b * sc) for a, b in nodes]
    for a, b in edges:
        col = hi_stroke if (hi_stroke and a == 2) else color
        s.line(*P[a], *P[b], stroke=col, cls='w1')
    for k, (px, py) in enumerate(P):
        on = hi and k in (2, 5)
        s.circle(px, py, 4.4 * sc, fill=hi if on else fill, stroke=hi_stroke if on else color, cls='w1')


def glyph_tournament(s, x, y, w=98, h=42):
    s.rect(x, y, w, h, fill=C['white'], stroke=C['line'], rx=6, cls='w1 dash')
    for k in range(3):
        tx = x + 20 + k * 30
        tiny_tree(s, tx, y + 9, 0.55)
    s.circle(x + 50 + 1, y + 21, 17, fill='none', stroke=C['sel'], cls='w2')


def glyph_mutation(s, x, y):
    tiny_tree(s, x + 16, y + 6, 0.62)
    s.arrow([(x + 38, y + 18), (x + 56, y + 18)], cls='w15', head=6)
    tiny_tree(s, x + 76, y + 6, 0.62, hi=C['mut_t'], hi_stroke=C['mut'])


def glyph_loss(s, x, y, w=96, h=44):
    s.line(x + 4, y + h - 4, x + w - 4, y + h - 4, stroke=C['mute'], cls='w1')
    s.line(x + 4, y + 4, x + 4, y + h - 4, stroke=C['mute'], cls='w1')
    f = lambda u: 0.25 + 0.55 * u * u
    px = lambda u: x + 8 + u * (w - 16)
    py = lambda v: y + h - 6 - v * (h - 12)
    d = 'M' + ' L'.join(f'{px(u / 16):.1f},{py(f(u / 16)):.1f}' for u in range(17))
    for u, j in [(0.08, 0.10), (0.3, -0.09), (0.52, 0.1), (0.72, -0.08), (0.9, 0.07)]:
        s.line(px(u), py(f(u)), px(u), py(f(u) + j), stroke=C['loss'], cls='w15')
        s.circle(px(u), py(f(u) + j), 2.2, fill=C['ink2'], stroke='none', cls='w1')
    s.path(d, stroke=C['loss_d'], cls='w15')


def glyph_survival(s, x, y, n=4, sw=22, sh=28, gap=4):
    for k in range(n):
        sx = x + k * (sw + gap)
        if k == 1:
            s.rect(sx, y, sw, sh, fill=C['surv_t'], stroke=C['surv'], rx=4, cls='w15')
            tiny_tree(s, sx + sw / 2, y + 5, 0.42, hi=C['mut_t'], hi_stroke=C['mut'])
        else:
            s.rect(sx, y, sw, sh, fill=C['white'], stroke=C['line'], rx=4, cls='w1')
            tiny_tree(s, sx + sw / 2, y + 5, 0.42)


def island_loop(s, x, y, w, h, ghosts=2):
    """PySR-style inner loop on one island: a stadium track with four operator stations."""
    r = h / 2
    for g in range(ghosts, 0, -1):
        s.rect(x + 6 * g, y + 6 * g, w, h, fill=C['white'], stroke=C['line'], rx=r, cls='w1')
    s.rect(x, y, w, h, fill=C['white'], stroke=C['ink2'], rx=r, cls='w2')
    x0, x1 = x + r, x + w - r
    cx = (x0 + x1) / 2
    # arrowheads along the track
    chevron(s, cx, y, 0, size=11, color=C['ink2'])
    chevron(s, x + w, y + r, 90, size=11, color=C['ink2'])
    chevron(s, cx, y + h, 180, size=11, color=C['ink2'])
    chevron(s, x, y + r, 270, size=11, color=C['ink2'])
    # stations: two on top (selection -> mutation), two on bottom (loss -> survival)
    s.opchip(x0 + 8, y - 11, 'sel', size=13, center=True)
    s.opchip(x1 - 8, y - 11, 'mut', size=13, center=True)
    s.opchip(x1 - 8, y + h - 11, 'loss', size=13, center=True)
    s.opchip(x0 + 8, y + h - 11, 'surv', size=13, center=True)
    # what each operator does (PySR defaults)
    glyph_tournament(s, x0 + 8 - 54, y + 20)
    glyph_mutation(s, x1 - 8 - 44, y + 20)
    glyph_loss(s, x1 - 8 - 50, y + h - 66, 100, 44)
    glyph_survival(s, x0 + 8 - 50, y + h - 56)
    if ghosts:
        s.text(x + w + 6 * ghosts, y + h + 6 * ghosts + 16, '× islands', 's12 i', fill=C['mute'], anchor='e')


def evaluate_block(s, x, y, w, h, title='Evaluate: inner SR loop', score=False):
    s.panel(x, y, w, h, title)
    # input: one task, one seed
    tx, ty = x + 16, y + h / 2 - 20
    s.text(tx + 32, ty - 18, 'one task,', 's12', fill=C['ink2'], anchor='m')
    s.text(tx + 32, ty - 4, 'one seed', 's12', fill=C['ink2'], anchor='m')
    s.task_tile(tx, ty + 4, 64, 44, kind=1, stroke=C['ink2'])
    s.arrow([(tx + 68, ty + 26), (tx + 98, ty + 26)], cls='w2', head=8)
    lw = w - 16 - 104 - (150 if score else 104)
    island_loop(s, x + 124, y + 40, lw, h - 86)
    ox = x + 124 + lw + 12 + 12
    s.arrow([(x + 124 + lw + 14, ty + 26), (ox + 4, ty + 26)], cls='w2', head=8)
    pareto(s, ox + 6, ty, 70, 54)
    s.text(ox + 41, ty + 72, 'Pareto front', 's12', fill=C['ink2'], anchor='m')
    return (tx + 32, ty + 48)


# =====================================================================
# bottom-row pieces
# =====================================================================
def seeds_ir(s, x, y, n_init, n_re, r=3.6, gap=9.2):
    for k in range(n_init + n_re):
        s.circle(x + k * gap, y, r, fill=INIT if k < n_init else REEV, stroke='none', cls='w1')
    return x + (n_init + n_re - 1) * gap


def prog(s, x, y, hi=None, dim=False):
    s.code_card(x, y - 11, 18, 22, header=False, bars=1, hi=hi, dim=dim)


def budget_tray(s, x, y, B=10, cols=5, label=True):
    rows = math.ceil(B / cols)
    w, h = cols * 11 + 12, rows * 11 + 12
    s.rect(x, y, w, h, fill=C['white'], stroke=REEV, rx=6, cls='w15')
    for k in range(B):
        s.circle(x + 11 + (k % cols) * 11, y + 11 + (k // cols) * 11, 3.8, fill=REEV, stroke='none', cls='w1')
    if label:
        s.text(x + w / 2, y + h + 16, 'budget B', 's12 b', fill=C['ink2'], anchor='m')
        s.text(x + w / 2, y + h + 30, 'per generation', 's12', fill=C['ink2'], anchor='m')
    return w, h


def strategy_pill(s, x, y, center=True):
    return s.chip(x, y, 'reeval strategy', C['white'], C['ink2'], C['ink'], size=13, h=26, pad=10, center=center)


def legend_seeds(s, x, y):
    s.circle(x, y, 4, fill=INIT, stroke='none', cls='w1')
    s.text(x + 8, y + 4.5, 'initial', 's12', fill=C['ink2'])
    s.circle(x + 62, y, 4, fill=REEV, stroke='none', cls='w1')
    s.text(x + 70, y + 4.5, 'reeval', 's12', fill=C['ink2'])


def population_block(s, x, y, w, h):
    s.panel(x, y, w, h, 'Population', tsize=15)
    wins = [True, False, True, True, False]
    for k in range(5):
        cx = x + 20 + k * 36
        s.code_card(cx, y + 24, 24, 32, header=False, bars=1)
        if wins[k]:
            s.task_tile(cx - 1, y + 62, 26, 18, kind=k, stroke=C['gold'])
    s.lines(x + 16, y + 98, ['keep each task’s best,', 'fill the rest by fitness'], 's12', lh=15, fill=C['ink2'])


def reeval_rows(s, x, y, rows, rowh=25):
    """rows: (n_init, n_prev_reeval, n_new) -> returns list of (end_x, y) for allocation arrows."""
    ends = []
    for k, (ni, nprev, nnew) in enumerate(rows):
        yy = y + k * rowh
        prog(s, x, yy)
        e = seeds_ir(s, x + 28, yy, ni, nprev + nnew)
        ends.append((e, yy, nnew))
    return ends


def alloc_arrows(s, src, ends, side='right'):
    sx, sy = src
    for (ex, ey, n) in ends:
        if n <= 0:
            continue
        tx = ex + 10
        s.carrow((sx, sy), (sx - 30, sy), (tx + 30, ey), (tx, ey), color=REEV, cls='w15', head=7)
        s.text(tx + 14, ey - 5, f'+{n}', 's12 b', fill=REEV)


def base(s):
    """Track, meta-mutation, offspring, interior label and tasks (shared)."""
    d = (f'M{XL:.1f},{CY - R:.1f} L{XR:.1f},{CY - R:.1f} A{R},{R} 0 0 1 {XR:.1f},{CY + R:.1f} '
         f'L{XL:.1f},{CY + R:.1f} A{R},{R} 0 0 1 {XL:.1f},{CY - R:.1f} Z')
    s.path(d, stroke=TRACK, cls='w18')
    s.path(d, stroke=C['ink2'], cls='w15 ldash')
    meta_mutation(s, 20, 30, 312, 184)
    offspring(s, 399, CY - R - 6)


def arrows(s, bottom_gaps):
    for x, y, a in [(348, CY - R, 0), (451, CY - R, 0), (XR + R, CY, 90), (XL - R, CY, 270)] + \
            [(gx, CY + R, 180) for gx in bottom_gaps]:
        chevron(s, x, y, a, size=16, color=C['ink'])


def interior(s, task_x):
    s.lines(300, 266, ['Meta-evolution', 'G generations'], 's16 b i', lh=19, fill=C['ink2'], anchor='m')
    task_stack(s, task_x, 246, w=46, h=30, label=False, lab2=False)
    s.text(task_x + 72, 266, 'training tasks', 's13 b')
    s.text(task_x + 72, 282, 'T × N seeds', 's13', fill=C['ink2'])


# =====================================================================
# J. separate Initial-eval and Reevaluation blocks
# =====================================================================
def mock_J():
    s = Svg(W, H)
    base(s)
    tin = evaluate_block(s, 466, 26, 504, 212)
    interior(s, 488)
    s.arrow([(tin[0] + 0, 250), (tin[0], 238 - 4)], cls='w2', head=8) if False else None
    s.arrow([(514, 254), (514, 232)], cls='w2', head=8)
    # initial evaluation
    ib = (668, 300, 302, 156)
    s.panel(*ib, 'Initial evaluation', tsize=15)
    for k, op in enumerate(['mut', 'sel', 'loss']):
        yy = ib[1] + 36 + k * 26
        prog(s, ib[0] + 16, yy, hi=op)
        seeds_ir(s, ib[0] + 44, yy, 3, 0)
    s.text(ib[0] + 16, ib[1] + 126, 'offspring: ' + s.sub('N', 'init') + 'seeds', 's12', fill=C['ink2'])
    s.text(ib[0] + 150, ib[1] + 34, 'meta-fitness', 's13', fill=C['ink2'])
    fitness_chips(s, ib[0] + 150, ib[1] + 42, size=12, h=21, gap=4)
    s.lines(ib[0] + 150, ib[1] + 90, ['mean over', 'tasks × seeds'], 's12', lh=15, fill=C['ink2'])
    # reevaluation
    rb = (262, 300, 382, 156)
    s.panel(*rb, 'Reevaluation', tsize=15)
    ends = reeval_rows(s, rb[0] + 16, rb[1] + 32, [(3, 6, 3), (3, 4, 2), (3, 2, 0), (3, 0, 1)], rowh=24)
    tw, th = budget_tray(s, rb[0] + rb[2] - 86, rb[1] + 28)
    px = rb[0] + 222
    strategy_pill(s, px, rb[1] + 108, center=True)
    s.arrow([(rb[0] + rb[2] - 86 + tw / 2, rb[1] + 28 + th + 38), (rb[0] + rb[2] - 86 + tw / 2, rb[1] + 118),
             (px + 64, rb[1] + 121)], color=REEV, cls='w15', head=7) if False else None
    s.arrow([(rb[0] + rb[2] - 86, rb[1] + 28 + th / 2), (px + 52, rb[1] + 28 + th / 2), (px + 52, rb[1] + 106)],
            color=REEV, cls='w15', head=7)
    for (ex, ey, n) in ends:
        if n <= 0:
            continue
        s.carrow((px - 30, rb[1] + 108), (px - 30, ey + 20), (ex + 40, ey), (ex + 8, ey), color=REEV, cls='w12', head=6)
        s.text(ex + 18, ey - 5, f'+{n}', 's12 b', fill=REEV)
    legend_seeds(s, rb[0] + 20, rb[1] + 140)
    population_block(s, 20, 314, 218, 128)
    arrows(s, [656, 250])
    return s


# =====================================================================
# K. one combined "estimate fitness" block: offspring and population in one ledger
# =====================================================================
def mock_K():
    s = Svg(W, H)
    base(s)
    evaluate_block(s, 466, 26, 504, 212)
    interior(s, 488)
    s.arrow([(514, 254), (514, 232)], cls='w2', head=8)
    fb = (262, 300, 708, 156)
    s.panel(*fb, 'Initial evaluation + reevaluation', tsize=15)
    # right: meta-fitness + budget -> strategy
    rx = fb[0] + fb[2] - 190
    s.text(rx, fb[1] + 34, 'meta-fitness', 's13', fill=C['ink2'])
    fitness_chips(s, rx, fb[1] + 42, size=12, h=21, gap=4)
    s.text(rx, fb[1] + 88, 'mean over tasks × seeds', 's12', fill=C['ink2'])
    tw, th = budget_tray(s, rx, fb[1] + 100, B=10, cols=5, label=False)
    s.text(rx + tw + 10, fb[1] + 116, 'reeval budget B', 's12 b', fill=C['ink2'])
    s.text(rx + tw + 10, fb[1] + 131, 'per generation', 's12', fill=C['ink2'])
    # middle: ledger (population rows + new offspring)
    lx = fb[0] + 230
    s.text(lx, fb[1] + 30, 'population', 's12', fill=C['mute'])
    ends = reeval_rows(s, lx, fb[1] + 46, [(3, 6, 3), (3, 4, 2), (3, 2, 0), (3, 0, 1)], rowh=22)
    s.line(lx, fb[1] + 130, lx + 170, fb[1] + 130, stroke=C['line'], cls='w1 dash') if False else None
    # strategy pill between tray and ledger
    px = rx - 60
    strategy_pill(s, px, fb[1] + 104, center=True)
    s.arrow([(rx - 2, fb[1] + 117), (px + 58, fb[1] + 117)], color=REEV, cls='w15', head=7)
    for (ex, ey, n) in ends:
        if n <= 0:
            continue
        s.carrow((px - 12, fb[1] + 104), (px - 12, ey + 10), (ex + 36, ey), (ex + 8, ey), color=REEV, cls='w12', head=6)
        s.text(ex + 16, ey - 5, f'+{n}', 's12 b', fill=REEV)
    # left: new offspring, initial seeds only
    ox = fb[0] + 18
    s.text(ox, fb[1] + 30, 'new offspring', 's12', fill=C['mute'])
    for k, op in enumerate(['mut', 'sel', 'loss']):
        yy = fb[1] + 46 + k * 22
        prog(s, ox, yy, hi=op)
        seeds_ir(s, ox + 28, yy, 3, 0)
    s.text(ox, fb[1] + 128, s.sub('N', 'init') + 'seeds each', 's12', fill=C['ink2'])
    legend_seeds(s, ox + 2, fb[1] + 144)
    s.line(fb[0] + 208, fb[1] + 22, fb[0] + 208, fb[1] + 146, stroke=C['line'], cls='w1')
    population_block(s, 20, 314, 218, 128)
    arrows(s, [250])
    return s


# =====================================================================
# L. Evaluate scores offspring directly; the bottom row is reevaluation + population
# =====================================================================
def mock_L():
    s = Svg(W, H)
    base(s)
    E = (466, 26, 568, 212)
    s.panel(*E, 'Evaluate offspring: inner SR loop')
    tx, ty = E[0] + 16, E[1] + E[3] / 2 - 20
    s.text(tx + 32, ty - 18, 'one task,', 's12', fill=C['ink2'], anchor='m')
    s.text(tx + 32, ty - 4, 'one seed', 's12', fill=C['ink2'], anchor='m')
    s.task_tile(tx, ty + 4, 64, 44, kind=1, stroke=C['ink2'])
    s.arrow([(tx + 68, ty + 26), (tx + 98, ty + 26)], cls='w2', head=8)
    island_loop(s, E[0] + 124, E[1] + 40, 276, E[3] - 86)
    ox = E[0] + 124 + 276 + 24
    s.arrow([(ox - 10, ty + 26), (ox + 4, ty + 26)], cls='w2', head=8)
    pareto(s, ox + 6, ty - 26, 62, 46)
    s.text(ox + 37, ty + 34, 'Pareto front', 's12', fill=C['ink2'], anchor='m')
    s.arrow([(ox + 37, ty + 40), (ox + 37, ty + 58)], cls='w15', head=6)
    fitness_chips(s, ox - 6, ty + 62, size=12, h=21, gap=3)
    s.text(ox - 6, ty + 104, s.sub('N', 'init') + 'seeds each', 's12', fill=C['ink2'])
    s.circle(ox + 90, ty + 99.5, 3.6, fill=INIT, stroke='none', cls='w1')
    s.circle(ox + 99, ty + 99.5, 3.6, fill=INIT, stroke='none', cls='w1')
    s.circle(ox + 108, ty + 99.5, 3.6, fill=INIT, stroke='none', cls='w1')
    interior(s, 488)
    s.arrow([(514, 254), (514, 232)], cls='w2', head=8)
    # reevaluation, wide
    rb = (262, 300, 772, 156)
    s.panel(*rb, 'Reevaluate the population', tsize=15)
    tw, th = budget_tray(s, rb[0] + rb[2] - 100, rb[1] + 34)
    px = rb[0] + rb[2] - 230
    strategy_pill(s, px, rb[1] + 52, center=True)
    s.arrow([(rb[0] + rb[2] - 102, rb[1] + 34 + th / 2), (px + 62, rb[1] + 65)], color=REEV, cls='w15', head=7)
    lx = rb[0] + 24
    s.text(lx, rb[1] + 30, 'fitness estimate ± error', 's12', fill=C['mute'])
    s.text(lx + 136, rb[1] + 30, 'population + new offspring', 's12', fill=C['mute'])
    rows = [(3, 6, 3), (3, 4, 2), (3, 2, 0), (3, 0, 1), (3, 0, 0)]
    ends = []
    for k, (ni, npv, nn) in enumerate(rows):
        yy = rb[1] + 46 + k * 21
        n = ni + npv + nn
        fit = [0.66, 0.62, 0.58, 0.64, 0.55][k]
        bx, bw = lx, 110
        s.rect(bx, yy - 5, bw, 10, fill=C['panel2'], stroke='none', rx=3, cls='w1')
        s.rect(bx, yy - 5, bw * fit, 10, fill=C['ink2'], stroke='none', rx=3, cls='w1')
        e2 = 0.25 / math.sqrt(n)
        s.line(bx + bw * (fit - e2), yy, bx + bw * (fit + e2), yy, stroke=REEV, cls='w2')
        prog(s, lx + 136, yy, hi='sel' if k == 3 else ('mut' if k == 4 else None))
        e = seeds_ir(s, lx + 164, yy, ni, npv + nn, r=3.5, gap=9)
        ends.append((e, yy, nn))
    for (ex, ey, n) in ends:
        if n <= 0:
            continue
        s.carrow((px - 62, rb[1] + 65), (px - 110, rb[1] + 65), (ex + 90, ey), (ex + 8, ey), color=REEV, cls='w12', head=6)
        s.text(ex + 14, ey - 5, f'+{n}', 's12 b', fill=REEV)
    legend_seeds(s, rb[0] + rb[2] - 250 - 60, rb[1] + 132)
    population_block(s, 20, 314, 218, 128)
    arrows(s, [250])
    return s


MOCKS = {'J': mock_J, 'K': mock_K, 'L': mock_L}

if __name__ == '__main__':
    import cairosvg
    which = sys.argv[1:] or list(MOCKS)
    for k in which:
        s = MOCKS[k]()
        svg = s.svg(local=True)
        open(f'mock_{k}.svg', 'w').write(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f'mock_{k}.png', output_width=s.w * 2)
        print('wrote', k)
