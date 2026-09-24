"""Round-12 Figure 1 (refined X): shorter overall."""
import math
import sys
from svgkit import Svg, C, OPS, OPNAME, text_w, vc
from mockups2 import chevron, task_stack, TRACK
from mockups3 import glyph_loss, seeds_ir, strategy_pill, INIT, REEV
from mockups5 import robot, pareto2, prog
from mockups6 import llm_box

W, H = 1100, 442
MX0, MX1 = 16, 300
RX0, RX1 = 396, 1040
Y0, YM0, YM1, Y1 = 22, 242, 262, 428
ECY = (Y0 + YM0) // 2
TOP, BOT = 86, 374
XT, RC = 1066, 18


# ------------------------------------------------------------------ expression trees of different shapes
SHAPES = {
    # node offsets (x, y) in units of the tree scale, and edges
    'a': ([(0, 0), (-14, 16), (14, 16), (-22, 32), (-6, 32), (20, 32)], [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]),
    'b': ([(0, 0), (0, 14), (-12, 30), (12, 30)], [(0, 1), (1, 2), (1, 3)]),
    'c': ([(0, 0), (-14, 15), (14, 15), (-20, 31), (-8, 31), (8, 31), (20, 31)],
          [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
    'd': ([(0, 0), (-12, 16), (12, 16), (12, 32)], [(0, 1), (0, 2), (2, 3)]),
    'e': ([(0, 0), (-13, 14), (13, 14), (4, 28), (22, 28), (22, 40)], [(0, 1), (0, 2), (2, 3), (2, 4), (4, 5)]),
    'f': ([(0, 0), (0, 14), (0, 28)], [(0, 1), (1, 2)]),
    'g': ([(0, 0), (-12, 15), (12, 15), (-12, 30)], [(0, 1), (0, 2), (1, 3)]),
}


def tree(s, x, y, shape='a', sc=0.5, color=C['ink2'], fill=C['white'], hi=(), hi_fill=None, hi_stroke=None):
    nodes, edges = SHAPES[shape]
    P = [(x + a * sc, y + b * sc) for a, b in nodes]
    for a, b in edges:
        on = b in hi
        s.line(*P[a], *P[b], stroke=(hi_stroke if on and hi_stroke else color), cls='w1')
    for k, (px, py) in enumerate(P):
        on = k in hi
        s.circle(px, py, 4.4 * sc, fill=(hi_fill if on and hi_fill else fill),
                 stroke=(hi_stroke if on and hi_stroke else color), cls='w1')


def glyph_selection(s, x, y):
    fits, shapes = [0.45, 0.85, 0.6], ['d', 'a', 'b']
    for k, (f, sh) in enumerate(zip(fits, shapes)):
        tx = x + k * 32
        win = k == 1
        s.rect(tx, y, 26, 30, fill=C['sel_t'] if win else C['white'], stroke=C['sel'] if win else C['line'],
               rx=4, cls='w2' if win else 'w1')
        tree(s, tx + 13, y + 6, sh, 0.44)
        s.rect(tx + 3, y + 36, 20, 5, fill=C['panel2'], stroke='none', rx=2, cls='w1')
        s.rect(tx + 3, y + 36, 20 * f, 5, fill=C['sel_d'] if win else C['mute'], stroke='none', rx=2, cls='w1')


def glyph_mutation(s, x, y):
    tree(s, x + 16, y + 4, 'a', 0.62)
    s.arrow([(x + 38, y + 16), (x + 56, y + 16)], cls='w15', head=6)
    # same tree, but the right subtree has been replaced by a bigger one
    tree(s, x + 76, y + 4, 'e', 0.62, hi=(2, 3, 4, 5), hi_fill=C['mut_t'], hi_stroke=C['mut'])


def glyph_survival(s, x, y, sw=22, sh=28, gap=4):
    shapes = ['g', 'e', 'c', 'f']
    for k, shp in enumerate(shapes):
        sx = x + k * (sw + gap)
        new = k == 1
        s.rect(sx, y, sw, sh, fill=C['surv_t'] if new else C['white'], stroke=C['surv'] if new else C['line'],
               rx=4, cls='w15' if new else 'w1')
        tree(s, sx + sw / 2 - (2 if shp == 'e' else 0), y + 5, shp, 0.42 if shp != 'e' else 0.38,
             color=C['surv_d'] if new else C['ink2'])


def island_loop(s, x, y, w, h):
    r = h / 2
    s.rect(x, y, w, h, fill=C['white'], stroke=C['ink2'], rx=r, cls='w2')
    x0, x1 = x + r, x + w - r
    cx = (x0 + x1) / 2
    chevron(s, cx, y, 0, size=11, color=C['ink2'])
    chevron(s, x + w, y + r, 90, size=11, color=C['ink2'])
    chevron(s, cx, y + h, 180, size=11, color=C['ink2'])
    chevron(s, x, y + r, 270, size=11, color=C['ink2'])
    s.opchip(x0 + 10, y - 11, 'sel', size=13, center=True)
    s.opchip(x1 - 10, y - 11, 'mut', size=13, center=True)
    s.opchip(x1 - 10, y + h - 11, 'loss', size=13, center=True)
    s.opchip(x0 + 10, y + h - 11, 'surv', size=13, center=True)
    glyph_selection(s, x0 + 10 - 45, y + 22)
    glyph_mutation(s, x1 - 10 - 46, y + 24)
    glyph_loss(s, x1 - 10 - 52, y + h - 70, 104, 46)
    glyph_survival(s, x0 + 10 - 50, y + h - 58)


# ------------------------------------------------------------------ prompt pieces
def mode_grid(s, x, y, active='refine', size=12, h=20, colw=88, rowh=24):
    for k, m in enumerate(['explore', 'refine', 'simplify', 'crossover']):
        xx, yy = x + (k % 2) * colw, y + (k // 2) * rowh
        if m == active:
            s.chip(xx, yy, m, C['llm_t'], C['llm'], C['llm_d'], size=size, h=h, pad=7, cls='b')
        else:
            s.chip(xx, yy, m, C['white'], C['line'], C['ink2'], size=size, h=h, pad=7, cls='')


def op_grid(s, x, y, active='surv', size=12, h=20, colw=88, rowh=28):
    for k, op in enumerate(['mut', 'sel', 'surv', 'loss']):
        xx, yy = x + (k % 2) * colw, y + (k // 2) * rowh
        w = text_w(OPNAME[op], size, 700 if op == active else 400) + 14
        if op == active:
            s.rect(xx, yy, w, h, fill=C[op + '_t'], stroke=C['ink'], rx=h / 2, cls='w15')
            s.text(xx + w / 2, yy + h / 2 + vc(size), OPNAME[op], f's{size} b', fill=C[op + '_d'], anchor='m')
        else:
            s.rect(xx, yy, w, h, fill=C['white'], stroke=C['line'], rx=h / 2, cls='w12')
            s.text(xx + w / 2, yy + h / 2 + vc(size), OPNAME[op], f's{size}', fill=C['ink2'], anchor='m')


def pareto_history(s, x, y, w, h):
    """Pareto fronts of an unsolved task at three checkpoints (light = early, dark = late)."""
    s.rect(x, y, w, h, fill=C['white'], stroke=C['line'], rx=4, cls='w1')
    ax0, ay0, ax1, ay1 = x + 7, y + 5, x + w - 5, y + h - 6
    s.line(ax0, ay1, ax1, ay1, stroke=C['mute'], cls='w1')
    s.line(ax0, ay0, ax0, ay1, stroke=C['mute'], cls='w1')
    fronts = [([(0.05, 0.95), (0.3, 0.8), (0.6, 0.7), (0.95, 0.66)], '#a4adb9'),
              ([(0.05, 0.9), (0.25, 0.62), (0.5, 0.48), (0.95, 0.42)], '#687282'),
              ([(0.05, 0.86), (0.2, 0.5), (0.42, 0.3), (0.7, 0.14), (0.95, 0.1)], C['ink'])]
    for pts, col in fronts:
        P = [(ax0 + 3 + u * (ax1 - ax0 - 6), ay1 - 2 - v * (ay1 - ay0 - 4)) for u, v in pts]
        d = f'M{P[0][0]:.1f},{P[0][1]:.1f}'
        for (xa, ya), (xb, yb) in zip(P, P[1:]):
            d += f' L{xb:.1f},{ya:.1f} L{xb:.1f},{yb:.1f}'
        s.path(d, stroke=col, cls='w12')
        for px, py in P:
            s.circle(px, py, 1.9, fill=col, stroke='none', cls='w1')


def reference_doc(s, x, y, w=28, h=34):
    """Document glyph standing for the SR reference text (API, examples, requirements) in every prompt."""
    f = 9
    s.path(f'M{x},{y} L{x + w - f},{y} L{x + w},{y + f} L{x + w},{y + h} L{x},{y + h} Z',
           stroke=C['ink2'], fill=C['white'], cls='w12')
    s.path(f'M{x + w - f},{y} L{x + w - f},{y + f} L{x + w},{y + f}', stroke=C['ink2'], cls='w1')
    rows = [(0.55, C['ink2']), (0.72, C['line']), (0.62, C['line']), (0.5, C['ink2']),
            (0.68, C['line']), (0.58, C['line'])]
    block = (w - 10) * max(ww for ww, _ in rows)
    bx = x + (w - block) / 2
    for k, (ww, col) in enumerate(rows):
        s.rect(bx, y + 12 + k * 4.6, (w - 10) * ww, 2.3, fill=col, stroke='none', rx=1, cls='w1')


# ------------------------------------------------------------------ blocks
CW, CH, CG = 26, 32, 5            # card size / gap shared by the offspring and meta-selection columns


def track(s):
    xl = 160
    d = (f'M{xl},{TOP} L{XT - RC},{TOP} A{RC},{RC} 0 0 1 {XT},{TOP + RC} L{XT},{BOT - RC} '
         f'A{RC},{RC} 0 0 1 {XT - RC},{BOT} L{xl},{BOT} Z')
    s.path(d, stroke=TRACK, cls='w18')
    s.path(d, stroke=C['ink2'], cls='w3')


def meta_mutation(s):
    x, y, w, h = MX0, Y0, MX1 - MX0, Y1 - Y0
    s.panel(x, y, w, h, 'Meta-mutation', tab=C['llm_t'])
    llm_box(s, x + 12, TOP - 29, 104, 58)
    s.arrow([(x + 118, TOP), (x + 130, TOP)], cls='w2', head=7)
    s.code_card(x + 132, TOP - 25, 36, 50, header=False, bars=1, hi='surv')
    s.arrow([(x + 170, TOP), (x + 182, TOP)], cls='w2', head=7)
    s.rect(x + 184, TOP - 13, 84, 26, fill=C['surv_t'], stroke=C['surv'], rx=13, cls='w12')
    s.check(x + 199, TOP, 5.5)
    s.text(x + 210, TOP + 4.5, 'validate', 's13 b', fill=C['surv_d'])
    s.arrow([(x + 270, TOP), (x + w - 2, TOP)], cls='w2', head=6)
    # prompt
    px, py, pw = x + 14, TOP + 50, w - 28
    pb = y + h - 20                        # prompt box bottom
    s.arrow([(x + 64, py - 2), (x + 64, TOP + 31)], cls='w2', head=8)
    s.rect(px, py, pw, pb - py, fill=C['white'], stroke=C['line'], rx=8, cls='w12')
    title_base = py + 24
    s.text(px + 12, title_base, 'Prompt', 's16 b')
    # section heights: mode (2 chip rows), operator (2 chip rows), feedback (label + plot), parent card
    mode_h, op_h, fb_h, par_top = 20 + 24, 20 + 24, 9 + 9 + 34, BOT - CH / 2
    g_end = pb - (BOT + CH / 2)            # whitespace below the parent row ...
    mode_y = title_base + g_end            # ... equals whitespace above the mode row
    g = (par_top - (mode_y + mode_h) - op_h - fb_h) / 3
    op_y = mode_y + mode_h + g
    fb_y = op_y + op_h + g                 # top of the feedback label's capitals
    s.text(px + 12, mode_y + 14.5, 'mode', 's13', fill=C['ink2'])
    mode_grid(s, px + 78, mode_y, size=12, h=20, rowh=24, colw=88)
    s.text(px + 12, op_y + 14.5, 'operator', 's13', fill=C['ink2'])
    op_grid(s, px + 78, op_y, 'surv', size=12, h=20, colw=88, rowh=24)
    half = px + pw / 2 + 2
    s.text(px + 12, fb_y + 9, 'execution feedback', 's12', fill=C['ink2'])
    pareto_history(s, px + 12, fb_y + 18, 100, 34)
    dcx = half + 12 + text_w('SR reference', 12) / 2      # label and page share a centre line
    s.text(dcx, fb_y + 9, 'SR reference', 's12', fill=C['ink2'], anchor='m')
    reference_doc(s, dcx - 14, fb_y + 18)
    # the selected parent, arriving from meta-selection on the bottom loop line
    s.text(px + 12, BOT + 5, 'parent(s)', 's13', fill=C['ink2'])
    s.code_card(px + 78, BOT - CH / 2, CW, CH, header=False, bars=1)
    s.rect(px + 78, BOT - CH / 2, CW, CH, fill='none', stroke=C['llm'], rx=6, cls='w25')


def gap_column(s, cx, cy, label, his=(None, None, None), picked=None, label_above=False):
    top = cy - (3 * CH + 2 * CG) / 2
    for k in range(3):
        yy = top + k * (CH + CG)
        faded = picked is not None and k not in picked
        if faded:
            s.raw('<g opacity="0.45">')
        s.code_card(cx - CW / 2, yy, CW, CH, header=False, bars=1, hi=his[k])
        if faded:
            s.raw('</g>')
        if picked is not None and k in picked:
            s.rect(cx - CW / 2, yy, CW, CH, fill='none', stroke=C['llm'], rx=6, cls='w25')
    ly = top - 9 if label_above else top + 3 * CH + 2 * CG + 16
    s.text(cx, ly, label, 's12 b', anchor='m')


def evaluate(s):
    x, y, w, h = RX0, Y0, RX1 - RX0, YM0 - Y0
    s.panel(x, y, w, h, 'Evaluate: inner SR loop')
    lx, lw = x + 140, 320
    ly, lh = y + 58, 132
    cy = ly + lh / 2
    task_stack(s, x + 18, cy - 36, w=62, h=42, label=False, lab2=False)
    s.text(x + 55, cy + 44, 'training', 's14 b', anchor='m')
    s.text(x + 55, cy + 61, 'tasks', 's14 b', anchor='m')
    s.arrow([(lx - 34, cy), (lx - 8, cy)], cls='w2', head=8)
    s.text(lx + lw / 2, y + 38, 'PySR or BasicSR', 's14 b', fill=C['ink2'], anchor='m')
    island_loop(s, lx, ly, lw, lh)
    ox = lx + lw + 40
    s.arrow([(lx + lw + 10, cy), (ox - 4, cy)], cls='w2', head=8)
    pareto2(s, ox, cy - 44, 104, 82, labels=False)
    s.text(ox + 52, cy + 56, 'Pareto front', 's13', fill=C['ink2'], anchor='m')


def reevaluation(s, x, y, w, h):
    s.panel(x, y, w, h, 'Reevaluation', tsize=15)
    alloc = [5, 3, 2, 0]                     # reeval seeds handed out this generation (sum = B)
    fits = [0.66, 0.63, 0.60, 0.52]
    n_init = 3
    s.text(x + 14, y + 32, 'fitness ± err', 's12', fill=C['mute'])
    s.text(x + 106, y + 32, 'archive', 's12', fill=C['mute'])
    rows_y = [y + 52, y + 78, y + 104, y + 144]   # extra space around the cut line
    bw = 78
    ends = []
    for k, (nn, f, yy) in enumerate(zip(alloc, fits, rows_y)):
        dropped = k == 3
        if dropped:
            s.raw('<g opacity="0.6">')
        n = n_init + nn
        half = 0.30 / math.sqrt(n) * bw          # standard error  ~  1 / sqrt(N seeds)
        s.rect(x + 14, yy - 5, bw, 10, fill=C['panel2'], stroke='none', rx=3, cls='w1')
        s.rect(x + 14, yy - 5, bw * f, 10, fill=C['ink2'], stroke='none', rx=3, cls='w1')
        mx = x + 14 + bw * f
        s.line(mx - half, yy, mx + half, yy, stroke=REEV, cls='w2')
        s.line(mx - half, yy - 3.5, mx - half, yy + 3.5, stroke=REEV, cls='w15')
        s.line(mx + half, yy - 3.5, mx + half, yy + 3.5, stroke=REEV, cls='w15')
        prog(s, x + 106, yy, hi=['mut', 'sel', 'surv', 'loss'][k])
        e = seeds_ir(s, x + 138, yy, n_init, nn)
        if dropped:
            s.raw('</g>')
        ends.append((e, yy, nn))
    cut = (rows_y[2] + rows_y[3]) / 2
    s.line(x + 10, cut, x + 232, cut, stroke=C['mute'], cls='w1 dash')
    # budget -> strategy -> allocation
    B = sum(alloc)
    cols = 5
    bw_, bh_ = cols * 11 + 12, math.ceil(B / cols) * 11 + 12
    pcx = x + w - 76
    bx, by = pcx - bw_ / 2, y + 44
    s.text(pcx, by - 7, 'budget B', 's12 b', fill=C['ink2'], anchor='m')
    s.rect(bx, by, bw_, bh_, fill=C['white'], stroke=REEV, rx=6, cls='w15')
    for k in range(B):
        s.circle(bx + 11 + (k % cols) * 11, by + 11 + (k // cols) * 11, 3.8, fill=REEV, stroke='none', cls='w1')
    pw = strategy_pill(s, pcx, by + bh_ + 22, center=True)
    s.arrow([(pcx, by + bh_ + 2), (pcx, by + bh_ + 20)], color=REEV, cls='w15', head=6)
    sx, sy = pcx - pw / 2, by + bh_ + 35
    for (ex, ey, n) in ends:
        if n <= 0:
            continue
        s.carrow((sx, sy), (sx - 18, sy), (ex + 30, ey), (ex + 8, ey), color=REEV, cls='w12', head=6)
        s.text(ex + 12, ey - 6, f'+{n}', 's12 b', fill=REEV)


def initial_eval(s, x, y, w, h):
    s.panel(x, y, w, h, 'Initial evaluation', tsize=15)
    cy = y + 14 + (h - 14) / 2
    for k, op in enumerate(['mut', 'sel', 'loss']):
        yy = cy - 28 + k * 28
        prog(s, x + 14, yy, hi=op)
        seeds_ir(s, x + 46, yy, 3, 0)
        s.text(x + 88, yy + 4.5, ['0.61', '0.55', '0.48'][k], 's13', fill=C['ink2'])
    cx = x + 132
    top = cy - 56
    s.text(cx, top + 12, 'meta-', 's13 b', fill=C['ink'])
    s.text(cx, top + 28, 'fitness', 's13 b', fill=C['ink'])
    for k, lab in enumerate(['GT', 'R²', 'GT-R²']):
        act = k == 0
        s.chip(cx, top + 38 + k * 26, lab, C['ink2'] if act else C['white'], C['ink2'],
               C['white'] if act else C['ink2'], size=12, h=21, pad=8)


def mock_Y():
    s = Svg(W, H)
    track(s)
    meta_mutation(s)
    evaluate(s)
    gx = (MX1 + RX0) / 2
    gap_column(s, gx, TOP, 'offspring', his=('mut', 'sel', 'loss'))
    gap_column(s, gx, BOT, 'meta-selection', picked=(1,), label_above=True)
    iw = 194
    rw = RX1 - RX0 - iw - 20
    reevaluation(s, RX0, YM1, rw, Y1 - YM1)
    initial_eval(s, RX0 + rw + 20, YM1, iw, Y1 - YM1)
    for cx, cy, a, sz in [(MX1 + 16, TOP, 0, 14), (RX0 - 16, TOP, 0, 14), (XT, (TOP + BOT) / 2 - 70, 90, 17), (XT, (TOP + BOT) / 2 + 70, 90, 17),
                          (RX0 + rw + 10, BOT, 180, 17), (RX0 - 16, BOT, 180, 14), (MX1 + 16, BOT, 180, 14)]:
        chevron(s, cx, cy, a, size=sz, color=C['ink'])
    return s


MOCKS = {'Y': mock_Y}

if __name__ == '__main__':
    import cairosvg
    for k in sys.argv[1:] or list(MOCKS):
        s = MOCKS[k]()
        svg = s.svg(local=True)
        open(f'mock_{k}.svg', 'w').write(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f'mock_{k}.png', output_width=s.w * 2)
        print('wrote', k)
