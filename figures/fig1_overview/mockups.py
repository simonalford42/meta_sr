"""Five Figure 1 mockups for the meta-SR paper (1100 px wide ~ 5.5 in text width)."""
import math
import sys
from svgkit import Svg, C, OPS, OPNAME, text_w

W = 1100


# =====================================================================
# shared composite pieces
# =====================================================================
def inner_ring(s, cx, cy, r, chip_size=13, center='trees', label=True):
    """Inner SR loop: 4 operator stations on a clockwise ring."""
    s.circle(cx, cy, r, fill=C['white'], stroke='none')
    # arcs between stations (clockwise: top Selection -> right Mutation -> bottom Loss -> left Survival)
    for a0 in (-90, 0, 90, 180):
        s.arc(cx, cy, r, a0 + 24, a0 + 66, color=C['ink2'], cls='w2', head=8)
    pos = {'sel': (cx, cy - r), 'mut': (cx + r, cy), 'loss': (cx, cy + r), 'surv': (cx - r, cy)}
    for op, (x, y) in pos.items():
        s.opchip(x, y - 11, op, size=chip_size, center=True)
    if center == 'trees':
        s.tree(cx - 12, cy - 16, 0.55)
        s.tree(cx + 13, cy - 2, 0.55, hi=C['mut_t'])
    elif center:
        s.lines(cx, cy - 2, center, 's12', lh=15, fill=C['ink2'], anchor='m')
    return pos


def pop_row(s, x, y, fit, n, n_new=0, dropped=False, tag=None, barw=92, hi=None):
    s.code_card(x, y - 13, 30, 26, header=False, bars=1, dim=dropped, hi=hi)
    bx = x + 40
    s.rect(bx, y - 6, barw, 12, fill=C['panel2'], stroke='none', rx=3, cls='w1')
    s.rect(bx, y - 6, barw * fit, 12, fill=C['line'] if dropped else C['ink2'], stroke='none', rx=3, cls='w1')
    # error bar ~ 1/sqrt(n)
    e = 0.22 / math.sqrt(n + n_new)
    ex0, ex1 = bx + barw * max(fit - e, 0), bx + barw * min(fit + e, 1)
    s.line(ex0, y, ex1, y, stroke=C['warn'] if not dropped else C['line'], cls='w2')
    s.line(ex0, y - 4, ex0, y + 4, stroke=C['warn'] if not dropped else C['line'], cls='w15')
    s.line(ex1, y - 4, ex1, y + 4, stroke=C['warn'] if not dropped else C['line'], cls='w15')
    s.seeds(bx + barw + 14, y, n, n_new, color=C['line'] if dropped else C['ink2'])
    if tag:
        s.text(bx + barw + 14 + (n + n_new) * 9.5 + 4, y + 4.5, tag, 's12 i', fill=C['mute'])


def objective_box(s, x, y, w, title='Meta-objective J', size=13):
    s.rect(x, y, w, 96, fill=C['white'], stroke=C['ink2'], rx=8, cls='w12')
    s.text(x + w / 2, y + 20, title, 's13 b', anchor='m')
    rows = ['GT recovery', 'R² (black-box)', 'GT + R² blend']
    for k, r in enumerate(rows):
        yy = y + 42 + k * 20
        s.circle(x + 14, yy - 4, 4.5, fill=C['ink2'] if k == 0 else C['white'], stroke=C['ink2'], cls='w12')
        s.text(x + 25, yy, r, f's{size}')


def trace_plot(s, x, y, w, h):
    """Execution-feedback mini plot: loss vs. time for an unsolved task."""
    s.rect(x, y, w, h, fill=C['white'], stroke=C['line'], rx=4, cls='w1')
    s.line(x + 8, y + h - 7, x + w - 6, y + h - 7, stroke=C['mute'], cls='w1')
    s.line(x + 8, y + 6, x + 8, y + h - 7, stroke=C['mute'], cls='w1')
    pts = [(0, .1), (.15, .45), (.3, .55), (.5, .62), (.7, .64), (1, .65)]
    d = 'M' + ' L'.join(f'{x + 8 + u * (w - 16):.1f},{y + h - 7 - v * (h - 14):.1f}' for u, v in pts)
    s.path(d, stroke=C['mut'], cls='w15')
    s.line(x + 8, y + 9, x + w - 6, y + 9, stroke=C['surv'], cls='w1 dash')


# =====================================================================
# A. Three-stage cycle (LLM-SR structure, lighter styling)
# =====================================================================
def mock_A():
    H = 470
    s = Svg(W, H)
    # ---------------- panel a: LLM meta-mutation
    ax, aw = 14, 318
    s.panel(ax, 30, aw, 368, 'LLM writes a new operator', letter='a')
    px, pw = 28, 146
    s.rect(px, 56, pw, 330, fill=C['white'], stroke=C['line'], rx=8, cls='w12')
    s.text(px + 10, 76, 'Prompt', 's15 b')
    s.text(px + 10, 98, 'parent algorithm', 's12', fill=C['ink2'])
    s.code_card(px + 10, 105, 60, 62, hi='sel')
    s.code_card(px + 78, 105, 60, 62, hi='sel', shadow=False)
    s.text(px + 10, 190, 'mode', 's12', fill=C['ink2'])
    s.chip(px + 10, 197, 'explore', C['llm_t'], C['llm'], C['llm_d'], size=11, h=20, pad=6)
    s.chip(px + 70, 197, 'refine', C['white'], C['line'], C['ink2'], size=11, h=20, pad=6)
    s.chip(px + 10, 222, 'simplify', C['white'], C['line'], C['ink2'], size=11, h=20, pad=6)
    s.chip(px + 70, 222, 'crossover', C['white'], C['line'], C['ink2'], size=11, h=20, pad=6)
    s.text(px + 10, 265, 'target operator', 's12', fill=C['ink2'])
    s.opchip(px + 10, 272, 'sel', size=12, h=20)
    s.text(px + 10, 315, 'execution feedback', 's12', fill=C['ink2'])
    trace_plot(s, px + 10, 322, pw - 20, 52)
    # LLM
    s.arrow([(px + pw + 4, 215), (190, 215)], cls='w25')
    s.llm(190, 188, 64, 54)
    s.arrow([(256, 215), (268, 215)], cls='w25', head=8)
    s.code_card(270, 160, 52, 110, hi='sel')
    s.text(296, 290, 'child', 's13 b', anchor='m')
    s.check(284, 308, 6)
    s.text(294, 312, 'valid', 's12', fill=C['surv_d'])

    # ---------------- panel b: evaluation
    bx, bw = 346, 410
    s.panel(bx, 30, bw, 368, 'Evaluate on training tasks', letter='b')
    s.arrow([(324, 215), (bx + 14, 215)], cls='w25')
    s.text(bx + 22, 66, 'T tasks × N seeds, run in parallel', 's13', fill=C['ink2'])
    gx, gy, tw, th = bx + 22, 76, 54, 40
    for r in range(2):
        for c in range(4):
            s.task_tile(gx + c * (tw + 8), gy + r * (th + 8), tw, th, kind=r * 4 + c,
                        stroke=C['ink2'] if (r, c) == (1, 1) else C['line'])
    # zoom wedge from tile (1,1) to ring
    tx0, ty = gx + 1 * (tw + 8), gy + th + 8 + th
    rcx, rcy, rr = bx + 116, 290, 58
    s.line(tx0, ty, rcx - rr - 8, rcy - 30, stroke=C['mute'], cls='w1 dot')
    s.line(tx0 + tw, ty, rcx + rr + 8, rcy - 30, stroke=C['mute'], cls='w1 dot')
    inner_ring(s, rcx, rcy, rr, chip_size=12)
    s.text(bx + 250, 262, 'Inner SR loop', 's15 b')
    s.lines(bx + 250, 282, ['evolves equations', 'with the child’s', 'operators;', '~10⁶ evals, no LLM'],
            's12', lh=16, fill=C['ink2'])
    # scores -> objective
    s.arrow([(gx + 4 * (tw + 8) + 2, 120), (bx + 290, 120)], cls='w2')
    objective_box(s, bx + 286, 70, 116, size=12)
    s.arrow([(bx + bw - 8, 215), (bx + bw + 20, 215)], cls='w25')

    # ---------------- panel c: reevaluation & selection
    cx, cw = 770, 316
    s.panel(cx, 30, cw, 368, 'Reevaluate &amp; select', letter='c')
    s.text(cx + 16, 66, 'program', 's12', fill=C['mute'])
    s.text(cx + 62, 66, 'fitness ± err', 's12', fill=C['mute'])
    s.text(cx + 166, 66, 'seeds run so far', 's12', fill=C['mute'])
    rows = [(0.66, 9, 0, None), (0.63, 7, 0, None), (0.69, 0, 3, 'new'), (0.60, 5, 0, None),
            (0.55, 3, 0, 'dropped')]
    for k, (f, n, nn, tag) in enumerate(rows):
        pop_row(s, cx + 16, 96 + k * 42, f, n, nn, dropped=(tag == 'dropped'), tag=tag, hi='sel' if tag == 'new' else None)
    # legend
    ly = 318
    s.seeds(cx + 22, ly, 1)
    s.text(cx + 32, ly + 4.5, 'past evaluation', 's12', fill=C['ink2'])
    s.seeds(cx + 150, ly, 0, 1)
    s.text(cx + 160, ly + 4.5, 'new: ' + s.sub('N', 'init'), 's12', fill=C['ink2'])
    s.lines(cx + 16, 346, ['Survivors get ' + s.sub('N', 'reeval') + 'more seeds each', 'generation; lucky offspring fall out.'],
            's12', lh=16, fill=C['ink2'])

    # ---------------- loop-back arrow + init
    s.arrow([(cx + 90, 398), (cx + 90, 440), (px + pw / 2, 440), (px + pw / 2, 388)], cls='w3', head=11)
    lab = 'tournament-selected parents → next generation'
    lw = text_w(lab, 14) + 20
    s.rect(560 - lw / 2, 428, lw, 24, fill=C['white'], stroke='none', rx=4, cls='w1')
    s.text(560, 445, lab, 's14 i', anchor='m')
    s.chip(cx + 150, 430, 'init: BasicSR or PySR', C['white'], C['ink2'], size=13, h=24)
    s.arrow([(cx + 230, 430), (cx + 230, 400)], cls='w2', head=8)
    return s



# ---------------------------------------------------------------------
def mode_chips(s, x, y, active='explore', size=12, h=20, cols=2, colw=64, rowh=25):
    for k, m in enumerate(['explore', 'refine', 'simplify', 'crossover']):
        xx, yy = x + (k % cols) * colw, y + (k // cols) * rowh
        if m == active:
            s.chip(xx, yy, m, C['llm_t'], C['llm'], C['llm_d'], size=size, h=h, pad=7)
        else:
            s.chip(xx, yy, m, C['white'], C['line'], C['ink2'], size=size, h=h, pad=7)


def op_row(s, x, y, active='sel', size=11, h=19, gap=5):
    xx = x
    for op in OPS:
        if op == active:
            w = s.opchip(xx, y, op, size=size, h=h)
        else:
            w = s.chip(xx, y, OPNAME[op], C['white'], C['line'], C['mute'], size=size, h=h)
        xx += w + gap


def tiles(s, x, y, cols, rows, tw, th, gap=6, hi=None):
    for r in range(rows):
        for c in range(cols):
            s.task_tile(x + c * (tw + gap), y + r * (th + gap), tw, th, kind=r * cols + c,
                        stroke=C['ink2'] if hi == (r, c) else C['line'])


# =====================================================================
# B. Concentric loops (LaSR-style centre wheel + side panels)
# =====================================================================
def mock_B():
    H = 470
    s = Svg(W, H)
    # ---------- left panel: propose
    lx, lw = 14, 290
    s.panel(lx, 30, lw, 426, 'Meta-mutation', tab=C['llm_t'])
    s.rect(28, 52, 262, 214, fill=C['white'], stroke=C['line'], rx=8, cls='w12')
    s.text(40, 72, 'parent(s)', 's12', fill=C['ink2'])
    s.code_card(40, 80, 44, 54, hi='sel')
    s.code_card(92, 80, 44, 54, hi='sel')
    s.text(152, 72, 'execution feedback', 's12', fill=C['ink2'])
    trace_plot(s, 152, 80, 126, 54)
    s.text(40, 158, 'mode', 's12', fill=C['ink2'])
    mode_chips(s, 40, 166, cols=4, colw=0) if False else None
    xx = 40
    for k, m in enumerate(['explore', 'refine', 'simplify', 'crossover']):
        if k == 2:
            xx = 40
        yy = 166 if k < 2 else 191
        act = m == 'refine'
        xx += s.chip(xx, yy, m, C['llm_t'] if act else C['white'], C['llm'] if act else C['line'],
                     C['llm_d'] if act else C['ink2'], size=12, h=20, pad=7) + 6
    s.text(152, 180, 'one of four', 's12 i', fill=C['mute']) if False else None
    s.text(40, 232, 'operator to rewrite', 's12', fill=C['ink2'])
    op_row(s, 40, 240, 'sel', size=11, h=18, gap=4)
    s.arrow([(159, 268), (159, 288)], cls='w25', head=8)
    s.llm(112, 290, 94, 44)
    s.arrow([(159, 336), (159, 356)], cls='w25', head=8)
    s.code_card(40, 358, 60, 84, hi='sel')
    s.text(112, 378, 'child program', 's14 b')
    s.check(118, 400, 6)
    s.text(130, 404, 'compiles &amp; runs', 's12', fill=C['surv_d'])
    s.check(118, 424, 6)
    s.text(130, 428, 'passes smoke test', 's12', fill=C['surv_d'])

    # ---------- centre wheel
    cx, cy, R, r = 550, 250, 168, 74
    for a0, a1 in ((180, 270), (270, 360), (0, 90), (90, 180)):
        s.arc(cx, cy, R, a0 + 17, a1 - 17, color=C['ink'], cls='w4', head=14)
    st = {'propose': (cx - R, cy), 'eval': (cx, cy - R), 'reeval': (cx + R, cy), 'select': (cx, cy + R)}
    s.chip(st['propose'][0], cy - 14, 'LLM proposes', C['llm_t'], C['llm'], C['llm_d'], size=14, h=28, center=True)
    s.chip(st['eval'][0], st['eval'][1] - 14, 'Evaluate', C['tab'], C['ink2'], size=14, h=28, center=True)
    s.chip(st['reeval'][0], cy - 14, 'Reevaluate', C['tab'], C['ink2'], size=14, h=28, center=True)
    s.chip(st['select'][0], st['select'][1] - 14, 'Population', C['tab'], C['ink2'], size=14, h=28, center=True)
    inner_ring(s, cx, cy + 6, r, chip_size=12, center=None)
    s.lines(cx, cy + 2, ['Inner SR', 'loop'], 's14 b', lh=17, anchor='m')
    s.line(cx, cy - R + 16, cx, cy + 6 - r - 13, stroke=C['ink2'], cls='w15 dash')
    s.rect(cx - 62, cy - R + 30, 124, 36, fill=C['white'], stroke='none', rx=4, cls='w1')
    s.lines(cx, cy - R + 44, ['one inner run per', 'task × seed'], 's12 i', lh=15, fill=C['ink2'], anchor='m')
    s.lines(322, 62, ['Outer loop:', 'evolve SR programs'], 's14 b', lh=17, fill=C['ink'])
    s.lines(cx, cy + 6 + r + 34, ['Inner loop: evolve equations'], 's12 i', lh=15, fill=C['mute'], anchor='m')
    # init
    s.chip(322, 436, 'init: BasicSR or PySR', C['white'], C['ink2'], size=13, h=24)
    s.arrow([(478, 448), (cx - 22, 448), (cx - 22, cy + R + 16)], cls='w2', head=8)
    # connectors to side panels
    s.line(lx + lw, cy, st['propose'][0] - 58, cy, stroke=C['mute'], cls='w15 dot')
    s.line(st['reeval'][0] + 50, cy, 796, cy, stroke=C['mute'], cls='w15 dot')

    # ---------- right panel: evaluation & reevaluation
    rx0, rw = 796, 290
    s.panel(rx0, 30, rw, 426, 'Noisy evaluation')
    s.text(rx0 + 14, 60, 'T training tasks × N seeds, in parallel', 's12', fill=C['ink2'])
    tiles(s, rx0 + 14, 68, 5, 2, 46, 32, gap=6)
    s.text(rx0 + 14, 170, 'meta-objective J', 's12', fill=C['ink2'])
    xx = rx0 + 14
    for k, lab in enumerate(['GT recovery', 'R²', 'GT + R²']):
        xx += s.chip(xx, 178, lab, C['ink2'] if k == 0 else C['white'], C['ink2'],
                     C['white'] if k == 0 else C['ink2'], size=12, h=20, pad=8) + 6
    s.text(rx0 + 14, 228, 'running mean over all seeds so far', 's12', fill=C['ink2'])
    rows = [(0.66, 9, 0, None), (0.63, 6, 0, None), (0.70, 0, 3, 'new'), (0.52, 3, 0, 'dropped')]
    for k, (f, n, nn, tag) in enumerate(rows):
        pop_row(s, rx0 + 14, 256 + k * 38, f, n, nn, dropped=(tag == 'dropped'), tag=tag, barw=78,
                hi='sel' if tag == 'new' else None)
    s.lines(rx0 + 14, 416, ['offspring: ' + s.sub('N', 'init') + 'seeds; survivors get',
                            s.sub('N', 'reeval') + 'more every generation'], 's12', lh=16, fill=C['ink2'])
    return s


# =====================================================================
# C. Algorithm anatomy (the program is the hero)
# =====================================================================
def mock_C():
    H = 470
    s = Svg(W, H)
    # ---------- centre: the SR algorithm with evolvable slots
    kx, ky, kw, kh = 346, 52, 408, 262
    # file tabs
    s.rect(kx + 14, ky - 22, 96, 30, fill=C['white'], stroke=C['ink2'], rx=6, cls='w12')
    s.text(kx + 62, ky - 3, 'PySR', 's14 b', anchor='m')
    s.rect(kx + 116, ky - 20, 96, 28, fill=C['panel2'], stroke=C['line'], rx=6, cls='w12')
    s.text(kx + 164, ky - 2, 'BasicSR', 's14', fill=C['mute'], anchor='m')
    s.text(kx + kw - 6, ky - 6, 'fixed skeleton + evolvable operators', 's12 i', fill=C['mute'], anchor='e')
    s.rect(kx, ky, kw, kh, fill=C['white'], stroke=C['ink2'], rx=8, cls='w15')
    cw = 8.43  # mono char width at 14px
    code = [
        ('function SR(data, budget)', None, ''),
        ('  pops ← random expressions', None, ''),
        ('  while budget remains', None, ''),
        ('    for P in pops', None, ''),
        ('      parent ← ', 'sel', '(P)'),
        ('      child  ← ', 'mut', '(parent)'),
        ('      score  ← ', 'loss', '(child, data)'),
        ('      P      ← ', 'surv', '(P, child)'),
        ('  return Pareto front(pops)', None, ''),
    ]
    for k, (pre, op, post) in enumerate(code):
        y = ky + 32 + k * 26
        ind = len(pre) - len(pre.lstrip(' '))
        if op == 'sel':
            s.rect(kx + 4, y - 19, kw - 8, 27, fill=C['sel_t'], stroke='none', rx=4, cls='w1')
            s.text(kx + kw - 12, y, '← rewritten', 's12 i b', fill=C['sel_d'], anchor='e')
        s.text(kx + 18 + ind * cw, y, pre.strip(), 's14 mono', fill=C['ink2'] if op is None else C['ink'])
        if op:
            x0 = kx + 18 + len(pre) * cw
            w = s.opchip(x0, y - 16, op, size=13, h=22)
            s.text(x0 + w + 2, y, post, 's14 mono', fill=C['ink'])
    # highlight the slot being rewritten
    sel_y = ky + 32 + 4 * 26 - 16
    # ---------- left: LLM rewrites one slot
    s.panel(14, 30, 316, 284, 'LLM rewrites one operator', tab=C['llm_t'])
    s.text(28, 60, 'prompt: parent code, mode, execution trace', 's12', fill=C['ink2'])
    xx = 28
    for k, m in enumerate(['explore', 'refine', 'simplify', 'crossover']):
        act = m == 'explore'
        xx += s.chip(xx, 68, m, C['llm_t'] if act else C['white'], C['llm'] if act else C['line'],
                     C['llm_d'] if act else C['ink2'], size=12, h=20, pad=7) + 5
    s.llm(28, 102, 70, 48)
    s.arrow([(100, 126), (118, 126)], cls='w25', head=8)
    # code snippet of a new selection operator
    sx, sy, sw, sh = 120, 98, 198, 150
    s.rect(sx, sy, sw, sh, fill=C['sel_t'], stroke=C['sel'], rx=6, cls='w15')
    snip = ['function select(P)', '  pool ← sample(P, 1.5k)', '  if rand() &lt; 0.1', '    return rarest(pool)',
            '  return tournament(', '    pool, parsimony)', 'end']
    for k, ln in enumerate(snip):
        ind = len(ln) - len(ln.lstrip(' '))
        s.text(sx + 10 + ind * 7.22, sy + 20 + k * 18, ln.strip(), 's12 mono', fill=C['ink'])
    s.check(126, 270, 6)
    s.text(138, 274, 'validated: compiles &amp; runs', 's12', fill=C['surv_d'])
    s.text(28, 180, 'new', 's12', fill=C['ink2'])
    s.text(28, 196, 'Selection', 's12 b', fill=C['sel_d'])
    s.text(28, 212, 'operator', 's12', fill=C['ink2'])
    s.carrow((sx + sw, sy + 60), (sx + sw + 18, sy + 60), (kx - 20, sel_y + 11), (kx + 2, sel_y + 11),
             color=C['sel_d'], cls='w25', head=10)
    # ---------- right: evaluate + reevaluate
    s.panel(770, 30, 316, 284, 'Score on training tasks')
    s.text(784, 60, 'T tasks × N seeds, in parallel', 's12', fill=C['ink2'])
    tiles(s, 784, 68, 4, 2, 52, 36, gap=6)
    s.arrow([(kx + kw, 180), (782, 180)], cls='w25') if False else None
    objective_box(s, 1010 - 6, 68, 78) if False else None
    s.text(784, 172, 'J = mean GT recovery / R²', 's13 b')
    rows = [(0.66, 9, 0, None), (0.70, 0, 3, 'new'), (0.60, 5, 0, None), (0.52, 3, 0, 'dropped')]
    for k, (f, n, nn, tag) in enumerate(rows):
        pop_row(s, 784, 200 + k * 30, f, n, nn, dropped=(tag == 'dropped'), tag=tag, barw=80,
                hi='sel' if tag == 'new' else None)
    s.arrow([(kx + kw + 2, 150), (768, 150)], cls='w25')
    # ---------- bottom: meta-evolution trajectory strip
    s.arrow([(928, 316), (928, 334), (172, 334), (172, 316)], cls='w25', head=9)
    lab = 'reevaluate, select parents, repeat for G generations'
    lw = text_w(lab, 13) + 16
    s.rect(550 - lw / 2, 324, lw, 20, fill=C['white'], stroke='none', rx=3, cls='w1')
    s.text(550, 338, lab, 's13 i', anchor='m')
    s.panel(14, 360, 1072, 100, None)
    ox, oy, ow, oh = 150, 372, 880, 76
    s.text(28, 392, 'Meta-evolution', 's13 b')
    s.text(28, 408, 'fitness per', 's12', fill=C['ink2'])
    s.text(28, 423, 'generation', 's12', fill=C['ink2'])
    import random
    rng = random.Random(3)
    best = 0.47
    pts = []
    anc = {3: 'mut', 8: 'loss', 14: 'sel', 21: 'mut', 27: 'surv', 34: 'loss', 40: 'sel'}
    for g in range(46):
        if g in anc:
            best += rng.uniform(0.02, 0.05)
        for o in range(6):
            f = max(0.05, min(0.98, best - abs(rng.gauss(0, 0.12))))
            px = ox + g / 45 * ow + rng.uniform(-3, 3)
            py = oy + oh - (f - 0.25) / 0.6 * oh
            if oy <= py <= oy + oh:
                s.circle(px, py, 2.2, fill=C['line'], stroke='none', cls='w1')
        pts.append((ox + g / 45 * ow, oy + oh - (best - 0.25) / 0.6 * oh))
        if g in anc:
            s.circle(pts[-1][0], pts[-1][1], 5, fill=C[anc[g]], stroke=C['ink'], cls='w1')
    d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts)
    s.path(d, stroke=C['ink'], cls='w15')
    s.star(pts[-1][0] + 8, pts[-1][1], 9)
    s.text(ox - 6, pts[0][1] + 4, 'init', 's12 i', fill=C['mute'], anchor='e')
    s.text(pts[-1][0] - 4, pts[-1][1] + 26, 'evolved SR algorithm', 's12 b', anchor='e')
    s.text(ox + ow / 2, oy + oh + 8, '', 's11')
    return s


# =====================================================================
# D. Two timescales (swimlanes: outer programs / inner equations / test time)
# =====================================================================
def mock_D():
    H = 470
    s = Svg(W, H)
    # ---------- top lane
    s.panel(14, 26, 1072, 176, 'Outer loop · meta-evolution of SR programs (LLM, train time)', tab=C['llm_t'],
            tab_x=28)
    # init
    s.code_card(34, 78, 50, 58, dim=False)
    s.lines(59, 156, ['BasicSR', 'or PySR'], 's12 b', lh=14, anchor='m')
    blocks = [130, 386, 642]
    modes = ['explore', 'refine', 'crossover']
    hiop = ['mut', 'loss', 'sel']
    popseeds = [[3, 3, 3], [6, 4, 3], [9, 7, 5]]
    for b, bx in enumerate(blocks):
        s.text(bx + 4, 56, f'generation {b + 1}' if b < 2 else 'generation g', 's12 b', fill=C['ink2'])
        for k in range(3):
            y = 66 + k * 42
            s.code_card(bx, y, 32, 30, header=False, bars=1, hi=hiop[b - 1] if (b and k == 2) else None)
            s.seeds(bx + 42, y + 15, popseeds[b][k], 0, r=3, gap=8)
        # LLM pill + arrow to offspring
        s.arrow([(bx + 112, 108), (bx + 132, 108)], cls='w2', head=7)
        s.chip(bx + 134, 97, 'LLM', C['llm_t'], C['llm'], C['llm_d'], size=12, h=22, pad=7)
        s.chip(bx + 124, 124, modes[b], C['white'], C['line'], C['ink2'], size=11, h=18, pad=5)
        s.arrow([(bx + 178, 108), (bx + 190, 108)], cls='w2', head=7)
        for k in range(2):
            y = 80 + k * 44
            s.code_card(bx + 192, y, 32, 30, header=False, bars=1, hi=hiop[b])
            s.seeds(bx + 208 - 9, y + 40, 0, 3, r=2.6, gap=7)
        if b < 2:
            s.arrow([(bx + 232, 108), (blocks[b + 1] - 8, 108)], cls='w25', head=9)
    s.text(900, 112, '…', 's24 b', fill=C['ink2'], anchor='m')
    s.arrow([(870, 108), (886, 108)], cls='w25', head=8) if False else None
    s.code_card(950, 70, 56, 64, hi=None)
    s.star(1006, 72, 10)
    s.lines(978, 156, ['best evolved', 'SR program'], 's12 b', lh=14, anchor='m')
    s.arrow([(916, 104), (944, 104)], cls='w25', head=8)
    s.arrow([(bx + 232, 108), (886, 108)], cls='w25', head=9)
    # ---------- zoom from one offspring to the inner lane
    zx, zy = 386 + 192, 124 + 30
    s.path(f'M{zx:.1f},{zy:.1f} L300,236 M{zx + 32:.1f},{zy:.1f} L740,236', stroke=C['mute'], cls='w1 dot')
    # ---------- bottom lane: inner SR loop
    s.panel(14, 236, 740, 222, 'Inner loop · symbolic regression with that program (no LLM)', tab_x=28)
    for k in range(3):
        s.task_tile(34 + k * 6, 290 - k * 6, 78, 56, kind=k)
    s.lines(34, 368, ['one task,', 'one seed'], 's12', lh=15, fill=C['ink2'])
    s.lines(34, 410, ['× T tasks', '× N seeds', 'in parallel'], 's12 b', lh=15, fill=C['ink2'])
    s.arrow([(124, 318), (158, 318)], cls='w2', head=8)
    # loop track
    tx0, ty0, tx1, ty1 = 170, 286, 560, 430
    s.rect(tx0, ty0, tx1 - tx0, ty1 - ty0, fill=C['white'], stroke=C['ink2'], rx=60, cls='w2')
    s.arrow([(300, ty0), (420, ty0)], color=C['ink2'], cls='w2', head=9)
    s.arrow([(tx1, 340), (tx1, 380)], color=C['ink2'], cls='w2', head=9)
    s.arrow([(420, ty1), (300, ty1)], color=C['ink2'], cls='w2', head=9)
    s.arrow([(tx0, 380), (tx0, 340)], color=C['ink2'], cls='w2', head=9)
    s.opchip(250, ty0 - 11, 'sel', size=13, center=True)
    s.opchip(480, ty0 - 11, 'mut', size=13, center=True)
    s.opchip(480, ty1 - 11, 'loss', size=13, center=True)
    s.opchip(250, ty1 - 11, 'surv', size=13, center=True)
    for k, (x, y) in enumerate([(236, 318), (286, 330), (336, 316), (386, 332), (436, 318), (486, 330)]):
        s.tree(x, y, 0.72, hi=C['mut_t'] if k == 4 else None)
    s.text(365, 406, 'population of equations', 's13 i', fill=C['ink2'], anchor='m')
    s.arrow([(564, 358), (590, 358)], cls='w2', head=8)
    # pareto front plot
    px0, py0, pw0, ph0 = 596, 290, 146, 110
    s.rect(px0, py0, pw0, ph0, fill=C['white'], stroke=C['line'], rx=4, cls='w1')
    steps = [(0.05, 0.9), (0.2, 0.6), (0.35, 0.42), (0.55, 0.15), (0.75, 0.12), (0.95, 0.1)]
    d = 'M'
    for k, (u, v) in enumerate(steps):
        X, Y = px0 + 8 + u * (pw0 - 16), py0 + 8 + (1 - v) * 0 + v * (ph0 - 16)
        if k:
            d += f' L{X:.1f},{prevY:.1f}'
        d += f' L{X:.1f},{Y:.1f}' if k else f'{X:.1f},{Y:.1f}'
        prevY = Y
        s.circle(X, Y, 3, fill=C['ink2'], stroke='none', cls='w1')
    s.path(d, stroke=C['ink2'], cls='w12')
    s.check(px0 + 8 + 0.55 * (pw0 - 16) + 10, py0 + 8 + 0.15 * (ph0 - 16) - 12, 6)
    s.text(px0 + pw0 / 2, py0 + ph0 + 16, 'Pareto front → GT / R²', 's12', fill=C['ink2'], anchor='m')
    s.text(px0 + 6, py0 - 4, 'loss vs. size', 's11', fill=C['mute']) if False else None
    # score goes up to the offspring's seed dots
    s.carrow((px0 + pw0 - 10, py0), (px0 + pw0 - 10, 250), (zx + 60, 230), (zx + 30, 178), color=C['warn'],
             cls='w2', head=8)
    s.text(676, 224, 'fitness from this run', 's12 i', fill=C['warn'], anchor='e')
    # ---------- test-time panel
    s.panel(770, 236, 316, 222, 'Test time', tab_x=784)
    s.task_tile(790, 290, 78, 56, kind=4, stroke=C['ink2'])
    s.text(829, 364, 'new problem', 's12', fill=C['ink2'], anchor='m')
    s.arrow([(874, 318), (906, 318)], cls='w2', head=8)
    s.code_card(912, 286, 52, 64)
    s.star(962, 288, 8)
    s.arrow([(970, 318), (1000, 318)], cls='w2', head=8)
    s.text(1004, 314, 'equations', 's12 b')
    s.lines(790, 400, ['Evolved SR runs like PySR:', 'no LLM calls, any operators,', 'data, or loss.'],
            's12', lh=16, fill=C['ink2'])
    s.arrow([(1040, 180), (1040, 250), (938, 250), (938, 280)], cls='w2', head=8)
    return s


# =====================================================================
# E. Storyboard strip (five numbered steps, very shallow)
# =====================================================================
def mock_E():
    H = 400
    s = Svg(W, H)
    P = [(14, 176), (202, 216), (430, 276), (718, 160), (890, 196)]
    top, ph = 70, 316
    titles = [['Start from an', 'SR algorithm'], ['LLM rewrites', 'one operator'], ['Run on training tasks', 'in parallel'],
              ['Score', ''], ['Reevaluate &amp;', 'select']]
    for k, ((x, w), tt) in enumerate(zip(P, titles)):
        s.rect(x, top, w, ph, fill=C['panel'], stroke=C['ink2'], rx=12, cls='w15')
        s.badge(x + 20, top + 22, str(k + 1), r=11, size=13, fill=C['llm'] if k == 1 else C['ink'])
        s.lines(x + 38, top + 20, tt, 's14 b', lh=16)
        if k < 4:
            nx = P[k + 1][0]
            s.arrow([(x + w + 1, top + ph / 2), (nx - 1, top + ph / 2)], cls='w25', head=8)
    # loop arrow on top
    s.arrow([(988, top), (988, 34), (310, 34), (310, top - 2)], cls='w3', head=11)
    lab = 'next generation (repeat G times)'
    lw = text_w(lab, 14) + 18
    s.rect(650 - lw / 2, 22, lw, 24, fill=C['white'], stroke='none', rx=3, cls='w1')
    s.text(650, 39, lab, 's14 i', anchor='m')
    # 1: start
    x = 14
    s.code_card(x + 18, top + 56, 64, 96)
    s.text(x + 50, top + 170, 'BasicSR', 's12 b', anchor='m')
    s.code_card(x + 96, top + 56, 64, 96)
    s.text(x + 128, top + 170, 'PySR', 's12 b', anchor='m')
    s.text(x + 18, top + 202, 'evolvable operators:', 's12', fill=C['ink2'])
    for k, op in enumerate(OPS):
        s.opchip(x + 18 + (k % 2) * 76, top + 212 + (k // 2) * 26, op, size=11, h=20)
    s.text(x + 18, top + 284, '(BasicSR: + crossover,', 's11', fill=C['mute'])
    s.text(x + 18, top + 298, ' acceptance, …)', 's11', fill=C['mute'])
    # 2: LLM
    x = 202
    s.text(x + 16, top + 58, 'mode', 's12', fill=C['ink2'])
    mode_chips(s, x + 16, top + 64, active='simplify', colw=82, rowh=24)
    s.text(x + 16, top + 132, '+ parent code', 's12', fill=C['ink2'])
    s.text(x + 16, top + 148, '+ execution feedback', 's12', fill=C['ink2'])
    s.llm(x + 16, top + 168, 80, 50)
    s.arrow([(x + 98, top + 193), (x + 124, top + 193)], cls='w25', head=8)
    s.code_card(x + 128, top + 158, 70, 96, hi='surv')
    s.check(x + 22, top + 280, 6)
    s.text(x + 34, top + 284, 'validated before running', 's12', fill=C['surv_d'])
    # 3: evaluate
    x = 430
    s.text(x + 16, top + 58, 'T tasks × N seeds', 's12', fill=C['ink2'])
    tiles(s, x + 16, top + 66, 4, 1, 56, 40, gap=8, hi=(0, 1))
    rcx, rcy, rr = x + 138, top + 198, 64
    tx, ty = x + 16 + 64, top + 106
    s.line(tx, ty, rcx - rr - 8, rcy - 20, stroke=C['mute'], cls='w1 dot')
    s.line(tx + 56, ty, rcx + rr + 8, rcy - 20, stroke=C['mute'], cls='w1 dot')
    inner_ring(s, rcx, rcy, rr, chip_size=11, center=None)
    s.lines(rcx, rcy - 3, ['inner', 'SR loop'], 's12 b', lh=15, anchor='m')
    s.text(rcx, top + 298, 'each run: ~10⁶ equation evals, no LLM', 's11', fill=C['mute'], anchor='m')
    # 4: score
    x = 718
    s.text(x + 16, top + 64, 'meta-objective', 's12', fill=C['ink2'])
    for k, lab in enumerate(['GT recovery', 'R² (black box)', 'GT + R²']):
        s.chip(x + 16, top + 74 + k * 28, lab, C['ink2'] if k == 0 else C['white'], C['ink2'],
               C['white'] if k == 0 else C['ink2'], size=12, h=22, pad=8)
    s.text(x + 16, top + 184, 'mean over', 's12', fill=C['ink2'])
    s.text(x + 16, top + 200, 'tasks &amp; seeds', 's12', fill=C['ink2'])
    s.text(x + 16, top + 244, 'Ĵ(π) = mean L(S)', 's14 b i', fill=C['ink'])
    s.text(x + 16, top + 264, 'S ~ π(task, seed)', 's12 i', fill=C['ink2'])
    # 5: reevaluate & select
    x = 890
    rows = [(0.66, 7, 0, None), (0.63, 5, 0, None), (0.70, 0, 3, 'new'), (0.52, 3, 0, None)]
    for k, (f, n, nn, tag) in enumerate(rows):
        pop_row(s, x + 12, top + 70 + k * 36, f, n, nn, dropped=(k == 3), tag=None, barw=52,
                hi='surv' if tag == 'new' else None)
    s.lines(x + 14, top + 226, ['survivors: +' + s.sub('N', 'reeval') + 'seeds', 'offspring: ' + s.sub('N', 'init') + 'seeds',
                                'lucky ones fall out'], 's12', lh=16, fill=C['ink2'])
    s.chip(x + 14, top + 284, 'tournament → parents', C['white'], C['ink2'], size=11, h=20, pad=6)
    return s


MOCKS = {'A': mock_A, 'B': mock_B, 'C': mock_C, 'D': mock_D, 'E': mock_E}


if __name__ == '__main__':
    import cairosvg
    which = sys.argv[1:] or list(MOCKS)
    for k in which:
        s = MOCKS[k]()
        svg = s.svg(local=True)
        open(f'mock_{k}.svg', 'w').write(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f'mock_{k}.png', output_width=s.w * 2)
        print('wrote', k)
