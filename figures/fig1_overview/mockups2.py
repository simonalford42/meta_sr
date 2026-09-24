"""Round-2 Figure 1 mockups: one big outer loop (A-style boxes) with the inner SR loop inside Evaluate."""
import math
import sys
from svgkit import Svg, C, OPS, OPNAME, text_w
from mockups import inner_ring, trace_plot

W = 1100
TRACK = '#e2e7ee'
NEW = C['warn']          # evaluations run this generation
OLD = '#5b6678'          # evaluations from earlier generations


# =====================================================================
# components
# =====================================================================
def chevron(s, x, y, ang, size=13, color=C['ink2']):
    a = math.radians(ang)
    ux, uy = math.cos(a), math.sin(a)
    px, py = -uy, ux
    tip = (x + ux * size * 0.6, y + uy * size * 0.6)
    b = (x - ux * size * 0.4, y - uy * size * 0.4)
    s.poly([tip, (b[0] + px * size * 0.55, b[1] + py * size * 0.55),
            (b[0] - px * size * 0.55, b[1] - py * size * 0.55)], fill=color)


def op_squares(s, x, y, active='sel', sz=15, gap=6, label=True):
    for k, op in enumerate(OPS):
        xx = x + k * (sz + gap)
        if op == active:
            s.rect(xx - 2.5, y - 2.5, sz + 5, sz + 5, fill='none', stroke=C['ink'], rx=5, cls='w15')
            s.rect(xx, y, sz, sz, fill=C[op], stroke=C[op], rx=3, cls='w1')
        else:
            s.rect(xx, y, sz, sz, fill=C[op + '_t'], stroke=C[op], rx=3, cls='w1')
    if label:
        s.text(x + 4 * (sz + gap) + 4, y + sz * 0.8, OPNAME[active], 's13 b', fill=C[active + '_d'])


def mode_grid(s, x, y, active='refine', size=12, h=21, colw=None, rowh=25, cols=2):
    ws = {}
    for m in ['explore', 'refine', 'simplify', 'crossover']:
        ws[m] = text_w(m, size) * 1.07 + 14
    cw = colw or max(ws['explore'], ws['simplify']) + 6
    for k, m in enumerate(['explore', 'refine', 'simplify', 'crossover']):
        xx, yy = x + (k % cols) * cw, y + (k // cols) * rowh
        act = m == active
        s.chip(xx, yy, m, C['llm_t'] if act else C['white'], C['llm'] if act else C['line'],
               C['llm_d'] if act else C['ink2'], size=size, h=h, pad=7)


def validate_chip(s, x, y, label='validate'):
    w = text_w(label, 13) * 1.07 + 34
    s.rect(x, y, w, 24, fill=C['surv_t'], stroke=C['surv'], rx=12, cls='w12')
    s.check(x + 14, y + 12, 5.5)
    s.text(x + 24, y + 16.5, label, 's13 b', fill=C['surv_d'])
    return w


def meta_mutation(s, x, y, w, h, title='Meta-mutation', compact=False, active_op='sel'):
    s.panel(x, y, w, h, title, tab=C['llm_t'])
    if compact:
        s.text(x + 12, y + 32, 'mode', 's13', fill=C['ink2'])
        mode_grid(s, x + 52, y + 18, size=12, h=20, rowh=24)
        s.text(x + 12, y + 88, 'operator', 's13', fill=C['ink2'])
        op_squares(s, x + 72, y + 76, active_op, sz=13, gap=5, label=False)
        s.text(x + 12, y + 116, '+ exec. feedback', 's13', fill=C['ink2'])
        s.llm(x + w - 66, y + 96, 56, 40, size=17)
        s.check(x + w - 56, y + 148, 5)
        s.text(x + w - 46, y + 152, 'valid', 's13 b', fill=C['surv_d'])
        return
    pw = w - 104
    px, py = x + 12, y + 22
    s.rect(px, py, pw, h - 34, fill=C['white'], stroke=C['line'], rx=8, cls='w12')
    s.text(px + 10, py + 21, 'Prompt', 's15 b')
    s.code_card(px + pw - 58, py + 7, 22, 26, header=False, bars=1)
    s.code_card(px + pw - 32, py + 7, 22, 26, header=False, bars=1)
    s.text(px + 10, py + 46, 'mode', 's13', fill=C['ink2'])
    mode_grid(s, px + 52, py + 32, size=12, h=20, rowh=24)
    s.text(px + 10, py + 104, 'operator', 's13', fill=C['ink2'])
    op_squares(s, px + 72, py + 92, active_op, sz=13, gap=5, label=False)
    s.text(px + 10, py + 134, 'feedback', 's13', fill=C['ink2'])
    trace_plot(s, px + 72, py + 118, pw - 82, 24)
    lx = x + w - 80
    s.arrow([(px + pw + 2, y + h / 2 - 8), (lx - 2, y + h / 2 - 8)], cls='w2', head=8)
    s.llm(lx, y + h / 2 - 32, 64, 48, size=18)
    s.arrow([(lx + 32, y + h / 2 + 18), (lx + 32, y + h / 2 + 34)], cls='w2', head=7)
    s.check(lx + 14, y + h / 2 + 48, 5.5)
    s.text(lx + 24, y + h / 2 + 53, 'valid', 's13 b', fill=C['surv_d'])


def offspring(s, cx, cy, label=True, sc=1.0):
    ops = ['mut', 'sel', 'loss']
    w, h = 34 * sc, 44 * sc
    for k, (dx, ang) in enumerate([(-16, -10), (0, 0), (16, 10)]):
        x, y = cx + dx * sc - w / 2, cy - h / 2
        s.raw(f'<g transform="rotate({ang} {cx + dx * sc:.1f} {cy + h / 2:.1f})">')
        s.code_card(x, y, w, h, header=False, bars=1, hi=ops[k])
        s.raw('</g>')
    if label:
        s.text(cx, cy + h / 2 + 22, 'offspring', 's15 b', anchor='m')


def seeds2(s, x, y, n_old, n_new, r=3.6, gap=9.5):
    for k in range(n_old + n_new):
        s.circle(x + k * gap, y, r, fill=OLD if k < n_old else NEW, stroke='none', cls='w1')


def fitness_chips(s, x, y, size=13, h=23, active=0, gap=6):
    xx = x
    for k, lab in enumerate(['GT', 'R²', 'GT-R²']):
        act = k == active
        xx += s.chip(xx, y, lab, C['ink2'] if act else C['white'], C['ink2'], C['white'] if act else C['ink2'],
                     size=size, h=h, pad=9) + gap
    return xx - x


def reeval_chips(s, x, y, size=12, h=21):
    w = s.chip(x, y, 'fixed ' + 'N', C['ink2'], C['ink2'], C['white'], size=size, h=h, pad=8)
    s.chip(x + w + 5, y, 'TTTS', C['white'], C['ink2'], C['ink2'], size=size, h=h, pad=8)


def ledger(s, x, y, rows, rowh=26, icon=True):
    for k, (n_old, n_new, hi, faded) in enumerate(rows):
        yy = y + k * rowh
        if icon:
            s.code_card(x, yy - 10, 18, 20, header=False, bars=1, hi=hi, dim=faded)
        seeds2(s, x + 30, yy, n_old, n_new)


def heatmap(s, x, y, cw=22, ch=17, gap=2, icons=True):
    M = [[.62, .20, .55, .30, .72, .41],
         [.35, .80, .40, .28, .52, .66],
         [.58, .45, .50, .52, .60, .50],
         [.30, .35, .78, .74, .40, .33],
         [.25, .30, .35, .20, .31, .28]]
    kept = [True, True, True, True, False]
    ox = x + (26 if icons else 0)
    colmax = [max(range(5), key=lambda r: M[r][c]) for c in range(6)]

    def shade(v):
        a = (int('f1', 16), int('f3', 16), int('f6', 16))
        b = (int('3d', 16), int('4a', 16), int('5c', 16))
        t = min(max((v - 0.15) / 0.7, 0), 1)
        return '#%02x%02x%02x' % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
    for r in range(5):
        yy = y + r * (ch + gap)
        if icons:
            s.code_card(x, yy - 1, 18, ch + 2, header=False, bars=1, dim=not kept[r])
        for c in range(6):
            s.rect(ox + c * (cw + gap), yy, cw, ch, fill=shade(M[r][c]) if kept[r] else '#f4f5f7',
                   stroke='none', rx=2, cls='w1')
    for c, r in enumerate(colmax):
        s.rect(ox + c * (cw + gap) - 1.5, y + r * (ch + gap) - 1.5, cw + 3, ch + 3, fill='none',
               stroke=C['gold'], rx=3, cls='w25')
    return ox + 6 * (cw + gap), y + 5 * (ch + gap)


def pareto(s, x, y, w, h, check=True):
    s.rect(x, y, w, h, fill=C['white'], stroke=C['line'], rx=4, cls='w1')
    steps = [(0.05, 0.92), (0.22, 0.62), (0.38, 0.44), (0.56, 0.16), (0.76, 0.12), (0.95, 0.10)]
    d, prev = '', None
    for k, (u, v) in enumerate(steps):
        X, Y = x + 7 + u * (w - 14), y + 6 + (1 - v) * 0 + v * (h - 12)
        Y = y + 6 + (1 - (1 - v)) * (h - 12)
        d += (f'M{X:.1f},{Y:.1f}' if k == 0 else f' L{X:.1f},{prev:.1f} L{X:.1f},{Y:.1f}')
        prev = Y
    s.path(d, stroke=C['ink2'], cls='w12')
    for u, v in steps:
        s.circle(x + 7 + u * (w - 14), y + 6 + v * (h - 12), 2.4, fill=C['ink2'], stroke='none', cls='w1')
    if check:
        s.check(x + 7 + 0.56 * (w - 14) + 1, y + 6 + 0.16 * (h - 12) - 10, 5)


def task_stack(s, x, y, w=62, h=42, n=3, label=True, lab2=True):
    for k in range(n):
        s.task_tile(x + (n - 1 - k) * 6, y + k * 6 - (n - 1) * 6 + 12, w, h, kind=k + 1)
    if label:
        s.text(x + w / 2 + 6, y + h + 32, 'training tasks', 's13 b', anchor='m')
    if lab2:
        s.text(x + w / 2 + 6, y + h + 48, 'T × N seeds', 's13', fill=C['ink2'], anchor='m')


def deploy(s, x, y, w, h):
    s.panel(x, y, w, h, 'After training', tab=C['sel_t'], tsize=15)
    cx = x + w / 2
    s.code_card(cx - 20, y + 24, 40, 50)
    s.star(cx + 19, y + 26, 9)
    s.text(cx, y + 94, 'evolved SR algorithm', 's13 b', anchor='m')
    yy = y + 112
    for k, (lab, icon) in enumerate([('new domains', 'arrow'), ('fine-tune', 'cycle')]):
        by = yy + k * 30
        s.rect(x + 12, by, w - 24, 24, fill=C['white'], stroke=C['line'], rx=12, cls='w12')
        if icon == 'arrow':
            s.arrow([(x + 22, by + 12), (x + 38, by + 12)], cls='w15', head=6)
        else:
            s.arc(x + 30, by + 12, 6.5, 200, 480, color=C['ink'], cls='w15', head=5)
        s.text(x + 46, by + 16.5, lab, 's13', fill=C['ink'])
    s.text(cx, y + h - 12, 'no LLM calls', 's12 i', fill=C['mute'], anchor='m')


def deploy_small(s, x, y, w, h):
    """Compact version: star card + two labelled arrows."""
    s.panel(x, y, w, h, 'Use', tab=C['sel_t'], tsize=15)
    cx = x + 36
    s.code_card(cx - 20, y + 30, 40, 48)
    s.star(cx + 18, y + 32, 8)
    s.arrow([(cx + 26, y + 46), (cx + 50, y + 40)], cls='w15', head=7)
    s.arrow([(cx + 26, y + 64), (cx + 50, y + 74)], cls='w15', head=7)
    s.text(cx + 56, y + 44, 'new domains', 's13', fill=C['ink2'])
    s.text(cx + 56, y + 80, 'fine-tune', 's13', fill=C['ink2'])
    s.text(x + w / 2, y + h - 12, 'no LLM calls at test time', 's12 i', fill=C['mute'], anchor='m')


def eval_contents(s, x, y, w, h, r=52, tasks_from=None):
    """Inner ring + Pareto output inside an Evaluate box."""
    rcx, rcy = x + 16 + 40 + r, y + h / 2 + 8
    inner_ring(s, rcx, rcy, r, chip_size=12, center=None)
    s.lines(rcx, rcy - 3, ['inner', 'SR loop'], 's13 b', lh=15, anchor='m')
    return rcx, rcy


# ------------------------------------------------ box contents for the bottom row
def fitness_box(s, x, y, w, h, title='Initial eval &amp; meta-fitness'):
    s.panel(x, y, w, h, title, tsize=15)
    s.text(x + 14, y + 34, 'meta-fitness', 's13', fill=C['ink2'])
    fitness_chips(s, x + 14, y + 42)
    s.text(x + 14, y + 90, 'offspring:', 's13', fill=C['ink2'])
    s.text(x + 14, y + 106, s.sub('N', 'init') + 'seeds', 's13', fill=C['ink2'])
    seeds2(s, x + 104, y + 96, 0, 3, r=4.2, gap=11)
    s.text(x + 14, y + h - 16, 'mean score over tasks, seeds', 's12 i', fill=C['mute'])


def reeval_box(s, x, y, w, h, title='Reevaluate'):
    s.panel(x, y, w, h, title, tsize=15)
    rows = [(8, 2, None, False), (5, 2, None, False), (3, 2, None, False), (0, 3, 'sel', False)]
    ledger(s, x + 16, y + 34, rows, rowh=24)
    s.text(x + 180, y + 64, '', 's12')
    lx = x + w - 96
    s.text(lx, y + 38, 'strategy', 's13', fill=C['ink2'])
    s.chip(lx, y + 46, 'fixed', C['ink2'], C['ink2'], C['white'], size=12, h=21, pad=8)
    s.chip(lx + 44, y + 46, 'TTTS', C['white'], C['ink2'], C['ink2'], size=12, h=21, pad=8)
    # legend
    s.circle(x + 20, y + h - 20, 4, fill=OLD, stroke='none', cls='w1')
    s.text(x + 28, y + h - 15.5, 'earlier', 's12', fill=C['ink2'])
    s.circle(x + 82, y + h - 20, 4, fill=NEW, stroke='none', cls='w1')
    s.text(x + 90, y + h - 15.5, 'this generation', 's12', fill=C['ink2'])


def pop_box(s, x, y, w, h, title='Task-based population'):
    s.panel(x, y, w, h, title, tsize=15)
    hx, hy = x + 14, y + 26
    s.text(hx + 26, hy + 8, 'tasks →', 's12', fill=C['mute'])
    ex, ey = heatmap(s, hx, hy + 16, cw=20, ch=16, gap=2)
    s.rect(ex + 10, hy + 22, 14, 12, fill='none', stroke=C['gold'], rx=2, cls='w25')
    s.lines(ex + 30, hy + 32, ['best on', 'a task'], 's12', lh=14, fill=C['ink2'])
    s.lines(ex + 10, hy + 76, ['keep each', 'task’s best;', 'fill by mean'], 's12', lh=14, fill=C['ink2'])


# =====================================================================
# F. Racetrack: stadium band behind the six stations
# =====================================================================
def mock_F(style='track'):
    H = 470
    s = Svg(W, H)
    xl, xr, cy, r = 190, 740, 238, 128
    d = (f'M{xl:.1f},{cy - r:.1f} L{xr:.1f},{cy - r:.1f} A{r},{r} 0 0 1 {xr:.1f},{cy + r:.1f} '
         f'L{xl:.1f},{cy + r:.1f} A{r},{r} 0 0 1 {xl:.1f},{cy - r:.1f} Z')
    if style == 'track':
        s.path(d, stroke=TRACK, cls='w18')
        s.path(d, stroke=C['ink2'], cls='w15 ldash')
    else:
        s.path(d, stroke=C['ink'], cls='w4')
    # stations
    meta_mutation(s, 22, 26, 312, 172)
    offspring(s, 400, 102)
    E = (466, 26, 388, 172)
    s.panel(*E, 'Evaluate')
    rcx, rcy, rr = E[0] + 110, E[1] + 90, 58
    inner_ring(s, rcx, rcy, rr, chip_size=12, center='trees')
    if style != 'track':
        for a0 in (-90, 0, 90, 180):
            s.arc(rcx, rcy, rr, a0 + 24, a0 + 66, color=C['ink'], cls='w3', head=10)
        inner_ring(s, rcx, rcy, rr, chip_size=12, center='trees') if False else None
    s.text(E[0] + 222, E[1] + 40, 'Inner SR loop', 's15 b')
    s.text(E[0] + 222, E[1] + 58, 'one run per task × seed', 's12', fill=C['ink2'])
    s.arrow([(rcx + rr + 44, rcy + 14), (E[0] + 252, rcy + 14)], cls='w2', head=8)
    pareto(s, E[0] + 256, E[1] + 76, 104, 62)
    s.text(E[0] + 308, E[1] + 156, 'Pareto front', 's12', fill=C['ink2'], anchor='m')
    fitness_box(s, 640, 292, 214, 162)
    reeval_box(s, 342, 292, 268, 162)
    pop_box(s, 22, 292, 290, 162)
    # tasks fed from the loop's interior
    task_stack(s, 610, 212, w=54, h=36, label=False, lab2=False)
    s.arrow([(640, 212), (640, 200)], cls='w2', head=7) if False else None
    s.arrow([(rcx + 0, 214), (rcx + 0, 200)], cls='w2', head=7) if False else None
    s.arrow([(676, 214), (676, 200)], cls='w2', head=8)
    s.text(600, 236, 'training tasks', 's13 b', anchor='e')
    s.text(600, 252, 'T tasks × N seeds', 's13', fill=C['ink2'], anchor='e')
    # track arrows
    big = 16 if style == 'track' else 20
    for x, y, a in [(349, 110, 0), (452, 110, 0), (xr + r, cy, 90), (626, 366, 180), (327, 366, 180), (xl - r, cy, 270)]:
        chevron(s, x, y, a, size=big, color=C['ink'])
    s.lines(250, 232, ['Meta-evolution', 'G generations'], 's16 b i', lh=19, fill=C['ink2'], anchor='m')
    # deploy
    s.arrow([(xr + r + 14, cy), (912, cy)], cls='w15 dash', head=8)
    deploy(s, 918, 130, 168, 216)
    return s


def mock_G():
    return mock_F(style='bold')


# =====================================================================
# H. Loop in a loop: stations ride an elliptical track; inner SR loop sits at its centre
# =====================================================================
def mock_H():
    H = 470
    s = Svg(W, H)
    ecx, ecy, rx, ry = 456, 246, 356, 170
    s.raw(f'<ellipse cx="{ecx}" cy="{ecy}" rx="{rx}" ry="{ry}" fill="none" stroke="{TRACK}" class="w18"/>')
    s.raw(f'<ellipse cx="{ecx}" cy="{ecy}" rx="{rx}" ry="{ry}" fill="none" stroke="{C["ink2"]}" class="w15 ldash"/>')

    def ept(a):
        return ecx + rx * math.cos(math.radians(a)), ecy + ry * math.sin(math.radians(a))

    def etan(a):
        dx, dy = -rx * math.sin(math.radians(a)), ry * math.cos(math.radians(a))
        return math.degrees(math.atan2(dy, dx))
    for a in (236, 290, 346, 44, 112, 176):
        x, y = ept(a)
        chevron(s, x, y, etan(a), size=16, color=C['ink'])
    # stations (clockwise from upper left)
    meta_mutation(s, 16, 40, 222, 160, compact=True)
    x, y = ept(266)
    offspring(s, x, y - 2, sc=0.9)
    Ex, Ey = ept(322)
    s.chip(Ex, Ey - 15, 'Evaluate', C['tab'], C['ink2'], size=15, h=30, center=True)
    Fx, Fy = ept(16)
    fb = (Fx - 86, Fy - 36, 178, 100)
    s.panel(*fb, 'Meta-fitness', tsize=15)
    fitness_chips(s, fb[0] + 14, fb[1] + 22, size=13)
    s.text(fb[0] + 14, fb[1] + 72, s.sub('N', 'init') + 'seeds', 's13', fill=C['ink2'])
    seeds2(s, fb[0] + 94, fb[1] + 67, 0, 3, r=4, gap=10)
    Rx, Ry = ept(84)
    rb = (Rx - 118, Ry - 48, 236, 96)
    s.panel(*rb, 'Reevaluate', tsize=15)
    ledger(s, rb[0] + 16, rb[1] + 24, [(8, 2, None, False), (4, 2, None, False), (0, 3, 'sel', False)], rowh=22)
    s.chip(rb[0] + 160, rb[1] + 18, 'fixed', C['ink2'], C['ink2'], C['white'], size=12, h=20, pad=7)
    s.chip(rb[0] + 160, rb[1] + 44, 'TTTS', C['white'], C['ink2'], C['ink2'], size=12, h=20, pad=7)
    pb = (16, 262, 232, 124)
    s.panel(*pb, 'Task-based population', tsize=15)
    ex, ey = heatmap(s, pb[0] + 14, pb[1] + 22, cw=18, ch=15, gap=2)
    s.lines(ex + 10, pb[1] + 42, ['keep each', 'task’s best'], 's12', lh=14, fill=C['ink2'])
    s.rect(ex + 10, pb[1] + 68, 13, 11, fill='none', stroke=C['gold'], rx=2, cls='w25')
    s.text(ex + 28, pb[1] + 78, 'best', 's12', fill=C['ink2'])
    # inner loop at the centre of the outer loop
    icx, icy, ir = 500, 226, 64
    inner_ring(s, icx, icy, ir, chip_size=13, center='trees')
    s.path(f'M{Ex - 30:.1f},{Ey + 16:.1f} C{Ex - 70:.1f},{Ey + 50:.1f} {icx + 110:.1f},{icy - 70:.1f} {icx + 58:.1f},{icy - 58:.1f}',
           stroke=C['ink2'], cls='w15 dash')
    s.arrow([(icx + ir + 36, icy + 26), (fb[0] - 6, fb[1] + 34)], cls='w2', head=8)
    task_stack(s, 306, 196, w=54, h=36, label=False, lab2=False)
    s.arrow([(374, icy), (icx - ir - 42, icy)], cls='w2', head=8)
    s.text(336, 278, 'training tasks', 's13 b', anchor='m')
    s.text(336, 294, 'T × N seeds', 's13', fill=C['ink2'], anchor='m')
    s.text(icx, icy + ir + 34, 'Inner loop: evolve equations', 's13 i', fill=C['ink2'], anchor='m')
    s.text(ecx + 20, 26, 'Outer loop: evolve SR programs with an LLM', 's15 b i', fill=C['ink2'], anchor='m')
    deploy(s, 918, 130, 168, 216)
    s.arrow([(ecx + rx + 14, ecy - 70), (912, ecy - 70)], cls='w15 dash', head=8) if False else None
    return s


# =====================================================================
# I. Ring + lens: outer loop as a ring; a magnifier shows the inner loop fed by tasks
# =====================================================================
def mock_I():
    H = 470
    s = Svg(W, H)
    ocx, ocy, R = 290, 240, 168
    s.circle(ocx, ocy, R, fill='none', stroke=TRACK, cls='w18')
    s.circle(ocx, ocy, R, fill='none', stroke=C['ink2'], cls='w15 ldash')
    for a in (292, 30, 150, 208):
        x, y = ocx + R * math.cos(math.radians(a)), ocy + R * math.sin(math.radians(a))
        chevron(s, x, y, a + 90, size=16, color=C['ink'])
    # lens
    lcx, lcy, lr = 712, 238, 184
    Ex, Ey = ocx + R, ocy
    for sgn in (-1, 1):
        s.line(Ex + 36, Ey + sgn * 16, lcx - lr * 0.55, lcy + sgn * lr * 0.83, stroke=C['mute'], cls='w1 dot')
    s.circle(lcx, lcy, lr, fill=C['panel'], stroke=C['ink2'], cls='w2')
    s.text(lcx, lcy - lr + 30, 'Evaluate', 's16 b i', anchor='m')
    icx, icy = lcx + 6, lcy - 34
    inner_ring(s, icx, icy, 62, chip_size=13, center='trees')
    task_stack(s, lcx - 150, lcy + 4, w=52, h=34, label=False, lab2=False)
    s.arrow([(lcx - 90, lcy + 16), (icx - 70, icy + 40)], cls='w2', head=8)
    s.lines(lcx - 118, lcy + 66, ['T tasks', '× N seeds'], 's13', lh=15, fill=C['ink2'], anchor='m')
    s.arrow([(icx + 62, icy + 50), (lcx + 86, lcy + 42)], cls='w2', head=8)
    pareto(s, lcx + 80, lcy + 44, 64, 46)
    s.text(lcx + 112, lcy + 106, 'Pareto front', 's12', fill=C['ink2'], anchor='m')
    s.text(lcx - 6, lcy + 104, 'meta-fitness', 's13', fill=C['ink2'], anchor='m')
    fitness_chips(s, lcx - 72, lcy + 112, size=13)
    s.text(lcx - 58, lcy + 158, s.sub('N', 'init') + 'seeds each', 's13', fill=C['ink2'])
    seeds2(s, lcx + 42, lcy + 153, 0, 3, r=4, gap=10)
    # ring stations
    meta_mutation(s, 20, 26, 230, 164, compact=True)
    s.chip(Ex, Ey - 15, 'Evaluate', C['tab'], C['ink2'], size=15, h=30, center=True)
    offspring(s, ocx + 116, ocy - 142, sc=0.85)
    rb = (ocx - 40, ocy + 118, 220, 100)
    s.panel(*rb, 'Reevaluate', tsize=15)
    ledger(s, rb[0] + 14, rb[1] + 26, [(8, 2, None, False), (4, 2, None, False), (0, 3, 'sel', False)], rowh=22)
    s.chip(rb[0] + 150, rb[1] + 20, 'fixed', C['ink2'], C['ink2'], C['white'], size=12, h=20, pad=7)
    s.chip(rb[0] + 150, rb[1] + 46, 'TTTS', C['white'], C['ink2'], C['ink2'], size=12, h=20, pad=7)
    pb = (20, 238, 176, 132)
    s.panel(*pb, 'Population', tsize=15)
    heatmap(s, pb[0] + 14, pb[1] + 26, cw=17, ch=15, gap=2, icons=True)
    s.text(pb[0] + 14, pb[1] + 122, 'best on ≥1 task', 's12', fill=C['ink2'])
    s.lines(ocx + 10, ocy - 8, ['Meta-', 'evolution'], 's16 b i', lh=19, fill=C['ink2'], anchor='m')
    deploy(s, 918, 130, 168, 216)
    return s


MOCKS = {'F': mock_F, 'G': mock_G, 'H': mock_H, 'I': mock_I}

if __name__ == '__main__':
    import cairosvg
    which = sys.argv[1:] or list(MOCKS)
    for k in which:
        s = MOCKS[k]()
        svg = s.svg(local=True)
        open(f'mock_{k}.svg', 'w').write(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f'mock_{k}.png', output_width=s.w * 2)
        print('wrote', k)
