"""Round-5 Figure 1: refined Q (3x2 grid).  Column 1 = meta-mutation; row 1 cols 2-3 =
Evaluate (taller); row 2 cols 2-3 = population/reevaluation + initial evaluation."""
import math
import sys
from svgkit import Svg, C, OPS, OPNAME, text_w
from mockups2 import chevron, fitness_chips, task_stack, mode_grid, op_squares, TRACK
from mockups3 import tiny_tree, glyph_mutation, glyph_loss, seeds_ir, strategy_pill, legend_seeds, INIT, REEV

W, H = 1100, 470
MX0, MX1 = 16, 300                 # meta-mutation column
RX0, RX1 = 372, 1040               # right two columns
Y0, YM0, YM1, Y1 = 22, 270, 290, 456
TOP, BOT = (Y0 + YM0) // 2, (YM1 + Y1) // 2 + 4
XT, RC = 1066, 18                  # ] -shaped turnaround


# ------------------------------------------------------------------ small pieces
def robot(s, x, y, sc=1.0, color=C['llm_d']):
    """Line-drawn robot head, top-left at (x, y); ~22 x 22 at sc=1."""
    w, h = 20 * sc, 15 * sc
    s.line(x + w / 2, y + 1 * sc, x + w / 2, y + 6 * sc, stroke=color, cls='w15')
    s.circle(x + w / 2, y + 1 * sc, 2.2 * sc, fill=color, stroke='none', cls='w1')
    s.rect(x, y + 6 * sc, w, h, fill=C['white'], stroke=color, rx=4 * sc, cls='w15')
    s.circle(x + w * 0.32, y + 6 * sc + h * 0.45, 2.1 * sc, fill=color, stroke='none', cls='w1')
    s.circle(x + w * 0.68, y + 6 * sc + h * 0.45, 2.1 * sc, fill=color, stroke='none', cls='w1')
    s.line(x + w * 0.34, y + 6 * sc + h * 0.78, x + w * 0.66, y + 6 * sc + h * 0.78, stroke=color, cls='w12')
    s.line(x - 2.5 * sc, y + 10 * sc, x - 2.5 * sc, y + 17 * sc, stroke=color, cls='w2')
    s.line(x + w + 2.5 * sc, y + 10 * sc, x + w + 2.5 * sc, y + 17 * sc, stroke=color, cls='w2')


def llm_box(s, x, y, w, h):
    s.rect(x, y, w, h, fill=C['llm_t'], stroke=C['llm'], rx=12, cls='w2')
    robot(s, x + 13, y + h / 2 - 12, 1.05)
    s.text(x + w - 11, y + h / 2 + 6.5, 'LLM', 's18 b', fill=C['llm_d'], anchor='e')


def pareto2(s, x, y, w, h, labels=True, check=True, frame=True):
    """Loss (y) vs. complexity (x): a decreasing staircase."""
    if frame:
        s.rect(x, y, w, h, fill=C['white'], stroke=C['line'], rx=4, cls='w1')
    ax0, ay1 = x + (16 if labels else 7), y + h - (14 if labels else 6)
    ax1, ay0 = x + w - 6, y + 6
    s.line(ax0, ay1, ax1, ay1, stroke=C['mute'], cls='w1')
    s.line(ax0, ay0, ax0, ay1, stroke=C['mute'], cls='w1')
    pts = [(0.05, 0.92), (0.22, 0.60), (0.40, 0.42), (0.58, 0.16), (0.78, 0.12), (0.96, 0.09)]
    P = [(ax0 + 4 + u * (ax1 - ax0 - 8), ay1 - 3 - v * (ay1 - ay0 - 6)) for u, v in pts]
    d = f'M{P[0][0]:.1f},{P[0][1]:.1f}'
    for (x0, y0), (x1, y1) in zip(P, P[1:]):
        d += f' L{x1:.1f},{y0:.1f} L{x1:.1f},{y1:.1f}'
    s.path(d, stroke=C['ink2'], cls='w12')
    for px, py in P:
        s.circle(px, py, 2.4, fill=C['ink2'], stroke='none', cls='w1')
    if check:
        s.circle(P[3][0], P[3][1], 4.6, fill='none', stroke=C['surv'], cls='w2')
    if labels:
        s.text(ax0 - 4, ay0 + 4, 'loss', 's11', fill=C['mute'], anchor='e') if False else None
        s.raw(f'<text x="{x + 10:.1f}" y="{(ay0 + ay1) / 2:.1f}" class="t s11 m" fill="{C["mute"]}" '
              f'transform="rotate(-90 {x + 10:.1f} {(ay0 + ay1) / 2:.1f})">loss</text>')
        s.text((ax0 + ax1) / 2, y + h - 3, 'complexity', 's11', fill=C['mute'], anchor='m')


def glyph_selection(s, x, y):
    """Tournament: three sampled candidates with fitness bars; the fittest is picked."""
    fits = [0.45, 0.85, 0.6]
    for k, f in enumerate(fits):
        tx = x + k * 32
        win = k == 1
        s.rect(tx, y, 26, 30, fill=C['sel_t'] if win else C['white'], stroke=C['sel'] if win else C['line'],
               rx=4, cls='w2' if win else 'w1')
        tiny_tree(s, tx + 13, y + 6, 0.44)
        s.rect(tx + 3, y + 36, 20, 5, fill=C['panel2'], stroke='none', rx=2, cls='w1')
        s.rect(tx + 3, y + 36, 20 * f, 5, fill=C['sel_d'] if win else C['mute'], stroke='none', rx=2, cls='w1')


def glyph_survival2(s, x, y, n=4, sw=22, sh=28, gap=4):
    for k in range(n):
        sx = x + k * (sw + gap)
        new = k == 1
        s.rect(sx, y, sw, sh, fill=C['surv_t'] if new else C['white'], stroke=C['surv'] if new else C['line'],
               rx=4, cls='w15' if new else 'w1')
        tiny_tree(s, sx + sw / 2, y + 5, 0.42, color=C['surv_d'] if new else C['ink2'])


def island_loop2(s, x, y, w, h, ghosts=2):
    r = h / 2
    for g in range(ghosts, 0, -1):
        s.rect(x + 6 * g, y + 6 * g, w, h, fill=C['white'], stroke=C['line'], rx=r, cls='w1')
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
    glyph_survival2(s, x0 + 10 - 50, y + h - 58)


def prog(s, x, y, hi=None, dim=False, w=20, h=26):
    s.code_card(x, y - h / 2, w, h, header=False, bars=1, hi=hi, dim=dim)


# ------------------------------------------------------------------ blocks
def track(s):
    xl = 160
    d = (f'M{xl},{TOP} L{XT - RC},{TOP} A{RC},{RC} 0 0 1 {XT},{TOP + RC} L{XT},{BOT - RC} '
         f'A{RC},{RC} 0 0 1 {XT - RC},{BOT} L{xl},{BOT} Z')
    s.path(d, stroke=TRACK, cls='w18')
    s.path(d, stroke=C['ink2'], cls='w15 ldash')


def meta_mutation(s):
    x, y, w, h = MX0, Y0, MX1 - MX0, Y1 - Y0
    s.panel(x, y, w, h, 'Meta-mutation', tab=C['llm_t'])
    s.text(x + 18, TOP - 52, 'LLM rewrites one operator', 's13 i', fill=C['ink2'])
    s.text(x + 18, TOP - 36, 'of a parent program', 's13 i', fill=C['ink2'])
    # LLM -> child -> validate -> out
    llm_box(s, x + 12, TOP - 25, 98, 50)
    s.arrow([(x + 112, TOP), (x + 124, TOP)], cls='w2', head=7)
    s.code_card(x + 126, TOP - 19, 28, 38, header=False, bars=1, hi='sel')
    s.arrow([(x + 156, TOP), (x + 168, TOP)], cls='w2', head=7)
    s.rect(x + 170, TOP - 13, 86, 26, fill=C['surv_t'], stroke=C['surv'], rx=13, cls='w12')
    s.check(x + 186, TOP, 5.5)
    s.text(x + 198, TOP + 4.5, 'validate', 's13 b', fill=C['surv_d'])
    s.arrow([(x + 258, TOP), (x + w - 2, TOP)], cls='w2', head=7)
    # prompt
    px, py, pw, ph = x + 14, TOP + 44, w - 28, 150
    s.arrow([(x + 62, py - 2), (x + 62, TOP + 27)], cls='w2', head=8)
    s.rect(px, py, pw, ph, fill=C['white'], stroke=C['line'], rx=8, cls='w12')
    s.text(px + 10, py + 21, 'Prompt', 's15 b')
    s.text(px + 10, py + 46, 'mode', 's13', fill=C['ink2'])
    mode_grid(s, px + 78, py + 32, size=12, h=20, rowh=24)
    s.text(px + 10, py + 98, 'operator', 's13', fill=C['ink2'])
    op_squares(s, px + 78, py + 86, 'sel', sz=14, gap=6, label=False)
    s.text(px + 10, py + 122, 'execution', 's13', fill=C['ink2'])
    s.text(px + 10, py + 138, 'feedback', 's13', fill=C['ink2'])
    pareto2(s, px + 78, py + 110, 62, 34, labels=False, check=False)
    s.text(px + 148, py + 124, 'best eqn:', 's11', fill=C['mute'])
    s.text(px + 148, py + 139, '1.3·sin(x₀)', 's12 mono', fill=C['ink'])
    # parents (meta-selection) at the loop's entry
    cy = BOT + 4
    for k in range(4):
        cx = x + 36 + k * 36
        picked = k in (0, 2)
        if picked:
            s.rect(cx - 4, cy - 20, 30, 40, fill='none', stroke=C['llm'], rx=6, cls='w2')
            s.arrow([(cx + 11, cy - 22), (cx + 11, py + ph + 2)], cls='w2', head=7)
        s.code_card(cx, cy - 16, 22, 32, header=False, bars=1, dim=not picked)
    s.text(x + 18, cy + 42, 'meta-selection', 's14 b', fill=C['ink2'])


def offspring_column(s, cx, cy):
    ops = ['mut', 'sel', 'loss']
    cw, ch, gap = 38, 34, 7
    top = cy - (3 * ch + 2 * gap) / 2
    for k, op in enumerate(ops):
        s.code_card(cx - cw / 2, top + k * (ch + gap), cw, ch, header=False, bars=1, hi=op)
    s.text(cx, top + 3 * ch + 2 * gap + 18, 'offspring', 's13 b', anchor='m')


def evaluate(s):
    x, y, w, h = RX0, Y0, RX1 - RX0, YM0 - Y0
    s.panel(x, y, w, h, 'Evaluate: inner SR loop')
    task_stack(s, x + 16, TOP - 30, w=52, h=36, label=False, lab2=False)
    s.text(x + 48, TOP + 42, 'training', 's13 b', anchor='m')
    s.text(x + 48, TOP + 58, 'tasks', 's13 b', anchor='m')
    s.arrow([(x + 88, TOP), (x + 114, TOP)], cls='w2', head=8)
    lx, lw = x + 122, w - 122 - 136
    ly, lh = y + 64, h - 106
    s.text(lx + lw / 2, y + 44, '[ PySR or BasicSR ]', 's14 b', fill=C['ink2'], anchor='m')
    island_loop2(s, lx, ly, lw, lh)
    ox = lx + lw + 24
    s.arrow([(ox - 10, TOP), (ox + 6, TOP)], cls='w2', head=8)
    pareto2(s, ox + 8, TOP - 44, 98, 80)
    s.text(ox + 57, TOP + 54, 'Pareto front', 's12', fill=C['ink2'], anchor='m')


def population(s, x, y, w, h):
    s.panel(x, y, w, h, 'Population: reevaluate &amp; select', tsize=15)
    rows = [(3, 5, 3), (3, 3, 2), (3, 1, 1), (3, 0, 0)]
    fits = [0.66, 0.63, 0.60, 0.52]
    s.text(x + 14, y + 32, 'fitness ± err', 's12', fill=C['mute'])
    s.text(x + 106, y + 32, 'archive', 's12', fill=C['mute'])
    ends = []
    for k, (ni, npv, nn) in enumerate(rows):
        yy = y + 52 + k * 28
        n = ni + npv + nn
        bw = 78
        dropped = k == 3
        s.rect(x + 14, yy - 5, bw, 10, fill=C['panel2'], stroke='none', rx=3, cls='w1')
        s.rect(x + 14, yy - 5, bw * fits[k], 10, fill=C['line'] if dropped else C['ink2'], stroke='none', rx=3, cls='w1')
        e2 = 0.25 / math.sqrt(n)
        s.line(x + 14 + bw * (fits[k] - e2), yy, x + 14 + bw * (fits[k] + e2), yy, stroke=REEV, cls='w2')
        prog(s, x + 106, yy, dim=dropped, hi='sel' if k == 2 else None)
        e = seeds_ir(s, x + 138, yy, ni, npv + nn)
        ends.append((e, yy, nn))
    cut = y + 52 + 2.5 * 28
    s.line(x + 10, cut, x + 262, cut, stroke=C['mute'], cls='w1 dash')
    s.text(x + 266, cut + 4, 'cut', 's11 i', fill=C['mute'])
    B = sum(n for _, _, n in ends)
    cols = 3
    bx, by = x + w - 94, y + 44
    bw_, bh_ = cols * 11 + 12, math.ceil(B / cols) * 11 + 12
    s.text(bx + bw_ / 2, by - 6, f'budget B', 's12 b', fill=C['ink2'], anchor='m')
    s.rect(bx, by, bw_, bh_, fill=C['white'], stroke=REEV, rx=6, cls='w15')
    for k in range(B):
        s.circle(bx + 11 + (k % cols) * 11, by + 11 + (k // cols) * 11, 3.8, fill=REEV, stroke='none', cls='w1')
    pcx = bx + bw_ / 2
    pw = strategy_pill(s, pcx - 10, by + bh_ + 22, center=True)
    s.arrow([(pcx, by + bh_ + 2), (pcx, by + bh_ + 20)], color=REEV, cls='w15', head=6)
    sy = by + bh_ + 35
    for (ex, ey, n) in ends:
        if n <= 0:
            continue
        s.carrow((pcx - 10 - pw / 2, sy), (pcx - 10 - pw / 2 - 26, sy), (ex + 40, ey), (ex + 8, ey),
                 color=REEV, cls='w12', head=6)
        s.text(ex + 12, ey - 6, f'+{n}', 's12 b', fill=REEV)
    legend_seeds(s, x + 300, y + h - 14)


def initial_eval(s, x, y, w, h):
    s.panel(x, y, w, h, 'Initial evaluation', tsize=15)
    s.text(x + 14, y + 32, 'offspring', 's12', fill=C['mute'])
    for k, op in enumerate(['mut', 'sel', 'loss']):
        yy = y + 52 + k * 28
        prog(s, x + 14, yy, hi=op)
        seeds_ir(s, x + 46, yy, 3, 0)
        s.text(x + 88, yy + 4.5, ['0.61', '0.55', '0.48'][k], 's13', fill=C['ink2'])
    s.text(x + 134, y + 52, 'meta-', 's13', fill=C['ink2'])
    s.text(x + 134, y + 68, 'fitness', 's13', fill=C['ink2'])
    for k, lab in enumerate(['GT', 'R²', 'GT-R²']):
        act = k == 0
        s.chip(x + 134, y + 80 + k * 26, lab, C['ink2'] if act else C['white'], C['ink2'],
               C['white'] if act else C['ink2'], size=12, h=21, pad=8)


def mock_R():
    s = Svg(W, H)
    track(s)
    meta_mutation(s)
    evaluate(s)
    offspring_column(s, (MX1 + RX0) / 2, TOP - 6)
    population(s, RX0, YM1, 452, Y1 - YM1)
    initial_eval(s, RX0 + 452 + 20, YM1, RX1 - RX0 - 452 - 20, Y1 - YM1)
    for cx, cy, a in [(MX1 + 9, TOP, 0), (RX0 - 9, TOP, 0), (XT, (TOP + BOT) / 2, 90),
                      (RX0 + 462, BOT, 180), ((MX1 + RX0) / 2, BOT, 180)]:
        chevron(s, cx, cy, a, size=15, color=C['ink'])
    return s


MOCKS = {'R': mock_R}

if __name__ == '__main__':
    import cairosvg
    for k in sys.argv[1:] or list(MOCKS):
        s = MOCKS[k]()
        svg = s.svg(local=True)
        open(f'mock_{k}.svg', 'w').write(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f'mock_{k}.png', output_width=s.w * 2)
        print('wrote', k)
