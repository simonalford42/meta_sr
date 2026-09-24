"""Round-6 Figure 1 (refined R)."""
import math
import sys
from svgkit import Svg, C, OPS, OPNAME, text_w
from mockups2 import chevron, task_stack, mode_grid
from mockups3 import tiny_tree, glyph_mutation, glyph_loss, seeds_ir, strategy_pill, legend_seeds, INIT, REEV
from mockups5 import robot, pareto2, glyph_selection, glyph_survival2, prog

W, H = 1100, 470
MX0, MX1 = 16, 300                 # meta-mutation column
RX0, RX1 = 360, 1040               # right two columns
Y0, YM0, YM1, Y1 = 22, 270, 290, 456
ECY = (Y0 + YM0) // 2              # centre of the Evaluate row
TOP, BOT = 98, 402                 # outer-loop lines
XT, RC = 1066, 18                  # ] -shaped turnaround
LOOP = C['ink2']


def llm_box(s, x, y, w, h):
    s.rect(x, y, w, h, fill=C['llm_t'], stroke=C['llm'], rx=12, cls='w2')
    robot(s, x + 15, y + h / 2 - 13, 1.15)
    s.text(x + w - 12, y + h / 2 + 7, 'LLM', 's20 b', fill=C['llm_d'], anchor='e')


def op_grid(s, x, y, active='sel', size=12, h=21, colw=80, rowh=26):
    for k, op in enumerate(['mut', 'sel', 'surv', 'loss']):
        xx, yy = x + (k % 2) * colw, y + (k // 2) * rowh
        if op == active:
            w = text_w(OPNAME[op], size) * 1.07 + 14
            s.rect(xx - 3, yy - 3, w + 6, h + 6, fill='none', stroke=C['ink'], rx=(h + 6) / 2, cls='w15')
            s.chip(xx, yy, OPNAME[op], C[op + '_t'], C[op], C[op + '_d'], size=size, h=h, pad=7)
        else:
            s.chip(xx, yy, OPNAME[op], C['white'], C[op], C[op + '_d'], size=size, h=h, pad=7, cls='')


def population_glyph(s, x, y, w, h):
    s.rect(x, y, w, h, fill=C['panel'], stroke=C['line'], rx=8, cls='w1')
    for r in range(2):
        for c in range(3):
            tiny_tree(s, x + w / 6 + c * w / 3, y + 10 + r * 30, 0.5)
    s.text(x + w / 2, y + h + 15, 'population', 's12 i', fill=C['ink2'], anchor='m')


def island_loop3(s, x, y, w, h):
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
    sgx, sgy = x0 + 10 - 45, y + 22
    glyph_selection(s, sgx, sgy)
    glyph_mutation(s, x1 - 10 - 46, y + 24)
    glyph_loss(s, x1 - 10 - 52, y + h - 70, 104, 46)
    svx, svy = x0 + 10 - 50, y + h - 58
    glyph_survival2(s, svx, svy)
    # the island's population sits in the middle: sampled by selection, updated by survival
    pw, ph = 112, 62
    px, py = cx - pw / 2, y + r - ph / 2 - 6
    population_glyph(s, px, py, pw, ph)
    s.arrow([(px - 2, py + 14), (sgx + 94, sgy + 20)], color=C['sel_d'], cls='w15', head=6)
    s.arrow([(svx + 106, svy + 14), (px - 2, py + ph - 12)], color=C['surv_d'], cls='w15', head=6)


def track(s):
    xl = 160
    d = (f'M{xl},{TOP} L{XT - RC},{TOP} A{RC},{RC} 0 0 1 {XT},{TOP + RC} L{XT},{BOT - RC} '
         f'A{RC},{RC} 0 0 1 {XT - RC},{BOT} L{xl},{BOT} Z')
    s.path(d, stroke=LOOP, cls='w3')


def meta_mutation(s):
    x, y, w, h = MX0, Y0, MX1 - MX0, Y1 - Y0
    s.panel(x, y, w, h, 'Meta-mutation', tab=C['llm_t'])
    # LLM -> child -> validate -> out (on the top loop line)
    llm_box(s, x + 12, TOP - 29, 104, 58)
    s.arrow([(x + 118, TOP), (x + 130, TOP)], cls='w2', head=7)
    s.code_card(x + 132, TOP - 25, 36, 50, header=False, bars=1, hi='sel')
    s.arrow([(x + 170, TOP), (x + 182, TOP)], cls='w2', head=7)
    s.rect(x + 184, TOP - 13, 84, 26, fill=C['surv_t'], stroke=C['surv'], rx=13, cls='w12')
    s.check(x + 199, TOP, 5.5)
    s.text(x + 210, TOP + 4.5, 'validate', 's13 b', fill=C['surv_d'])
    s.arrow([(x + 270, TOP), (x + w - 2, TOP)], cls='w2', head=6)
    # prompt
    px, py, pw = x + 14, TOP + 50, w - 28
    ph = BOT - 50 - py
    s.arrow([(x + 64, py - 2), (x + 64, TOP + 31)], cls='w2', head=8)
    s.rect(px, py, pw, ph, fill=C['white'], stroke=C['line'], rx=8, cls='w12')
    s.text(px + 12, py + 24, 'Prompt', 's16 b')
    s.text(px + 12, py + 54, 'mode', 's13', fill=C['ink2'])
    mode_grid(s, px + 78, py + 40, size=12, h=21, rowh=26, colw=84)
    s.text(px + 12, py + 112, 'operator', 's13', fill=C['ink2'])
    op_grid(s, px + 78, py + 98, 'sel', size=12, h=21, colw=84, rowh=26)
    s.text(px + 12, py + 168, 'execution', 's13', fill=C['ink2'])
    s.text(px + 12, py + 184, 'feedback', 's13', fill=C['ink2'])
    pareto2(s, px + 78, py + 152, pw - 92, ph - 164, labels=False, check=False)
    # parents: meta-selection from the population (on the bottom loop line), centred
    n, cw, chh, gap = 4, 26, 38, 16
    tot = n * cw + (n - 1) * gap
    cx0 = x + (w - tot) / 2
    for k in range(n):
        cx = cx0 + k * (cw + gap)
        picked = k in (0, 2)
        if picked:
            s.rect(cx - 4, BOT - chh / 2 - 4, cw + 8, chh + 8, fill='none', stroke=C['llm'], rx=6, cls='w2')
            s.arrow([(cx + cw / 2, BOT - chh / 2 - 6), (cx + cw / 2, py + ph + 2)], cls='w2', head=7)
        s.code_card(cx, BOT - chh / 2, cw, chh, header=False, bars=1, dim=not picked)
    s.text(x + w / 2, BOT + chh / 2 + 24, 'meta-selection', 's14 b', fill=C['ink2'], anchor='m')


def offspring_column(s, cx, cy):
    ops = ['mut', 'sel', 'loss']
    cw, ch, gap = 26, 34, 6
    top = cy - (3 * ch + 2 * gap) / 2
    for k, op in enumerate(ops):
        s.code_card(cx - cw / 2, top + k * (ch + gap), cw, ch, header=False, bars=1, hi=op)
    s.text(cx, top + 3 * ch + 2 * gap + 16, 'offspring', 's12 b', anchor='m')


def evaluate(s):
    x, y, w, h = RX0, Y0, RX1 - RX0, YM0 - Y0
    s.panel(x, y, w, h, 'Evaluate: inner SR loop')
    task_stack(s, x + 16, ECY - 30, w=52, h=36, label=False, lab2=False)
    s.text(x + 48, ECY + 42, 'training', 's13 b', anchor='m')
    s.text(x + 48, ECY + 58, 'tasks', 's13 b', anchor='m')
    s.arrow([(x + 88, ECY), (x + 114, ECY)], cls='w2', head=8)
    lx, lw = x + 122, w - 122 - 130
    ly, lh = y + 64, h - 100
    s.text(lx + lw / 2, y + 44, 'PySR or BasicSR', 's14 b', fill=C['ink2'], anchor='m')
    island_loop3(s, lx, ly, lw, lh)
    ox = lx + lw + 22
    s.arrow([(ox - 10, ECY), (ox + 6, ECY)], cls='w2', head=8)
    pareto2(s, ox + 8, ECY - 42, 94, 76, labels=False)
    s.text(ox + 55, ECY + 52, 'Pareto front', 's12', fill=C['ink2'], anchor='m')


def reevaluation(s, x, y, w, h):
    s.panel(x, y, w, h, 'Reevaluation', tsize=15)
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
    s.text(bx + bw_ / 2, by - 6, 'budget B', 's12 b', fill=C['ink2'], anchor='m')
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
    s.text(x + 136, y + 52, 'meta-', 's13', fill=C['ink2'])
    s.text(x + 136, y + 68, 'fitness', 's13', fill=C['ink2'])
    for k, lab in enumerate(['GT', 'R²', 'GT-R²']):
        act = k == 0
        s.chip(x + 136, y + 80 + k * 26, lab, C['ink2'] if act else C['white'], C['ink2'],
               C['white'] if act else C['ink2'], size=12, h=21, pad=8)


def mock_S():
    s = Svg(W, H)
    track(s)
    meta_mutation(s)
    evaluate(s)
    offspring_column(s, (MX1 + RX0) / 2, TOP + 10)
    rw = 458
    reevaluation(s, RX0, YM1, rw, Y1 - YM1)
    initial_eval(s, RX0 + rw + 20, YM1, RX1 - RX0 - rw - 20, Y1 - YM1)
    for cx, cy, a in [(XT, (TOP + BOT) / 2, 90), (RX0 + rw + 10, BOT, 180), ((MX1 + RX0) / 2, BOT, 180)]:
        chevron(s, cx, cy, a, size=17, color=C['ink'])
    return s


MOCKS = {'S': mock_S}

if __name__ == '__main__':
    import cairosvg
    for k in sys.argv[1:] or list(MOCKS):
        s = MOCKS[k]()
        svg = s.svg(local=True)
        open(f'mock_{k}.svg', 'w').write(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f'mock_{k}.png', output_width=s.w * 2)
        print('wrote', k)
