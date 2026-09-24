"""Tiny SVG drawing kit for the Figure 1 mockups.

Only single-word SVG attributes + CSS classes are emitted, so the same markup
works in cairosvg (local preview) and inside a React-rendered design canvas.
"""
import math
import os
import re

# Atkinson Hyperlegible Next (Braille Institute): designed for legibility, incl. at small sizes.
FONT = "'Atkinson Hyperlegible Next', sans-serif"
FONT_DIR = os.path.expanduser('~/.local/share/fonts/atkinson-next')
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
FS = 1.05          # global font-size scale applied to every .sNN class
CAP = 0.68         # cap height / em of the font, used to centre text vertically

C = dict(
    mut='#d62728', loss='#2878c8', sel='#e4bc24', surv='#2c9b49',
    mut_d='#b3201f', loss_d='#1f62a6', sel_d='#9a7a08', surv_d='#23803b',
    mut_t='#fbe3e2', loss_t='#e0ecf8', sel_t='#fbf2cf', surv_t='#dff1e3',
    llm='#6b4fc0', llm_d='#4f389a', llm_t='#ebe6f8',
    ink='#1d2430', ink2='#343c4a', mute='#636d7d', line='#99a2af', faint='#cbd1da',
    panel='#f7f8fa', panel2='#e3e7ed', white='#ffffff', tab='#e2e7ee',
    gold='#c98a12', warn='#e07b24',
)
OPS = ['sel', 'mut', 'loss', 'surv']
OPNAME = dict(sel='Selection', mut='Mutation', loss='Loss', surv='Survival')

CSS = f"""
.t{{font-family:@FONT@}}
.mono{{font-family:@MONO@}}
.b{{font-weight:700}} .sb{{font-weight:700}} .i{{font-style:italic}}
.m{{text-anchor:middle}} .e{{text-anchor:end}}
{' '.join(f'.s{n}{{font-size:{n * FS:.2f}px}}' for n in (10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28))}
.ln{{fill:none;stroke-linecap:round;stroke-linejoin:round}}
.w05{{stroke-width:0.6}} .w1{{stroke-width:1}} .w12{{stroke-width:1.25}} .w15{{stroke-width:1.5}}
.w2{{stroke-width:2}} .w25{{stroke-width:2.5}} .w3{{stroke-width:3}} .w4{{stroke-width:4}} .w5{{stroke-width:5}}
.w1b{{stroke-width:1}} .w15b{{stroke-width:1.5}} .w18{{stroke-width:18}} .w14{{stroke-width:14}}
.dash{{stroke-dasharray:5 4}} .dot{{stroke-dasharray:1.5 3.5}} .ldash{{stroke-dasharray:9 6}}
"""


def css(local=False):
    return CSS.replace('@FONT@', FONT).replace('@MONO@', MONO)


def vc(size):
    """Baseline offset that vertically centres text of nominal `size` on a line."""
    return size * FS * CAP / 2


def weight_of(cls):
    words = cls.split()
    return 700 if ('b' in words or 'sb' in words) else 400


class Svg:
    def __init__(self, w, h, bg=C['white']):
        self.w, self.h = w, h
        self.els = []
        if bg:
            self.rect(0, 0, w, h, fill=bg, stroke='none', rx=0)

    # ------------------------------------------------------------ primitives
    def raw(self, s):
        self.els.append(s)

    def rect(self, x, y, w, h, fill='none', stroke=C['ink'], rx=8, cls='w15', opacity=None):
        op = f' opacity="{opacity}"' if opacity is not None else ''
        self.raw(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" class="{cls}"{op}/>')

    def circle(self, cx, cy, r, fill='none', stroke=C['ink'], cls='w15', opacity=None):
        op = f' opacity="{opacity}"' if opacity is not None else ''
        self.raw(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" class="{cls}"{op}/>')

    def line(self, x1, y1, x2, y2, stroke=C['ink'], cls='w15'):
        self.raw(f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" stroke="{stroke}" class="ln {cls}"/>')

    def path(self, d, stroke=C['ink'], fill='none', cls='w15'):
        self.raw(f'<path d="{d}" stroke="{stroke}" fill="{fill}" class="ln {cls}"/>')

    def poly(self, pts, fill=C['ink'], stroke='none', cls='w1'):
        p = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        self.raw(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" class="{cls}"/>')

    def text(self, x, y, s, cls='s14', fill=None, anchor=None):
        f = f' fill="{fill or C["ink"]}"'
        a = {'m': ' m', 'e': ' e'}.get(anchor, '')
        self.raw(f'<text x="{x:.1f}" y="{y:.1f}" class="t {cls}{a}"{f}>{s}</text>')

    def sub(self, base, sub, size=10):
        """Inline subscript markup, e.g. N_init."""
        return f'{base}<tspan dy="3" class="s{size}">{sub}</tspan><tspan dy="-3"> </tspan>'

    def lines(self, x, y, rows, cls='s14', lh=17, fill=None, anchor=None):
        for k, r in enumerate(rows):
            self.text(x, y + k * lh, r, cls, fill, anchor)

    # ------------------------------------------------------------ arrows
    def _head(self, tip, frm, color, size):
        dx, dy = tip[0] - frm[0], tip[1] - frm[1]
        L = math.hypot(dx, dy) or 1
        ux, uy = dx / L, dy / L
        px, py = -uy, ux
        b = (tip[0] - ux * size, tip[1] - uy * size)
        self.poly([tip, (b[0] + px * size * 0.5, b[1] + py * size * 0.5),
                   (b[0] - px * size * 0.5, b[1] - py * size * 0.5)], fill=color)
        return b

    def arrow(self, pts, color=C['ink'], cls='w2', head=9, both=False):
        pts = list(pts)
        b = self._head(pts[-1], pts[-2], color, head)
        # stop the shaft inside the head
        q = pts[:-1] + [((pts[-1][0] + b[0]) / 2, (pts[-1][1] + b[1]) / 2)]
        if both:
            b0 = self._head(pts[0], pts[1], color, head)
            q[0] = ((pts[0][0] + b0[0]) / 2, (pts[0][1] + b0[1]) / 2)
        d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in q)
        self.path(d, stroke=color, cls=cls)

    def carrow(self, p0, c1, c2, p3, color=C['ink'], cls='w2', head=9):
        b = self._head(p3, c2, color, head)
        e = ((p3[0] + b[0]) / 2, (p3[1] + b[1]) / 2)
        self.path(f'M{p0[0]:.1f},{p0[1]:.1f} C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {e[0]:.1f},{e[1]:.1f}',
                  stroke=color, cls=cls)

    def arc(self, cx, cy, r, a0, a1, color=C['ink'], cls='w2', head=None):
        """Arc from angle a0 to a1 (degrees, 0 = east, clockwise positive in screen coords)."""
        def pt(a):
            return (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
        p0, p1 = pt(a0), pt(a1)
        sweep = 1 if a1 > a0 else 0
        large = 1 if abs(a1 - a0) > 180 else 0
        if head:
            # pull the end back a little and put a tangent head there
            back = math.degrees(head * 0.6 / r) * (1 if a1 > a0 else -1)
            pe = pt(a1 - back)
            self.path(f'M{p0[0]:.1f},{p0[1]:.1f} A{r},{r} 0 {large} {sweep} {pe[0]:.1f},{pe[1]:.1f}', stroke=color, cls=cls)
            tang = math.radians(a1 + (90 if a1 > a0 else -90))
            frm = (p1[0] - math.cos(tang) * 10, p1[1] - math.sin(tang) * 10)
            self._head(p1, frm, color, head)
        else:
            self.path(f'M{p0[0]:.1f},{p0[1]:.1f} A{r},{r} 0 {large} {sweep} {p1[0]:.1f},{p1[1]:.1f}', stroke=color, cls=cls)

    # ------------------------------------------------------------ widgets
    def panel(self, x, y, w, h, title=None, tab=C['tab'], fill=C['panel'], stroke=C['ink2'],
              title_cls='s16 i b', rx=12, dashed=False, tab_x=None, letter=None, tsize=16):
        self.rect(x, y, w, h, fill=fill, stroke=stroke, rx=rx, cls='w15 dash' if dashed else 'w15')
        if title:
            extra = 24 if letter else 0
            tw = text_w(title, tsize, weight_of(title_cls), 'i' in title_cls.split()) + 26 + extra
            tx = tab_x if tab_x is not None else x + w / 2 - tw / 2
            self.rect(tx, y - 13, tw, 26, fill=tab, stroke=stroke, rx=13, cls='w12')
            if letter:
                self.badge(tx + 14, y, letter, r=9, size=12)
            self.text(tx + (tw + extra) / 2, y + vc(tsize), title, title_cls, anchor='m')

    def star(self, cx, cy, r, fill=C['gold'], stroke=C['ink']):
        pts = []
        for k in range(10):
            a = -math.pi / 2 + k * math.pi / 5
            rr = r if k % 2 == 0 else r * 0.45
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        self.poly(pts, fill=fill, stroke=stroke, cls='w1')

    def chip(self, x, y, s, fill, stroke, color=None, size=13, h=22, pad=9, cls='sb', center=False):
        w = text_w(s, size, weight_of(cls), 'i' in cls.split()) + 2 * pad
        if center:
            x = x - w / 2
        self.rect(x, y, w, h, fill=fill, stroke=stroke, rx=h / 2, cls='w12')
        self.text(x + w / 2, y + h / 2 + vc(size), s, f's{size} {cls}', fill=color or C['ink'], anchor='m')
        return w

    def opchip(self, x, y, op, size=13, h=22, label=None, center=False):
        return self.chip(x, y, label or OPNAME[op], C[op + '_t'], C[op], C[op + '_d'], size=size, h=h, center=center)

    def llm(self, x, y, w, h, label='LLM', size=20):
        self.rect(x, y, w, h, fill=C['llm_t'], stroke=C['llm'], rx=12, cls='w2')
        self.text(x + w / 2, y + h / 2 + size * 0.36, label, f's{size} b', fill=C['llm_d'], anchor='m')
        self.sparkle(x + w - 13, y + 13, 7, C['llm'])

    def sparkle(self, cx, cy, r, color):
        k = r * 0.28
        pts = [(cx, cy - r), (cx + k, cy - k), (cx + r, cy), (cx + k, cy + k),
               (cx, cy + r), (cx - k, cy + k), (cx - r, cy), (cx - k, cy - k)]
        self.poly(pts, fill=color)

    def code_card(self, x, y, w, h, ops=OPS, hi=None, dim=False, shadow=False, header=True,
                  bars=2, fixed_rows=0):
        """A light 'program' card: one block per operator; `hi` highlights one block."""
        if shadow:
            self.rect(x + 4, y + 4, w, h, fill=C['faint'], stroke=C['line'], rx=6, cls='w1')
        self.rect(x, y, w, h, fill=C['white'], stroke=C['ink2'], rx=6, cls='w12')
        top = y + 6
        if header:
            for k in range(3):
                self.circle(x + 8 + k * 7, y + 7, 2.2, fill=C['line'], stroke='none', cls='w1')
            top = y + 14
        n = len(ops) + fixed_rows
        bh = (y + h - 5 - top) / max(n, 1)
        row = 0
        for op in ops:
            by = top + row * bh
            if hi == op:
                self.rect(x + 3, by, w - 6, bh - 1.5, fill=C[op + '_t'], stroke=C[op], rx=3, cls='w1')
            col = C['line'] if dim else C[op]
            lh = min(4.0, bh / (bars + 1.6))
            # single-bar cards: centre the bar in its slot so a highlight box sits evenly around it
            bar_y = by + (bh - 1.5 - lh) / 2 if bars == 1 else by + bh * 0.22
            self.rect(x + 7, bar_y, (w - 14) * 0.42, lh, fill=col, stroke='none', rx=1.5, cls='w1')
            for k in range(1, bars):
                frac = [0.72, 0.55, 0.64][k % 3]
                self.rect(x + 13, bar_y + k * (lh + 2.2), (w - 20) * frac, lh,
                          fill=C['faint'], stroke='none', rx=1.5, cls='w1')
            row += 1
        for f in range(fixed_rows):
            by = top + row * bh
            self.rect(x + 7, by + bh * 0.3, (w - 14) * 0.6, 3.5, fill=C['faint'], stroke='none', rx=1.5, cls='w1')
            row += 1

    def task_tile(self, x, y, w, h, kind=0, stroke=C['line']):
        self.rect(x, y, w, h, fill=C['white'], stroke=stroke, rx=4, cls='w1')
        fns = [lambda u: 0.5 + 0.35 * math.sin(6 * u),
               lambda u: 0.15 + 0.7 * u * u,
               lambda u: 0.85 - 0.7 * math.exp(-4 * u) * 0 - 0.6 * u + 0.25 * math.sin(9 * u) * u,
               lambda u: 0.2 + 0.6 / (1 + math.exp(-10 * (u - 0.5))),
               lambda u: 0.8 * math.exp(-3 * u) + 0.1,
               lambda u: 0.5 + 0.3 * math.cos(4 * u) * (1 - u)]
        f = fns[kind % len(fns)]
        px = lambda u: x + 5 + u * (w - 10)
        py = lambda v: y + h - 5 - v * (h - 10)
        d = 'M' + ' L'.join(f'{px(u / 20):.1f},{py(f(u / 20)):.1f}' for u in range(21))
        self.path(d, stroke=C['ink2'], cls='w12')
        rnd = [0.07, 0.31, 0.52, 0.66, 0.83, 0.2, 0.43, 0.94]
        jit = [0.06, -0.05, 0.04, -0.07, 0.05, -0.03, 0.07, -0.04]
        for u, j in zip(rnd, jit):
            self.circle(px(u), py(f(u) + j), 1.7, fill=C['mute'], stroke='none', cls='w1')

    def seeds(self, x, y, n, n_new=0, r=3.6, gap=9.5, color=C['ink2'], new_color=None):
        for k in range(n + n_new):
            if k < n:
                self.circle(x + k * gap, y, r, fill=color, stroke='none', cls='w1')
            else:
                self.circle(x + k * gap, y, r, fill=C['white'], stroke=new_color or color, cls='w12')

    def tree(self, x, y, s=1.0, color=C['ink2'], fill=C['white'], hi=None):
        """Small expression-tree glyph with root at (x, y)."""
        nodes = [(0, 0), (-14, 16), (14, 16), (-22, 32), (-6, 32), (20, 32)]
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
        P = [(x + a * s, y + b * s) for a, b in nodes]
        for a, b in edges:
            self.line(*P[a], *P[b], stroke=color, cls='w1')
        for k, (px, py) in enumerate(P):
            f = hi if (hi and k in (2, 5)) else fill
            self.circle(px, py, 4.2 * s, fill=f, stroke=color, cls='w1')

    def check(self, x, y, s=7, color=C['surv']):
        self.path(f'M{x - s:.1f},{y:.1f} L{x - s * 0.3:.1f},{y + s * 0.7:.1f} L{x + s:.1f},{y - s * 0.8:.1f}',
                  stroke=color, cls='w25')

    def cross(self, x, y, s=6, color=C['mut']):
        self.path(f'M{x - s:.1f},{y - s:.1f} L{x + s:.1f},{y + s:.1f} M{x + s:.1f},{y - s:.1f} L{x - s:.1f},{y + s:.1f}',
                  stroke=color, cls='w25')

    def badge(self, cx, cy, label, r=12, fill=C['ink'], color=C['white'], size=14):
        self.circle(cx, cy, r, fill=fill, stroke='none', cls='w1')
        self.text(cx, cy + size * 0.36, label, f's{size} b', fill=color, anchor='m')

    def gear(self, cx, cy, r, color=C['ink2'], teeth=8):
        pts = []
        for k in range(teeth * 2):
            a = math.pi * k / teeth
            rr = r if k % 2 == 0 else r * 0.78
            for da in (-0.18, 0.18):
                pts.append((cx + rr * math.cos(a + da * math.pi / teeth * 2),
                            cy + rr * math.sin(a + da * math.pi / teeth * 2)))
        self.poly(pts, fill=color)
        self.circle(cx, cy, r * 0.35, fill=C['white'], stroke='none', cls='w1')

    # ------------------------------------------------------------ output
    def svg(self, inline_css=True, local=False):
        style = f'<style>{css(local)}</style>' if inline_css else ''
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}">{style}' + ''.join(self.els) + '</svg>')


_FONTS = {}


def _font(weight, italic):
    key = (weight, italic)
    if key not in _FONTS:
        path = os.path.join(FONT_DIR, f"AtkinsonHyperlegibleNext-{weight}{'Italic' if italic else ''}.ttf")
        try:
            from PIL import ImageFont
            _FONTS[key] = ImageFont.truetype(path, 100)
        except (ImportError, OSError):
            _FONTS[key] = None
    return _FONTS[key]


def text_w(s, size, weight=400, italic=False):
    """Rendered width of `s` at nominal `size` (after the global FS scale), measured from the font."""
    s = re.sub(r'<[^>]+>', '', s)
    s = s.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    f = _font(weight, italic)
    if f is not None:
        return f.getlength(s) / 100 * size * FS
    # fallback: rough Helvetica estimate
    wide = sum(1 for ch in s if ch in 'mwMW@%')
    narrow = sum(1 for ch in s if ch in 'iljtf.,:;|!\' ()')
    caps = sum(1 for ch in s if ch.isupper())
    n = len(s)
    return size * FS * (0.52 * (n - wide - narrow - caps) + 0.82 * wide + 0.28 * narrow + 0.66 * caps)
