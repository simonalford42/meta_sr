"""Build Figure 1 (meta-evolution overview) as an Inkscape-friendly SVG plus a PDF.

The layout lives in mockups12.py (final design round); the other modules are the
helpers it imports.  This script re-assembles that layout with one labelled
Inkscape layer per block and every CSS class inlined as a style attribute, so
the SVG can be hand-aligned in Inkscape.

    python figures/fig1_overview/build_fig1_overview.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cairosvg  # noqa: E402

import svgkit  # noqa: E402
from svgkit import Svg, C  # noqa: E402
import mockups12 as L  # noqa: E402

FONT = 'Helvetica, Arial, sans-serif'
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"


def class_styles():
    css = svgkit.CSS.replace('@FONT@', FONT).replace('@MONO@', MONO)
    return {name: body.strip().rstrip(';') for name, body in re.findall(r'\.([\w-]+)\{([^}]*)\}', css)}


def inline_classes(svg):
    """Replace class="..." with style="..."; drop presentation attributes the style overrides
    (in CSS the class rule wins over the attribute, so the rendering is unchanged)."""
    styles = class_styles()

    def repl(m):
        tag, rest, close = m.group(1), m.group(2), m.group(3)
        attrs = re.findall(r'([\w:-]+)="([^"]*)"', rest)
        classes = next(v for k, v in attrs if k == 'class').split()
        props = {}
        for c in classes:
            for decl in styles[c].split(';'):
                if decl.strip():
                    k, v = decl.split(':', 1)
                    props[k.strip()] = v.strip()
        kept = ''.join(f' {k}="{v}"' for k, v in attrs if k != 'class' and k not in props)
        style = ';'.join(f'{k}:{v}' for k, v in props.items())
        return f'<{tag}{kept} style="{style}"{close}>'

    return re.sub(r'<(\w+)(\s[^<>]*?\bclass="[^"]*"[^<>]*?)(/?)>', repl, svg)


def build():
    s = Svg(L.W, L.H)
    s.els = ['<g id="background" inkscape:label="Background" inkscape:groupmode="layer" '
             'sodipodi:insensitive="true">' + ''.join(s.els) + '</g>']

    def layer(lid, label, fn, *a, **kw):
        i = len(s.els)
        fn(*a, **kw)
        block = s.els[i:]
        del s.els[i:]
        s.els.append(f'<g id="{lid}" inkscape:label="{label}" inkscape:groupmode="layer">' + ''.join(block) + '</g>')

    gx = (L.MX1 + L.RX0) / 2
    iw = 194
    rw = L.RX1 - L.RX0 - iw - 20
    layer('outer_loop', 'Outer loop', L.track, s)
    layer('meta_mutation', 'Meta-mutation', L.meta_mutation, s)
    layer('evaluate', 'Evaluate', L.evaluate, s)
    layer('offspring', 'Offspring', L.gap_column, s, gx, L.TOP, 'offspring', his=('mut', 'sel', 'loss'))
    layer('meta_selection', 'Meta-selection', L.gap_column, s, gx, L.BOT, 'meta-selection', picked=(1,),
          label_above=True)
    layer('reevaluation', 'Reevaluation', L.reevaluation, s, L.RX0, L.YM1, rw, L.Y1 - L.YM1)
    layer('initial_evaluation', 'Initial evaluation', L.initial_eval, s, L.RX0 + rw + 20, L.YM1, iw, L.Y1 - L.YM1)

    def arrowheads():
        mid = (L.TOP + L.BOT) / 2
        for cx, cy, a, sz in [(L.MX1 + 16, L.TOP, 0, 14), (L.RX0 - 16, L.TOP, 0, 14),
                              (L.XT, mid - 70, 90, 17), (L.XT, mid + 70, 90, 17),
                              (L.RX0 + rw + 10, L.BOT, 180, 17), (L.RX0 - 16, L.BOT, 180, 14),
                              (L.MX1 + 16, L.BOT, 180, 14)]:
            L.chevron(s, cx, cy, a, size=sz, color=C['ink'])
    layer('loop_arrowheads', 'Loop arrowheads', arrowheads)

    body = inline_classes(''.join(s.els))
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
            f'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" '
            f'width="{L.W}" height="{L.H}" viewBox="0 0 {L.W} {L.H}">\n{body}\n</svg>\n')


if __name__ == '__main__':
    svg = build()
    out_svg = HERE / 'fig1_overview.svg'
    out_pdf = HERE / 'fig1_overview.pdf'
    out_svg.write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(out_pdf))
    print('wrote', out_svg)
    print('wrote', out_pdf)
