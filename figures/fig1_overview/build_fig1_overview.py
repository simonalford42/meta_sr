"""Build Figure 1 (meta-evolution overview) as a PDF.

The layout lives in mockups12.py (final design round); the other modules are the
helpers it imports.  The figure is assembled with one labelled group per block and
every CSS class inlined as a style attribute, then cropped horizontally to the
drawing plus a small margin.

Text is set in Atkinson Hyperlegible Next, expected as static TTFs in
~/.local/share/fonts/atkinson-next/ (Google Fonts); cairo embeds it in the PDF.

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

MARGIN = 4          # horizontal white space kept on each side
X0 = L.MX0 - MARGIN
X1 = L.XT + 9 + MARGIN   # outer edge of the loop's 18 px halo


def class_styles():
    css = svgkit.css()
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
    w = X1 - X0
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
            f'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" '
            f'width="{w}" height="{L.H}" viewBox="{X0} 0 {w} {L.H}">\n{body}\n</svg>\n')


if __name__ == '__main__':
    out_pdf = HERE / 'fig1_overview.pdf'
    cairosvg.svg2pdf(bytestring=build().encode(), write_to=str(out_pdf))
    print('wrote', out_pdf)
